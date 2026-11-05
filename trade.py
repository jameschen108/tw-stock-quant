# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""虛擬下單：使用者自己用虛擬資金買賣，照台股的交易時段成交，帳只追加不改寫。

用法:
  uv run trade.py                            # 先撮合到期的預約單，再列目前帳戶的現金、持股、預約單與損益
  uv run trade.py open --capital 1000000     # 開新帳戶（舊帳戶保留，之後顯示新的）
  uv run trade.py buy 2330 1000              # 買 1 張；零股寫 1–999 股
  uv run trade.py sell 2330 500
  uv run trade.py cancel O3                  # 取消還沒成交的預約單

規則（照證交所的交易時段；國定假日看證交所報價的日期是不是今天）:
  - 盤中 09:00–13:30：整股與盤中零股都用即時報價的成交價成交（realtime.quotes，延遲約 5 秒）；
    今天還沒有成交時，買用最佳賣價、賣用最佳買價，沒有對手單（例如漲停鎖死）就不能下單
  - 盤後定價 14:00–14:30（整股）、盤後零股 13:40–14:30：用今天的收盤價成交
  - 其他時間（盤前、13:30 之後不在上述時段、晚上、週末、假日）：變成預約單，下一個交易日用開盤價成交。
    下單時用最後價格當參考價先檢查、保留現金（買）或股數（賣）；開盤後撮合時用實際開盤價重算，
    漲停開盤買不到、跌停開盤賣不掉、開盤沒有成交或現金不夠，就記成未成交。還沒成交前可以取消
  - 整股是 1,000 股的倍數，零股 1–999 股（和交易所一樣，不能混在同一筆）
  - 成本：同 strategy.DEFAULTS["cost"]（手續費 0.1425%、每筆最低 20 元）；賣出證交稅股票 0.3%、ETF（00 開頭）0.1%
  - 漲停買不到、跌停賣不掉：即時報價看證交所給的漲跌停價；用資料表的價格時看是否達前一天收盤 ±9.5%（同 strategy._limits）
  - 現金不夠不能買，持股不夠不能賣（不做融資融券、不能放空）；交割不分 T+2，成交當下就扣款入帳
  - 持股成本用平均成本（含買進手續費）；賣出時的已實現損益 = 賣得金額 − 手續費 − 稅 − 平均成本 × 股數
  - 除權息：除權息日前一天收盤還持有的，每股（前收盤 − 參考價）當成現金入帳（dividends；來源 price_events.parquet，
    晚上更新資料後才會出現；配股也折成現金、不增加股數；不扣股利所得稅與二代健保）。減資、變更面額不處理
  - 同期 0050：開帳戶時記下 0050 價格，之後用現在價格算同期報酬，除息用 前收盤 ÷ 參考價 還原（含息）

紀錄 data/paper/my_trades.jsonl（只追加不改寫），一行一筆：type = open（開帳戶）、trade（成交）、
order（預約單）、cancel（預約單取消或未成交）。
"""
import argparse
import json
from datetime import datetime, time
from pathlib import Path

import pandas as pd

import realtime
import strategy as S
from build_panel import OUT

LEDGER = OUT / "paper" / "my_trades.jsonl"
COST = S.DEFAULTS["cost"]
TAX_ETF = 0.001
LIMIT = 0.10 - 0.005            # 用資料表價格時的漲跌停近似，同 strategy._limits
DEFAULT_CAPITAL = 1_000_000
BENCH = S.ETF
LOT = 1000
OPEN, CLOSE = time(9, 0), time(13, 30)
AFTER = {"整股": (time(14, 0), time(14, 30)), "零股": (time(13, 40), time(14, 30))}
LIVE, AFTER_FIXED, AFTER_ODD, RESERVE = "盤中", "盤後定價", "盤後零股", "預約"


def costs(code: str, side: str, amount: float) -> tuple[float, float]:
    """(手續費, 證交稅)，元，四捨五入到整數。"""
    fee = max(COST["min_fee"], amount * COST["fee_rate"] * COST["fee_discount"]) if amount > 0 else 0.0
    tax = amount * (TAX_ETF if code.startswith("00") else COST["tax"]) if side == "sell" else 0.0
    return float(round(fee)), float(round(tax))


def lot_kind(shares: int) -> str:
    """整股或零股；不符合交易單位丟 ValueError。"""
    if shares <= 0 or int(shares) != shares:
        raise ValueError("股數要是正整數")
    if shares < LOT:
        return "零股"
    if shares % LOT:
        raise ValueError("整股要是 1,000 股的倍數，零股 1–999 股，要分開下")
    return "整股"


def phase(now: datetime, kind: str, traded_today: bool) -> str:
    """現在下單會怎麼成交：盤中、盤後定價、盤後零股，或預約（下一個交易日開盤）。
    traded_today：證交所報價的日期是今天（國定假日、盤前都不是）。"""
    if now.weekday() >= 5 or not traded_today:
        return RESERVE
    t = now.time()
    if OPEN <= t < CLOSE:
        return LIVE
    lo, hi = AFTER[kind]
    if lo <= t < hi:
        return AFTER_ODD if kind == "零股" else AFTER_FIXED
    return RESERVE


def last_close(code: str) -> dict | None:
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "name", "close", "ret"],
                         filters=[("code", "==", code),
                                  ("date", ">=", pd.Timestamp.today().normalize() - pd.Timedelta(days=60))])
    px = px.dropna(subset=["close"]).sort_values("date")
    if px.empty:
        return None
    r = px.iloc[-1]
    return {"code": code, "name": r["name"], "price": float(r["close"]), "date": r["date"],
            "up": bool(r["ret"] >= LIMIT), "down": bool(r["ret"] <= -LIMIT)}


def mark(code: str, live: dict | None = None) -> dict | None:
    """估值用的現價 {price, src, date}：證交所報價不比資料表舊就用報價（盤中是即時價，收盤後是今天收盤），
    否則用資料表最後收盤。查不到回傳 None。"""
    q = (live if live is not None else realtime.quotes([code])).get(code)
    c = last_close(code)
    if q and q.get("last") and q.get("date") is not None and (c is None or q["date"] >= c["date"]):
        return {"price": float(q["last"]), "src": f"證交所 {q['date']:%m-%d} {q.get('time') or ''}".strip(), "date": q["date"]}
    if c is None:
        return None
    return {"price": c["price"], "src": f"收盤價 {c['date']:%Y-%m-%d}", "date": c["date"]}


def quote(code: str, side: str = "buy", now: datetime | None = None, live: dict | None = None,
          kind: str = "整股") -> dict:
    """{price, src, name, phase, live, date, up, down}。phase 是預約時 price 只是參考價（最後的價格）。
    live 是已經查好的 realtime.quotes 結果（前端快取用）。"""
    now = now or datetime.now()
    q = (live if live is not None else realtime.quotes([code])).get(code)
    qd = q.get("date") if q else None
    ph = phase(now, kind, qd is not None and qd.date() == now.date())

    def _from_quote(px, src):
        return {"price": float(px), "src": src, "name": q.get("name"), "phase": ph, "live": ph == LIVE, "date": qd,
                "up": q.get("limit_up") is not None and px >= q["limit_up"],
                "down": q.get("limit_down") is not None and px <= q["limit_down"]}

    if ph == LIVE:
        px, src = q.get("last"), f"即時 {q.get('time') or ''}".strip()
        if q.get("indicative"):
            px = q.get("ask") if side == "buy" else q.get("bid")
            src += "（今天還沒成交，用最佳" + ("賣價" if side == "buy" else "買價") + "）"
        if px:
            return _from_quote(px, src)
        raise ValueError(f"現在沒有{'賣單' if side == 'buy' else '買單'}可以成交（可能{'漲停' if side == 'buy' else '跌停'}）")
    if ph in (AFTER_FIXED, AFTER_ODD):
        if not q.get("last") or q.get("indicative"):
            raise ValueError("今天沒有收盤價，盤後交易不能成交")
        return _from_quote(q["last"], f"收盤價 {qd:%Y-%m-%d}（{ph}）")
    c = last_close(code)
    if qd is not None and q.get("last") and not q.get("indicative") and (c is None or qd > c["date"]):
        r = _from_quote(q["last"], f"參考價：{qd:%Y-%m-%d} 收盤（證交所）")
    elif c is not None:
        r = {"price": c["price"], "src": f"參考價：{c['date']:%Y-%m-%d} 收盤", "name": c["name"], "phase": ph,
             "live": False, "date": c["date"]}
    else:
        raise ValueError(f"查不到 {code} 的行情")
    return {**r, "up": False, "down": False}      # 預約單的漲跌停到開盤撮合時才看


# ---------------------------------------------------------------- 帳本

def load(path: Path = LEDGER) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(ln))
        except ValueError:      # 空行，或寫到一半中斷的最後一行
            continue
    return rows


def _append(rec: dict, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def _ts(now: datetime | None) -> str:
    return (now or datetime.now()).isoformat(timespec="seconds")


def current(rows: list[dict]) -> str | None:
    """最後開的帳戶。"""
    opened = [r["account"] for r in rows if r["type"] == "open"]
    return opened[-1] if opened else None


def open_account(capital: float = DEFAULT_CAPITAL, path: Path = LEDGER, now: datetime | None = None,
                 bench: float | None = None) -> dict:
    if capital <= 0:
        raise ValueError("資金要大於 0")
    rows = load(path)
    name = f"帳戶 {sum(r['type'] == 'open' for r in rows) + 1}"
    if bench is None:
        m = mark(BENCH)
        bench = m["price"] if m else None
    return _append({"type": "open", "ts": _ts(now), "account": name, "capital": float(capital), "bench_price": bench}, path)


def dividends(rows: list[dict], account: str | None = None) -> pd.DataFrame:
    """除權息入帳：除權息日前一天收盤還持有的股數 ×（前收盤 − 參考價），當成現金。
    配股也折成現金（除權後的總價值相同，只是不增加股數）；不扣股利所得稅與二代健保。"""
    cols = ["date", "code", "name", "shares", "per_share", "amount"]
    account = account or current(rows)
    op = next((r for r in rows if r["type"] == "open" and r["account"] == account), None)
    tr = pd.DataFrame([r for r in rows if r["type"] == "trade" and r["account"] == account])
    path = OUT / "price_events.parquet"
    if op is None or tr.empty or not path.exists():
        return pd.DataFrame(columns=cols)
    ev = pd.read_parquet(path, columns=["date", "code", "name", "prev_close", "ref", "kind"],
                         filters=[("code", "in", sorted(set(tr["code"]))), ("kind", "==", "div"),
                                  ("date", ">", pd.Timestamp(op["ts"]).normalize())])
    tr["day"] = pd.to_datetime(tr["ts"])
    tr["dsh"] = tr["shares"].where(tr["side"] == "buy", -tr["shares"])
    out = []
    for e in ev.sort_values("date").itertuples():
        sh = tr.loc[(tr["code"] == e.code) & (tr["day"] < e.date), "dsh"].sum()     # 除權息當天買的不算
        if sh > 0:
            per = e.prev_close - e.ref
            out.append({"date": e.date, "code": e.code, "name": e.name, "shares": int(sh), "per_share": per,
                        "amount": float(round(per * sh))})
    return pd.DataFrame(out, columns=cols)


def bench_factor(since: str | pd.Timestamp) -> pd.Series:
    """0050 除息還原：開帳戶之後每次除息的 前收盤 ÷ 參考價，累乘；索引是除息日。"""
    path = OUT / "price_events.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    ev = pd.read_parquet(path, columns=["date", "code", "prev_close", "ref", "kind"],
                         filters=[("code", "==", BENCH), ("date", ">", pd.Timestamp(since).normalize())])
    ev = ev[ev["ref"] > 0].sort_values("date")
    return (ev["prev_close"] / ev["ref"]).set_axis(ev["date"]).cumprod()


def state(rows: list[dict], account: str | None = None, divs: pd.DataFrame | None = None) -> dict:
    """帳戶的現金、持股（股數、平均成本）、已實現損益、成交紀錄、還沒成交的預約單（orders）與取消／未成交紀錄；
    divs（dividends 的結果）有給就把配息加進現金。reserved = 預約買單保留的現金，pending_sell = 預約賣單保留的股數。"""
    account = account or current(rows)
    st = {"account": account, "capital": 0.0, "cash": 0.0, "realized": 0.0, "fees": 0.0, "dividends": 0.0,
          "opened": None, "bench_price": None, "pos": {}, "trades": [], "orders": {}, "cancels": []}
    for r in rows:
        if r.get("account") != account:
            continue
        if r["type"] == "open":
            st.update(capital=r["capital"], cash=r["capital"], opened=r["ts"], bench_price=r.get("bench_price"))
            continue
        if r["type"] == "order":
            st["orders"][r["id"]] = r
            continue
        if r["type"] == "cancel":
            o = st["orders"].pop(r["order_id"], None)
            st["cancels"].append({**(o or {}), **r})
            continue
        st["orders"].pop(r.get("order_id"), None)
        p = st["pos"].setdefault(r["code"], {"name": r.get("name"), "shares": 0, "cost": 0.0})
        if r["side"] == "buy":
            st["cash"] -= r["amount"] + r["fee"]
            p["shares"] += r["shares"]
            p["cost"] += r["amount"] + r["fee"]
        else:
            avg = p["cost"] / p["shares"]
            st["cash"] += r["amount"] - r["fee"] - r["tax"]
            st["realized"] += r["amount"] - r["fee"] - r["tax"] - avg * r["shares"]
            p["shares"] -= r["shares"]
            p["cost"] -= avg * r["shares"]
        st["fees"] += r["fee"] + r["tax"]
        st["trades"].append(r)
    st["pos"] = {c: p for c, p in st["pos"].items() if p["shares"] > 0}
    if divs is not None and len(divs):
        st["dividends"] = float(divs["amount"].sum())
        st["cash"] += st["dividends"]
    st["reserved"] = sum(o["est_amount"] + o["est_fee"] for o in st["orders"].values() if o["side"] == "buy")
    st["pending_sell"] = {}
    for o in st["orders"].values():
        if o["side"] == "sell":
            st["pending_sell"][o["code"]] = st["pending_sell"].get(o["code"], 0) + o["shares"]
    return st


def check(st: dict, code: str, side: str, shares: int, q: dict) -> dict:
    """試算一筆單：金額、手續費、稅；不能成交時丟 ValueError（原因給使用者看）。現金與股數先扣掉預約單保留的部分。"""
    if st["account"] is None:
        raise ValueError("還沒有帳戶")
    if side not in ("buy", "sell"):
        raise ValueError("買或賣")
    lot_kind(shares)
    if side == "buy" and q["up"]:
        raise ValueError("漲停，買不到")
    if side == "sell" and q["down"]:
        raise ValueError("跌停，賣不掉")
    amount = round(q["price"] * shares)
    fee, tax = costs(code, side, amount)
    cash = st["cash"] - st.get("reserved", 0.0)
    if side == "buy" and amount + fee > cash + 1e-6:
        held_back = f"（另有 {st['reserved']:,.0f} 元保留給預約單）" if st.get("reserved") else ""
        raise ValueError(f"現金不夠：需要 {amount + fee:,.0f} 元，可用 {cash:,.0f} 元{held_back}")
    held = st["pos"].get(code, {}).get("shares", 0) - st.get("pending_sell", {}).get(code, 0)
    if side == "sell" and shares > held:
        raise ValueError(f"持股不夠：可賣 {held:,} 股" + ("（其他已掛預約賣單）" if st.get("pending_sell", {}).get(code) else ""))
    net = -(amount + fee) if side == "buy" else amount - fee - tax
    return {"amount": float(amount), "fee": fee, "tax": tax, "net": float(net)}


def place(code: str, side: str, shares: int, path: Path = LEDGER, now: datetime | None = None,
          q: dict | None = None) -> dict:
    """下單並寫入帳本：可以成交的時段直接成交（type=trade），其他時間記成預約單（type=order）。q 沒給就現查價格。"""
    rows = load(path)
    st = state(rows, divs=dividends(rows))
    q = q or quote(code, side, now=now, kind=lot_kind(shares))
    c = check(st, code, side, shares, q)
    base = {"ts": _ts(now), "account": st["account"], "code": code, "name": q.get("name"), "side": side,
            "shares": int(shares)}
    if q.get("phase") == RESERVE:
        n = sum(r["type"] == "order" for r in rows) + 1
        return _append({"type": "order", "id": f"O{n}", **base, "ref_price": q["price"], "ref_src": q["src"],
                        "est_amount": c["amount"], "est_fee": c["fee"], "est_tax": c["tax"]}, path)
    return _append({"type": "trade", **base, "phase": q.get("phase", LIVE), "price": q["price"], "src": q["src"],
                    "amount": c["amount"], "fee": c["fee"], "tax": c["tax"],
                    "cash_after": round(st["cash"] + c["net"], 2)}, path)


def cancel(order_id: str, path: Path = LEDGER, now: datetime | None = None, reason: str = "自己取消") -> dict:
    rows = load(path)
    o = next((r for r in rows if r["type"] == "order" and r["id"] == order_id), None)
    if o is None or order_id not in state(rows, o["account"])["orders"]:
        raise ValueError(f"沒有還沒成交的預約單 {order_id}")
    return _append({"type": "cancel", "ts": _ts(now), "account": o["account"], "order_id": order_id, "reason": reason}, path)


def opening(code: str, after: datetime, now: datetime, live: dict | None = None) -> dict | None:
    """after 之後的第一個開盤：{date, price, up, down}；還沒開盤（或資料、報價都還沒有那天）回傳 None。
    price 是 None 代表那天開盤沒有成交。先看 prices_daily，再看今天的證交所報價。"""
    first = after.date() if after.time() < OPEN else after.date() + pd.Timedelta(days=1)
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "open", "close"],
                         filters=[("code", "==", code), ("date", ">=", pd.Timestamp(first) - pd.Timedelta(days=15))])
    px = px.sort_values("date").reset_index(drop=True)
    hit = px.index[px["date"] >= pd.Timestamp(first)]
    if len(hit):
        r, prev = px.loc[hit[0]], px.loc[hit[0] - 1, "close"] if hit[0] > 0 else None
        op = float(r["open"]) if pd.notna(r["open"]) and r["open"] > 0 else None
        lim = lambda s: op is not None and prev is not None and s * (op / prev - 1) >= LIMIT
        return {"date": r["date"], "price": op, "up": lim(1), "down": lim(-1)}
    if now.date() < first or (now.date() == first and now.time() < OPEN) or now.weekday() >= 5:
        return None
    q = (live if live is not None else realtime.quotes([code])).get(code)
    if not q or q.get("date") is None or q["date"].date() != now.date() or now.date() < first:
        return None
    op = q.get("open")
    return {"date": q["date"], "price": op,
            "up": op is not None and q.get("limit_up") is not None and op >= q["limit_up"],
            "down": op is not None and q.get("limit_down") is not None and op <= q["limit_down"]}


def settle(path: Path = LEDGER, now: datetime | None = None, live: dict | None = None) -> list[dict]:
    """撮合已經到開盤的預約單：用開盤價成交，不能成交就記未成交。回傳這次新寫入的紀錄。"""
    now = now or datetime.now()
    rows = load(path)
    out = []
    pending = [r for r in rows if r["type"] == "order"]
    for o in pending:
        st = state(rows, o["account"], divs=dividends(rows, o["account"]))
        if o["id"] not in st["orders"]:
            continue
        op = opening(o["code"], datetime.fromisoformat(o["ts"]), now, live)
        if op is None:
            continue
        st["orders"].pop(o["id"])           # 撮合自己時不要再保留自己
        st["reserved"] -= (o["est_amount"] + o["est_fee"]) if o["side"] == "buy" else 0
        if o["side"] == "sell":
            st["pending_sell"][o["code"]] -= o["shares"]
        ts = datetime.combine(op["date"].date(), OPEN).isoformat(timespec="seconds")
        try:
            if op["price"] is None:
                raise ValueError("開盤沒有成交")
            q = {"price": op["price"], "up": op["up"], "down": op["down"]}
            c = check(st, o["code"], o["side"], o["shares"], q)
            rec = {"type": "trade", "ts": ts, "account": o["account"], "code": o["code"], "name": o.get("name"),
                   "side": o["side"], "shares": o["shares"], "phase": "預約單開盤", "order_id": o["id"],
                   "price": op["price"], "src": f"開盤價 {op['date']:%Y-%m-%d}（預約單 {o['id']}）",
                   "amount": c["amount"], "fee": c["fee"], "tax": c["tax"], "cash_after": round(st["cash"] + c["net"], 2)}
        except ValueError as e:
            rec = {"type": "cancel", "ts": ts, "account": o["account"], "order_id": o["id"], "reason": f"未成交：{e}"}
        out.append(_append(rec, path))
        rows.append(rec)
    return out


def max_shares(cash: float, price: float, code: str) -> int:
    """現金最多買得起幾股（含手續費）。"""
    if price <= 0 or cash <= 0:
        return 0
    n = int(cash // (price * (1 + COST["fee_rate"] * COST["fee_discount"])))
    while n > 0 and round(n * price) + costs(code, "buy", round(n * price))[0] > cash:
        n -= 1
    return n


def nav(rows: list[dict], account: str | None = None) -> pd.DataFrame:
    """帳戶每天收盤的總資產（現金 + 配息 + 持股 × 收盤價）與同期 0050（開帳戶時買進持有、含息），到最新資料日。
    週末或盤後的成交算在下一個交易日；資料還沒更新到的成交不畫。"""
    account = account or current(rows)
    op = next((r for r in rows if r["type"] == "open" and r["account"] == account), None)
    if op is None:
        return pd.DataFrame(columns=["nav", "bench"])
    tr = pd.DataFrame([r for r in rows if r["type"] == "trade" and r["account"] == account],
                      columns=["ts", "code", "side", "shares", "amount", "fee", "tax"])
    start = pd.Timestamp(op["ts"]).normalize()
    codes = sorted(set(tr["code"]) | {BENCH})
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "close"],
                         filters=[("code", "in", codes), ("date", ">=", start - pd.Timedelta(days=15))])
    close = px.pivot_table(index="date", columns="code", values="close").sort_index().ffill()
    days = close.index[close.index >= start]
    if days.empty:
        return pd.DataFrame(columns=["nav", "bench"])
    out = pd.DataFrame(index=days)
    cash = pd.Series(op["capital"], index=days)
    hold = pd.Series(0.0, index=days)
    if len(tr):
        tr["day"] = days[days.searchsorted(pd.to_datetime(tr["ts"]).dt.normalize()).clip(0, len(days) - 1)]
        tr = tr[pd.to_datetime(tr["ts"]).dt.normalize() <= days[-1]]
        buy = tr["side"] == "buy"
        tr["dcash"] = (-(tr["amount"] + tr["fee"])).where(buy, tr["amount"] - tr["fee"] - tr["tax"])
        tr["dsh"] = tr["shares"].where(buy, -tr["shares"])
        cash += tr.groupby("day")["dcash"].sum().reindex(days, fill_value=0).cumsum()
        sh = tr.pivot_table(index="day", columns="code", values="dsh", aggfunc="sum").reindex(days, fill_value=0).cumsum()
        hold = (sh * close.reindex(index=days, columns=sh.columns)).sum(axis=1)
    dv = dividends(rows, account)
    if len(dv):
        cash += dv.groupby("date")["amount"].sum().reindex(days, fill_value=0).cumsum()
    out["nav"] = cash + hold
    bp = op.get("bench_price")
    fac = bench_factor(op["ts"]).reindex(days).ffill().fillna(1.0)
    out["bench"] = close[BENCH].reindex(days) / bp * op["capital"] * fac if bp and BENCH in close else float("nan")
    return out


def holdings(st: dict, prices: dict[str, float]) -> pd.DataFrame:
    """持股明細；prices = {代號: 現價}，沒有的用平均成本（損益算 0）。"""
    rows = []
    for code, p in st["pos"].items():
        avg = p["cost"] / p["shares"]
        now = prices.get(code) or avg
        rows.append({"code": code, "name": p["name"], "shares": p["shares"], "avg_cost": avg, "price": now,
                     "value": now * p["shares"], "pnl": (now - avg) * p["shares"], "ret": now / avg - 1})
    return pd.DataFrame(rows, columns=["code", "name", "shares", "avg_cost", "price", "value", "pnl", "ret"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("open").add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    for s in ("buy", "sell"):
        p = sub.add_parser(s)
        p.add_argument("code")
        p.add_argument("shares", type=int)
    sub.add_parser("cancel").add_argument("order_id")
    a = ap.parse_args()
    for r in settle():
        print(f"預約單 {r['order_id']}：" + (f"{r['src']} 成交 {r['shares']:,} 股 @ {r['price']}" if r["type"] == "trade"
                                           else r["reason"]))
    try:
        if a.cmd == "open":
            r = open_account(a.capital)
            print(f"已開「{r['account']}」，資金 {r['capital']:,.0f} 元")
        elif a.cmd in ("buy", "sell"):
            r = place(a.code, a.cmd, a.shares)
            act = "買進" if a.cmd == "buy" else "賣出"
            if r["type"] == "order":
                print(f"預約單 {r['id']}：{act} {r['code']} {r['name']} {r['shares']:,} 股，下一個交易日開盤價成交"
                      f"（{r['ref_src']} {r['ref_price']}）")
            else:
                print(f"{act} {r['code']} {r['name']} {r['shares']:,} 股 @ {r['price']}（{r['src']}），"
                      f"手續費 {r['fee']:,.0f}、稅 {r['tax']:,.0f}，現金剩 {r['cash_after']:,.0f}")
        elif a.cmd == "cancel":
            cancel(a.order_id)
            print(f"已取消預約單 {a.order_id}")
    except ValueError as err:
        raise SystemExit(f"失敗：{err}")
    rows = load()
    st = state(rows, divs=dividends(rows))
    if st["account"] is None:
        print("還沒有帳戶（uv run trade.py open）")
        return
    live = realtime.quotes(list(st["pos"]))
    h = holdings(st, {c: m["price"] for c in st["pos"] if (m := mark(c, live))})
    total = st["cash"] + h["value"].sum()
    print(f"[{st['account']}] 現金 {st['cash']:,.0f}、持股市值 {h['value'].sum():,.0f}、總資產 {total:,.0f}"
          f"（{total / st['capital'] - 1:+.2%}），已實現 {st['realized']:,.0f}，配息 {st['dividends']:,.0f}，成本合計 {st['fees']:,.0f}")
    for r in h.itertuples():
        print(f"  {r.code} {r.name} {r.shares:,} 股  成本 {r.avg_cost:.2f}  現價 {r.price:.2f}  損益 {r.pnl:+,.0f}（{r.ret:+.2%}）")
    for o in st["orders"].values():
        print(f"  預約單 {o['id']}：{'買進' if o['side'] == 'buy' else '賣出'} {o['code']} {o['name']} {o['shares']:,} 股")


if __name__ == "__main__":
    main()
