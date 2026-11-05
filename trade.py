# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""虛擬下單：使用者自己用虛擬資金買賣，成交價用現在的股價，帳只追加不改寫。

用法:
  uv run trade.py                            # 目前帳戶的現金、持股與損益
  uv run trade.py open --capital 1000000     # 開新帳戶（舊帳戶保留，之後顯示新的）
  uv run trade.py buy 2330 1000              # 買 1000 股（1 張）；零股直接寫股數
  uv run trade.py sell 2330 500

規則:
  - 成交價：盤中（平日 09:00–13:35）用證交所即時報價的成交價（realtime.quotes，延遲約 5 秒）；
    今天還沒有成交時，買用最佳賣價、賣用最佳買價。盤後用最後收盤價成交（類似盤後定價交易）：
    證交所報價的日期比 prices_daily 新（收盤後到晚上更新資料前）就用報價的收盤價，否則用 prices_daily
    最後一天；盤中查不到今天的報價也用 prices_daily。紀錄寫明價格來源與日期
  - 成本：同 strategy.DEFAULTS["cost"]（手續費 0.1425%、每筆最低 20 元）；賣出證交稅股票 0.3%、ETF（00 開頭）0.1%
  - 漲停買不到、跌停賣不掉：即時報價看證交所給的漲跌停價；收盤價成交時看當天報酬是否達 ±9.5%（同 strategy._limits）
  - 現金不夠不能買，持股不夠不能賣（不做融資融券、不能放空）
  - 持股成本用平均成本（含買進手續費）；賣出時的已實現損益 = 賣得金額 − 手續費 − 稅 − 平均成本 × 股數
  - 除權息：除權息日前一天收盤還持有的，每股（前收盤 − 參考價）當成現金入帳（dividends；來源 price_events.parquet，
    晚上更新資料後才會出現；配股也折成現金、不增加股數；不扣股利所得稅與二代健保）。減資、變更面額不處理
  - 同期 0050：開帳戶時記下 0050 價格，之後用現在價格算同期報酬，除息用 前收盤 ÷ 參考價 還原（含息）

紀錄 data/paper/my_trades.jsonl（只追加不改寫），一行一筆：type = open（開帳戶）或 trade（成交）。
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import realtime
import strategy as S
from build_panel import OUT

LEDGER = OUT / "paper" / "my_trades.jsonl"
COST = S.DEFAULTS["cost"]
TAX_ETF = 0.001
LIMIT = 0.10 - 0.005            # 收盤價成交時的漲跌停近似，同 strategy._limits
DEFAULT_CAPITAL = 1_000_000
BENCH = S.ETF


def costs(code: str, side: str, amount: float) -> tuple[float, float]:
    """(手續費, 證交稅)，元，四捨五入到整數。"""
    fee = max(COST["min_fee"], amount * COST["fee_rate"] * COST["fee_discount"]) if amount > 0 else 0.0
    tax = amount * (TAX_ETF if code.startswith("00") else COST["tax"]) if side == "sell" else 0.0
    return float(round(fee)), float(round(tax))


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


def quote(code: str, side: str = "buy", now: datetime | None = None, live: dict | None = None) -> dict:
    """成交價與來源：{price, src, name, live, up, down}。live 是已經查好的 realtime.quotes 結果（前端快取用）。"""
    now = now or datetime.now()
    q = (live if live is not None else realtime.quotes([code])).get(code)
    qd = q.get("date") if q else None

    def _from_quote(px, src, is_live):
        return {"price": float(px), "src": src, "name": q.get("name"), "live": is_live, "date": qd,
                "up": q.get("limit_up") is not None and px >= q["limit_up"],
                "down": q.get("limit_down") is not None and px <= q["limit_down"]}

    if realtime.session_open(now) and qd is not None and qd.date() == now.date():
        px, src = q.get("last"), f"即時 {q.get('time') or ''}".strip()
        if q.get("indicative"):
            px = q.get("ask") if side == "buy" else q.get("bid")
            src += "（今天還沒成交，用最佳" + ("賣價" if side == "buy" else "買價") + "）"
        if px:
            return _from_quote(px, src, True)
        raise ValueError(f"現在沒有{'賣單' if side == 'buy' else '買單'}可以成交（可能{'漲停' if side == 'buy' else '跌停'}）")
    c = last_close(code)
    if not realtime.session_open(now) and qd is not None and q.get("last") and not q.get("indicative") \
            and (c is None or qd > c["date"]):
        return _from_quote(q["last"], f"收盤價 {qd:%Y-%m-%d}（證交所）", False)
    if c is None:
        raise ValueError(f"查不到 {code} 的行情")
    return {"price": c["price"], "src": f"收盤價 {c['date']:%Y-%m-%d}", "name": c["name"], "live": False, "date": c["date"],
            "up": c["up"], "down": c["down"]}


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
        try:
            bench = quote(BENCH, now=now)["price"]
        except ValueError:
            bench = None
    return _append({"type": "open", "ts": (now or datetime.now()).isoformat(timespec="seconds"), "account": name,
                    "capital": float(capital), "bench_price": bench}, path)


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
    """帳戶的現金、持股（股數、平均成本）、已實現損益與成交紀錄；divs（dividends 的結果）有給就把配息加進現金。"""
    account = account or current(rows)
    st = {"account": account, "capital": 0.0, "cash": 0.0, "realized": 0.0, "fees": 0.0, "dividends": 0.0,
          "opened": None, "bench_price": None, "pos": {}, "trades": []}
    for r in rows:
        if r.get("account") != account:
            continue
        if r["type"] == "open":
            st.update(capital=r["capital"], cash=r["capital"], opened=r["ts"], bench_price=r.get("bench_price"))
            continue
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
    return st


def check(st: dict, code: str, side: str, shares: int, q: dict) -> dict:
    """試算一筆單：金額、手續費、稅；不能成交時丟 ValueError（原因給使用者看）。"""
    if st["account"] is None:
        raise ValueError("還沒有帳戶")
    if side not in ("buy", "sell"):
        raise ValueError("買或賣")
    if shares <= 0 or int(shares) != shares:
        raise ValueError("股數要是正整數")
    if side == "buy" and q["up"]:
        raise ValueError("漲停，買不到")
    if side == "sell" and q["down"]:
        raise ValueError("跌停，賣不掉")
    amount = round(q["price"] * shares)
    fee, tax = costs(code, side, amount)
    if side == "buy" and amount + fee > st["cash"] + 1e-6:
        raise ValueError(f"現金不夠：需要 {amount + fee:,.0f} 元，現金 {st['cash']:,.0f} 元")
    held = st["pos"].get(code, {}).get("shares", 0)
    if side == "sell" and shares > held:
        raise ValueError(f"持股不夠：持有 {held:,} 股")
    net = -(amount + fee) if side == "buy" else amount - fee - tax
    return {"amount": float(amount), "fee": fee, "tax": tax, "net": float(net)}


def place(code: str, side: str, shares: int, path: Path = LEDGER, now: datetime | None = None,
          q: dict | None = None) -> dict:
    """下單並寫入帳本；q 沒給就現查價格。"""
    rows = load(path)
    st = state(rows, divs=dividends(rows))
    q = q or quote(code, side, now=now)
    c = check(st, code, side, shares, q)
    return _append({"type": "trade", "ts": (now or datetime.now()).isoformat(timespec="seconds"),
                    "account": st["account"], "code": code, "name": q.get("name"), "side": side, "shares": int(shares),
                    "price": q["price"], "src": q["src"], "amount": c["amount"], "fee": c["fee"], "tax": c["tax"],
                    "cash_after": round(st["cash"] + c["net"], 2)}, path)


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
    a = ap.parse_args()
    try:
        if a.cmd == "open":
            r = open_account(a.capital)
            print(f"已開「{r['account']}」，資金 {r['capital']:,.0f} 元")
        elif a.cmd in ("buy", "sell"):
            r = place(a.code, a.cmd, a.shares)
            print(f"{'買進' if a.cmd == 'buy' else '賣出'} {r['code']} {r['name']} {r['shares']:,} 股 @ {r['price']}（{r['src']}），"
                  f"手續費 {r['fee']:,.0f}、稅 {r['tax']:,.0f}，現金剩 {r['cash_after']:,.0f}")
    except ValueError as err:
        raise SystemExit(f"失敗：{err}")
    rows = load()
    st = state(rows, divs=dividends(rows))
    if st["account"] is None:
        print("還沒有帳戶（uv run trade.py open）")
        return
    prices = {}
    for c in st["pos"]:
        try:
            prices[c] = quote(c, "sell")["price"]
        except ValueError:
            pass
    h = holdings(st, prices)
    total = st["cash"] + h["value"].sum()
    print(f"[{st['account']}] 現金 {st['cash']:,.0f}、持股市值 {h['value'].sum():,.0f}、總資產 {total:,.0f}"
          f"（{total / st['capital'] - 1:+.2%}），已實現 {st['realized']:,.0f}，配息 {st['dividends']:,.0f}，成本合計 {st['fees']:,.0f}")
    for r in h.itertuples():
        print(f"  {r.code} {r.name} {r.shares:,} 股  成本 {r.avg_cost:.2f}  現價 {r.price:.2f}  損益 {r.pnl:+,.0f}（{r.ret:+.2%}）")


if __name__ == "__main__":
    main()
