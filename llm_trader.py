# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15", "anthropic>=2.1"]
# ///
"""LLM 決策的模擬交易（學術研究，虛擬資金）：量化規則給候選，LLM 讀事實表後決定買 / 不買 / 等待。

用法:
  uv run llm_trader.py candidates             # 列出這次的候選與待重評的「等待」，不呼叫 API
  uv run llm_trader.py run --dry-run          # 估算這次要花多少，不呼叫 API
  uv run llm_trader.py run [--model haiku]    # 決策、寫紀錄、更新三組模擬持股
  uv run llm_trader.py status                 # 三組模擬持股與決策統計
  uv run llm_trader.py try 2330 --asof 2025-06-30   # 只看單檔決策，不寫任何紀錄（除錯用）

流程:
  - 候選 = 月營收創歷史新高且單月年增 >= 30%、20 日均成交值 >= 5,000 萬，公告日在最近 CAND_DAYS 天內、還沒決策過。
    event_study.py：這類事件公告後 60 個交易日平均超額約 5%，但勝率不到一半，報酬靠少數大贏家。
  - LLM 只拿到 analyst.stock_facts 的事實表（資料時點 = 最新資料日），輸出 buy / skip / wait 與理由。
    wait 的股票之後每次 run 重評，第一次決策後超過 WAIT_DAYS 個交易日就放棄。
  - 訊號日 = 決策當天（不是資料日），成交在決策之後的第一個交易日收盤，不回填到決策前。
  - 三組模擬持股（mine.py 的帳本，成本與漲跌停規則同回測），持有 HOLD_DAYS 個交易日、等權：
      LLM篩選-<model>  LLM 說 buy 的
      量化全買          所有候選（不經 LLM）
      隨機對照-<model>  每批候選隨機挑和 LLM 一樣多檔，排除「只是持股比較少」的效果
    0050 是 mine.py 內建的基準。只有 LLM 組穩定贏量化全買與隨機對照，才代表 LLM 有加分。
  - 決策紀錄 data/paper/llm_decisions.jsonl 只追加，含模型、prompt 雜湊、輸入 token 與費用；
    模型看到的使用者訊息原文存在 data/paper/llm_inputs/（gz），紀錄裡有路徑與 sha256。
    第一批（2026-10-22 資料）的輸入是事後用同一份未更新的資料重建的，紀錄標 input_backfilled。

注意:
  - 模型知識截止 2026-06，用更早的 --asof 測試時模型可能記得後來的股價，try 的結果不能當績效證據。
  - 全部是虛擬資金，不連任何券商。
"""
import argparse
import gzip
import hashlib
import json
import random
from datetime import datetime

import anthropic
import pandas as pd

import analyst as A
import mine as M
from build_panel import OUT
from report import load_dotenv

CAND_DAYS = 35      # 公告日在最近幾個日曆天內才算新候選
WAIT_DAYS = 20      # wait 最多等幾個交易日（從第一次決策算）
HOLD_DAYS = 60      # 買進後持有幾個交易日
MIN_VALUE = 5e7
DECISIONS = OUT / "paper" / "llm_decisions.jsonl"
INPUTS = OUT / "paper" / "llm_inputs"
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}   # 每百萬 token（輸入, 輸出）
QUANT = "量化全買"

SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["buy", "skip", "wait"]},
        "conviction": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "summary": {"type": "string"},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "wait_for": {"type": "string"},
    },
    "required": ["decision", "conviction", "summary", "reasons", "risks", "wait_for"],
    "additionalProperties": False,
}

SYSTEM = """你是學術研究用的台股模擬交易決策者，資金全是虛擬的。

背景：這檔股票是量化規則挑出的候選——最新月營收創歷史新高，且單月年增 >= 30%。
過去（2013 起）這類事件公告後 60 個交易日，平均比同期流動股票池多賺約 5%，但只有不到一半的事件贏過股票池，
報酬集中在少數大贏家。所以「全買」本身已經是有正期望值的基準；你的工作是判斷這一檔該不該買、現在買還是再等，
目標是挑出比全買更好的子集合。

決策:
- buy：現在買進，持有約 60 個交易日。
- skip：不買。例如營收創高看起來是一次性（單月暴增但前後月份平平、低基期）、獲利沒有跟上（毛利率或營益率明顯惡化）、
  重大訊息有負面事件、或風險明顯偏高。
- wait：基本面可以，但現在進場時機不好，之後會再問你一次；wait_for 寫清楚你要等什麼。
  等待有成本：歷史上這類事件的超額報酬在公告後就開始累積，等越久吃到的越少；股價近期大漲本身也不是避開的理由
  （過去動能強的股票平均表現較好）。所以只有在你預期等待能明顯改善結果時才選 wait，基本面可以就直接 buy，不可以就 skip。
  等待條件必須在約 20 個交易日內觀察得到（例如股價、法人買賣超、處置或注意股狀態），
  不要等下個月營收或下一季財報；從第一次決策起超過約 20 個交易日仍未改善就視同放棄。

規則:
- 只能用使用者提供的 JSON 事實表，不得引用表外的新聞、產業知識、或你記得的這家公司後來的股價與事件。
- reasons、risks 各 2~4 點，每點帶具體數字與期別。summary 2~3 句。conviction 1~5，5 = 最有把握。
- 不給目標價或報酬預測。用繁體中文。"""

PROMPT_HASH = hashlib.sha256((SYSTEM + json.dumps(SCHEMA, sort_keys=True)).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- 候選

def trading_days() -> pd.DatetimeIndex:
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date"], filters=[("code", "==", "0050")])
    return pd.DatetimeIndex(px["date"].sort_values().unique())


def liquid_codes(asof: pd.Timestamp) -> set[str]:
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "value"],
                         filters=[("date", ">", asof - pd.Timedelta(days=45)), ("date", "<=", asof)])
    last20 = px[px["date"] >= sorted(px["date"].unique())[-20]]
    v = last20.groupby("code")["value"].mean()
    return set(v[v >= MIN_VALUE].index)


def events(asof: pd.Timestamp, days: int) -> pd.DataFrame:
    """公告日在 (asof - days, asof] 內、創新高且年增 >= 30% 的月營收。"""
    rv = pd.read_parquet(OUT / "tej_rev.parquet",
                         columns=["code", "name", "period", "avail_date", "單月營收成長率％", "與歷史最高單月營收比%"])
    rv = rv[(rv["avail_date"] > asof - pd.Timedelta(days=days)) & (rv["avail_date"] <= asof)]
    rv = rv[(rv["與歷史最高單月營收比%"] >= 0) & (rv["單月營收成長率％"] >= 30)]
    rv = rv[rv["code"].isin(liquid_codes(asof))]
    rv = rv.rename(columns={"單月營收成長率％": "yoy", "與歷史最高單月營收比%": "vs_high"})
    return rv.sort_values("avail_date").drop_duplicates("code", keep="last").reset_index(drop=True)


def load_decisions() -> pd.DataFrame:
    if not DECISIONS.exists():
        return pd.DataFrame(columns=["code", "period", "decision", "model", "asof"])
    return pd.DataFrame([json.loads(l) for l in DECISIONS.read_text().splitlines() if l.strip()])


def todo(asof: pd.Timestamp, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(新候選, 要重評的 wait)。同一檔同一營收月份，同一模型只決策一次，wait 例外。"""
    ev = events(asof, CAND_DAYS + 60)   # wait 從第一次決策起算，可能超過 CAND_DAYS
    dec = load_decisions()
    dec = dec[dec["model"] == model] if len(dec) else dec
    recent = ev["avail_date"] > asof - pd.Timedelta(days=CAND_DAYS)
    if not len(dec):
        return ev[recent].reset_index(drop=True), ev.iloc[0:0]
    ev["key"] = ev["code"] + "|" + ev["period"].dt.strftime("%Y-%m")
    dec["key"] = dec["code"] + "|" + dec["period"]
    last = dec.groupby("key").last()
    seen = ev["key"].isin(last.index)
    waiting = ev[seen & ev["key"].map(last["decision"]).eq("wait")]
    first = dec.groupby("key")["asof"].min()
    days = trading_days()
    age = waiting["key"].map(lambda k: ((days > pd.Timestamp(first[k])) & (days <= asof)).sum())
    return ev[~seen & recent].drop(columns="key"), waiting[age <= WAIT_DAYS].drop(columns="key")


# ---------------------------------------------------------------- 決策

def facts_json(code: str, asof: pd.Timestamp) -> str:
    A.ASOF = asof
    A.names.cache_clear()
    A.universe.cache_clear()
    return A.stock_facts.func(code)


def decide(client, model: str, ev: pd.Series, asof: pd.Timestamp, prior: str | None = None) -> dict:
    facts = facts_json(ev["code"], asof)
    head = (f"候選：{ev['code']} {ev['name']}，{ev['period']:%Y-%m} 月營收 {ev['avail_date']:%Y-%m-%d} 公告，"
            f"單月年增 {ev['yoy']:.1f}%、比前歷史高點多 {ev['vs_high']:.1f}%。資料時點 {asof:%Y-%m-%d}。")
    if prior:
        head += f"\n你上次對這檔的決定是 wait，當時要等的是：{prior}。請依最新資料重新判斷。"
    user = head + "\n\n事實表：\n" + facts
    resp = client.messages.create(
        model=model, max_tokens=4000, system=SYSTEM,
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    u = resp.usage
    pin, pout = PRICES[model]
    out = {"decision": "skip", "conviction": 1, "summary": "（模型拒絕回答，記為 skip）", "reasons": [], "risks": [], "wait_for": ""}
    if resp.stop_reason != "refusal":
        out = json.loads(next(b.text for b in resp.content if b.type == "text"))
    return {**out, "usage": {"input": u.input_tokens, "output": u.output_tokens},
            "cost_usd": round((u.input_tokens * pin + u.output_tokens * pout) / 1e6, 5), "_input": user}


def save_input(text: str, model: str, asof: pd.Timestamp, code: str, period: str, stamp: str) -> tuple[str, str]:
    """模型看到的使用者訊息原文存成 gz（事實表會隨資料更新改變，之後無法重建）；回傳相對路徑與 sha256。"""
    path = INPUTS / model / f"{asof:%Y-%m-%d}" / f"{code}_{period}_{stamp.replace(':', '')}.txt.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(text.encode("utf-8")))
    return str(path.relative_to(OUT)), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def log_decision(ev: pd.Series, asof: pd.Timestamp, model: str, d: dict) -> None:
    DECISIONS.parent.mkdir(exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")
    period = f"{ev['period']:%Y-%m}"
    d = dict(d)
    path, sha = save_input(d.pop("_input"), model, asof, ev["code"], period, now)
    rec = {"time": now, "asof": str(asof.date()), "model": model,
           "prompt": PROMPT_HASH, "code": ev["code"], "name": ev["name"], "period": period,
           "avail_date": str(ev["avail_date"].date()), "yoy": round(float(ev["yoy"]), 2),
           "input_file": path, "input_sha256": sha, **d}
    with DECISIONS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 模擬持股

def holdings(entries: dict[str, pd.Timestamp], asof: pd.Timestamp) -> list[str]:
    """還在持有期內的股票（訊號日到 asof 未滿 HOLD_DAYS 個交易日）。"""
    days = trading_days()
    return sorted(c for c, s in entries.items() if ((days > s) & (days <= asof)).sum() < HOLD_DAYS)


def signal_dates(dec: pd.DataFrame) -> pd.Series:
    """訊號日 = 決策當天的日期（不是資料日）：成交在決策之後的第一個交易日，不回填到決策前。"""
    return pd.to_datetime(dec["time"]).dt.normalize()


def entries_from(dec: pd.DataFrame, picked: pd.Series) -> dict[str, pd.Timestamp]:
    """每檔最後一次被買進的訊號日。"""
    d = dec[picked]
    return signal_dates(d).groupby(d["code"]).max().to_dict()


def random_picks(dec: pd.DataFrame, model: str) -> pd.Series:
    """每個 asof 批次在「最終決定」的候選裡隨機挑和 LLM buy 一樣多檔；種子固定，重跑結果相同。"""
    final = dec[dec["decision"] != "wait"]
    picked = pd.Series(False, index=dec.index)
    for asof, g in final.groupby("asof"):
        n = int((g["decision"] == "buy").sum())
        chosen = random.Random(f"{model}|{asof}").sample(sorted(g.index), n)
        picked[chosen] = True
    return picked


def sync_portfolios(model: str, asof: pd.Timestamp, short: str, today: pd.Timestamp | None = None) -> list[str]:
    """today：這次調整的訊號日，預設今天（決策剛做完）。"""
    dec = load_decisions()
    dec = dec[dec["model"] == model].reset_index(drop=True)
    # 量化全買不分模型：同一檔同一營收月份只算第一次出現
    first = dec.drop_duplicates(["code", "period"])
    groups = {
        f"LLM篩選-{short}": entries_from(dec, dec["decision"] == "buy"),
        QUANT: entries_from(first, pd.Series(True, index=first.index)),
        f"隨機對照-{short}": entries_from(dec, random_picks(dec, model)),
    }
    today = today if today is not None else pd.Timestamp(datetime.now().date())
    meta, msgs = M.load_meta(), []
    led = M.load_ledger()
    for name, ent in groups.items():
        hold = holdings(ent, asof)
        w = pd.Series(1 / len(hold), index=hold) if hold else pd.Series(dtype=float)
        if name not in meta:
            if not hold:
                msgs.append(f"[{name}] 還沒有持股，先不建立")
                continue
            M.create(name, w, signal_date=today)
            msgs.append(f"[{name}] 建立，{len(hold)} 檔")
            continue
        cur = M._latest_signals(led, name)
        prev = cur[max(cur)] if cur else pd.Series(dtype=float)
        if set(prev[prev > 0].index) == set(hold):
            msgs.append(f"[{name}] 持股不變（{len(hold)} 檔）")
            continue
        M.edit(name, w if hold else pd.Series({"0050": 0.0}), signal_date=today)
        msgs.append(f"[{name}] 更新為 {len(hold)} 檔" + ("（空手）" if not hold else ""))
    return msgs


# ---------------------------------------------------------------- 指令

def cmd_candidates(asof, model):
    new, wait = todo(asof, model)
    print(f"資料時點 {asof:%Y-%m-%d}；新候選 {len(new)} 檔，待重評 wait {len(wait)} 檔")
    for tag, d in (("新", new), ("重評", wait)):
        for r in d.itertuples():
            print(f"  [{tag}] {r.code} {r.name}  {r.period:%Y-%m} 營收 {r.avail_date:%m-%d} 公告  年增 {r.yoy:.1f}%  比前高 +{r.vs_high:.1f}%")
    return new, wait


def cmd_run(asof, model, short, dry, limit):
    new, wait = cmd_candidates(asof, model)
    jobs = [(r, None) for _, r in new.iterrows()]
    if len(wait):
        dec = load_decisions()
        last = dec[dec["model"] == model].groupby(["code", "period"]).last()
        jobs += [(r, last.loc[(r["code"], f"{r['period']:%Y-%m}"), "wait_for"]) for _, r in wait.iterrows()]
    if limit:
        jobs = jobs[:limit]
    if not jobs:
        print("沒有要決策的股票")
    elif dry:
        sample = len(facts_json(jobs[0][0]["code"], asof)) + len(SYSTEM)
        tok_in = sample / 1.6 * len(jobs)   # 中文 JSON 粗估每 token 約 1.6 字元
        pin, pout = PRICES[model]
        est = (tok_in * pin + 1500 * len(jobs) * pout) / 1e6
        print(f"預估：{len(jobs)} 次呼叫、輸入約 {tok_in / 1e3:.0f}k token，費用約 US${est:.3f}（{model}）")
        return
    else:
        load_dotenv()
        client = anthropic.Anthropic()
        total = 0.0
        for ev, prior in jobs:
            d = decide(client, model, ev, asof, prior)
            log_decision(ev, asof, model, d)
            total += d["cost_usd"]
            print(f"  {ev['code']} {ev['name']}: {d['decision']}（信心 {d['conviction']}）{d['summary']}")
        print(f"實際費用 US${total:.4f}")
    if limit:
        print("試跑（--limit）：決策已記錄，下次不會重問；模擬持股等不加 --limit 跑完再更新")
        return
    for m in sync_portfolios(model, asof, short):
        print(m)


def cmd_status(model, short):
    dec = load_decisions()
    if not len(dec):
        print("還沒有任何決策")
        return
    d = dec[dec["model"] == model]
    print(f"{model}：{len(d)} 筆決策，" + "、".join(f"{k} {v}" for k, v in d["decision"].value_counts().items())
          + f"，累計費用 US${d['cost_usd'].sum():.4f}")
    meta = M.load_meta()
    for name in (f"LLM篩選-{short}", QUANT, f"隨機對照-{short}"):
        if name not in meta:
            continue
        res = M.replay(name)
        if res is None:
            print(f"[{name}] 等下一個交易日收盤進場")
            continue
        b = res["daily"]
        tot = lambda k: (1 + b[k]).prod() - 1
        print(f"[{name}] 自 {meta[name]['created']} 起累積 {tot('strategy'):+.2%}（0050 {tot('etf'):+.2%}），資產 {res['nav'].iloc[-1]:,.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["candidates", "run", "status", "try"])
    ap.add_argument("code", nargs="?")
    ap.add_argument("--model", choices=["haiku", "sonnet"], default="haiku")
    ap.add_argument("--asof", help="只給 try 用；run 一律用最新資料日")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, help="這次最多決策幾檔（試跑用）")
    a = ap.parse_args()
    model = A.MODELS[a.model]
    asof = M.data_date()
    if a.cmd == "candidates":
        cmd_candidates(asof, model)
    elif a.cmd == "run":
        cmd_run(asof, model, a.model, a.dry_run, a.limit)
    elif a.cmd == "status":
        cmd_status(model, a.model)
    elif a.cmd == "try":
        if not a.code:
            raise SystemExit("try 要給股票代號")
        at = pd.Timestamp(a.asof) if a.asof else asof
        if at < pd.Timestamp("2026-07-01"):
            print("⚠ 資料時點早於模型知識截止（2026-06），模型可能記得後來的走勢，結果不能當績效證據")
        ev = events(at, 400)
        hit = ev[ev["code"] == a.code]
        row = hit.iloc[-1] if len(hit) else pd.Series({"code": a.code, "name": A.names().loc[a.code, "name"],
                                                       "period": at, "avail_date": at, "yoy": float("nan"), "vs_high": float("nan")})
        load_dotenv()
        d = decide(anthropic.Anthropic(), model, row, at)
        d.pop("_input")
        print(json.dumps(d, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
