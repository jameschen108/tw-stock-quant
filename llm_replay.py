# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15", "anthropic>=2.1"]
# ///
"""LLM 決策的準前瞻回放（學術研究，虛擬資金）：用模型知識截止（2026-06）之後的事件，重現 llm_trader.py 的決策流程。

用法:
  uv run llm_replay.py run --dry-run          # 列出各批候選與估價，不呼叫 API
  uv run llm_replay.py run --limit 3          # 試跑：每批第一輪只問前 3 檔
  uv run llm_replay.py run [--model haiku]    # 跑全部（已問過的會跳過，中斷後重跑會接續）
  uv run llm_replay.py report                 # 選股檢定：LLM buy 對全部候選、隨機挑同樣檔數

設計:
  - 每個月一批，資料時點 = 該月第一個 >= 10 日的交易日（月營收法定 10 日前公布）；
    候選 = 上一批資料時點之後到這批之間公告、創歷史新高且年增 >= 30%、20 日均成交值 >= 5,000 萬（同 llm_trader）。
    第一批只收 2026-07-01 起公告的，避免模型知識截止前的事件。
  - prompt、事實表、schema 都和 llm_trader 相同；事實表用 analyst.stock_facts 在該資料時點的版本，只含當時已公布的資料。
  - wait 的股票在資料時點後第 10、20 個交易日各重問一次（同 llm_trader 的 WAIT_DAYS），還是 wait 就算放棄。
  - 決策紀錄 data/paper/llm_replay.jsonl（和前向的 llm_decisions.jsonl 分開），輸入原文存在 data/paper/llm_inputs/replay/。
  - 報酬：訊號日 = 決策的資料時點，隔一個交易日收盤進場，看之後 20、60 個交易日（不扣成本、等權）；
    資料還不夠長的事件，那個天期不計入。
  - 比較：
      全部候選      量化全買，從批次資料時點進場
      LLM buy（選股）  LLM 最後說 buy 的，也從批次資料時點進場：只看挑股，不含等待的時點
      LLM buy（含等待）  從實際說 buy 的那一輪進場
      隨機          每批從全部候選隨機挑和 LLM buy 一樣多檔，重複 N_RANDOM 次
    0050 超額 = 個股報酬 − 同期間 0050 報酬。

注意:
  - 這不是真正的前瞻：事實表是事後用現在的資料重建的，公布日對齊靠 avail_date；但模型沒看過這段期間之後的股價。
  - Haiku 的知識截止是 2026-06；換成截止更晚的模型就不能用這段期間。
"""
import argparse
import gzip
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import anthropic
import pandas as pd

import analyst as A
import llm_trader as T
from build_panel import OUT
from custom import load_rets
from report import load_dotenv

MONTHS = ["2026-07", "2026-08", "2026-09"]
FIRST_AVAIL = pd.Timestamp("2026-07-01")
REASK = [10, 20]        # wait 在資料時點後第幾個交易日重問
HORIZONS = [20, 60]
LOG = OUT / "paper" / "llm_replay.jsonl"
INPUTS = T.INPUTS / "replay"
WORKERS = 4
N_RANDOM = 1000


def batch_dates() -> list[pd.Timestamp]:
    days = T.trading_days()
    return [days[days >= pd.Timestamp(f"{m}-10")][0] for m in MONTHS]


def batch_events() -> dict[pd.Timestamp, pd.DataFrame]:
    out, start = {}, FIRST_AVAIL - pd.Timedelta(days=1)
    for b in batch_dates():
        ev = T.events(b, (b - start).days)
        out[b] = ev[ev["avail_date"] >= FIRST_AVAIL].reset_index(drop=True)
        start = b
    return out


def load_log() -> pd.DataFrame:
    if not LOG.exists():
        return pd.DataFrame(columns=["batch", "round", "asof", "model", "code", "period", "decision", "wait_for"])
    return pd.DataFrame([json.loads(l) for l in LOG.read_text().splitlines() if l.strip()])


def save_input(text: str, model: str, asof: pd.Timestamp, code: str, period: str, rnd: int) -> tuple[str, str]:
    path = INPUTS / model / f"{asof:%Y-%m-%d}" / f"{code}_{period}_r{rnd}.txt.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(text.encode("utf-8")))
    return str(path.relative_to(OUT)), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def ask(client, model: str, batch: pd.Timestamp, rnd: int, asof: pd.Timestamp, jobs: list[tuple[pd.Series, str | None]]) -> float:
    """jobs：(候選, 上次的 wait_for)。事實表先依序算好，API 呼叫再平行。"""
    facts = [T.facts_json(ev["code"], asof) for ev, _ in jobs]
    with ThreadPoolExecutor(WORKERS) as pool:
        futs = [pool.submit(T.decide, client, model, ev, asof, prior, f) for (ev, prior), f in zip(jobs, facts)]
        total = 0.0
        for (ev, _), fut in zip(jobs, futs):
            d = fut.result()
            period = f"{ev['period']:%Y-%m}"
            path, sha = save_input(d.pop("_input"), model, asof, ev["code"], period, rnd)
            rec = {"time": datetime.now().isoformat(timespec="seconds"), "batch": str(batch.date()), "round": rnd,
                   "asof": str(asof.date()), "model": model, "prompt": T.PROMPT_HASH, "code": ev["code"], "name": ev["name"],
                   "period": period, "avail_date": str(ev["avail_date"].date()), "yoy": round(float(ev["yoy"]), 2),
                   "vs_high": round(float(ev["vs_high"]), 2), "input_file": path, "input_sha256": sha, **d}
            with LOG.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += d["cost_usd"]
            print(f"  [{asof:%m-%d} 第{rnd}輪] {ev['code']} {ev['name']}: {d['decision']}（信心 {d['conviction']}）")
    return total


def cmd_run(model: str, dry: bool, limit: int | None):
    days = T.trading_days()
    client = None
    if not dry:
        load_dotenv()
        client = anthropic.Anthropic()
    total = 0.0
    for b, ev in batch_events().items():
        print(f"批次 {b:%Y-%m-%d}：候選 {len(ev)} 檔")
        if dry:
            continue
        for rnd, asof in enumerate([b] + [days[days.get_loc(b) + k] for k in REASK]):
            log = load_log()
            log = log[(log["model"] == model) & (log["batch"] == str(b.date()))] if len(log) else log
            done = set(zip(log["code"], log["round"])) if len(log) else set()
            if rnd == 0:
                jobs = [(r, None) for _, r in ev.iterrows() if (r["code"], 0) not in done]
                jobs = jobs[:limit] if limit else jobs
            else:
                prev = log[log["round"] == rnd - 1].set_index("code")
                prev = prev[prev["decision"] == "wait"]
                jobs = [(r, prev.loc[r["code"], "wait_for"]) for _, r in ev.iterrows()
                        if r["code"] in prev.index and (r["code"], rnd) not in done]
            if jobs:
                total += ask(client, model, b, rnd, asof, jobs)
    if dry:
        n = sum(len(e) for e in batch_events().values())
        pin, pout = T.PRICES[model]
        print(f"預估：第一輪 {n} 次呼叫，加上 wait 重問約 {n * 1.15:.0f} 次；費用約 US${n * 1.15 * (8500 * pin + 1500 * pout) / 1e6:.2f}")
    else:
        print(f"這次實際費用 US${total:.4f}")


# ---------------------------------------------------------------- 評估

def finals(model: str) -> pd.DataFrame:
    """每批每檔的最終決定：最後一輪的決策；最後一輪還是 wait 就算放棄（不買）。"""
    log = load_log()
    log = log[log["model"] == model]
    last = log.sort_values("round").groupby(["batch", "code"]).last().reset_index()
    last.loc[last["decision"] == "wait", "decision"] = "放棄"
    return last


def window_returns(rets: pd.DataFrame, code: str, signal: pd.Timestamp) -> dict:
    days = rets.index
    after = days[days > signal]
    out = {}
    for h in HORIZONS:
        if len(after) <= h or code not in rets:
            out[h] = (float("nan"), float("nan"))
            continue
        w = (days > after[0]) & (days <= after[h])
        r = (1 + rets.loc[w, code].fillna(0)).prod() - 1
        e = (1 + rets.loc[w, T.M.S.ETF].fillna(0)).prod() - 1
        out[h] = (r, r - e)
    return out


def evaluate(model: str) -> dict | None:
    """回放的報酬與選股檢定：{"ev": 每檔明細, "batches": 各批 20 日, "tests": {天期: 檢定}}；還沒有紀錄回傳 None。"""
    f = finals(model)
    if not len(f):
        return None
    rets = load_rets(pd.Timestamp(f["batch"].min()))
    rows = []
    for r in f.itertuples():
        base = window_returns(rets, r.code, pd.Timestamp(r.batch))
        timed = window_returns(rets, r.code, pd.Timestamp(r.asof))
        row = {"batch": r.batch, "code": r.code, "name": r.name, "decision": r.decision, "round": r.round}
        for h in HORIZONS:
            row[f"r{h}"], row[f"x{h}"] = base[h]
            row[f"t{h}"] = timed[h][0]
        rows.append(row)
    ev = pd.DataFrame(rows)
    batches = []
    for b, g in ev.groupby("batch"):
        buy = g["decision"] == "buy"
        batches.append({"batch": b, "n": len(g), "n_buy": int(buy.sum()), "all": g["r20"].mean(),
                        "buy": g.loc[buy, "r20"].mean(), "not_buy": g.loc[~buy, "r20"].mean(), "x_all": g["x20"].mean()})
    tests = {}
    for h in HORIZONS:
        d = ev.dropna(subset=[f"r{h}"])
        if not len(d) or not (d["decision"] == "buy").any():
            continue
        groups = [(g[f"r{h}"].to_numpy(), int((g["decision"] == "buy").sum())) for _, g in d.groupby("batch")]
        rng = random.Random(f"{model}|replay|{h}")
        draws = []
        for _ in range(N_RANDOM):
            picked = [x for r, n in groups for x in rng.sample(list(r), n)]
            draws.append(sum(picked) / len(picked))
        draws = pd.Series(draws)
        buy = d["decision"] == "buy"
        llm = d.loc[buy, f"r{h}"].mean()
        tests[h] = {"n_batch": d["batch"].nunique(), "n": len(d), "n_buy": int(buy.sum()),
                    "all": d[f"r{h}"].mean(), "x_all": d[f"x{h}"].mean(), "win_etf": (d[f"x{h}"] > 0).mean(),
                    "llm": llm, "x_llm": d.loc[buy, f"x{h}"].mean(), "llm_timed": d.loc[buy, f"t{h}"].mean(),
                    "not_buy": d.loc[~buy, f"r{h}"].mean(), "rand_p5": draws.quantile(0.05),
                    "rand_p50": draws.median(), "rand_p95": draws.quantile(0.95),
                    "pctile": float((draws < llm).mean()), "p_one_sided": float((draws >= llm).mean())}
    log = load_log()
    return {"ev": ev, "batches": pd.DataFrame(batches), "tests": tests,
            "cost": float(log.loc[log["model"] == model, "cost_usd"].sum())}


def cmd_report(model: str):
    res = evaluate(model)
    if res is None:
        print("還沒有回放紀錄")
        return
    ev = res["ev"]
    ev.to_csv(OUT / "paper" / "llm_replay_returns.csv", index=False)
    print(f"{model}：{len(ev)} 檔，" + "、".join(f"{k} {v}" for k, v in ev["decision"].value_counts().items())
          + f"；經過 wait 才定案 {int((ev['round'] > 0).sum())} 檔；累計費用 US${res['cost']:.3f}")
    print("\n各批（20 日報酬，不扣成本）:")
    for b in res["batches"].itertuples():
        print(f"  {b.batch}：候選 {b.n}、buy {b.n_buy}　全部 {b.all:+.2%}　buy {b.buy:+.2%}"
              f"　不買 {b.not_buy:+.2%}　0050 超額（全部）{b.x_all:+.2%}")
    for h in HORIZONS:
        t = res["tests"].get(h)
        if t is None:
            print(f"\n{h} 日：資料還不夠")
            continue
        print(f"\n{h} 日（{t['n_batch']} 批、{t['n']} 檔有完整報酬，{t['n_buy']} 檔 buy）:\n"
              f"  全部候選 {t['all']:+.2%}（0050 超額 {t['x_all']:+.2%}，勝 0050 比例 {t['win_etf']:.0%}）\n"
              f"  LLM buy（選股）{t['llm']:+.2%}（0050 超額 {t['x_llm']:+.2%}）"
              f"　LLM buy（含等待）{t['llm_timed']:+.2%}　不買 {t['not_buy']:+.2%}\n"
              f"  隨機挑同樣檔數 {N_RANDOM} 次：中位數 {t['rand_p50']:+.2%}，90% 區間 {t['rand_p5']:+.2%} ~ {t['rand_p95']:+.2%}\n"
              f"  LLM 在隨機分布的第 {t['pctile'] * 100:.0f} 百分位（單尾 p = {t['p_one_sided']:.3f}）")
    print("\n明細：data/paper/llm_replay_returns.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--model", choices=["haiku", "sonnet"], default="haiku")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, help="試跑：每批第一輪最多問幾檔")
    a = ap.parse_args()
    model = A.MODELS[a.model]
    if a.cmd == "run":
        cmd_run(model, a.dry_run, a.limit)
    else:
        cmd_report(model)


if __name__ == "__main__":
    main()
