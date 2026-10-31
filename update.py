# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""每日資料更新：抓新資料 → 重建 parquet / 因子 / 綜合分數 → 健康檢查。

用法:
  uv run update.py                 # 完整流程
  uv run update.py --skip-fetch    # 只重建 + 檢查

步驟:
  1. 日行情    fetch_prices.py（上市、上櫃平行），從最新資料日往前 5 天抓到今天
               （17:30 前只抓到昨天；最近 7 天被記成空資料的平日會刪掉紀錄重抓，CLOSED 列的臨時休市日除外）
  2. 除權息    fetch_events.py（只重抓涵蓋今天的那一段）；處置／注意／全額交割 fetch_restricted.py（同上，另存當天名單）
  3. 營收財報  fetch_public.py（OpenAPI 快照）；fetch_cashflow.py（新季度的營業現金流，每次最多 200 家）；重大訊息 fetch_mops.py（近 14 天沒抓過的日子 + 最近 3 天）
  4. 重建      build_prices →（籌碼 fetch_chips：三大法人、融資融券、外資持股，再 --build）→ build_panel → factors → composite → outlook（短中長期標籤）
  5. 檢查      最新交易日、資料列數不減少、近 30 日可疑報酬、月營收覆蓋率、財報季覆蓋率（季底到期限後 30 天），
               結果寫 data/update_status.json
  6. 技術面    technicals.py --part latest（最新一天的技術訊號值）、patterns.py --latest（成立中的價格形態）、
               refprice.py --latest（參考價與參考區間，會用到形態的價位）；都沿用既有校準
     模擬持股  paper.py：新月份記一期綜合分數前 30 名與 paper_strategies.txt 各策略的目標權重，重算淨值；mine.py：自選組合（月調的記新月份）
  7. 每日速覽  daily.py：觀察名單（watchlist.txt）的變化 + 財報季新公布的公司 → reports/daily_YYYYMMDD.html

紀錄寫到 logs/update_YYYYMMDD.log；任何一步失敗或檢查不過，結束碼非 0。
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
LOGS = ROOT / "logs"
STATUS = DATA / "update_status.json"
PUBLISH_HOUR = (17, 30)   # 證交所、櫃買每日行情大約 14:00–16:00 公布，留點餘裕

log_file = None


def log(msg: str):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


def run(cmd: list[str], name: str, cwd: Path = ROOT) -> bool:
    log(f"▶ {name}")
    t0 = datetime.now()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    log_file.write(p.stdout + p.stderr)
    tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
    ok = p.returncode == 0
    log(f"{'✓' if ok else '✗'} {name}（{(datetime.now() - t0).seconds}s）" + ("" if ok else " ｜ " + " / ".join(tail)))
    return ok


def run_parallel(jobs: dict[str, list[str]], cwd: Path) -> bool:
    log("▶ " + "、".join(jobs))
    procs = {n: subprocess.Popen(c, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for n, c in jobs.items()}
    ok = True
    for n, p in procs.items():
        out, _ = p.communicate()
        log_file.write(out)
        good = p.returncode == 0
        ok &= good
        log(f"{'✓' if good else '✗'} {n}" + ("" if good else " ｜ " + " / ".join(out.strip().splitlines()[-2:])))
    return ok


def fetch_end() -> str:
    now = datetime.now()
    end = now.date() if (now.hour, now.minute) >= PUBLISH_HOUR else now.date() - timedelta(days=1)
    return end.isoformat()


# 交易所行事曆以外的臨時休市（颱風假等）：不重抓、不算進落後天數
CLOSED = {"2026-10-23": "颱風休市"}


def retry_recent_empty(days: int = 7):
    """行情的 manifest 把當天還沒公布的平日記成 empty 後就不會再抓：最近幾天的 empty 紀錄刪掉重試。"""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")
    for m in ("twse", "tpex"):
        f = DATA / "prices" / m / "manifest.jsonl"
        if not f.exists():
            continue
        lines = f.read_text().splitlines()
        keep = [l for l in lines if not (l.strip() and json.loads(l)["status"] == "empty"
                                          and json.loads(l)["date"] >= cutoff
                                          and pd.Timestamp(json.loads(l)["date"]).dayofweek < 5
                                          and str(pd.Timestamp(json.loads(l)["date"]).date()) not in CLOSED)]
        if len(keep) != len(lines):
            f.write_text("\n".join(keep) + "\n")
            log(f"{m}: 最近 {days} 天有 {len(lines) - len(keep)} 個平日是空資料，重抓")


def last_price_date() -> pd.Timestamp:
    d = pd.read_parquet(DATA / "prices_daily.parquet", columns=["date"])["date"]
    return d.max()


# 季報法定期限（月, 日）：Q1 5/15、Q2 8/14、Q3 11/14、年報 隔年 3/31（金融業等另有規定，這裡只做提醒）
DEADLINES = {3: (5, 15), 6: (8, 14), 9: (11, 14), 12: (3, 31)}


def season_status(today: pd.Timestamp | None = None) -> dict:
    """財報季覆蓋率：季底過後到期限後 30 天內，報告這一季已公布幾家、營業現金流補到幾家。"""
    today = today or pd.Timestamp.today().normalize()
    q = (today - pd.offsets.QuarterEnd(0) if today.is_quarter_end else today - pd.offsets.QuarterEnd(1)).normalize()
    m, d = DEADLINES[q.month]
    deadline = pd.Timestamp(year=q.year + (q.month == 12), month=m, day=d)
    if today > deadline + pd.Timedelta(days=30):
        return {}
    fin = pd.read_parquet(DATA / "tej_fin.parquet", columns=["code", "period", "來自營運之現金流量"])
    px = pd.read_parquet(DATA / "prices_daily.parquet", columns=["date", "code"])
    alive = set(px.loc[px["date"] >= px["date"].max() - pd.Timedelta(days=10), "code"])
    prev_q = (q - pd.offsets.QuarterEnd(1)).normalize()
    base = set(fin.loc[fin["period"] == prev_q, "code"]) & alive        # 上一季有財報、仍在交易
    cur = fin[(fin["period"] == q) & fin["code"].isin(base)]
    cf = pd.read_parquet(DATA / "cashflow_public.parquet") if (DATA / "cashflow_public.parquet").exists() else None
    cf_codes = set(cf.loc[cf["period"] == q, "code"]) if cf is not None else set()
    has_cfo = set(cur.loc[cur["來自營運之現金流量"].notna(), "code"]) | cf_codes
    return {"quarter": f"{q.year}Q{q.quarter}", "deadline": str(deadline.date()),
            "days_to_deadline": int((deadline - today).days),
            "expected": len(base), "reported": int(cur["code"].nunique()),
            "cashflow_filled": len(set(cur["code"]) & has_cfo),
            "cashflow_pending": len(set(cur["code"]) - has_cfo)}


def checks(prev: dict) -> tuple[dict, list[str]]:
    px = pd.read_parquet(DATA / "prices_daily.parquet", columns=["date", "code", "ret_suspect"])
    rev = pd.read_parquet(DATA / "tej_rev.parquet", columns=["code", "period", "source"])
    fin = pd.read_parquet(DATA / "tej_fin.parquet", columns=["code", "period", "source"])
    fm = pd.read_parquet(DATA / "factors_monthly.parquet", columns=["date", "code", "rev_period", "fin_period"])
    last = px["date"].max()
    recent = px[px["date"] > last - pd.Timedelta(days=30)]
    cur = fm[fm["date"] == fm["date"].max()]
    st = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "last_price_date": str(last.date()),
        "price_rows": int(len(px)),
        "rev_rows": int(len(rev)), "rev_public_rows": int((rev["source"] == "public").sum()),
        "fin_rows": int(len(fin)), "fin_public_rows": int((fin["source"] == "public").sum()),
        "latest_rev_period": str(rev["period"].max().date()), "latest_fin_period": str(fin["period"].max().date()),
        "suspect_30d": int(recent["ret_suspect"].sum()),
        "factor_date": str(cur["date"].max().date()),
        # 換股日已知營收落後不超過 2 個月的比例（月營收每月 10 日前公布）
        "rev_fresh_share": round(float(((cur["date"] - cur["rev_period"]).dt.days <= 70).mean()), 3),
        "fin_fresh_share": round(float(((cur["date"] - cur["fin_period"]).dt.days <= 170).mean()), 3),
    }
    problems = []
    bdays = pd.bdate_range(last, datetime.now().date()).difference(pd.to_datetime(list(CLOSED)))
    if len(bdays) > 4:
        problems.append(f"行情停在 {last.date()}，落後 {len(bdays) - 1} 個工作天")
    for k in ["price_rows", "rev_rows", "fin_rows"]:
        if prev.get(k) and st[k] < prev[k]:
            problems.append(f"{k} 從 {prev[k]} 減少到 {st[k]}")
    if st["suspect_30d"] > 10:
        problems.append(f"近 30 日可疑報酬 {st['suspect_30d']} 筆（可能缺除權息 / 減資事件）")
    if st["factor_date"] != st["last_price_date"]:
        problems.append(f"因子表日期 {st['factor_date']} ≠ 最新行情 {st['last_price_date']}")
    if st["rev_fresh_share"] < 0.85:
        problems.append(f"只有 {st['rev_fresh_share']:.0%} 股票有近 2 個月的月營收")
    if ss := season_status():
        st["season"] = ss
        if ss["days_to_deadline"] < 0 and ss["reported"] < 0.95 * ss["expected"]:
            problems.append(f"{ss['quarter']} 財報期限已過，只有 {ss['reported']}/{ss['expected']} 家有資料")
        if ss["cashflow_pending"] and ss["days_to_deadline"] < -7:
            problems.append(f"{ss['quarter']} 還有 {ss['cashflow_pending']} 家缺營業現金流（fetch_cashflow.py --max 0 可一次補完）")
    return st, problems


def main():
    global log_file
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true")
    a = ap.parse_args()
    LOGS.mkdir(exist_ok=True)
    log_file = open(LOGS / f"update_{datetime.now():%Y%m%d}.log", "a", encoding="utf-8")
    log(f"===== update {datetime.now():%Y-%m-%d %H:%M} =====")
    prev = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    ok = True

    if not a.skip_fetch:
        retry_recent_empty()
        start = (last_price_date() - timedelta(days=5)).date().isoformat()
        end = fetch_end()
        ok &= run_parallel({f"行情 {m}": ["uv", "run", "-q", "fetch_prices.py", "--market", m, "--start", start, "--end", end]
                            for m in ("twse", "tpex")}, cwd=ROOT)
        for src in ("twse", "tpex"):
            ok &= run(["uv", "run", "-q", "fetch_events.py", "--source", src], f"除權息 {src}")
            ok &= run(["uv", "run", "-q", "fetch_restricted.py", "--source", src], f"處置／注意／全額交割 {src}")
        ok &= run(["uv", "run", "-q", "fetch_public.py"], "營收 / 財報 OpenAPI")
        ok &= run(["uv", "run", "-q", "fetch_cashflow.py"], "營業現金流（新季度，最多 200 家）")
        ok &= run(["uv", "run", "-q", "fetch_mops.py", "--days", "14"], "重大訊息")
        if not ok:
            log("抓取有失敗，仍用現有資料重建")

    for script in ("build_prices.py", "build_panel.py", "factors.py", "composite.py", "outlook.py"):
        if not run(["uv", "run", "-q", script], script):
            log("重建失敗，停止")
            sys.exit(1)
        if script == "build_prices.py" and not a.skip_fetch:   # 法人買賣超的交易日取自剛重建的行情
            for kind, label in (("insti", "三大法人"), ("margin", "融資融券"), ("qfii", "外資持股")):
                ok &= run_parallel({f"{label} {m}": ["uv", "run", "-q", "fetch_chips.py", "--market", m, "--kind", kind]
                                    for m in ("twse", "tpex")}, cwd=ROOT)
            ok &= run(["uv", "run", "-q", "fetch_chips.py", "--build"], "籌碼整理成 parquet")

    st, problems = checks(prev)
    st["fetch_ok"] = ok
    st["problems"] = problems
    STATUS.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    log("狀態: " + json.dumps({k: v for k, v in st.items() if k not in ("problems", "season")}, ensure_ascii=False))
    if ss := st.get("season"):
        log(f"財報季 {ss['quarter']}（期限 {ss['deadline']}，還有 {ss['days_to_deadline']} 天）："
            f"已公布 {ss['reported']}/{ss['expected']} 家，營業現金流已補 {ss['cashflow_filled']}、待補 {ss['cashflow_pending']}")
    for p in problems:
        log(f"⚠ {p}")
    ok &= run(["uv", "run", "-q", "technicals.py", "--part", "latest"], "最新技術訊號")
    ok &= run(["uv", "run", "-q", "patterns.py", "--latest"], "最新價格形態")
    ok &= run(["uv", "run", "-q", "refprice.py", "--latest"], "參考價")
    ok &= run(["uv", "run", "-q", "paper.py"], "模擬持股")
    ok &= run(["uv", "run", "-q", "mine.py"], "自選組合")
    ok &= run(["uv", "run", "-q", "daily.py"], "每日速覽")
    log("完成" if ok and not problems else "完成（有警告）")
    sys.exit(0 if ok and not problems else 2)


if __name__ == "__main__":
    main()
