# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""補 TEJ 之後新季度的營業活動現金流（公開資訊觀測站舊版 t164sb05，依公司 × 季度查）。

用法:
  uv run fetch_cashflow.py                      # 公開資料補上、TEJ 沒有的季度，每次最多 200 家
  uv run fetch_cashflow.py --max 0              # 不設上限
  uv run fetch_cashflow.py --codes 2330,2317 --period 2026-06-30   # 指定（驗證用）
  uv run fetch_cashflow.py --parse-only

OpenAPI 沒有現金流量表，這裡只抓「營業活動之淨現金流入（流出）」一列（年初累計，仟元）。
原始頁面存 data/raw/cashflow/{代號}_{民國年}Q{季}.html；查無資料的也存（之後不再查，除非 --retry-empty）。
整理成 data/cashflow_public.parquet：code、period、cfo_cum、fetched_at。public_data.merge_fin 用它算單季現金流。
請求間隔 4 秒（MOPS 會封鎖太頻繁的查詢）；連續失敗 5 次就停，重跑會接續。
"""
import argparse
import html
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw" / "cashflow"
OUT = ROOT / "data" / "cashflow_public.parquet"
URL = "https://mopsov.twse.com.tw/mops/web/ajax_t164sb05"
HEADERS = {"User-Agent": "Mozilla/5.0 (research)", "Content-Type": "application/x-www-form-urlencoded"}
CFO_RE = re.compile(r"營業活動之淨現金流[入出]（流[出入]）\|+\s*(-?[\d,]+)")
NO_DATA = ("查無", "無應編製", "不存在")


def fetch(code: str, period: pd.Timestamp) -> str:
    data = {"encodeURIComponent": 1, "step": 1, "firstin": 1, "off": 1, "queryName": "co_id", "inpuType": "co_id",
            "TYPEK": "all", "isnew": "false", "co_id": code, "year": period.year - 1911, "season": f"{period.quarter:02d}"}
    page = _post(data)
    # 金融業等部分公司先回一個「詳細資料」按鈕（step=2 的表單），照表單參數再查一次
    m = re.search(r"name='fm1'.*?name=\"TYPEK\"[^>]*value=\"(\w+)\".*?name=\"step\"[^>]*value=\"2\"", page, re.S)
    if m and not CFO_RE.search(re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "|", page)))):
        time.sleep(2)
        page = _post({"step": 2, "firstin": 1, "TYPEK": m.group(1), "year": period.year - 1911,
                      "season": period.quarter, "co_id": code})
    return page


def _post(data: dict) -> str:
    req = urllib.request.Request(URL, data=urllib.parse.urlencode(data).encode(), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def parse(page: str) -> tuple[str, float | None]:
    """回傳 (狀態, 營業現金流年初累計)。狀態：ok / empty / blocked。"""
    text = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "|", page)))
    m = CFO_RE.search(text)
    if m:
        return "ok", float(m.group(1).replace(",", ""))
    if any(k in text for k in NO_DATA):
        return "empty", None
    return "blocked", None


def raw_path(code: str, period: pd.Timestamp) -> Path:
    return RAW / f"{code}_{period.year - 1911}Q{period.quarter}.html"


def targets() -> list[tuple[str, pd.Timestamp]]:
    """OpenAPI 快照裡有、TEJ 匯出檔沒有的 (公司, 季度)。"""
    from public_data import public_fin_cum
    cum = public_fin_cum()
    tej = pd.read_parquet(ROOT / "data" / "tej_fin.parquet", columns=["code", "period", "source"])
    have = set(zip(tej.loc[tej["source"] != "public", "code"], tej.loc[tej["source"] != "public", "period"]))
    return sorted({(c, p) for c, p in zip(cum["code"], cum["period"]) if (c, p) not in have}, key=lambda x: (x[1], x[0]))


def build() -> pd.DataFrame:
    rows = []
    for f in sorted(RAW.glob("*.html")):
        code, q = f.stem.rsplit("_", 1)
        roc, season = q.split("Q")
        st, v = parse(f.read_text(encoding="utf-8"))
        if st == "ok":
            period = pd.Timestamp(year=int(roc) + 1911, month=int(season) * 3, day=1) + pd.offsets.MonthEnd(0)
            rows.append({"code": code, "period": period, "cfo_cum": v,
                         "fetched_at": pd.Timestamp(datetime.fromtimestamp(f.stat().st_mtime))})
    df = pd.DataFrame(rows, columns=["code", "period", "cfo_cum", "fetched_at"])
    df.to_parquet(OUT, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes")
    ap.add_argument("--period", help="季底日期，例如 2026-09-30（搭配 --codes）")
    ap.add_argument("--max", type=int, default=200, help="這次最多查幾家，0 = 不限")
    ap.add_argument("--sleep", type=float, default=4)
    ap.add_argument("--retry-empty", action="store_true", help="重查之前查無資料的")
    ap.add_argument("--parse-only", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if not a.parse_only:
        if a.codes:
            todo = [(c, pd.Timestamp(a.period)) for c in a.codes.split(",")]
        else:
            todo = targets()

        def need(c, p):
            f = raw_path(c, p)
            return not f.exists() or (a.retry_empty and parse(f.read_text(encoding="utf-8"))[0] == "empty")

        todo = [(c, p) for c, p in todo if need(c, p)]
        if a.max:
            todo = todo[: a.max]
        print(f"要查 {len(todo)} 家（約 {len(todo) * a.sleep / 60:.0f} 分鐘）", flush=True)
        fails = 0
        for c, p in todo:
            try:
                page = fetch(c, p)
                st, v = parse(page)
                if st == "blocked":
                    raise RuntimeError("回應不是結果頁（可能查詢太頻繁）")
            except Exception as e:
                fails += 1
                print(f"{c} {p.date()} 失敗（第 {fails} 次）：{type(e).__name__} {str(e)[:60]}", flush=True)
                if fails >= 5:
                    sys.exit("連續失敗 5 次，可能被暫時封鎖；稍後重跑會接續")
                time.sleep(60 * fails)
                continue
            fails = 0
            raw_path(c, p).write_text(page, encoding="utf-8")
            print(f"{c} {p.year}Q{p.quarter}: {st} {'' if v is None else f'{v:,.0f}'}", flush=True)
            time.sleep(a.sleep)
    df = build()
    print(f"cashflow_public.parquet: {len(df)} 筆")


if __name__ == "__main__":
    main()
