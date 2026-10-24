# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""公開資訊觀測站重大訊息（全市場、依日期），存原始頁面並整理成 data/announcements.parquet。

用法:
  uv run fetch_mops.py                 # 近 180 天，已抓過的日子跳過（最近 3 天每次重抓）
  uv run fetch_mops.py --days 400
  uv run fetch_mops.py --parse-only    # 只重新整理 parquet

來源是舊版網站 mopsov 的 t05st02（依日期查全市場），每天一個請求；頁面的隱藏欄位附有完整說明內文。
若改用 t05st01 依「公司 × 年度」查，全市場要上千個請求，所以這裡用依日期查詢。
一天的查詢涵蓋前一個晚上到當天的公告，同一則會出現在相鄰兩天，整理時去重。
原始頁面存 data/raw/mops/YYYYMMDD.html；請求間隔 4 秒（MOPS 會封鎖太頻繁的查詢）。
"""
import argparse
import html
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw" / "mops"
OUT = ROOT / "data" / "announcements.parquet"
URL = "https://mopsov.twse.com.tw/mops/web/ajax_t05st02"
HEADERS = {"User-Agent": "Mozilla/5.0 (research)", "Content-Type": "application/x-www-form-urlencoded"}
NO_DATA = "查無需求資料"
REFRESH_DAYS = 3

FORM_RE = re.compile(r"<form action='/mops/web/ajax_t05st02'.*?</form>", re.S)
HIDDEN_RE = re.compile(r"name='h\d*?(\d)' value='(.*?)'>", re.S)
TYPEK_RE = re.compile(r'TYPEK\.value="(\w+)"')
MARKET = {"sii": "上市", "otc": "上櫃", "rotc": "興櫃", "pub": "公開發行"}


def fetch(d: date) -> str:
    data = {"encodeURIComponent": 1, "step": 0, "firstin": "true", "off": 1, "TYPEK": "all",
            "year": d.year - 1911, "month": f"{d.month:02d}", "day": f"{d.day:02d}"}
    req = urllib.request.Request(URL, data=urllib.parse.urlencode(data).encode(), headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def parse(page: str) -> pd.DataFrame:
    """每則一列：code、name、market、time、subject、fact_date、content。"""
    rows = []
    for f in FORM_RE.findall(page):
        h = {k: html.unescape(v).strip() for k, v in HIDDEN_RE.findall(f)}
        t = TYPEK_RE.search(f)
        if not h.get("1") or not h.get("2"):
            continue
        rows.append({"code": h["1"], "name": h.get("0"), "market": MARKET.get(t.group(1) if t else "", None),
                     "time": pd.to_datetime(h["2"] + h.get("3", "000000").zfill(6), format="%Y%m%d%H%M%S",
                                            errors="coerce"),
                     "seq": h.get("5"), "subject": h.get("4"), "fact_date": pd.to_datetime(h.get("7"), format="%Y%m%d",
                                                                                       errors="coerce"),
                     "content": h.get("8")})
    return pd.DataFrame(rows, columns=["code", "name", "market", "time", "seq", "subject", "fact_date", "content"])


def build() -> pd.DataFrame:
    frames = [parse(p.read_text(encoding="utf-8")) for p in sorted(RAW.glob("*.html"))]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df = df.drop_duplicates(["code", "time", "seq", "subject"]).sort_values(["code", "time"])
    # 更名等「公告期間」型訊息在期間內每天 07:00 重複出現：標題與內文都相同的只留第一則
    key = df["subject"].str.replace(r"\s+", "", regex=True) + "|" + df["content"].fillna("").str.replace(r"\s+", "", regex=True)
    df = df[~key.duplicated() | df["code"].ne(df["code"].shift())].copy()
    df = df[~df.assign(k=key).duplicated(["code", "k"])].reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--sleep", type=float, default=4)
    ap.add_argument("--parse-only", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if not a.parse_only:
        today = date.today()
        days = [today - timedelta(days=k) for k in range(a.days, -1, -1)]
        todo = [d for d in days if (today - d).days < REFRESH_DAYS or not (RAW / f"{d:%Y%m%d}.html").exists()]
        print(f"要抓 {len(todo)} 天", flush=True)
        fails = 0
        for d in todo:
            try:
                page = fetch(d)
                if "hasBorder" not in page and NO_DATA not in page and "<form action='/mops/web/ajax_t05st02'" not in page:
                    raise RuntimeError("回應不是結果頁（可能查詢太頻繁）")
            except Exception as e:
                fails += 1
                print(f"{d} 失敗（第 {fails} 次）：{type(e).__name__} {str(e)[:80]}", flush=True)
                if fails >= 5:
                    sys.exit("連續失敗 5 次，可能被暫時封鎖；稍後重跑會接續")
                time.sleep(60 * fails)
                continue
            fails = 0
            (RAW / f"{d:%Y%m%d}.html").write_text(page, encoding="utf-8")
            print(f"{d}: {len(parse(page))} 則", flush=True)
            time.sleep(a.sleep)
    df = build()
    print(f"announcements.parquet: {len(df):,} 則，{df['code'].nunique()} 家，{df['time'].min()} ~ {df['time'].max()}")


if __name__ == "__main__":
    main()
