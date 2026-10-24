# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2"]
# ///
"""下載上市（證交所 MI_INDEX）與上櫃（櫃買中心 dailyQuotes）每日全部股票的收盤行情。

用法:
  uv run fetch_prices.py --market twse
  uv run fetch_prices.py --market tpex --start 2026-10-01 --end 2026-10-22

每個平日與週六（補班日）一個請求，非交易日回空資料也會記下來。檔案存到 data/prices/{market}/YYYYMMDD.json，
manifest.jsonl 記錄每一天的網址、抓取時間、筆數與 sha256；中斷後重跑會跳過已經記錄的日子。
請求間隔預設 4.5 秒：兩個交易所都會封鎖太頻繁的請求。
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
PRICES = ROOT / "data" / "prices"

URLS = {
    "twse": "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX?date={d:%Y%m%d}&type=ALLBUT0999&response=json",
    "tpex": "https://www.tpex.org.tw/www/zh-tw/afterTrading/dailyQuotes?date={d:%Y}%2F{d:%m}%2F{d:%d}&type=EW&response=json",
}
HEADERS = {"User-Agent": "Mozilla/5.0 (research)"}


def n_rows(market: str, js: dict) -> int:
    """行情表的筆數；非交易日為 0。"""
    for t in js.get("tables") or []:
        fields = t.get("fields") or []
        if market == "twse" and "證券代號" in fields:
            return len(t.get("data") or [])
        if market == "tpex" and "代號" in fields:
            return len(t.get("data") or [])
    return 0


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=list(URLS), required=True)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=datetime.now().date().isoformat())
    ap.add_argument("--sleep", type=float, default=4.5)
    a = ap.parse_args()
    out_dir = PRICES / a.market
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    done = {p.stem for p in out_dir.glob("[0-9]*.json")}
    if manifest.exists():
        done |= {json.loads(line)["date"] for line in manifest.read_text().splitlines() if line.strip()}

    # 平日加週六：補班日的週六照常交易，一般週六會回空資料
    days = [d for d in pd.date_range(a.start, a.end)
            if d.dayofweek != 6 and d.strftime("%Y%m%d") not in done]
    print(f"{a.market}: 要抓 {len(days)} 天（已完成 {len(done)} 天），預估 {len(days) * a.sleep / 3600:.1f} 小時", flush=True)
    fails = 0
    for i, d in enumerate(days):
        url = URLS[a.market].format(d=d)
        try:
            raw = fetch(url)
            js = json.loads(raw)
        except Exception as e:   # 被擋時通常回 HTML 或連線錯誤：等久一點再試，連續失敗就停
            fails += 1
            print(f"{d.date()} 失敗（第 {fails} 次）：{type(e).__name__} {str(e)[:80]}", flush=True)
            if fails >= 5:
                sys.exit("連續失敗 5 次，可能被暫時封鎖；稍後重跑會從這裡接續")
            time.sleep(60 * fails)
            continue
        fails = 0
        n = n_rows(a.market, js)
        key = d.strftime("%Y%m%d")
        if n:
            (out_dir / f"{key}.json").write_bytes(raw)
        rec = {"date": key, "status": "ok" if n else "empty", "n_rows": n, "url": url,
               "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
               "fetched_on": datetime.now().astimezone().isoformat(timespec="seconds")}
        with open(manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 100 == 0:
            print(f"{d.date()} {rec['status']} {n} 筆（{i + 1}/{len(days)}）", flush=True)
        time.sleep(a.sleep)
    print(f"{a.market}: 完成", flush=True)


if __name__ == "__main__":
    main()
