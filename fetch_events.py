# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2"]
# ///
"""下載除權息、減資、變更面額、ETF 分割恢復買賣資料（還原股價用），原始 JSON 存到 data/raw/events/{source}/。

    uv run fetch_events.py --source twse      # TWT49U 除權息 + TWTAUU 減資 + TWTB8U 變更面額 + TWTCAU ETF 分割 / 反分割
    uv run fetch_events.py --source tpex      # exDailyQ 除權息 + revivt 減資 + pvChgRslt 變更面額 + etfSplitRslt ETF 分割

已存在的檔案會跳過；涵蓋到今天的那一段每次都重抓。被擋時多半回 HTML，連續失敗 5 次就停，重跑會接續。
"""
import argparse
import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "data" / "raw" / "events"
HEADERS = {"User-Agent": "Mozilla/5.0 (research)"}
START = "2010-01-01"

# (名稱, 網址, 分段頻率)；分段太長時交易所會截斷筆數
SOURCES = {
    "twse": [
        ("twt49u", "https://www.twse.com.tw/rwd/zh/exRight/TWT49U?startDate={s:%Y%m%d}&endDate={e:%Y%m%d}&response=json", "QS"),
        ("twtauu", "https://www.twse.com.tw/rwd/zh/reducation/TWTAUU?startDate={s:%Y%m%d}&endDate={e:%Y%m%d}&response=json", "YS"),
        ("twtb8u", "https://www.twse.com.tw/rwd/zh/change/TWTB8U?startDate={s:%Y%m%d}&endDate={e:%Y%m%d}&response=json", "YS"),
        ("twtcau", "https://www.twse.com.tw/rwd/zh/split/TWTCAU?startDate={s:%Y%m%d}&endDate={e:%Y%m%d}&response=json", "YS"),
    ],
    "tpex": [
        ("exdailyq", "https://www.tpex.org.tw/www/zh-tw/bulletin/exDailyQ?startDate={s:%Y}%2F{s:%m}%2F{s:%d}&endDate={e:%Y}%2F{e:%m}%2F{e:%d}&response=json", "MS"),
        ("revivt", "https://www.tpex.org.tw/www/zh-tw/bulletin/revivt?startDate={s:%Y}%2F{s:%m}%2F{s:%d}&endDate={e:%Y}%2F{e:%m}%2F{e:%d}&response=json", "YS"),
        ("pvchgrslt", "https://www.tpex.org.tw/www/zh-tw/bulletin/pvChgRslt?startDate={s:%Y}%2F{s:%m}%2F{s:%d}&endDate={e:%Y}%2F{e:%m}%2F{e:%d}&response=json", "YS"),
        ("etfsplitrslt", "https://www.tpex.org.tw/www/zh-tw/bulletin/etfSplitRslt?startDate={s:%Y}%2F{s:%m}%2F{s:%d}&endDate={e:%Y}%2F{e:%m}%2F{e:%d}&response=json", "YS"),
    ],
}


def chunks(freq: str):
    today = pd.Timestamp(date.today())
    starts = pd.date_range(START, today, freq=freq)
    for s in starts:
        e = min(s + pd.tseries.frequencies.to_offset(freq) - pd.Timedelta(days=1), today)
        yield s, e, e == today


def rows(js: dict) -> list:
    if "tables" in js:
        return [r for t in js["tables"] for r in (t.get("data") or [])]
    return js.get("data") or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), required=True)
    ap.add_argument("--sleep", type=float, default=6)
    a = ap.parse_args()
    for name, url_fmt, freq in SOURCES[a.source]:
        out_dir = OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        fails = 0
        for s, e, is_open in chunks(freq):
            path = out_dir / f"{s:%Y%m%d}_{e:%Y%m%d}.json"
            if path.exists() and not is_open:
                continue
            url = url_fmt.format(s=s, e=e)
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                    raw = r.read()
                js = json.loads(raw)
            except Exception as ex:
                fails += 1
                print(f"{name} {s.date()} 失敗（第 {fails} 次）：{type(ex).__name__} {str(ex)[:80]}", flush=True)
                if fails >= 5:
                    sys.exit("連續失敗 5 次，可能被暫時封鎖；稍後重跑會接續")
                time.sleep(60 * fails)
                continue
            fails = 0
            # 開放中的那一段可能每次不同，舊檔先刪掉
            for old in out_dir.glob(f"{s:%Y%m%d}_*.json"):
                old.unlink()
            path.write_bytes(raw)
            print(f"{name} {s.date()}~{e.date()}: {len(rows(js))} 筆", flush=True)
            time.sleep(a.sleep)
        print(f"{name}: 完成", flush=True)


if __name__ == "__main__":
    main()
