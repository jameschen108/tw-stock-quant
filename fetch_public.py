# /// script
# requires-python = ">=3.11"
# ///
"""抓證交所 / 櫃買 OpenAPI 的最新月營收、季財報、公司基本資料，存成每日快照。

用法: uv run fetch_public.py

OpenAPI 只提供最新一期（月營收最新月份、財報最新一季、基本資料目前狀態），所以要每天抓、
靠快照累積歷史。快照存在 data/raw/openapi/{name}/{YYYYMMDD}.json；內容和上一份相同就不存。
某檔公司某一期第一次出現在快照的日期，就是它的「可用日」（public_data.py 用）。
"""
import hashlib
import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

OUT = Path(__file__).parent / "data" / "raw" / "openapi"
HEADERS = {"User-Agent": "Mozilla/5.0 (research)"}
TWSE = "https://openapi.twse.com.tw/v1/opendata/"
TPEX = "https://www.tpex.org.tw/openapi/v1/"
TEMPLATES = ["ci", "basi", "bd", "fh", "ins", "mim"]   # 一般業、銀行、證券、金控、保險、其他

ENDPOINTS = {
    "rev_L": TWSE + "t187ap05_L", "rev_O": TPEX + "mopsfin_t187ap05_O",
    "basic_L": TWSE + "t187ap03_L", "basic_O": TPEX + "mopsfin_t187ap03_O",
    **{f"is_{t}_L": TWSE + f"t187ap06_L_{t}" for t in TEMPLATES},
    **{f"is_{t}_O": TPEX + f"mopsfin_t187ap06_O_{t}" for t in TEMPLATES},
    **{f"bs_{t}_L": TWSE + f"t187ap07_L_{t}" for t in TEMPLATES},
    **{f"bs_{t}_O": TPEX + f"mopsfin_t187ap07_O_{t}" for t in TEMPLATES},
}


def fetch(url: str) -> bytes:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                raw = r.read()
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("不是 JSON 陣列")
            return raw
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  重試 {url}：{type(e).__name__} {str(e)[:60]}", flush=True)
            time.sleep(10 * (attempt + 1))


def main():
    today = date.today().strftime("%Y%m%d")
    failed = []
    for name, url in ENDPOINTS.items():
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        try:
            raw = fetch(url)
        except Exception as e:
            print(f"{name}: 失敗 {type(e).__name__} {str(e)[:80]}", flush=True)
            failed.append(name)
            continue
        prev = sorted(d.glob("*.json"))
        same = prev and hashlib.sha256(prev[-1].read_bytes()).digest() == hashlib.sha256(raw).digest()
        n = len(json.loads(raw))
        if same:
            print(f"{name}: {n} 筆，與上次相同", flush=True)
        else:
            (d / f"{today}.json").write_bytes(raw)
            print(f"{name}: {n} 筆，已存 {today}", flush=True)
        time.sleep(1)
    if failed:
        sys.exit(f"失敗: {', '.join(failed)}")


if __name__ == "__main__":
    main()
