# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""即時報價：證交所基本市況報導（mis.twse.com.tw，上市、上櫃都有，延遲約 5 秒）。只給前端顯示，不進任何計算或紀錄。

用法:
  uv run realtime.py 2330 6488      # 印出即時報價（測試用）

規則:
  - 上市查 tse_、上櫃查 otc_；市場看 prices_daily 最近的 market 欄，查不到的兩邊都查
  - 盤中還沒成交（成交價是 "-"）時用最佳買價當參考價，標 indicative
  - 一次最多 50 檔一個請求；連線失敗或格式不對回傳空 dict，前端照常顯示收盤資料
"""
import json
import sys
import time
import urllib.request
from datetime import datetime, time as dtime

import pandas as pd

from build_panel import OUT

URL = "https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch={}&json=1&delay=0&_={}"
HEADERS = {"User-Agent": "Mozilla/5.0 (research)", "Referer": "https://mis.twse.com.tw/stock/index.jsp"}
OPEN, CLOSE = dtime(9, 0), dtime(13, 35)   # 13:30 收盤，多留幾分鐘等最後一筆
BATCH = 50


def session_open(now: datetime | None = None) -> bool:
    """平日 09:00–13:35（國定假日看不出來，假日查到的報價日期會是前一個交易日）。"""
    now = now or datetime.now()
    return now.weekday() < 5 and OPEN <= now.time() < CLOSE


def markets(codes: list[str]) -> dict[str, str]:
    """{代號: "tse" | "otc"}，看 prices_daily 最近 30 天的 market 欄。"""
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "market"],
                         filters=[("code", "in", list(codes)),
                                  ("date", ">=", pd.Timestamp.today().normalize() - pd.Timedelta(days=30))])
    last = px.sort_values("date").groupby("code")["market"].last()
    return {c: ("otc" if m == "TPEX" else "tse") for c, m in last.items()}


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v > 0 else None


def _parse(m: dict) -> dict:
    bids = [_num(x) for x in (m.get("b") or "").split("_") if _num(x)]
    asks = [_num(x) for x in (m.get("a") or "").split("_") if _num(x)]
    last = _num(m.get("z"))
    prev = _num(m.get("y"))
    ref = last or (bids[0] if bids else None) or (asks[0] if asks else None)
    d = m.get("d") or ""
    return {"code": m.get("c"), "name": m.get("n"), "last": ref, "indicative": last is None and ref is not None,
            "prev": prev, "chg": (ref / prev - 1) if ref and prev else None,
            "open": _num(m.get("o")), "high": _num(m.get("h")), "low": _num(m.get("l")),
            # 沒有漲跌幅限制的（例如部分 ETF）證交所給漲停 9999.95、跌停空白
            "limit_up": _num(m.get("u")) if (_num(m.get("u")) or 0) < 9999 else None,
            "limit_down": _num(m.get("w")), "volume": _num(m.get("v")),
            "bid": bids[0] if bids else None, "ask": asks[0] if asks else None,
            "date": pd.Timestamp(d) if len(d) == 8 else None, "time": m.get("t")}


def quotes(codes: list[str]) -> dict[str, dict]:
    """{代號: 報價}；last 是成交價（沒有成交時是最佳買價、indicative=True），chg 是相對昨收。"""
    codes = [str(c) for c in dict.fromkeys(codes) if c]
    if not codes:
        return {}
    try:
        mk = markets(codes)
    except Exception:
        mk = {}
    keys = []
    for c in codes:
        keys += [f"{mk[c]}_{c}.tw"] if c in mk else [f"tse_{c}.tw", f"otc_{c}.tw"]
    out = {}
    for i in range(0, len(keys), BATCH):
        url = URL.format("|".join(keys[i:i + BATCH]), int(time.time() * 1000))
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=8) as r:
                arr = json.loads(r.read().decode("utf-8")).get("msgArray", [])
        except (OSError, ValueError):
            continue
        for m in arr:
            q = _parse(m)
            if q["code"]:
                out[q["code"]] = q
    return out


def main():
    codes = sys.argv[1:] or ["2330", "0050"]
    q = quotes(codes)
    for c in codes:
        if c not in q:
            print(f"{c}：查不到")
            continue
        x = q[c]
        chg = f"{x['chg']:+.2%}" if x["chg"] is not None else "—"
        print(f"{c} {x['name']}  {x['last']}{'（參考）' if x['indicative'] else ''}  {chg}  "
              f"昨收 {x['prev']}  漲停 {x['limit_up']} 跌停 {x['limit_down']}  {x['date'] and x['date'].date()} {x['time']}")


if __name__ == "__main__":
    main()
