# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""抓籌碼日報，原始 JSON 存 data/raw/{kind 目錄}/{market}/YYYYMMDD.json。

  insti   三大法人買賣超（證交所 T86、櫃買 dailyTrade）→ data/raw/chips/
  margin  融資融券餘額（證交所 MI_MARGN、櫃買 margin/balance）→ data/raw/margin/
  qfii    外資及陸資持股比率（證交所 MI_QFIIS、櫃買 insti/qfii）→ data/raw/qfii/

用法:
  uv run fetch_chips.py --market twse                    # 三大法人，近 180 天的交易日，已抓過的跳過
  uv run fetch_chips.py --market tpex --kind margin --days 730
  uv run fetch_chips.py --build                          # 整理成 data/{insti,margin,qfii}_daily.parquet

交易日取自 prices_daily.parquet（有行情的日子），所以要在 build_prices.py 之後跑。
load_chips() / load_margin() / load_qfii() 把原始檔整理成 一列 = 股票 × 日。
"""
import argparse
import json
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
OUT = RAW / "chips"
DIRS = {"insti": RAW / "chips", "margin": RAW / "margin", "qfii": RAW / "qfii"}
PARQUET = {k: ROOT / "data" / f"{k}_daily.parquet" for k in DIRS}   # --build 產生，load_* 優先讀
HEADERS = {"User-Agent": "Mozilla/5.0 (research)"}
_TPEX_D = "date={d:%Y}%2F{d:%m}%2F{d:%d}&response=json"
URLS = {
    "insti": {"twse": "https://www.twse.com.tw/rwd/zh/fund/T86?date={d:%Y%m%d}&selectType=ALLBUT0999&response=json",
              "tpex": "https://www.tpex.org.tw/www/zh-tw/insti/dailyTrade?type=Daily&sect=EW&" + _TPEX_D},
    "margin": {"twse": "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?date={d:%Y%m%d}&selectType=ALL&response=json",
               "tpex": "https://www.tpex.org.tw/www/zh-tw/margin/balance?" + _TPEX_D},
    "qfii": {"twse": "https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?date={d:%Y%m%d}&selectType=ALLBUT0999&response=json",
             "tpex": "https://www.tpex.org.tw/www/zh-tw/insti/qfii?" + _TPEX_D},
}
# 欄位位置：外資（含外資自營商）、投信、自營商、三大法人合計
COLS = {
    "twse": {"foreign": [4, 7], "trust": [10], "dealer": [11], "total": [18]},
    "tpex": {"foreign": [10], "trust": [13], "dealer": [22], "total": [23]},
}


# 融資融券 / 外資持股：(代號, 欄位) 的位置；張、股、% 依來源
MARGIN_COLS = {"twse": {"margin_bal": 6, "margin_limit": 7, "short_bal": 12, "offset": 14},
               "tpex": {"margin_bal": 6, "margin_limit": 9, "short_bal": 14, "offset": 18}}
QFII_COLS = {"twse": {"code": 0, "shares_issued": 3, "foreign_shares": 5, "foreign_pct": 7, "limit_pct": 8},
             "tpex": {"code": 1, "shares_issued": 3, "foreign_shares": 5, "foreign_pct": 7, "limit_pct": 8}}


def rows(market: str, js: dict, kind: str = "insti") -> list:
    if js.get("data"):              # 證交所 T86、MI_QFIIS 的資料在最外層
        return js["data"]
    tables = js.get("tables") or []
    if market == "twse" and kind == "margin":   # 第一張是全市場統計，第二張才是個股
        tables = [t for t in tables if "彙總" in (t.get("title") or "")]
    return [r for t in tables for r in (t.get("data") or [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["twse", "tpex"])
    ap.add_argument("--build", action="store_true", help="只把原始檔整理成 data/{insti,margin,qfii}_daily.parquet")
    ap.add_argument("--kind", choices=list(URLS), default="insti")
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--sleep", type=float, default=3)
    a = ap.parse_args()
    if a.build:
        for k, f in {"insti": load_chips, "margin": load_margin, "qfii": load_qfii}.items():
            df = f(raw=True)
            df.to_parquet(PARQUET[k], index=False)
            print(f"{PARQUET[k].name}: {len(df):,} 列，{df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}" if len(df)
                  else f"{PARQUET[k].name}: 0 列")
        return
    if not a.market:
        ap.error("要指定 --market（或用 --build）")
    out = DIRS[a.kind] / a.market
    out.mkdir(parents=True, exist_ok=True)
    px = pd.read_parquet(ROOT / "data" / "prices_daily.parquet", columns=["date", "market"])
    days = sorted(px.loc[px["market"] == a.market.upper(), "date"].unique())
    days = [pd.Timestamp(d) for d in days if pd.Timestamp(d) >= pd.Timestamp(date.today() - timedelta(days=a.days))]
    todo = [d for d in days if not (out / f"{d:%Y%m%d}.json").exists()]
    print(f"{a.kind} {a.market}: {len(todo)} 天要抓", flush=True)
    fails = 0
    for d in todo:
        try:
            with urllib.request.urlopen(urllib.request.Request(URLS[a.kind][a.market].format(d=d), headers=HEADERS),
                                        timeout=60) as r:
                raw = r.read()
            n = len(rows(a.market, json.loads(raw), a.kind))
        except Exception as e:
            fails += 1
            print(f"{d.date()} 失敗（第 {fails} 次）：{type(e).__name__}", flush=True)
            if fails >= 5:
                sys.exit("連續失敗 5 次，稍後重跑會接續")
            time.sleep(60 * fails)
            continue
        fails = 0
        if n:
            (out / f"{d:%Y%m%d}.json").write_bytes(raw)
        print(f"{d.date()}: {n} 筆", flush=True)
        time.sleep(a.sleep)
    print(f"{a.kind} {a.market}: 完成", flush=True)


def _num(x) -> float:
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return float("nan")


def _cached(kind: str, code: str | None) -> pd.DataFrame | None:
    if not PARQUET[kind].exists():
        return None
    return pd.read_parquet(PARQUET[kind], filters=[("code", "==", code)] if code else None)


def load_chips(code: str | None = None, raw: bool = False) -> pd.DataFrame:
    """一列 = 股票 × 日：foreign / trust / dealer / total 買賣超股數。"""
    if not raw and (df := _cached("insti", code)) is not None:
        return df
    recs = []
    for market, cols in COLS.items():
        for p in sorted((OUT / market).glob("*.json")):
            d = pd.Timestamp(p.stem)
            for r in rows(market, json.loads(p.read_text())):
                c = str(r[0]).strip()
                if code and c != code:
                    continue
                recs.append({"date": d, "code": c, **{k: sum(_num(r[i]) for i in idx) for k, idx in cols.items()}})
    return pd.DataFrame(recs, columns=["date", "code", *COLS["twse"]])


def _load(kind: str, cols: dict, code: str | None, code_col: dict) -> list[dict]:
    recs = []
    for market, idx in cols.items():
        for p in sorted((DIRS[kind] / market).glob("*.json")):
            d = pd.Timestamp(p.stem)
            for r in rows(market, json.loads(p.read_text()), kind):
                c = str(r[code_col[market]]).strip()
                if code and c != code:
                    continue
                recs.append({"date": d, "code": c, **{k: _num(str(r[i]).rstrip("%")) for k, i in idx.items()}})
    return recs


def load_margin(code: str | None = None, raw: bool = False) -> pd.DataFrame:
    """一列 = 股票 × 日：融資餘額、融資限額、融券餘額、資券互抵（張）。"""
    if not raw and (df := _cached("margin", code)) is not None:
        return df
    return pd.DataFrame(_load("margin", MARGIN_COLS, code, {"twse": 0, "tpex": 0}),
                        columns=["date", "code", *MARGIN_COLS["twse"]])


def load_qfii(code: str | None = None, raw: bool = False) -> pd.DataFrame:
    """一列 = 股票 × 日：發行股數、外資及陸資持股數、持股比率 %、法令上限 %。"""
    if not raw and (df := _cached("qfii", code)) is not None:
        return df
    cols = {m: {k: i for k, i in c.items() if k != "code"} for m, c in QFII_COLS.items()}
    return pd.DataFrame(_load("qfii", cols, code, {m: c["code"] for m, c in QFII_COLS.items()}),
                        columns=["date", "code", *cols["twse"]])


if __name__ == "__main__":
    main()
