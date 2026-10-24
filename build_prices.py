# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""把證交所 / 櫃買的每日全市場行情整理成 parquet，並用除權息、減資、變更面額、ETF 分割事件算還原因子。

用法: uv run build_prices.py

輸入:
  data/prices/{twse,tpex}/YYYYMMDD.json   每日收盤行情（fetch_prices.py 抓的）
  data/raw/events/*/*.json   除權息、減資、變更面額、ETF 分割 / 反分割（fetch_events.py）

輸出 (data/):
  price_events.parquet   一列 = 一次除權息 / 減資 / 變更面額恢復買賣，factor = 參考價 / 前收盤
  prices_daily.parquet   一列 = 股票 × 交易日，未還原 OHLCV 加上
                           ret         含息日報酬（除權息、減資、分割日改用參考價為基準）
                           adj_factor  後復權因子，adj_close = close * adj_factor（最新一天 = 1）
                           ret_suspect 報酬超過漲跌幅限制又沒有對應事件（多半是缺事件資料），回測時建議排除
"""
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DAILY = ROOT / "data" / "prices"
EVENTS = ROOT / "data" / "raw" / "events"
OUT = ROOT / "data"

# 普通股 / 特別股 (2881A)、ETF (0050, 00631L)、TDR (911613)；排除權證、牛熊證
CODE_RE = re.compile(r"^(\d{4}[A-Z]?|00\d{2,4}[A-Z]?|91\d{4})$")

TWSE_COLS = {"證券代號": "code", "證券名稱": "name", "開盤價": "open", "最高價": "high", "最低價": "low",
             "收盤價": "close", "成交股數": "volume", "成交金額": "value", "成交筆數": "trades",
             "漲跌(+/-)": "sign"}
TPEX_COLS = {"代號": "code", "名稱": "name", "開盤": "open", "最高": "high", "最低": "low", "收盤": "close",
             "成交股數": "volume", "成交金額(元)": "value", "成交筆數": "trades", "發行股數": "shares",
             "次日 參考價": "ref_next"}
NUM_COLS = ["open", "high", "low", "close", "volume", "value", "trades", "shares", "ref_next"]


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def roc_date(s: pd.Series) -> pd.Series:
    """'113/06/04'、'113年06月04日'、'1130604' -> Timestamp"""
    d = s.astype(str).str.replace(r"\D", "", regex=True).str.zfill(7)
    return pd.to_datetime((d.str[:3].astype(int) + 1911).astype(str) + d.str[3:], format="%Y%m%d")


def parse_day(args) -> pd.DataFrame | None:
    market, path = args
    js = json.loads(path.read_text())
    key, cols = ("證券代號", TWSE_COLS) if market == "twse" else ("代號", TPEX_COLS)
    frames = []
    for t in js.get("tables") or []:
        fields = t.get("fields") or []
        if key in fields and t.get("data"):
            df = pd.DataFrame(t["data"], columns=fields[:len(t["data"][0])])
            frames.append(df[[c for c in cols if c in df]].rename(columns=cols))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["code"] = df["code"].str.strip()
    df = df[df["code"].str.match(CODE_RE)].copy()
    df["name"] = df["name"].str.strip()
    for c in NUM_COLS:
        if c in df:
            df[c] = to_num(df[c])
    if "sign" in df:  # 證交所「X」= 不比價（除權息、減資恢復等）
        df["no_compare"] = df.pop("sign").str.contains("X", na=False)
    df.insert(0, "date", pd.Timestamp(path.stem))
    df.insert(2, "market", market.upper())
    return df


def load_daily() -> pd.DataFrame:
    jobs = [(m, p) for m in ("twse", "tpex") for p in sorted((DAILY / m).glob("[0-9]*.json"))]
    with ProcessPoolExecutor() as ex:
        parts = [d for d in ex.map(parse_day, jobs, chunksize=32) if d is not None]
    df = pd.concat(parts, ignore_index=True)
    df.loc[df["close"] <= 0, "close"] = np.nan
    # 同一天同代號只會出現一次；萬一重複（轉板當天）保留有成交的
    df = df.sort_values(["code", "date", "volume"]).drop_duplicates(["code", "date"], keep="last")
    df["market"] = df["market"].astype("category")
    df["no_compare"] = df["no_compare"].astype("boolean")
    return df.reset_index(drop=True)


def load_events() -> pd.DataFrame:
    # (資料夾, 市場, 類型, 前收盤欄位位置, 參考價欄位位置)
    specs = [("twt49u", "TWSE", "div", 3, 4), ("twtauu", "TWSE", "reduction", 3, 4),
             ("twtb8u", "TWSE", "par_change", 3, 4), ("twtcau", "TWSE", "split", 4, 5),
             ("exdailyq", "TPEX", "div", 3, 4), ("revivt", "TPEX", "reduction", 3, 4),
             ("pvchgrslt", "TPEX", "par_change", 3, 4), ("etfsplitrslt", "TPEX", "split", 3, 4)]
    frames = []
    for folder, market, kind, i_prev, i_ref in specs:
        rows = []
        for p in sorted((EVENTS / folder).glob("*.json")) if (EVENTS / folder).exists() else []:
            js = json.loads(p.read_text())
            data = [r for t in js.get("tables", [js]) for r in (t.get("data") or [])]
            rows += [[r[0], str(r[1]).strip(), str(r[2]).strip(), r[i_prev], r[i_ref], kind]
                     for r in data]
        df = pd.DataFrame(rows, columns=["date", "code", "name", "prev_close", "ref", "kind"])
        df["date"] = roc_date(df["date"])
        df["prev_close"], df["ref"] = to_num(df["prev_close"]), to_num(df["ref"])
        df.insert(2, "market", market)
        frames.append(df)
    ev = pd.concat(frames, ignore_index=True).drop_duplicates(["date", "code", "kind"])
    ev = ev[(ev["prev_close"] > 0) & (ev["ref"] > 0)]
    ev["factor"] = ev["ref"] / ev["prev_close"]
    return ev.sort_values(["code", "date"]).reset_index(drop=True)


def price_limit(date: pd.Series) -> pd.Series:
    """漲跌幅限制：2015-06-01 起 10%，之前 7%。"""
    return np.where(date >= pd.Timestamp("2015-06-01"), 0.10, 0.07)


def add_adjustment(df: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    # 每檔股票事件因子的累積乘積 C(t) = Π_{事件日 <= t} factor；停牌期間的事件也會算進恢復交易那天
    ev_day = ev.groupby(["code", "date"])["factor"].prod().rename("ev_factor").reset_index()
    df = df.merge(ev_day, on=["code", "date"], how="left")
    orphan = ev_day.merge(df[["code", "date"]], on=["code", "date"], how="left", indicator=True)
    orphan = orphan[orphan["_merge"] == "left_only"].drop(columns="_merge")
    # 事件日當天沒有行情列（停牌）：併到該股下一個交易日
    if len(orphan):
        nxt = pd.merge_asof(orphan.sort_values("date"), df[["code", "date"]].rename(columns={"date": "next"})
                            .sort_values("next"), left_on="date", right_on="next", by="code",
                            direction="forward").dropna(subset=["next"])
        extra = nxt.groupby(["code", "next"])["ev_factor"].prod().rename("ev_extra").reset_index()
        df = df.merge(extra.rename(columns={"next": "date"}), on=["code", "date"], how="left")
        df["ev_factor"] = df["ev_factor"].fillna(1) * df.pop("ev_extra").fillna(1)
    df["ev_factor"] = df["ev_factor"].fillna(1.0)

    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    g = df.groupby("code", sort=False)
    log_c = np.log(df["ev_factor"]).groupby(df["code"], sort=False).cumsum()
    tr = df["close"] / np.exp(log_c)                       # 含息總報酬指數（尚未標準化）
    prev_tr = tr.groupby(df["code"], sort=False).transform(lambda s: s.ffill().shift())
    df["ret"] = tr / prev_tr - 1
    df["adj_factor"] = np.exp(g["ev_factor"].transform(lambda s: np.log(s).sum()) - log_c)
    df["adj_close"] = df["close"] * df["adj_factor"]

    # 距離上次有成交隔了 n 列（中間是無成交日），上限按 n 天複利放寬
    row = g.cumcount()
    last_valid = row.where(df["close"].notna()).groupby(df["code"], sort=False).transform(
        lambda s: s.ffill().shift())
    n = (row - last_valid).clip(lower=1)
    lim = (1 + price_limit(df["date"]) + 0.005) ** n - 1
    big = (df["ret"] > lim) | (df["ret"] < (1 - price_limit(df["date"]) - 0.005) ** n - 1)
    first_days = row < 5                                    # 新上市前五日無漲跌幅限制
    is_etf = df["code"].str.startswith("00")                # 部分 ETF（國外成分）無漲跌幅限制
    df["ret_suspect"] = big & ~first_days & ~is_etf
    return df


def main():
    print("daily ..."); df = load_daily()
    print(f"  {len(df):,} rows, {df['code'].nunique()} codes, {df['date'].min().date()} ~ {df['date'].max().date()}")
    print("events ..."); ev = load_events()
    print(f"  {len(ev):,} events:", ev.groupby(["market", "kind"]).size().to_dict())
    df = add_adjustment(df, ev)
    ev.to_parquet(OUT / "price_events.parquet", index=False)
    df.to_parquet(OUT / "prices_daily.parquet", index=False)
    print(f"ret_suspect: {int(df['ret_suspect'].sum())} rows "
          f"({df.loc[df['ret_suspect'], 'code'].nunique()} codes)")


if __name__ == "__main__":
    main()
