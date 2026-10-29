# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""主題指數與強弱排行：依 themes.toml 的成分股算每日等權指數，列出各期間報酬與相對 0050 的超額。

用法: uv run themes.py [--asof 2026-10-22] [--theme memory_ic]

  - 主題指數 = 成分股每日報酬（還原股價，ret_suspect 的日子剔除）的等權平均，每天再平衡
  - 沒有行情的成分股當天不計入；上市前或停牌的股票不會拉低指數
  - 上漲比例 = 近 20 日報酬為正的成分股占比；成交值倍數 = 近 5 日平均成交值 / 近 60 日平均
  - 成分股是現在手挑的，歷史指數有事後偏差，只看當下相對強弱與資金輪動，不拿來回測
  - 結果另存 data/themes_latest.csv
"""
import argparse
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ETF = "0050"
HORIZONS = (1, 5, 20, 60)


def load_themes(path=ROOT / "themes.toml") -> dict[str, dict]:
    with open(path, "rb") as f:
        themes = tomllib.load(f)["themes"]
    for k, t in themes.items():
        codes = [str(c) for c in t["codes"]]
        if len(set(codes)) != len(codes):
            raise ValueError(f"主題 {k} 有重複代號")
        t["codes"] = codes
    return themes


def theme_returns(rets: pd.DataFrame, codes: list[str]) -> pd.Series:
    """成分股的等權日報酬；當天沒有任何成分有報酬則為 NaN。"""
    cols = [c for c in codes if c in rets.columns]
    return rets[cols].mean(axis=1, skipna=True) if cols else pd.Series(np.nan, index=rets.index)


def window_return(daily: pd.Series, n: int) -> float:
    """最近 n 個交易日的複利報酬。"""
    w = daily.dropna().iloc[-n:]
    return float((1 + w).prod() - 1) if len(w) == n else np.nan


def load_prices(asof=None, since=None):
    px = pd.read_parquet(ROOT / "data/prices_daily.parquet",
                         columns=["date", "code", "adj_close", "value", "ret_suspect"],
                         filters=[("date", ">=", pd.Timestamp(since))] if since is not None else None)
    if asof is not None:
        px = px[px["date"] <= pd.Timestamp(asof)]
    adj = px.pivot(index="date", columns="code", values="adj_close")
    sus = px.pivot(index="date", columns="code", values="ret_suspect").reindex_like(adj).fillna(False).astype(bool)
    val = px.pivot(index="date", columns="code", values="value").reindex_like(adj)
    rets = adj.pct_change(fill_method=None).mask(sus)
    return rets, val


def index_levels(daily: pd.Series, n: int) -> pd.Series:
    """最近 n 個交易日的累積指數，窗口起點前一天 = 1。"""
    return (1 + daily.dropna().iloc[-n:]).cumprod()


def summarize(themes: dict[str, dict], rets: pd.DataFrame, val: pd.DataFrame) -> pd.DataFrame:
    bench = rets[ETF] if ETF in rets.columns else pd.Series(np.nan, index=rets.index)
    mkt_val = val.sum(axis=1)
    rows = []
    for key, t in themes.items():
        have = [c for c in t["codes"] if c in rets.columns and rets[c].notna().iloc[-20:].any()]
        d = theme_returns(rets, have)
        row = {"theme": key, "name": t["name"], "n": len(t["codes"]), "n_data": len(have)}
        for h in HORIZONS:
            row[f"r{h}"] = window_return(d, h)
        for h in (5, 20, 60):
            row[f"ex{h}"] = row[f"r{h}"] - window_return(bench, h)
        r20 = rets[have].iloc[-20:].add(1).prod() - 1 if have else pd.Series(dtype=float)
        row["up_ratio"] = float((r20 > 0).mean()) if len(r20) else np.nan
        share = val[have].sum(axis=1) / mkt_val if have else pd.Series(np.nan, index=val.index)
        row["value_share_5d"] = float(share.iloc[-5:].mean())
        row["value_mult"] = float(share.iloc[-5:].mean() / share.iloc[-60:].mean()) if share.iloc[-60:].mean() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("r20", ascending=False).reset_index(drop=True)


def members(theme: dict, rets: pd.DataFrame, names: pd.Series) -> pd.DataFrame:
    rows = []
    for c in theme["codes"]:
        r = rets[c] if c in rets.columns else pd.Series(dtype=float)
        rows.append({"code": c, "name": names.get(c, "?"), **{f"r{h}": window_return(r, h) for h in HORIZONS}})
    return pd.DataFrame(rows).sort_values("r20", ascending=False)


def fmt(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in out.columns:
        if c[0] == "r" and c[1:].isdigit() or c.startswith("ex") or c in ("up_ratio", "value_share_5d"):
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x * 100:.1f}%")
        elif c == "value_mult":
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.2f}x")
    return out.to_string(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", help="只用這天以前的行情")
    ap.add_argument("--theme", help="列出某主題的成分股報酬（themes.toml 的 key）")
    a = ap.parse_args()
    themes = load_themes()
    rets, val = load_prices(a.asof)
    print(f"資料到 {rets.index.max().date()}")
    missing = {k: [c for c in t["codes"] if c not in rets.columns] for k, t in themes.items()}
    for k, m in missing.items():
        if m:
            print(f"警告：主題 {k} 有代號沒有行情：{m}")
    if a.theme:
        names = pd.read_parquet(ROOT / "data/prices_daily.parquet", columns=["date", "code", "name"]) \
            .sort_values("date").groupby("code")["name"].last()
        print(fmt(members(themes[a.theme], rets, names)))
        return
    s = summarize(themes, rets, val)
    s.to_csv(ROOT / "data/themes_latest.csv", index=False)
    print(fmt(s.drop(columns=["theme"])))
    print("\n成分股為手動挑選，有事後偏差，只看當下相對強弱，不構成投資建議。")


if __name__ == "__main__":
    main()
