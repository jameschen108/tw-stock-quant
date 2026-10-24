# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""市場狀態指標 + Top-N 曝險調整測試。

用法:
  uv run regime.py
  uv run regime.py --lookback 3 --tilt 0.5

狀態指標（換股日 t 當下可知，只用到 t 為止的報酬 rev_1m）:
  large_rs   上期市值前 20% 的股票，過去 k 個月等權報酬 − 全池等權報酬（累加）
             > 0 視為「大型股領漲」
產業動能: 每個產業過去 k 個月的成分股等權報酬（不指定 PCB / 記憶體，避免事後挑產業）

測試的調整（都在綜合分數 comp_equal 上加分，z 分數加權 tilt）:
  base            原綜合分數
  size_*          + tilt × 市值 z
  indmom_*        + tilt × 產業動能 z
  both_*          兩個都加
  *_always 一直加；*_regime 只在 large_rs > 0 時加
  big_regime      大型股領漲時只從市值前 40% 挑

判斷順序：先看 2011–2018（樣本內）哪個調整站得住，再看 2019 之後。
"""
import argparse

import numpy as np
import pandas as pd

from backtest import COST_ROUND_TRIP
from build_panel import OUT

POOL_MIN_VALUE = 2e7
PERIODS = [("2011-18", "2011-01", "2018-12"), ("2019-24", "2019-01", "2024-12"), ("2025-26", "2025-01", "2026-12")]


def zscore(s: pd.Series, by) -> pd.Series:
    g = s.groupby(by)
    return (s - g.transform("mean")) / g.transform("std")


def load() -> pd.DataFrame:
    f = pd.read_parquet(OUT / "factors_monthly.parquet",
                        columns=["date", "code", "industry", "mcap", "rev_1m", "avg_value", "n_obs", "close"])
    s = pd.read_parquet(OUT / "bt" / "composite_scores.parquet", columns=["date", "code", "fwd_ret", "comp_equal"])
    return f.merge(s, on=["date", "code"], how="left").sort_values(["code", "date"]).reset_index(drop=True)


def add_signals(d: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.Series]:
    # 個股過去 k 月報酬（連續 k 期都有資料才算）
    lr = np.log1p(d["rev_1m"])
    d["mom_k"] = np.expm1(lr.groupby(d["code"]).transform(lambda s: s.rolling(k, min_periods=k).sum()))

    # 產業動能：全市場（不限流動性）成分股等權，成分股少於 5 檔不算
    g = d.groupby(["date", "industry"])["mom_k"]
    d["ind_mom"] = g.transform("mean").where(g.transform("count") >= 5)

    # 大型股相對強弱：市值分組和流動性門檻都用上期值，避免「漲了才變大型股 / 才進池」的倖存者偏誤
    d["size_q_prev"] = d.groupby("code")["mcap"].shift().pipe(
        lambda s: s.groupby(d["date"]).rank(pct=True))
    liq = d.groupby("code")["avg_value"].shift() >= POOL_MIN_VALUE
    r_large = d[liq & (d["size_q_prev"] > 0.8)].groupby("date")["rev_1m"].mean()
    r_pool = d[liq & d["size_q_prev"].notna()].groupby("date")["rev_1m"].mean()
    large_rs = (r_large - r_pool).rolling(k, min_periods=k).sum()
    return d, large_rs


def select(d: pd.DataFrame, score: pd.Series, n: int, mask: pd.Series | None = None) -> pd.DataFrame:
    x = d.assign(s=score)
    if mask is not None:
        x = x[mask]
    picks = x.dropna(subset=["s"]).sort_values("s", ascending=False).groupby("date").head(n)
    ret = picks.groupby("date")["fwd_ret"].mean()
    sets = picks.groupby("date")["code"].apply(set)
    turn = pd.Series([1.0] + [1 - len(a & b) / n for b, a in zip(sets[:-1], sets[1:])], index=sets.index)
    big = picks.groupby("date")["size_pct"].mean()
    return pd.DataFrame({"ret": ret, "net": ret - turn * COST_ROUND_TRIP, "turn": turn, "size_pct": big})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=6, help="狀態指標與產業動能的回看月數")
    ap.add_argument("--tilt", type=float, default=0.5, help="加分權重（綜合分數 z 的倍數）")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()

    d, large_rs = add_signals(load(), a.lookback)
    d = d[(d["date"] >= "2011-01") & (d["avg_value"] >= POOL_MIN_VALUE) & d["fwd_ret"].notna()
          & (d["n_obs"] >= 120) & (d["close"] >= 5) & d["mcap"].notna()].copy()
    d["size_pct"] = d.groupby("date")["mcap"].rank(pct=True)
    on = d["date"].map(large_rs.gt(0))
    base = zscore(d["comp_equal"], d["date"])
    size_z = zscore(np.log(d["mcap"]), d["date"])
    ind_z = zscore(d["ind_mom"], d["date"]).fillna(0)
    t = a.tilt
    mkt = d.groupby("date")["fwd_ret"].mean()

    strategies = {
        "base": (base, None),
        "size_always": (base + t * size_z, None),
        "size_regime": (base + t * size_z * on, None),
        "indmom_always": (base + t * ind_z, None),
        "indmom_regime": (base + t * ind_z * on, None),
        "both_always": (base + t * (size_z + ind_z), None),
        "both_regime": (base + t * (size_z + ind_z) * on, None),
        "big_regime": (base, ~on | (d["size_pct"] > 0.6)),
    }

    reg = large_rs.reindex(mkt.index)
    print(f"回看 {a.lookback} 個月，tilt {t}，Top {a.top}，池 = 20 日均成交值 >= {POOL_MIN_VALUE / 1e6:.0f} 百萬\n")
    print("大型股領漲（large_rs > 0）的月份比例:",
          {p: f"{(reg.loc[s:e] > 0).mean():.0%}" for p, s, e in PERIODS})

    res = {k: select(d, sc, a.top, m) for k, (sc, m) in strategies.items()}

    # 1. 狀態指標有沒有用：base 在狀態開 / 關時的超額
    b = res["base"]
    ex = (b["net"] - mkt).dropna()
    st = reg.reindex(ex.index).gt(0).map({True: "大型股領漲", False: "其他"})
    print("\n[1] base Top-N 月均超額（%，扣成本），依狀態分:")
    tab = pd.DataFrame({p: ex.loc[s:e].groupby(st.loc[s:e]).mean() * 100 for p, s, e in PERIODS})
    print(tab.round(2).to_string())

    # 2. 各調整
    rows = []
    for k, r in res.items():
        e = (r["net"] - mkt).dropna()
        row = {"strategy": k}
        for p, s, en in PERIODS:
            x = e.loc[s:en]
            row[f"{p} 月超額%"] = x.mean() * 100
            row[f"{p} 勝率"] = (x > 0).mean()
        row["全期年化超額"] = (1 + e).prod() ** (12 / len(e)) - 1
        row["IR"] = e.mean() / e.std() * np.sqrt(12)
        row["換手"] = r["turn"].mean()
        row["市值百分位"] = r["size_pct"].mean()
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("strategy")
    fmt = {c: "{:+.2f}" for c in tab if "月超額" in c} | {c: "{:.0%}" for c in tab if "勝率" in c} | \
        {"全期年化超額": "{:+.1%}", "IR": "{:+.2f}", "換手": "{:.0%}", "市值百分位": "{:.2f}"}
    for c, f in fmt.items():
        tab[c] = tab[c].map(f.format)
    print("\n[2] Top-N 對同池等權的超額（扣成本）:")
    with pd.option_context("display.width", 220):
        print(tab.to_string())

    # 3. 每年
    print("\n[3] 每年超額（%，扣成本）:")
    yr = pd.DataFrame({k: (r["net"] - mkt).dropna().groupby(lambda x: x.year).apply(lambda s: (1 + s).prod() - 1)
                       for k, r in res.items()}).T * 100
    print(yr.round(1).to_string())


if __name__ == "__main__":
    main()
