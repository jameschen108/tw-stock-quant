# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""事件研究：月營收公告（與外資買超）後，個股相對同期流動股票池平均的超額報酬。

用法: uv run event_study.py [--min-value 5e7] [--start 2013-01-01]

時點規則：
  - 事件日 = 營收 avail_date；隔一個交易日收盤進場（和 factors.py 的 LAG=1 一致）
  - 持有 5 / 20 / 60 個交易日，窗內有 ret_suspect 的事件剔除
  - 超額 = 個股報酬 − 同進場日、同持有期、流動股票池（20 日均成交值 >= --min-value）的等權平均
  - 扣成本 = 來回 0.585%（手續費 0.1425%×2 + 證交稅 0.3%）
  - t 值：先對每個進場日取事件平均，再對「進場日」做 t（事件集中在每月 10 日前後，直接對事件算 t 會高估）
  - 營收公布日被推估的年份（avail_imputed）不納入
  - 籌碼資料只有 2024-10 起，籌碼條件的樣本只落在這段
"""
import argparse

import numpy as np
import pandas as pd

COST = 0.00585
HORIZONS = (5, 20, 60)


def load_prices(min_value):
    px = pd.read_parquet("data/prices_daily.parquet",
                         columns=["date", "code", "adj_close", "value", "ret_suspect"])
    adj = px.pivot(index="date", columns="code", values="adj_close")
    sus = px.pivot(index="date", columns="code", values="ret_suspect").reindex_like(adj).fillna(False).astype(float)
    val = px.pivot(index="date", columns="code", values="value").reindex_like(adj)
    liquid = val.rolling(20, min_periods=15).mean().shift(1) >= min_value
    return adj, sus, liquid


def fwd_excess(adj, sus, liquid, h):
    """每個 (進場日, 股票) 持有 h 日的報酬與超額報酬矩陣。"""
    fwd = adj.shift(-h) / adj - 1
    bad = sus.rolling(h, min_periods=1).sum().shift(-h) > 0
    fwd = fwd.mask(bad)
    bench = fwd.where(liquid).mean(axis=1)
    return fwd, fwd.sub(bench, axis=0)


def summarize(ev, label, tag):
    rows = []
    for h in HORIZONS:
        x = ev.dropna(subset=[f"ex{h}"])
        if len(x) < 30:
            rows.append(dict(事件=label, 區間=tag, 持有=h, 事件數=len(x)))
            continue
        by_date = x.groupby("entry")[f"ex{h}"].mean()
        by_m = x.groupby(x.entry.dt.to_period("M"))[f"ex{h}"].mean()  # 月份分群：持有期重疊時比逐日 t 保守
        t = by_m.mean() / (by_m.std(ddof=1) / np.sqrt(len(by_m))) if len(by_m) > 2 else np.nan
        rows.append(dict(事件=label, 區間=tag, 持有=h, 事件數=len(x), 進場日數=len(by_date),
                         平均超額=x[f"ex{h}"].mean(), 中位超額=x[f"ex{h}"].median(),
                         扣成本=x[f"ex{h}"].mean() - COST, 勝率=(x[f"ex{h}"] > 0).mean(), t=t))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-value", type=float, default=5e7)
    ap.add_argument("--start", default="2013-01-01")
    a = ap.parse_args()

    adj, sus, liquid = load_prices(a.min_value)
    dates = adj.index

    rev = pd.read_parquet("data/tej_rev.parquet")
    rev = rev[~rev.avail_imputed & (rev.avail_date >= a.start)].copy()
    rev = rev.rename(columns={"單月營收成長率％": "yoy", "累計營收成長率％": "cum_yoy",
                              "與歷史最高單月營收比%": "vs_high"})
    rev = rev[rev.code.isin(adj.columns)]
    pos = dates.searchsorted(rev.avail_date.values, side="right") + 1  # 公告日後第 2 個交易日收盤進場
    rev = rev[pos < len(dates)].copy()
    rev["entry"] = dates[pos[pos < len(dates)]]

    for h in HORIZONS:
        _, ex = fwd_excess(adj, sus, liquid, h)
        s = ex.stack()
        s.index.names = ["entry", "code"]
        rev = rev.join(s.rename(f"ex{h}"), on=["entry", "code"])
    rev = rev[liquid.stack().reindex(pd.MultiIndex.from_frame(rev[["entry", "code"]])).fillna(False).values]

    rev["is_high"] = rev.vs_high >= 0  # 欄位是相對「前」歷史高點的差距%，>=0 即創新高
    events = {
        "全部流動股公告": pd.Series(True, index=rev.index),
        "單月營收創歷史新高": rev.is_high,
        "年增>=30% 且 累計年增>=20%": (rev.yoy >= 30) & (rev.cum_yoy >= 20),
        "創新高 且 年增>=30%": rev.is_high & (rev.yoy >= 30),
        "年增>=30% 但未創新高": ~rev.is_high & (rev.yoy >= 30),
        "年增<=-20%": rev.yoy <= -20,
    }
    periods = {"2013–": ("2013-01-01", "2100-01-01"), "2013–2018": ("2013-01-01", "2019-01-01"),
               "2019–": ("2019-01-01", "2100-01-01"), "2024–": ("2024-01-01", "2100-01-01")}

    rows = []
    for name, m in events.items():
        for tag, (lo, hi) in periods.items():
            sel = rev[m & (rev.entry >= lo) & (rev.entry < hi)]
            rows += summarize(sel, name, tag)

    # 籌碼條件（2024-10 起）：外資近 5 日合計買超
    ins = pd.read_parquet("data/insti_daily.parquet")
    f = ins.pivot(index="date", columns="code", values="foreign")
    f5 = f.rolling(5, min_periods=5).sum()
    buy5 = (f.gt(0).rolling(5, min_periods=5).sum() == 5)
    sig = rev.entry.map(lambda d: dates[dates.get_loc(d) - 2])  # 公告日（進場前兩個交易日）
    ok = rev.entry >= "2024-11-01"
    r2 = rev[ok].copy()
    sd = pd.Series(sig[ok].values, index=r2.index)
    r2["f5"] = [f5.at[d, c] if (d in f5.index and c in f5.columns) else np.nan for d, c in zip(sd, r2.code)]
    r2["buy5"] = [buy5.at[d, c] if (d in buy5.index and c in buy5.columns) else np.nan for d, c in zip(sd, r2.code)]
    r2 = r2.dropna(subset=["f5"])
    hi_ev = r2.is_high & (r2.yoy >= 30)
    for name, m in {
        "（2024-11起）創新高且年增>=30%": hi_ev,
        "　＋外資5日合計買超": hi_ev & (r2.f5 > 0),
        "　＋外資連5日買超": hi_ev & (r2.buy5 == True),
        "　＋外資5日合計賣超": hi_ev & (r2.f5 < 0),
        "（2024-11起）年增>=30%（不看創高）": r2.yoy >= 30,
        "　＋外資5日合計買超（不看創高）": (r2.yoy >= 30) & (r2.f5 > 0),
        "　＋外資5日合計賣超（不看創高）": (r2.yoy >= 30) & (r2.f5 < 0),
    }.items():
        rows += summarize(r2[m], name, "2024-11–")

    # 逐年：主事件（年增>=30% 且 累計年增>=20%）
    main = events["年增>=30% 且 累計年增>=20%"]
    for y in range(2013, 2027):
        sel = rev[main & (rev.entry.dt.year == y)]
        rows += summarize(sel, "逐年：年增>=30%且累計>=20%", str(y))
        rows += summarize(rev[(rev.entry.dt.year == y)], "逐年：全部流動股公告", str(y))

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_rows", 500)
    fmt = out.copy()
    for c in ["平均超額", "中位超額", "扣成本", "勝率"]:
        if c in fmt:
            fmt[c] = (fmt[c] * 100).round(2)
    fmt["t"] = fmt["t"].round(2)
    print("超額報酬單位 %，相對同期流動股票池等權平均\n")
    print(fmt.to_string(index=False))
    out.to_csv("data/event_study.csv", index=False)


if __name__ == "__main__":
    main()
