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
  - 最後的「基準比較」表另外換三種基準，檢查超額是不是只來自小型股、高波動或產業：
      市值×波動  同進場日、同市值五分位 × 60 日波動五分位的流動股等權平均（5×5 = 25 格）
      產業        同進場日、同產業的流動股等權平均
      0050        同期 0050 含息報酬
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


def daily_from_monthly(col, dates, codes):
    """factors_monthly 的月底欄位攤到每個交易日（用最近一個已過的月底值）。"""
    fm = pd.read_parquet("data/factors_monthly.parquet", columns=["date", "code", col])
    m = fm.pivot(index="date", columns="code", values=col)
    return m.reindex(m.index.union(dates)).ffill().reindex(index=dates, columns=codes)


def characteristics(adj, liquid):
    """每個 (交易日, 股票) 的市值×波動格子（11~55）與產業，只對流動股給值。"""
    dates, codes = adj.index, adj.columns
    mcap = daily_from_monthly("mcap", dates, codes).shift(1)   # 月底市值，下一個交易日起才用
    vol = adj.pct_change(fill_method=None).rolling(60, min_periods=40).std()
    ok = liquid & mcap.notna() & vol.notna()
    q = lambda x: np.ceil(x.where(ok).rank(axis=1, pct=True) * 5)
    bucket = q(mcap) * 10 + q(vol)
    ind = daily_from_monthly("industry", dates, codes).where(liquid)
    return bucket, ind


def matched_excess(fwd, groups, liquid):
    """個股報酬 − 同進場日、同組流動股的等權平均。"""
    long = pd.DataFrame({"r": fwd.where(liquid).stack(), "g": groups.stack()}).dropna()
    bench = long.groupby([long.index.get_level_values(0), "g"])["r"].transform("mean")
    return (long["r"] - bench).rename_axis(["entry", "code"])


def summarize(ev, label, tag, col="ex"):
    rows = []
    for h in HORIZONS:
        x = ev.dropna(subset=[f"{col}{h}"])
        if col != "ex":
            x = x.drop(columns=f"ex{h}").rename(columns={f"{col}{h}": f"ex{h}"})
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

    bucket, ind = characteristics(adj, liquid)
    for h in HORIZONS:
        fwd, ex = fwd_excess(adj, sus, liquid, h)
        s = ex.stack()
        s.index.names = ["entry", "code"]
        rev = rev.join(s.rename(f"ex{h}"), on=["entry", "code"])
        rev = rev.join(matched_excess(fwd, bucket, liquid).rename(f"sv{h}"), on=["entry", "code"])
        rev = rev.join(matched_excess(fwd, ind, liquid).rename(f"ind{h}"), on=["entry", "code"])
        etf = fwd.sub(fwd["0050"], axis=0).stack()
        etf.index.names = ["entry", "code"]
        rev = rev.join(etf.rename(f"etf{h}"), on=["entry", "code"])
    rev = rev[liquid.stack().reindex(pd.MultiIndex.from_frame(rev[["entry", "code"]])).fillna(False).values]
    b = bucket.stack()
    b.index.names = ["entry", "code"]
    rev = rev.join(b.rename("bucket"), on=["entry", "code"])

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

    def show(df, title):
        fmt = df.copy()
        for c in ["平均超額", "中位超額", "扣成本", "勝率"]:
            if c in fmt:
                fmt[c] = (fmt[c] * 100).round(2)
        fmt["t"] = fmt["t"].round(2)
        print(f"{title}\n")
        print(fmt.to_string(index=False))
        print()

    show(out, "超額報酬單位 %，相對同期流動股票池等權平均")
    out.to_csv("data/event_study.csv", index=False)

    # 基準比較
    benches = {"ex": "流動股池", "sv": "市值×波動", "ind": "產業", "etf": "0050"}
    rows = []
    for name in ("全部流動股公告", "創新高 且 年增>=30%", "年增>=30% 但未創新高"):
        for tag, (lo, hi) in periods.items():
            sel = rev[events[name] & (rev.entry >= lo) & (rev.entry < hi)]
            for col, bn in benches.items():
                rows += [{**r, "基準": bn} for r in summarize(sel, name, tag, col)]
    cmp_ = pd.DataFrame(rows)
    cmp_ = cmp_[["事件", "區間", "持有", "基準", "事件數", "平均超額", "中位超額", "扣成本", "勝率", "t"]]
    show(cmp_, "基準比較（超額單位 %）")
    cmp_.to_csv("data/event_study_bench.csv", index=False)

    # 事件股落在哪些市值×波動格子（1 = 最小 / 最低）
    hi = rev[events["創新高 且 年增>=30%"]].dropna(subset=["bucket"])
    allq = rev.dropna(subset=["bucket"])
    print("創新高且年增>=30% 的市值五分位分布 vs 全部公告：")
    print(pd.DataFrame({"事件": (hi.bucket // 10).value_counts(normalize=True).sort_index().round(3),
                        "全部": (allq.bucket // 10).value_counts(normalize=True).sort_index().round(3)}).to_string())
    print("60 日波動五分位分布：")
    print(pd.DataFrame({"事件": (hi.bucket % 10).value_counts(normalize=True).sort_index().round(3),
                        "全部": (allq.bucket % 10).value_counts(normalize=True).sort_index().round(3)}).to_string())


if __name__ == "__main__":
    main()
