# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""把營收動能、品質、價值、趨勢四組因子合成綜合分數，切樣本內 / 樣本外回測。

用法:
  uv run composite.py                       # 樣本內 2011–2018，樣本外 2019 起
  uv run composite.py --split 2017-01 --top 50 --top-min-value 3e7
  uv run composite.py --trend none          # 不含趨勢組（2026-10 以前的三組版本）
  uv run composite.py --value replace       # 價值組改用 fv_gap + sp（stars.py）

合成步驟（每個換股日）:
  1. 各因子做產業 + 市值中性（同 backtest.py），再標準化成 z 分數
  2. 同一組內等權平均，得到各組分數（組內至少一半因子有值）:
       營收動能 rev_accel、rev_vs_high、rev_yoy_3m
       品質     roe、op_margin、cfo_ta、gm_chg、−accruals
       價值     ep、bp、sp
       趨勢     from_high（距 252 日最高收盤）、ma200_gap（收盤 / 200 日均線 − 1），由日價格計算
  3. 各組再加權成綜合分數，權重三種:
       equal    各組等權，不用任何報酬資料
       is_ic    用樣本內各組平均 IC 當權重（負的歸零），固定套用到樣本外
       wf_ic    滾動：每期只用到 t−2 期為止的 IC（t−1 期的報酬當時還沒走完），前 24 期用等權

趨勢組（--trend，預設 always）: 加入後前 30 名扣成本超額，樣本內 +8.9% → +13.5%、樣本外 +1.9% → +11.4%
（comp_equal；逐月配對 t 為 1.8 / 2.5）。趨勢組自身 IC 很低（樣本外 0.009），綜合 IC 也略降，
推測效果主要是把基本面好但股價破底的股票擠出前段（未逐檔驗證）。--trend bull 只在大盤收在 200 日線之上時加入，
和 always 沒有顯著差異。輸出檔名不加後綴的是預設設定（analyst.py、report.py、diagnose.py 讀這些檔）。

注意：因子挑選本身是看過 2011–2026 全期單因子結果後決定的，樣本外仍有一點選擇偏誤；
趨勢組的兩個訊號也是看過 technicals.py 全期結果後挑的。組內成員是按常見分類放的，沒有再依樣本外表現調整。
"""
import argparse

import numpy as np
import pandas as pd

from backtest import COST_ROUND_TRIP, N_Q, evaluate, neutralize, perf, universe
from build_panel import OUT

GROUPS = {
    "rev_mom": {"rev_accel": 1, "rev_vs_high": 1, "rev_yoy_3m": 1},
    "quality": {"roe": 1, "op_margin": 1, "cfo_ta": 1, "gm_chg": 1, "accruals": -1},
    "value":   {"ep": 1, "bp": 1, "sp": 1},
}
WF_MIN = 24
# --value 的價值組成員；fv_gap = −log(P/FV)，來自 stars.py 輸出的 data/stars.parquet。
# fv_gap 要 8 個完整年度才有值，2018-03 起才有，樣本內大多缺值（組內過半門檻下價值組會只剩其餘成員）
VALUE_SETS = {
    "base":    {"ep": 1, "bp": 1, "sp": 1},
    "replace": {"fv_gap": 1, "sp": 1},
    "add":     {"ep": 1, "bp": 1, "sp": 1, "fv_gap": 1},
    "sp_only": {"sp": 1},                     # 對照：拿掉 ep、bp 但不放 fv_gap
}
# --trend: 趨勢組（technicals.py 回測過的兩個訊號，取連續值）
#   always  一直加入
#   bull    只在大盤（股票池市值加權指數）收在 200 日線之上時加入；空頭時該組缺值，其餘組權重重新歸一
TREND = {"from_high": 1, "ma200_gap": 1}


def trend_features(dates) -> tuple[pd.DataFrame, pd.Series]:
    """換股日收盤時的 from_high（距 252 日最高收盤）、ma200_gap（收盤 / 200 日均線 − 1），與大盤是否多頭。"""
    import technicals as tc
    w = tc.load()
    uni = tc.universe(w)
    c = w["adj_close"].ffill(limit=5)
    dates = pd.DatetimeIndex(dates)
    feats = {
        "from_high": (c / c.rolling(252, min_periods=200).max() - 1).loc[dates],
        "ma200_gap": (c / tc.ma(c, 200) - 1).loc[dates],
    }
    out = pd.concat({k: v.stack() for k, v in feats.items()}, axis=1).rename_axis(["date", "code"]).reset_index()
    lvl = (1 + tc.market_index(w, uni)["cap"]).cumprod()
    bull = (lvl > tc.ma(lvl, 200)).where(tc.ma(lvl, 200).notna()).reindex(dates)
    return out, bull


def zscore(s: pd.Series, by) -> pd.Series:
    g = s.groupby(by)
    return (s - g.transform("mean")) / g.transform("std")


def group_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["date", "code", "fwd_ret", "avg_value"]].copy()
    for gname, members in GROUPS.items():
        z = pd.concat({f: sign * zscore(neutralize(df, f, "ind_size"), df["date"])
                       for f, sign in members.items()}, axis=1)
        enough = z.notna().sum(axis=1) >= len(members) / 2
        out[gname] = zscore(z.mean(axis=1).where(enough), df["date"])
    return out


def group_ic(gs: pd.DataFrame) -> pd.DataFrame:
    r = gs.groupby("date")["fwd_ret"].rank(pct=True)
    return pd.DataFrame({g: gs.assign(r=r).groupby("date").apply(
        lambda x: x[g].corr(x["r"]), include_groups=False) for g in GROUPS})


def weights(ic: pd.DataFrame, split: str) -> dict[str, pd.DataFrame]:
    dates, names = ic.index, list(GROUPS)
    eq = pd.DataFrame(1 / len(names), index=dates, columns=names)
    w_is = ic[dates < split].mean().clip(lower=0)
    is_ic = pd.DataFrame([w_is / w_is.sum()] * len(dates), index=dates)
    hist = ic.shift(2).expanding(min_periods=WF_MIN).mean().clip(lower=0)
    wf = hist.div(hist.sum(axis=1), axis=0).where(hist.sum(axis=1) > 0)
    wf = wf.fillna(eq)
    return {"equal": eq, "is_ic": is_ic, "wf_ic": wf}


def combine(gs: pd.DataFrame, w: pd.DataFrame) -> pd.Series:
    names = list(GROUPS)
    wt = w.reindex(gs["date"]).set_axis(gs.index)[names]
    z = gs[names]
    has = z.notna()
    # 某組缺值時，用其餘組的權重重新歸一
    return (z.fillna(0) * wt).sum(axis=1) / (wt * has).sum(axis=1).replace(0, np.nan)


def top_n(d: pd.DataFrame, n: int, min_value: float) -> pd.DataFrame:
    """流動性較高的股票中取分數前 n 名等權；回傳每期毛報酬、換手率、扣成本淨報酬。"""
    pool = d[(d["avg_value"] >= min_value)].dropna(subset=["fwd_ret"])
    mkt = pool.groupby("date")["fwd_ret"].mean()                 # 基準：同池全部等權（不論有無分數）
    picks = pool.dropna(subset=["score"]).sort_values("score", ascending=False).groupby("date").head(n)
    ret = picks.groupby("date")["fwd_ret"].mean()
    sets = picks.groupby("date")["code"].apply(set)
    turn = pd.Series([1.0] + [1 - len(a & b) / n for b, a in zip(sets[:-1], sets[1:])], index=sets.index)
    mkt = mkt.reindex(ret.index)
    return pd.DataFrame({"gross": ret, "turn": turn, "net": ret - turn * COST_ROUND_TRIP, "mkt": mkt})


def period_stats(res: dict, top: pd.DataFrame, mask_dates) -> dict:
    ic = res["ic"][res["ic"].index.isin(mask_dates)].dropna()
    q = res["qret"][res["qret"].index.isin(mask_dates)]
    turn = res["turnover"].reindex(q.index)
    ls = q[N_Q] - q[1]
    q5ex = q[N_Q] - q["mkt"] - turn.fillna(1) * COST_ROUND_TRIP
    t = top[top.index.isin(mask_dates)]
    return {
        "IC": ic.mean(), "t": ic.mean() / ic.std() * np.sqrt(len(ic)), "IC>0": (ic > 0).mean(),
        "LS_ann": perf(ls)["ann"], "LS_sharpe": perf(ls)["sharpe"],
        "Q5ex_net": perf(q5ex)["ann"], "Q5_turn": turn.mean(),
        "Top_net": perf(t["net"])["ann"], "Top_ex": perf(t["net"] - t["mkt"])["ann"],
        "Top_sharpe": perf(t["net"])["sharpe"], "Top_mdd": perf(t["net"])["mdd"],
        "Mkt": perf(t["mkt"])["ann"], "n_months": len(ic),
    }


FMT = {"IC": "{:+.3f}", "t": "{:+.1f}", "IC>0": "{:.0%}", "LS_ann": "{:+.1%}", "LS_sharpe": "{:+.2f}",
       "Q5ex_net": "{:+.1%}", "Q5_turn": "{:.0%}", "Top_net": "{:+.1%}", "Top_ex": "{:+.1%}",
       "Top_sharpe": "{:+.2f}", "Top_mdd": "{:.0%}", "Mkt": "{:+.1%}", "n_months": "{:.0f}"}


def show(rows: list[dict]):
    tab = pd.DataFrame(rows).set_index(["period", "score"])
    for c, f in FMT.items():
        tab[c] = tab[c].map(f.format)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(tab.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2011-01")
    ap.add_argument("--split", default="2019-01", help="樣本外起始月")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--top-min-value", type=float, default=2e7, help="Top-N 投組的 20 日均成交值下限（元）")
    ap.add_argument("--value", choices=list(VALUE_SETS), default="base", help="價值組成員")
    ap.add_argument("--fv-file", default="stars.parquet", help="fv_gap 來源（data/ 下，stars.py 輸出）")
    ap.add_argument("--trend", choices=["none", "always", "bull"], default="always", help="趨勢組")
    a = ap.parse_args()
    GROUPS["value"] = VALUE_SETS[a.value]
    if a.trend != "none":
        GROUPS["trend"] = TREND
    print(f"價值組: {list(GROUPS['value'])}；趨勢組: {a.trend}")

    df = pd.read_parquet(OUT / "factors_monthly.parquet")
    if "fv_gap" in GROUPS["value"]:
        st = pd.read_parquet(OUT / a.fv_file, columns=["date", "code", "p_fv"])
        df = df.merge(st.assign(fv_gap=-np.log(st["p_fv"]))[["date", "code", "fv_gap"]],
                      on=["date", "code"], how="left")
    if a.trend != "none":
        tf, bull = trend_features(df["date"].unique())
        df = df.merge(tf, on=["date", "code"], how="left")
    df = universe(df[df["date"] >= a.start], 5, 5e6, 120).reset_index(drop=True)
    gs = group_scores(df)
    if a.trend == "bull":
        gs["trend"] = gs["trend"].where(gs["date"].map(bull).fillna(0).astype(bool))
        print(f"大盤多頭（加入趨勢組）的月份比例: {bull[bull.index >= a.start].mean():.0%}")
    gic = group_ic(gs)
    W = weights(gic, a.split)

    dates = pd.DatetimeIndex(sorted(gs["date"].unique()))
    periods = {"IS": dates[dates < a.split], "OOS": dates[dates >= a.split]}
    print(f"樣本內 {periods['IS'][0]:%Y-%m}~{periods['IS'][-1]:%Y-%m}（{len(periods['IS'])} 期），"
          f"樣本外 {periods['OOS'][0]:%Y-%m}~{periods['OOS'][-1]:%Y-%m}（{len(periods['OOS'])} 期）\n")

    print("各組 IC（樣本內 / 樣本外）:")
    print(pd.DataFrame({p: gic[gic.index.isin(ds)].mean() for p, ds in periods.items()}).round(3).to_string())
    print("\n分組分數相關（全期平均）:")
    print(gs.groupby("date")[list(GROUPS)].corr().groupby(level=1).mean().loc[list(GROUPS)].round(2).to_string())
    print("\nis_ic 權重:", W["is_ic"].iloc[0].round(2).to_dict())
    print("wf_ic 權重（樣本外期間平均）:", W["wf_ic"].loc[periods["OOS"]].mean().round(2).to_dict())

    rows, yearly, navs = [], {}, {}
    candidates = {g: gs[g] for g in GROUPS} | {f"comp_{k}": combine(gs, w) for k, w in W.items()}
    for name, score in candidates.items():
        d = gs.assign(score=score)
        res = evaluate(d)
        top = top_n(d, a.top, a.top_min_value)
        for p, ds in periods.items():
            rows.append({"period": p, "score": name, **period_stats(res, top, ds)})
        yearly[name] = (top["net"] - top["mkt"]).groupby(top.index.year).apply(lambda r: (1 + r).prod() - 1)
        navs[name] = top["net"]
    navs["mkt"] = top["mkt"]

    print(f"\n回測（Top = 20 日均成交值 >= {a.top_min_value / 1e6:.0f} 百萬中前 {a.top} 名等權，扣成本）:")
    show(rows)
    print("\nIC/t/LS/Q5ex: 全股票池五分位；Top_net: Top-N 年化淨報酬；Top_ex: 對同池等權的年化超額；Mkt: 同池等權年化")
    print("\nTop-N 每年超額（扣成本，對同池等權）:")
    print(pd.DataFrame(yearly).T.map("{:+.1%}".format).to_string())

    out = OUT / "bt"
    out.mkdir(exist_ok=True)
    sfx = "" if a.value == "base" else f"_{a.value}"
    if a.value != "base" and a.fv_file != "stars.parquet":
        sfx += "_" + a.fv_file.removesuffix(".parquet").removeprefix("stars_")
    if a.trend != "always":
        sfx += f"_trend_{a.trend}"
    pd.DataFrame(navs).to_parquet(out / f"composite_top_returns{sfx}.parquet")
    gs.assign(**{k: v for k, v in candidates.items() if k.startswith("comp_")}) \
        .to_parquet(out / f"composite_scores{sfx}.parquet", index=False)


if __name__ == "__main__":
    main()
