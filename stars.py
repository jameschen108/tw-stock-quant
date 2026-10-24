# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""仿晨星星等：護城河 + 不確定性 + 公允價值 → 1–5 星，並回測。

用法:
  uv run stars.py                      # 最新一期星等分布，與幾檔個股的卡片
  uv run stars.py --code 2330,2412     # 指定個股卡片
  uv run stars.py --backtest           # 回測星等（與 ep、bp 比較）

公允價值（剩餘收益模型，每股淨值的倍數）:
  FV / B = 1 + Σ (ROE_t − r) × B_{t−1}/B_0 / (1+r)^t
  - 起點 ROE = 近四季 ROE 與 10 年 ROE 中位數各半，上限 ROE_CAP
  - ROE 在 FADE_YEARS[護城河] 年內線性回到資金成本 r，之後不再創造超額價值
  - 淨值以 ROE × (1 − PAYOUT) 成長
  - P/FV = 市值 / (FV/B × 淨值) = 1 / (bp × FV/B)

不確定性: 每期橫斷面，10 年 ROE 標準差、年營收成長率標準差、近一年日報酬波動度三者百分位平均，
依 UNC_CUTS 分成 Low / Medium / High / Very High。

星等: 依不確定性用不同的 P/FV 區間（STAR_BANDS），不確定性越高，要折價越多才給 5 星。
時點: 護城河與 10 年統計用 avail_date 之前已公布的年度；ep、bp 與價格沿用 factors_monthly 的時點規則。
"""
import argparse

import numpy as np
import pandas as pd

from backtest import COST_ROUND_TRIP, run_factor, summarize, universe
from build_panel import OUT
from moat import MIN_YEARS, annual, build

R = 0.09                                   # 權益資金成本
PAYOUT = 0.5
ROE_CAP = 0.40
FADE_YEARS = {"Wide": 20, "Narrow": 10, "None": 3}
UNC = ["Low", "Medium", "High", "Very High"]
UNC_CUTS = [0, 0.25, 0.60, 0.85, 1.0]
# 不確定性 -> P/FV 上界：(5 星, 4 星, 3 星, 2 星)；超過最後一個是 1 星
STAR_BANDS = {"Low": (0.80, 0.95, 1.05, 1.25), "Medium": (0.70, 0.90, 1.10, 1.30),
              "High": (0.60, 0.85, 1.15, 1.45), "Very High": (0.50, 0.80, 1.20, 1.80)}


def fv_multiple(roe0: np.ndarray, fade: np.ndarray) -> np.ndarray:
    """剩餘收益模型的 FV/B；roe0 與 fade（年）為等長陣列。"""
    out = np.ones_like(roe0)
    book = np.ones_like(roe0)
    for t in range(1, max(FADE_YEARS.values()) + 1):
        roe_t = np.where(t <= fade, roe0 + (R - roe0) * (t - 1) / fade, R)
        out += (roe_t - R) * book / (1 + R) ** t
        book *= 1 + roe_t * (1 - PAYOUT)
    return out


def stock_vol() -> pd.DataFrame:
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "ret", "ret_suspect"])
    px = px[px["code"].str.fullmatch(r"[1-9]\d{3}")]
    px["ret"] = px["ret"].where(~px["ret_suspect"].fillna(False).astype(bool))
    ret = px.pivot(index="date", columns="code", values="ret")
    vol = ret.rolling(250, min_periods=120).std() * np.sqrt(250)
    return vol.stack().rename("vol_1y").reset_index()


def rate(df: pd.DataFrame) -> pd.DataFrame:
    """df 需有 date, tier, roe_med, roe_std, rev_g_std, vol_1y, ep, bp；加上 unc, p_fv, stars。"""
    d = df.dropna(subset=["tier", "bp", "ep"]).copy()
    d = d[d["bp"] > 0]
    d["roe_ttm"] = d["ep"] / d["bp"]
    d["roe0"] = (0.5 * d["roe_ttm"] + 0.5 * d["roe_med"] / 100).clip(-ROE_CAP, ROE_CAP)
    d["fv_b"] = fv_multiple(d["roe0"].to_numpy(), d["tier"].map(FADE_YEARS).to_numpy(float))
    d = d[d["fv_b"] > 0]
    # 固定 r 的絕對估值會隨利率與市場情緒整體偏高或偏低（2026 年中位數 P/FV 約 2.5），
    # 星等改用「相對當期市場」：除以當期中位數，p_fv_abs 另外保留
    d["p_fv_abs"] = 1 / (d["bp"] * d["fv_b"])
    d["mkt_p_fv"] = d.groupby("date")["p_fv_abs"].transform("median")
    d["p_fv"] = d["p_fv_abs"] / d["mkt_p_fv"]

    risk = d[["roe_std", "rev_g_std", "vol_1y"]].groupby(d["date"]).rank(pct=True).mean(axis=1, skipna=True)
    d["unc"] = pd.cut(risk, UNC_CUTS, labels=UNC, include_lowest=True).astype(str)
    bands = np.array([STAR_BANDS[u] for u in d["unc"]])
    d["stars"] = 5 - (d["p_fv"].to_numpy()[:, None] > bands).sum(axis=1)
    return d


def load(start: str = "2011-01", min_years: int = MIN_YEARS) -> pd.DataFrame:
    m = build(annual(), min_years).dropna(subset=["avail_date"]).sort_values("avail_date")
    df = pd.read_parquet(OUT / "factors_monthly.parquet")
    df = df[df["date"] >= start].sort_values("date")
    df = pd.merge_asof(df, m[["code", "avail_date", "tier", "roe_med", "roe_std", "rev_g_std"]],
                       left_on="date", right_on="avail_date", by="code", allow_exact_matches=False)
    df = df.merge(stock_vol(), on=["date", "code"], how="left")
    return rate(df)


def card(r: pd.Series) -> str:
    return (f"{r['code']}  {r['date']:%Y-%m-%d}  {'★' * r['stars']}{'☆' * (5 - r['stars'])}  "
            f"收盤 {r['close']:,.1f}  公允價值 {r['close'] / r['p_fv']:,.1f}  P/FV {r['p_fv']:.2f}  "
            f"（絕對 {r['close'] / r['p_fv_abs']:,.1f}、{r['p_fv_abs']:.2f}）  "
            f"護城河 {r['tier']}  不確定性 {r['unc']}  "
            f"（ROE 起點 {r['roe0']:.1%}＝近四季 {r['roe_ttm']:.1%} 與 10 年中位數 {r['roe_med'] / 100:.1%} 平均）")


def show_latest(d: pd.DataFrame, codes: list[str]):
    cur = d[d["date"] == d["date"].max()]
    print(f"{cur['date'].iloc[0]:%Y-%m-%d}，有星等 {len(cur)} 家；絕對 P/FV 中位數 {cur['mkt_p_fv'].iloc[0]:.2f}"
          f"（r = {R:.0%}，這個數字越高代表市場整體越貴）")
    yr = d.groupby(d["date"].dt.year)["mkt_p_fv"].last()
    print("各年底絕對 P/FV 中位數:", " ".join(f"{y}:{v:.2f}" for y, v in yr.items()), "\n")
    print(pd.crosstab(cur["unc"], cur["stars"], margins=True).reindex(UNC + ["All"]).to_string(), "\n")
    print(pd.crosstab(cur["tier"], cur["stars"], margins=True).to_string(), "\n")
    for c in codes:
        r = cur[cur["code"] == c]
        print(card(r.iloc[0]) if len(r) else f"{c}: 沒有星等（完整年度不足或淨值為負）")


def backtest(d: pd.DataFrame, top_cost: bool = True):
    d = universe(d, 5, 5e6, 120).dropna(subset=["fwd_ret"])
    d = d[d.groupby("date")["code"].transform("size") >= 100]
    print(f"{d['date'].min():%Y-%m}~{d['date'].max():%Y-%m}，{d['date'].nunique()} 期，"
          f"平均每期 {d.groupby('date').size().mean():.0f} 檔（有星等、通過流動性門檻）\n")

    print("星等分布（各期平均占比）:")
    print(d.groupby("date")["stars"].value_counts(normalize=True).unstack().mean().round(3).to_string(), "\n")

    d["ex_mkt"] = d["fwd_ret"] - d.groupby("date")["fwd_ret"].transform("mean")
    d["ex_ind"] = d["fwd_ret"] - d.groupby(["date", "industry"])["fwd_ret"].transform("mean")
    rows = []
    for s in range(1, 6):
        g = d[d["stars"] == s]
        r = {"stars": s, "平均檔數": g.groupby("date").size().mean()}
        for k, lab in (("ex_mkt", "對全體"), ("ex_ind", "對同產業")):
            x = g.groupby("date")[k].mean()
            r[f"{lab}年化超額"] = x.mean() * 12
            r[f"{lab}t"] = x.mean() / x.std() * np.sqrt(len(x))
        rows.append(r)
    print("各星等下月報酬（等權、未扣成本）:")
    print(pd.DataFrame(rows).set_index("stars").round(3).to_string(), "\n")

    # 5 星投組（扣成本）：換手率 = 本期不在上期名單的比例
    five = d[d["stars"] == 5].groupby("date")["code"].apply(set)
    turn = pd.Series([1.0] + [1 - len(a & b) / max(len(b), 1) for a, b in zip(five[:-1], five[1:])],
                     index=five.index)
    ex5 = d[d["stars"] == 5].groupby("date")["ex_mkt"].mean() - turn * COST_ROUND_TRIP
    ls = d[d["stars"] == 5].groupby("date")["fwd_ret"].mean() - d[d["stars"] == 1].groupby("date")["fwd_ret"].mean()
    print(f"5 星投組：平均換手 {turn.mean():.0%}，扣成本對全體年化超額 {ex5.mean() * 12:+.1%}"
          f"（t {ex5.mean() / ex5.std() * np.sqrt(len(ex5)):+.1f}）；5 星 − 1 星 年化 {ls.mean() * 12:+.1%}"
          f"（t {ls.mean() / ls.std() * np.sqrt(len(ls)):+.1f}）\n")
    print("每年 5 星對全體超額（扣成本，月超額加總）與 5 星 − 1 星:")
    print(pd.DataFrame({"5星超額": ex5.groupby(ex5.index.year).sum(), "5星−1星": ls.groupby(ls.index.year).sum()})
          .T.map("{:+.1%}".format).to_string(), "\n")

    # 連續分數：與 ep、bp 在同一個樣本比較，產業 + 市值中性
    d["fv_gap"] = -np.log(d["p_fv"])
    d["stars_score"] = d["stars"] + d["fv_gap"] * 1e-3            # 同星等內再依折價排序
    rows = [summarize(f, run_factor(d, f, "ind_size")) for f in ["stars_score", "fv_gap", "ep", "bp"]]
    tab = pd.DataFrame(rows).set_index("factor")[["IC", "t", "IC>0", "LS_ann", "LS_sharpe", "Q5ex_net", "turn"]]
    print("連續分數比較（同樣本、產業 + 市值中性，五分位）:")
    print(tab.round(3).to_string())
    print("\nstars_score = 星等（同星等內依 P/FV 排序）；fv_gap = −log(P/FV)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--code", default="2330,2308,2412,2603,2408,6488,2884,3008")
    ap.add_argument("--min-years", type=int, default=MIN_YEARS,
                    help="護城河視窗內最少完整年度；非預設值時輸出 stars_my{N}.parquet")
    args = ap.parse_args()
    d = load(min_years=args.min_years)
    out = "stars.parquet" if args.min_years == MIN_YEARS else f"stars_my{args.min_years}.parquet"
    d[["date", "code", "close", "tier", "unc", "roe0", "fv_b", "p_fv_abs", "p_fv", "stars"]].to_parquet(OUT / out)
    if args.backtest:
        backtest(d)
    else:
        show_latest(d, args.code.split(","))


if __name__ == "__main__":
    main()
