# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""月頻橫斷面因子回測：rank IC 與五分位投組。

用法:
  uv run backtest.py                              # 全部因子，產業 + 市值中性
  uv run backtest.py --neutral none --start 2015-01
  uv run backtest.py --factors rev_yoy,ep --by-year

輸入 data/factors_monthly.parquet（factors.py）。輸出 data/bt/ 下的 IC 序列與分位報酬。

股票池（每個換股日）: 普通股、當天有成交、上市滿 120 個交易日、收盤 >= min_price、
20 日均成交值 >= min_value。
報酬: 等權；成本以 Q5 換手率計，買 0.1425% + 賣 0.1425% + 證交稅 0.3%。
"""
import argparse

import numpy as np
import pandas as pd

from build_panel import OUT
from factors import FACTORS

COST_ROUND_TRIP = 0.001425 * 2 + 0.003
N_Q = 5
MIN_NAMES = 100


def universe(df: pd.DataFrame, min_price: float, min_value: float, min_obs: int) -> pd.DataFrame:
    keep = (df["close"] >= min_price) & (df["avg_value"] >= min_value) & (df["n_obs"] >= min_obs)
    return df[keep].copy()


def rank_pct(s: pd.Series, by) -> pd.Series:
    return s.groupby(by).rank(pct=True)


def neutralize(df: pd.DataFrame, col: str, mode: str) -> pd.Series:
    """回傳中性化後的分數（先轉每期百分位，再扣產業平均 / 對市值回歸取殘差）。"""
    y = rank_pct(df[col], df["date"])
    if mode == "none":
        return y
    grp = [df["date"], df["industry"].fillna("NA")]
    y = y - y.groupby(grp).transform("mean")
    if mode == "ind":
        return y
    # Frisch–Waugh：在產業內去平均後，再對 log 市值做每期的單變數回歸
    s = df["size"] - df["size"].groupby(grp).transform("mean")
    ok = y.notna() & s.notna()
    ys, ss = y.where(ok), s.where(ok)
    beta = (ys * ss).groupby(df["date"]).transform("sum") / (ss ** 2).groupby(df["date"]).transform("sum")
    return y - beta * s


def run_factor(df: pd.DataFrame, col: str, mode: str) -> dict:
    d = df[list(dict.fromkeys(["date", "code", "industry", "size", col, "fwd_ret"]))].copy()
    d["score"] = neutralize(d, col, mode if col != "size" else ("ind" if mode != "none" else "none"))
    return evaluate(d)


def evaluate(d: pd.DataFrame) -> dict:
    """d 需有 date, code, score, fwd_ret；回傳每期 IC、分位報酬、Q5 換手率、檔數。"""
    d = d.dropna(subset=["score", "fwd_ret"])
    d = d[d.groupby("date")["code"].transform("size") >= MIN_NAMES]     # 資料太少的期別不算
    d["ret_rank"] = rank_pct(d["fwd_ret"], d["date"])

    ic = d.groupby("date").apply(lambda x: x["score"].corr(x["ret_rank"]), include_groups=False)
    d["q"] = np.ceil(d.groupby("date")["score"].rank(method="first", pct=True) * N_Q).astype(int)
    qret = d.groupby(["date", "q"])["fwd_ret"].mean().unstack()
    qret["mkt"] = d.groupby("date")["fwd_ret"].mean()

    # Q5 換手率：本期 Q5 中上期不在 Q5 的比例
    top = d[d["q"] == N_Q].groupby("date")["code"].apply(set)
    turn = pd.Series([np.nan] + [1 - len(a & b) / max(len(a), 1) for b, a in zip(top[:-1], top[1:])],
                     index=top.index)
    return {"ic": ic, "qret": qret, "turnover": turn, "n": d.groupby("date").size()}


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty:
        return {"ann": np.nan, "vol": np.nan, "sharpe": np.nan, "mdd": np.nan}
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    return {"ann": ann, "vol": vol, "sharpe": r.mean() * 12 / vol if vol else np.nan,
            "mdd": (nav / nav.cummax() - 1).min()}


def summarize(name: str, res: dict) -> dict:
    ic = res["ic"].dropna()
    q = res["qret"]
    ls = q[N_Q] - q[1]
    excess = q[N_Q] - q["mkt"] - res["turnover"].fillna(1) * COST_ROUND_TRIP
    mono = q[list(range(1, N_Q + 1))].mean()
    return {
        "factor": name,
        "IC": ic.mean(), "ICIR": ic.mean() / ic.std() * np.sqrt(12), "t": ic.mean() / ic.std() * np.sqrt(len(ic)),
        "IC>0": (ic > 0).mean(),
        "LS_ann": perf(ls)["ann"], "LS_sharpe": perf(ls)["sharpe"], "LS_mdd": perf(ls)["mdd"],
        "Q5ex_net": perf(excess)["ann"],
        "turn": res["turnover"].mean(),
        "mono": np.corrcoef(mono.rank().values, np.arange(N_Q))[0, 1],
        "n": res["n"].mean(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", default=",".join(FACTORS))
    ap.add_argument("--neutral", choices=["none", "ind", "ind_size"], default="ind_size")
    ap.add_argument("--start", default="2011-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--min-price", type=float, default=5)
    ap.add_argument("--min-value", type=float, default=5e6, help="20 日均成交值下限（元）")
    ap.add_argument("--min-obs", type=int, default=120)
    ap.add_argument("--by-year", action="store_true", help="另外印出每年 IC")
    a = ap.parse_args()

    df = pd.read_parquet(OUT / "factors_monthly.parquet")
    df = df[(df["date"] >= a.start) & ((df["date"] <= a.end) if a.end else True)]
    df = universe(df, a.min_price, a.min_value, a.min_obs)
    print(f"{df['date'].min():%Y-%m} ~ {df['date'].max():%Y-%m}，{df['date'].nunique()} 期，"
          f"平均每期 {df.groupby('date').size().mean():.0f} 檔，中性化: {a.neutral}\n")

    out_dir = OUT / "bt"
    out_dir.mkdir(exist_ok=True)
    rows, ics, qrets = [], {}, {}
    for f in a.factors.split(","):
        res = run_factor(df, f, a.neutral)
        rows.append(summarize(f, res))
        ics[f] = res["ic"]
        qrets[f] = res["qret"]
    tab = pd.DataFrame(rows).set_index("factor")
    tab.insert(0, "desc", [FACTORS[f] for f in tab.index])
    pd.DataFrame(ics).to_parquet(out_dir / f"ic_{a.neutral}.parquet")
    pd.concat(qrets, names=["factor"]).rename(columns=lambda c: f"Q{c}" if isinstance(c, int) else c) \
        .to_parquet(out_dir / f"qret_{a.neutral}.parquet")

    fmt = {"IC": "{:+.3f}", "ICIR": "{:+.2f}", "t": "{:+.1f}", "IC>0": "{:.0%}", "LS_ann": "{:+.1%}",
           "LS_sharpe": "{:+.2f}", "LS_mdd": "{:.0%}", "Q5ex_net": "{:+.1%}", "turn": "{:.0%}",
           "mono": "{:+.1f}", "n": "{:.0f}"}
    show = tab.sort_values("t", key=abs, ascending=False).copy()
    for c, f in fmt.items():
        show[c] = show[c].map(f.format)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(show.to_string())
    print("\nIC: 因子分數與下月報酬的 rank 相關，每月平均；t = IC 平均 / 標準誤；LS = Q5−Q1 年化；"
          "\nQ5ex_net = Q5 對全體等權的年化超額（扣成本）；mono = 五分位平均報酬的單調性（+1 完全遞增）")

    if a.by_year:
        icy = pd.DataFrame(ics)
        print("\n每年平均 IC:")
        print(icy.groupby(icy.index.year).mean().T.round(3).to_string())


if __name__ == "__main__":
    main()
