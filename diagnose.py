# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""拆解綜合分數 Top-N 在某段期間輸給同池等權的原因：產業配置 / 選股、市值、哪一組因子拖累。

用法:
  uv run diagnose.py                          # 2025-01 起 vs 2019-01~2024-12
  uv run diagnose.py --focus 2025-01 --base 2019-01 --score comp_equal --top 30

需先跑 composite.py（讀 data/bt/composite_scores.parquet）。
報酬拆解用每月算術超額加總（不複利），所以各項加起來等於總超額。
"""
import argparse

import numpy as np
import pandas as pd

from build_panel import OUT
from composite import GROUPS, combine

POOL_MIN_VALUE = 2e7


def load(score: str) -> pd.DataFrame:
    s = pd.read_parquet(OUT / "bt" / "composite_scores.parquet")
    f = pd.read_parquet(OUT / "factors_monthly.parquet", columns=["date", "code", "industry", "mcap"])
    name = pd.read_parquet(OUT / "prices_daily.parquet", columns=["code", "name"]).drop_duplicates("code", keep="last")
    d = s.merge(f, on=["date", "code"]).merge(name, on="code", how="left")
    d = d[(d["avg_value"] >= POOL_MIN_VALUE) & d["fwd_ret"].notna() & d["mcap"].notna()].copy()
    d["score"] = d[score]
    d["size_q"] = d.groupby("date")["mcap"].transform(lambda s: np.ceil(s.rank(pct=True) * 5)).astype(int)
    return d


def pick(d: pd.DataFrame, n: int, col: str = "score") -> pd.Series:
    top = d.dropna(subset=[col]).sort_values(col, ascending=False).groupby("date").head(n).index
    return d.index.isin(top)


def industry_labels(d: pd.DataFrame) -> pd.Series:
    """TEJ 產業代碼沒有名稱：用該產業最近一期市值最大的三家公司當標籤。"""
    last = d[d["date"] == d["date"].max()]
    return last.sort_values("mcap", ascending=False).groupby("industry")["name"].apply(
        lambda s: "、".join(s.head(3)))


def brinson(d: pd.DataFrame) -> pd.DataFrame:
    """每月：配置 = Σ(wp−wb)(Rb_i−Rb)，選股 = Σ wp (Rp_i−Rb_i)。等權。"""
    rows = []
    for dt, g in d.groupby("date"):
        p = g[g["picked"]]
        rb = g["fwd_ret"].mean()
        b = g.groupby("industry")["fwd_ret"].agg(["mean", "size"])
        b["wb"] = b["size"] / b["size"].sum()
        pp = p.groupby("industry")["fwd_ret"].agg(["mean", "size"])
        pp["wp"] = pp["size"] / pp["size"].sum()
        m = b.join(pp.add_suffix("_p").rename(columns={"wp_p": "wp"}), how="left").fillna({"wp": 0})
        alloc = ((m["wp"] - m["wb"]) * (m["mean"] - rb)).rename("alloc")
        selc = (m["wp"] * (m["mean_p"] - m["mean"])).fillna(0).rename("select")
        rows.append(pd.concat([alloc, selc], axis=1).assign(date=dt))
    return pd.concat(rows).reset_index()


def section(title: str):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focus", default="2025-01", help="要拆解的期間起點")
    ap.add_argument("--base", default="2019-01", help="對照期間起點（到 focus 前一月）")
    ap.add_argument("--score", default="comp_equal")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()

    d = load(a.score)
    d = d[d["date"] >= a.base].reset_index(drop=True)
    d["picked"] = pick(d, a.top)
    d["period"] = np.where(d["date"] >= a.focus, "focus", "base")
    labels = industry_labels(d)
    span = {p: f"{g['date'].min():%Y-%m}~{g['date'].max():%Y-%m}（{g['date'].nunique()} 期）"
            for p, g in d.groupby("period")}
    print(f"分數 {a.score}，Top {a.top}，池 = 20 日均成交值 >= {POOL_MIN_VALUE / 1e6:.0f} 百萬")
    print(f"base {span['base']}  focus {span['focus']}")

    # 1. 總覽：Top-N、同池等權、同池市值加權
    section("1. 月均報酬（%）")
    def overview(g):
        top = g[g["picked"]].groupby("date")["fwd_ret"].mean()
        ew = g.groupby("date")["fwd_ret"].mean()
        vw = g.groupby("date").apply(lambda x: np.average(x["fwd_ret"], weights=x["mcap"]), include_groups=False)
        return pd.Series({"TopN": top.mean(), "池等權": ew.mean(), "池市值加權": vw.mean(),
                          "Top−等權": (top - ew).mean(), "Top 勝率": (top > ew).mean()}) * 100
    print(d.groupby("period").apply(overview, include_groups=False).round(2).to_string())

    # 2. 產業：配置 vs 選股
    section("2. 產業拆解（每月超額加總，%）")
    br = brinson(d)
    br["period"] = np.where(br["date"] >= a.focus, "focus", "base")
    tot = br.groupby("period")[["alloc", "select"]].sum() * 100
    tot["months"] = br.groupby("period")["date"].nunique()
    tot["alloc/月"], tot["select/月"] = tot["alloc"] / tot["months"], tot["select"] / tot["months"]
    print(tot.round(2).to_string())
    f = br[br["period"] == "focus"].groupby("industry")[["alloc", "select"]].sum() * 100
    f["total"] = f.sum(axis=1)
    fd = d[d["period"] == "focus"]
    f["Top 權重"] = fd[fd["picked"]].groupby("industry").size() / fd["picked"].sum() * 100
    f["池權重"] = fd.groupby("industry").size() / len(fd) * 100
    f["產業月均報酬"] = fd.groupby("industry")["fwd_ret"].mean() * 100
    f["代表公司"] = labels.reindex(f.index)
    f = f.fillna({"Top 權重": 0})
    print("\nfocus 期間拖累最多的產業:")
    print(f.sort_values("total").head(8).round(2).to_string())
    print("\nfocus 期間貢獻最多的產業:")
    print(f.sort_values("total").tail(5).round(2).to_string())

    # 3. 市值
    section("3. 市值")
    print("Top-N 持股的市值五分位分布（%，1 = 最小）:")
    print((d[d["picked"]].groupby("period")["size_q"].value_counts(normalize=True).unstack() * 100).round(0).to_string())
    print("\n池內各市值五分位的月均報酬（%）:")
    print((d.groupby(["period", "size_q"])["fwd_ret"].mean().unstack() * 100).round(2).to_string())

    # 4. 各組分數
    section("4. 各組因子")
    names = list(GROUPS)
    r = d.groupby("date")["fwd_ret"].rank(pct=True)
    ic = pd.DataFrame({g: d.assign(r=r).groupby("date").apply(lambda x: x[g].corr(x["r"]), include_groups=False)
                       for g in names + ["score"]})
    ic["period"] = np.where(ic.index >= a.focus, "focus", "base")
    print("池內 rank IC:")
    print(ic.groupby("period").mean().round(3).to_string())
    print("\nTop-N 持股的平均組分數（z）:")
    print(d[d["picked"]].groupby("period")[names].mean().round(2).to_string())

    rows = {}
    for g in names:
        dg = d.copy()
        dg["q5"] = dg.groupby("date")[g].transform(lambda s: s.rank(pct=True) > 0.8)
        ex = dg[dg["q5"]].groupby("date")["fwd_ret"].mean() - dg.groupby("date")["fwd_ret"].mean()
        rows[g] = ex.groupby(np.where(ex.index >= a.focus, "focus", "base")).mean() * 100
    print("\n各組單獨最高 20% 對池等權的月均超額（%）:")
    print(pd.DataFrame(rows).round(2).to_string())

    # 5. 反事實：拿掉一組 / 只用一組
    section("5. 反事實 Top-N 月均超額（%）")
    ew = d.groupby("date")["fwd_ret"].mean()
    out = {}
    combos = {"三組": names} | {f"去掉 {g}": [x for x in names if x != g] for g in names} | {f"只用 {g}": [g] for g in names}
    for k, members in combos.items():
        w = pd.DataFrame({g: (1.0 if g in members else 0.0) for g in names}, index=pd.DatetimeIndex(sorted(d["date"].unique())))
        dd = d.copy()
        dd["s"] = combine(dd, w)
        sel = pick(dd, a.top, "s")
        ex = dd[sel].groupby("date")["fwd_ret"].mean() - ew
        out[k] = ex.groupby(np.where(ex.index >= a.focus, "focus", "base")).mean() * 100
    print(pd.DataFrame(out).T.round(2).to_string())

    # 6. 錯過的大漲股
    section("6. focus 期間池內累積漲幅前 15 名（持有期間都在池內才算）")
    cum = fd.groupby(["code", "name"]).agg(ret=("fwd_ret", lambda s: (1 + s).prod() - 1), n=("fwd_ret", "size"),
                                           ind=("industry", "last"))
    pr = fd.assign(pct=fd.groupby("date")["score"].rank(pct=True))
    cum["分數百分位(平均)"] = pr.groupby(["code", "name"])["pct"].mean()
    cum["入選月數"] = fd.groupby(["code", "name"])["picked"].sum()
    for g in names:
        cum[g] = fd.groupby(["code", "name"])[g].mean()
    full = cum[cum["n"] == fd["date"].nunique()]
    print(full.sort_values("ret", ascending=False).head(15).round(2).to_string())

    # 7. 選到的最差持股
    section("7. focus 期間 Top-N 中拖累最多的持股（月超額加總，%）")
    fp = fd[fd["picked"]].copy()
    fp["ex"] = fp["fwd_ret"] - fp["date"].map(fd.groupby("date")["fwd_ret"].mean())
    worst = fp.groupby(["code", "name"]).agg(ex=("ex", "sum"), months=("ex", "size"), ind=("industry", "last"))
    worst["ex"] = worst["ex"] * 100 / a.top
    for g in names:
        worst[g] = fp.groupby(["code", "name"])[g].mean()
    print(worst.sort_values("ex").head(12).round(2).to_string())


if __name__ == "__main__":
    main()
