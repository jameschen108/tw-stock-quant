# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""量化護城河：用過去 10 個年度的 ROE 與毛利率穩定度，把公司分成 Wide / Narrow / None。

用法:
  uv run moat.py                     # 最新一期分級分布與 Wide 名單
  uv run moat.py --code 2330,2603    # 看個別公司的年度數字與分級理由
  uv run moat.py --check             # 驗證：分級能否預測之後 3 年 ROE、之後的股價報酬

概念（仿晨星：護城河 = 能長期維持超額資本報酬）:
  - 年度數字由單季（或半年）加總；ROE = 年度歸母淨利 / 年內各期期末權益平均
  - 只用看得到的資料：某年度的分級在該年最後一期財報 avail_date 之後才能用
  - 門檻寫在下方常數；毛利率缺值（金融業等）時略過毛利率條件
  - 這只是量化門檻，不判斷護城河的來源（品牌、轉換成本…），那部分要另外看質化資料

輸出 data/moat_annual.parquet（公司 × 年度）、data/moat.parquet（公司 × 評估年度的分級）。
"""
import argparse

import numpy as np
import pandas as pd

from backtest import universe
from build_panel import OUT

WINDOW = 10            # 評估視窗（年）
MIN_YEARS = 8          # 視窗內至少要有幾個完整年度
HURDLE = 10.0          # ROE 門檻（%），約當台股權益資金成本
WIDE = {"hit": 0.9, "median": 15.0, "losses": 0, "gm_floor": 0.7, "recent": HURDLE}
NARROW = {"hit": 0.7, "median": HURDLE, "losses": 1, "gm_floor": 0.5, "recent": HURDLE}
TIERS = ["Wide", "Narrow", "None"]

FIN_COLS = ["code", "name", "period", "avail_date", "單季(Q)/單半年(H)", "營業收入淨額", "營業毛利",
            "營業利益", "歸屬母公司淨利（損）", "母公司股東權益合計"]


def annual() -> pd.DataFrame:
    """公司 × 年度：營收、毛利率、營益率、ROE，只留 12 個月都有資料的年度。"""
    f = pd.read_parquet(OUT / "tej_fin.parquet", columns=FIN_COLS)
    f = f[f["code"].str.fullmatch(r"[1-9]\d{3}")]
    f["year"] = f["period"].dt.year
    f["months"] = np.where(f["單季(Q)/單半年(H)"] == "H", 6, 3)
    g = f.sort_values("period").groupby(["code", "year"])
    a = g.agg(name=("name", "last"), months=("months", "sum"), avail_date=("avail_date", "max"),
              rev=("營業收入淨額", "sum"), gp=("營業毛利", lambda s: s.sum(min_count=len(s))),
              op=("營業利益", "sum"), ni=("歸屬母公司淨利（損）", "sum"),
              equity=("母公司股東權益合計", "mean"))
    a = a[a["months"] == 12].reset_index()
    rev = a["rev"].where(a["rev"] > 0)
    a["gm"] = a["gp"] / rev * 100
    a["opm"] = a["op"] / rev * 100
    a["roe"] = a["ni"] / a["equity"].where(a["equity"] > 0) * 100
    return a[["code", "name", "year", "avail_date", "rev", "gm", "opm", "roe"]]


def grade(w: pd.DataFrame, min_years: int = MIN_YEARS) -> dict:
    """w = 單一公司視窗內的年度資料（依年度排序）。回傳統計值與分級。"""
    roe = w["roe"].dropna()
    gm = w["gm"].dropna()
    s = {
        "years": len(roe),
        "roe_med": roe.median(),
        "roe_min": roe.min(),
        "roe_hit": (roe >= HURDLE).mean(),
        "losses": int((roe < 0).sum()),
        "roe_recent": roe.tail(3).median(),
        "roe_std": roe.std(),
        "rev_g_std": w["rev"].where(w["rev"] > 0).pct_change().std() * 100,
        "gm_med": gm.median() if len(gm) >= min_years else np.nan,
        "gm_floor": gm.min() / gm.median() if len(gm) >= min_years and gm.median() > 0 else np.nan,
    }
    s["tier"] = "None"
    if s["years"] < min_years:
        s["tier"] = None
        return s
    for tier, t in (("Wide", WIDE), ("Narrow", NARROW)):
        gm_ok = np.isnan(s["gm_floor"]) or s["gm_floor"] >= t["gm_floor"]
        if (s["roe_hit"] >= t["hit"] and s["roe_med"] >= t["median"] and s["losses"] <= t["losses"]
                and s["roe_recent"] >= t["recent"] and gm_ok):
            s["tier"] = tier
            break
    return s


def build(a: pd.DataFrame, min_years: int = MIN_YEARS) -> pd.DataFrame:
    """每家公司、每個評估年度 Y（視窗 Y-9..Y）一列；avail_date = Y 年度最後一期財報公布日。"""
    rows = []
    for code, g in a.groupby("code"):
        g = g.set_index("year").sort_index()
        for y in g.index:
            w = g.loc[y - WINDOW + 1:y]
            s = grade(w, min_years)
            if s["tier"] is None:
                continue
            rows.append({"code": code, "name": g["name"].iloc[-1], "year": y,
                         "avail_date": g.at[y, "avail_date"], **s})
    return pd.DataFrame(rows)


def explain(r: pd.Series) -> str:
    gm = "毛利率缺值" if np.isnan(r["gm_floor"]) else f"毛利率最低 / 中位數 {r['gm_floor']:.2f}"
    return (f"{r['code']} {r['name']}（{r['year'] - WINDOW + 1}–{r['year']}，{r['years']} 年）→ {r['tier']}："
            f"ROE 中位數 {r['roe_med']:.1f}%、最低 {r['roe_min']:.1f}%、{r['roe_hit']:.0%} 年度 ≥ {HURDLE:.0f}%、"
            f"虧損 {r['losses']} 年、近 3 年中位數 {r['roe_recent']:.1f}%；{gm}")


def show_latest(m: pd.DataFrame, a: pd.DataFrame, top: int):
    last_year = int(a.loc[a.groupby("code")["year"].transform("max") == a["year"], "year"].mode()[0])
    cur = m[m["year"] == last_year]
    ind = pd.read_parquet(OUT / "factors_monthly.parquet", columns=["date", "code", "industry", "mcap"])
    ind = ind[ind["date"] == ind["date"].max()].set_index("code")
    cur = cur.join(ind[["industry", "mcap"]], on="code")
    listed = cur[cur["mcap"].notna()]
    print(f"評估年度 {last_year}（視窗 {last_year - WINDOW + 1}–{last_year}），"
          f"可評估 {len(cur)} 家，其中目前仍交易 {len(listed)} 家")
    print(listed["tier"].value_counts().reindex(TIERS).to_string(), "\n")

    wide = listed[listed["tier"] == "Wide"].sort_values("mcap", ascending=False)
    cols = {"code": "代號", "name": "名稱", "industry": "產業", "mcap": "市值(億)", "roe_med": "ROE中位",
            "roe_min": "ROE最低", "roe_recent": "近3年ROE", "gm_med": "毛利率中位", "gm_floor": "毛利穩定"}
    t = wide[list(cols)].rename(columns=cols).head(top).copy()
    t["市值(億)"] = (t["市值(億)"] / 1e5).round(0)          # mcap 單位千元
    print(f"Wide（依市值，前 {min(top, len(wide))} / {len(wide)} 家）:")
    print(t.round(2).to_string(index=False))
    m.to_parquet(OUT / "moat.parquet")
    a.to_parquet(OUT / "moat_annual.parquet")


def check(m: pd.DataFrame, a: pd.DataFrame):
    # 1) 持續性：評估年度 Y 的分級，之後 Y+1..Y+3 的 ROE 中位數
    roe = a.pivot(index="code", columns="year", values="roe")
    last_full = int(a["year"].max()) - 1          # 最新年度可能還不完整，不拿來當未來
    rows = []
    for y in range(int(m["year"].min()), last_full - 2):
        fut = roe.loc[:, y + 1:y + 3]
        fut = fut[fut.notna().sum(axis=1) == 3].median(axis=1)
        s = m[m["year"] == y].set_index("code")
        s = s.assign(fut=fut).dropna(subset=["fut"])
        base = roe[y].reindex(s.index)            # 對照：只看當年 ROE ≥ 15%
        for tier, g in s.groupby("tier"):
            rows.append({"year": y, "group": tier, "n": len(g), "fut_med": g["fut"].median(),
                         "fut_hit": (g["fut"] >= HURDLE).mean()})
        g = s[base >= 15]
        rows.append({"year": y, "group": "對照:當年ROE≥15", "n": len(g), "fut_med": g["fut"].median(),
                     "fut_hit": (g["fut"] >= HURDLE).mean()})
    p = pd.DataFrame(rows)
    print(f"1) 之後 3 年 ROE（評估年度 {p['year'].min()}–{p['year'].max()}，各年平均）")
    print(p.groupby("group")[["n", "fut_med", "fut_hit"]].mean()
          .reindex(TIERS + ["對照:當年ROE≥15"]).rename(columns={
              "n": "平均家數", "fut_med": "之後3年ROE中位數", "fut_hit": f"之後3年ROE中位≥{HURDLE:.0f}%比例"})
          .round(2).to_string())

    # 2) 報酬：每月依當時已知的分級分組，等權下月報酬，對同池等權與同產業等權的超額
    df = pd.read_parquet(OUT / "factors_monthly.parquet",
                         columns=["date", "code", "close", "avg_value", "n_obs", "industry", "fwd_ret"])
    df = universe(df, 5, 5e6, 120).dropna(subset=["fwd_ret"])
    df = df.sort_values("date")
    mm = m.dropna(subset=["avail_date"]).sort_values("avail_date")
    df = pd.merge_asof(df, mm[["code", "avail_date", "tier"]], left_on="date", right_on="avail_date",
                       by="code", allow_exact_matches=False)
    df = df[df["date"] >= mm["avail_date"].min()]
    df["tier"] = df["tier"].fillna("未評估")
    df["ex_mkt"] = df["fwd_ret"] - df.groupby("date")["fwd_ret"].transform("mean")
    df["ex_ind"] = df["fwd_ret"] - df.groupby(["date", "industry"])["fwd_ret"].transform("mean")
    mon = df.groupby(["date", "tier"])[["ex_mkt", "ex_ind"]].mean().unstack("tier")
    order = [t for t in TIERS + ["未評估"] if t in df["tier"].unique()]
    rows = []
    for t in order:
        r = {"group": t, "平均檔數": df[df["tier"] == t].groupby("date").size().mean()}
        for k, lab in (("ex_mkt", "對全體"), ("ex_ind", "對同產業")):
            x = mon[(k, t)].dropna()
            r[f"{lab}年化超額"] = x.mean() * 12
            r[f"{lab}t"] = x.mean() / x.std() * np.sqrt(len(x))
        rows.append(r)
    print(f"\n2) 下月報酬（{df['date'].min():%Y-%m}~{df['date'].max():%Y-%m}，{df['date'].nunique()} 期，"
          f"等權、未扣成本）")
    print(pd.DataFrame(rows).set_index("group").round(3).to_string())
    yearly = mon["ex_mkt"].groupby(mon.index.year).sum()[[t for t in TIERS if t in mon["ex_mkt"]]]
    print("\n每年對全體超額（月超額加總）:")
    print(yearly.T.map("{:+.1%}".format).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", help="逗號分隔，印出個別公司的年度數字與最新分級")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    a = annual()
    m = build(a)
    if args.code:
        for c in args.code.split(","):
            g = a[a["code"] == c].set_index("year")[["gm", "opm", "roe"]].round(1)
            print(g.T.to_string())
            if (m["code"] == c).any():
                print(explain(m[m["code"] == c].iloc[-1]), "\n")
            else:
                print(f"{c}: 完整年度不足 {MIN_YEARS} 年，無法評估\n")
        return
    if args.check:
        check(m, a)
        return
    show_latest(m, a, args.top)


if __name__ == "__main__":
    main()
