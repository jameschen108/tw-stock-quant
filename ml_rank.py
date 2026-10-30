# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15", "scikit-learn>=1.5"]
# ///
"""自行訓練的選股模型（Elastic Net、梯度提升樹），和 composite.py 的固定權重綜合分數比較（學術研究用）。

依 Gu, Kelly & Xiu (2020, Review of Financial Studies 33(5), 2223–2273) 的比較框架改編：
同一組特徵、依時間滾動訓練、只用當時已到期的標籤，比較線性與非線性模型。原文的特徵、年數與宏觀交互項
遠多於本專案，這裡是範圍小很多的台股改編，不是複現。用途是當 LLM 選股的對照組：在相同資訊下，
簡單統計模型能做到多少。

用法:
  uv run ml_rank.py                       # 2015 起逐月預測，報 2015–2018 與樣本外 2019 起
  uv run ml_rank.py --top 30 --top-min-value 2e7

設計（固定在看結果之前）:
  特徵    composite.py 四組 13 個因子（營收動能、品質、價值、趨勢），前處理同 composite.py：
          每期產業 + 市值中性、z 分數；缺值補 0（當期平均），至少一半特徵有值才納入。
          資訊集合和綜合分數完全相同，差別只在「固定等權」對「模型學出來的組合」
  標籤    下期報酬（fwd_ret）在同期的百分位 − 0.5
  切分    每年 1 月重新訓練；在換股日 t 只能用 t−2 期以前的標籤（t−1 期的報酬在 t+1 才走完），
          同 composite.py 的 wf_ic。擴張視窗，訓練資料從 2011 起
  調參    訓練資料最後 24 期當驗證期，從下面固定的小網格選平均 IC 最高的，再用全部訓練資料重配
  比較    同股票池、同 Top-N 與成本（composite.top_n）、同五分位評估（backtest.evaluate）；
          另算模型 − comp_equal 的逐月 Top-N 淨報酬差，Newey-West t（lag 3）

注意：2019 年後的資料在開發綜合分數時已被反覆看過（因子與趨勢組是看全期結果挑的），所以樣本外對
綜合分數本身也不是乾淨的保留集；這裡兩種模型只各跑一個預先設定的網格，不再依結果調整。
輸出 data/bt/ml_scores.parquet（逐期預測）、data/bt/ml_top_returns.parquet、data/bt/ml_meta.json（每次重訓的參數）。
"""
import argparse
import json
from collections.abc import Callable

import numpy as np
import pandas as pd

import composite as C
import technicals as tc
from backtest import evaluate, neutralize, universe
from build_panel import OUT

FEATURES = {f: s for g in ("rev_mom", "quality", "value") for f, s in C.GROUPS[g].items()} | C.TREND
PRED_START = "2015-01"
SPLIT = "2019-01"
VAL_MONTHS = 24
LABEL_LAG = 2                       # 在 t 只能用到 t−LABEL_LAG 期的標籤
EN_GRID = [{"alpha": a, "l1_ratio": r} for a in (1e-4, 1e-3, 1e-2) for r in (0.1, 0.5, 0.9)]
GBT_GRID = [{"max_depth": d, "max_iter": n, "learning_rate": 0.05, "min_samples_leaf": 200}
            for d in (2, 3) for n in (100, 300)]

Fit = Callable[[pd.DataFrame, pd.Series], Callable[[pd.DataFrame], np.ndarray]]


# ---------------------------------------------------------------- 特徵與標籤

def features(df: pd.DataFrame) -> pd.DataFrame:
    """每期橫斷面前處理（只用同一期的資料，不會看到未來）：中性化 → z 分數 → 缺值補 0。"""
    z = pd.DataFrame({f: s * C.zscore(neutralize(df, f, "ind_size"), df["date"]) for f, s in FEATURES.items()},
                     index=df.index)
    enough = z.notna().sum(axis=1) >= len(FEATURES) / 2
    return z.fillna(0).where(enough)


def label(df: pd.DataFrame) -> pd.Series:
    return df.groupby("date")["fwd_ret"].rank(pct=True) - 0.5


def date_ic(pred: np.ndarray, y: pd.Series, dates: pd.Series) -> float:
    """每期 pred 與 y 的相關係數平均。"""
    d = pd.DataFrame({"p": pred, "y": y.to_numpy(), "d": dates.to_numpy()})
    return float(d.groupby("d").apply(lambda x: x["p"].corr(x["y"]), include_groups=False).mean())


# ---------------------------------------------------------------- 模型

def tuned(make: Callable[[dict], object], grid: list[dict], meta: list) -> Fit:
    """回傳 fit(X, y)：最後 VAL_MONTHS 期當驗證期選參數，再用全部資料重配。X 的 index 第一層是日期。"""
    def fit(X: pd.DataFrame, y: pd.Series):
        dates = X.index.get_level_values(0)
        cut = np.sort(dates.unique())[-VAL_MONTHS]
        tr, va = dates < cut, dates >= cut
        scores = []
        for p in grid:
            m = make(p).fit(X[tr].to_numpy(), y[tr].to_numpy())
            scores.append(date_ic(m.predict(X[va].to_numpy()), y[va], pd.Series(dates[va])))
        best = grid[int(np.nanargmax(scores))]
        m = make(best).fit(X.to_numpy(), y.to_numpy())
        info = {"train_end": str(pd.Timestamp(dates.max()).date()), "n_train": len(y), "params": best,
                "val_ic": round(float(np.nanmax(scores)), 4)}
        if hasattr(m, "coef_"):
            info["coef"] = dict(zip(X.columns, np.round(m.coef_, 5).tolist()))
        meta.append(info)
        return lambda Xp: m.predict(Xp.to_numpy())
    return fit


def elastic_net(meta: list) -> Fit:
    from sklearn.linear_model import ElasticNet
    return tuned(lambda p: ElasticNet(**p, max_iter=5000), EN_GRID, meta)


def gbt(meta: list) -> Fit:
    from sklearn.ensemble import HistGradientBoostingRegressor
    return tuned(lambda p: HistGradientBoostingRegressor(**p, random_state=0), GBT_GRID, meta)


def walk_forward(X: pd.DataFrame, y: pd.Series, fit: Fit, start: str = PRED_START) -> pd.Series:
    """X、y 的 index = (date, code)。每年 1 月（或第一個預測期）重訓，在 t 只用 t−LABEL_LAG 期以前有標籤的列。"""
    dates = np.sort(X.index.get_level_values(0).unique())
    lvl = X.index.get_level_values(0)
    ok = X.notna().all(axis=1)
    out, model, year = [], None, None
    for i, t in enumerate(dates):
        if t < pd.Timestamp(start) or i < LABEL_LAG:
            continue
        t = pd.Timestamp(t)
        if model is None or t.year != year:
            train = ok & y.notna() & (lvl <= dates[i - LABEL_LAG])
            model, year = fit(X[train], y[train]), t.year
        rows = ok & (lvl == t)
        if rows.any():
            out.append(pd.Series(model(X[rows]), index=X.index[rows]))
    return pd.concat(out) if out else pd.Series(dtype=float)


# ---------------------------------------------------------------- 主程式

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2011-01")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--top-min-value", type=float, default=2e7)
    a = ap.parse_args()

    C.GROUPS["trend"] = C.TREND
    df = pd.read_parquet(OUT / "factors_monthly.parquet")
    tf, _ = C.trend_features(df["date"].unique())
    df = df.merge(tf, on=["date", "code"], how="left")
    df = universe(df[df["date"] >= a.start], 5, 5e6, 120).reset_index(drop=True)

    gs = C.group_scores(df)
    W = C.weights(C.group_ic(gs), SPLIT)
    X = features(df).set_index([df["date"], df["code"]])
    y = label(df).set_axis(X.index)

    meta = {"en": [], "gbt": []}
    preds = {"ml_en": walk_forward(X, y, elastic_net(meta["en"])),
             "ml_gbt": walk_forward(X, y, gbt(meta["gbt"]))}
    key = pd.MultiIndex.from_frame(gs[["date", "code"]])
    cands = {"comp_equal": C.combine(gs, W["equal"]), "comp_wf_ic": C.combine(gs, W["wf_ic"])}
    cands |= {k: pd.Series(v.reindex(key).to_numpy(), index=gs.index) for k, v in preds.items()}
    first = preds["ml_en"].index.get_level_values(0).min()
    common = gs["date"] >= first                               # 四個分數都只比模型有預測的期間
    for k in cands:
        cands[k] = cands[k].where(common)

    dates = pd.DatetimeIndex(sorted(gs.loc[common, "date"].unique()))
    periods = {f"{dates[0]:%Y}–2018": dates[dates < SPLIT], "樣本外 2019–": dates[dates >= SPLIT]}
    rows, tops = [], {}
    for name, score in cands.items():
        d = gs.assign(score=score)
        res, top = evaluate(d), C.top_n(d, a.top, a.top_min_value)
        tops[name] = top
        for p, ds in periods.items():
            rows.append({"period": p, "score": name, **C.period_stats(res, top, ds)})
    print(f"特徵 {len(FEATURES)} 個：{list(FEATURES)}")
    print(f"預測期 {dates[0]:%Y-%m}~{dates[-1]:%Y-%m}；每年 1 月重訓，標籤落後 {LABEL_LAG} 期，驗證期 {VAL_MONTHS} 期\n")
    print(f"回測（Top = 20 日均成交值 >= {a.top_min_value / 1e6:.0f} 百萬中前 {a.top} 名等權，扣成本）:")
    C.show(rows)

    print("\n模型 − comp_equal：逐月 Top-N 淨報酬差（年化 = 月平均 × 12），Newey-West t（lag 3）")
    for name in ("comp_wf_ic", "ml_en", "ml_gbt"):
        diff = (tops[name]["net"] - tops["comp_equal"]["net"]).dropna()
        for p, ds in periods.items():
            x = diff[diff.index.isin(ds)]
            print(f"  {name:10s} {p:12s} {x.mean() * 12:+.1%}  t = {tc.nw_t(x, 3):+.2f}  （{len(x)} 期）")

    print("\n每次重訓選到的參數與驗證 IC:")
    for k, ms in meta.items():
        for m in ms:
            print(f"  {k:4s} {m['train_end']}  {m['params']}  val IC {m['val_ic']:+.3f}")
    last = meta["en"][-1].get("coef", {})
    print("\nElastic Net 最後一次重訓的係數（特徵已標準化，可直接比大小）:")
    print("  " + "、".join(f"{k} {v:+.4f}" for k, v in sorted(last.items(), key=lambda kv: -abs(kv[1]))))

    out = OUT / "bt"
    out.mkdir(exist_ok=True)
    gs.assign(**{k: v for k, v in cands.items()}).to_parquet(out / "ml_scores.parquet", index=False)
    pd.DataFrame({k: t["net"] for k, t in tops.items()} | {"mkt": tops["comp_equal"]["mkt"]}) \
        .to_parquet(out / "ml_top_returns.parquet")
    (out / "ml_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
