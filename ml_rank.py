# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15", "scikit-learn>=1.5", "shap>=0.46"]
# ///
"""自行訓練的選股模型（Elastic Net、梯度提升樹），和 composite.py 的固定權重綜合分數比較（學術研究用）。

依 Gu, Kelly & Xiu (2020, Review of Financial Studies 33(5), 2223–2273) 的比較框架改編：
同一組特徵、依時間滾動訓練、只用當時已到期的標籤，比較線性與非線性模型。原文的特徵、年數與宏觀交互項
遠多於本專案，這裡是範圍小很多的台股改編，不是複現。用途是當 LLM 選股的對照組：在相同資訊下，
簡單統計模型能做到多少。

用法:
  uv run ml_rank.py                       # 2015 起逐月預測，報 2015–2018 與樣本外 2019 起
  uv run ml_rank.py --top 30 --top-min-value 2e7
  uv run ml_rank.py --llm                 # 另做 LLM 同候選比較（讀 llm_replay.jsonl，不呼叫 API）
  uv run ml_rank.py --shap [--llm]        # 另做樹模型的 SHAP 解釋（加 --llm 時候選也附解釋）

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

LLM 同候選比較（--llm）:
  候選與 LLM 決定 = llm_replay.py 的回放紀錄（2026-07～09 三批，每批最後一輪；wait 到最後算放棄）。
  每批在資料時點重算全市場因子（factors.build，含當天公布的營收，和 LLM 看到的資訊對齊），
  用該年度重訓的模型打分；每批從同一群候選挑和 LLM buy 一樣多檔：Elastic Net、樹模型、comp_equal 各取前 k 名。
  報酬同 llm_replay.py：資料時點隔一個交易日收盤進場，之後 20、60 個交易日，不扣成本、等權；
  另附每批隨機挑 k 檔 N_RANDOM 次的分布，與批內「分數對報酬」的等級相關。三批、約 100 檔候選，樣本很小。

SHAP 解釋（--shap，Lundberg & Lee 2017, NeurIPS 30）:
  對每年重訓的樹模型，用 TreeExplainer（tree_path_dependent：背景分布就是當年的訓練資料，不用之後的資料）
  解釋該年度的預測：各特徵平均 |SHAP| 佔比（逐年）、特徵值和它的 SHAP 的相關（方向）、相鄰年度重要度排序的
  等級相關（穩定性），以及 from_high 和 ma200_gap 這對高度相關特徵的解釋是否互相抵銷。最新一期每檔的 SHAP 存檔。
  SHAP 解釋的是模型的預測，不是報酬的因果，也不能拿來說明 LLM 怎麼決策。

注意：2019 年後的資料在開發綜合分數時已被反覆看過（因子與趨勢組是看全期結果挑的），所以樣本外對
綜合分數本身也不是乾淨的保留集；這裡兩種模型只各跑一個預先設定的網格，不再依結果調整。
輸出 data/bt/ml_scores.parquet（逐期預測）、data/bt/ml_top_returns.parquet、data/bt/ml_meta.json（每次重訓的參數）。
"""
import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path

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

LLM_LOG = OUT / "paper" / "llm_replay.jsonl"
HORIZONS = [20, 60]
N_RANDOM = 1000

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
        predict = lambda Xp: m.predict(Xp.to_numpy())          # noqa: E731
        predict.model = m                                         # SHAP 要用原始模型
        return predict
    return fit


def elastic_net(meta: list) -> Fit:
    from sklearn.linear_model import ElasticNet
    return tuned(lambda p: ElasticNet(**p, max_iter=5000), EN_GRID, meta)


def gbt(meta: list) -> Fit:
    from sklearn.ensemble import HistGradientBoostingRegressor
    return tuned(lambda p: HistGradientBoostingRegressor(**p, random_state=0), GBT_GRID, meta)


def walk_forward(X: pd.DataFrame, y: pd.Series, fit: Fit, start: str = PRED_START,
                 models: dict | None = None) -> pd.Series:
    """X、y 的 index = (date, code)。每年 1 月（或第一個預測期）重訓，在 t 只用 t−LABEL_LAG 期以前有標籤的列。
    models 有給時，記下每年用的模型（{年: 預測函式}），給同年度的其他快照用。"""
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
            if models is not None:
                models[year] = model
        rows = ok & (lvl == t)
        if rows.any():
            out.append(pd.Series(model(X[rows]), index=X.index[rows]))
    return pd.concat(out) if out else pd.Series(dtype=float)


# ---------------------------------------------------------------- LLM 同候選比較

def llm_finals(path=LLM_LOG) -> pd.DataFrame:
    """回放紀錄每批每檔的最終決定（同 llm_replay.finals）：最後一輪；最後還是 wait 算放棄。只讀不寫。"""
    log = pd.DataFrame([json.loads(x) for x in path.read_text().splitlines() if x.strip()])
    last = log.sort_values("round").groupby(["model", "batch", "code"]).last().reset_index()
    last.loc[last["decision"] == "wait", "decision"] = "放棄"
    return last[["model", "batch", "code", "name", "decision"]]


def window_returns(rets: pd.DataFrame, codes, signal: pd.Timestamp) -> dict[int, pd.Series]:
    """同 llm_replay.window_returns：signal 之後第一個交易日收盤進場，持有 H 個交易日；資料不夠長給 NaN。"""
    days = rets.index
    after = days[days > signal]
    out = {}
    for h in HORIZONS:
        if len(after) <= h:
            out[h] = pd.Series(np.nan, index=codes)
            continue
        w = (days > after[0]) & (days <= after[h])
        out[h] = (1 + rets.loc[w].reindex(columns=codes).fillna(0)).prod() - 1
    return out


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return a.rank().corr(b.rank())


def compare_picks(ev: pd.DataFrame, scores: list[str], seed: str = "ml_vs_llm") -> dict:
    """ev：batch、code、decision、r{H}、各分數。每批 k = LLM buy 檔數，各分數取前 k 名（缺分數排最後），
    和 LLM buy、全部候選、隨機挑 k 檔比平均報酬；另算批內分數對報酬的 Spearman 平均。"""
    out = {}
    for h in HORIZONS:
        d = ev.dropna(subset=[f"r{h}"])
        if not len(d) or not (d["decision"] == "buy").any():
            continue
        picks = {"LLM buy": [], **{s: [] for s in scores}}
        groups, ic = [], {s: [] for s in [*scores, "LLM buy"]}
        for _, g in d.groupby("batch"):
            k = int((g["decision"] == "buy").sum())
            r = g[f"r{h}"]
            groups.append((r.to_numpy(), k))
            picks["LLM buy"] += r[g["decision"] == "buy"].tolist()
            ic["LLM buy"].append(_spearman((g["decision"] == "buy").astype(float), r))
            for s in scores:
                picks[s] += r[g[s].fillna(-np.inf).rank(method="first", ascending=False) <= k].tolist()
                ic[s].append(_spearman(g[s], r))
        rng = random.Random(f"{seed}|{h}")
        draws = pd.Series([np.mean([x for r, k in groups for x in rng.sample(list(r), k)]) for _ in range(N_RANDOM)])
        out[h] = {"n_batch": len(groups), "n": len(d), "k": sum(k for _, k in groups), "all": float(d[f"r{h}"].mean()),
                  "rand_p5": float(draws.quantile(0.05)), "rand_p50": float(draws.median()),
                  "rand_p95": float(draws.quantile(0.95)),
                  "picks": {s: {"mean": float(np.mean(v)), "pctile": float((draws < np.mean(v)).mean()),
                                "batch_ic": float(np.nanmean(ic[s]))} for s, v in picks.items()}}
    return out


def snapshot(dates: list[pd.Timestamp]) -> pd.DataFrame:
    """資料時點的全市場因子（含當天公布的營收、財報）、趨勢特徵，股票池同主程式。"""
    import factors as F
    df = F.build(reb=pd.DatetimeIndex(dates), same_day=True)
    tf, _ = C.trend_features(df["date"].unique())
    df = df.merge(tf, on=["date", "code"], how="left")
    return universe(df, 5, 5e6, 120).reset_index(drop=True)


def llm_compare(models: dict[str, dict], scores_out: Path, explain: bool = False) -> None:
    from custom import load_rets
    fin = llm_finals()
    batches = sorted(pd.Timestamp(b) for b in fin["batch"].unique())
    snap = snapshot(batches)
    gs = C.group_scores(snap)
    eq = pd.DataFrame(1 / len(C.GROUPS), index=pd.DatetimeIndex(batches), columns=list(C.GROUPS))
    snap["comp_equal"] = C.combine(gs, eq)
    Xs = features(snap)
    ok = Xs.notna().all(axis=1)
    for name, by_year in models.items():
        snap[name] = np.nan
        for b in batches:
            rows = ok & (snap["date"] == b)
            snap.loc[rows, name] = by_year[b.year](Xs[rows])
    rets = load_rets(batches[0])
    rows = []
    for (model, b), g in fin.groupby(["model", "batch"]):
        b = pd.Timestamp(b)
        sc = snap[snap["date"] == b].set_index("code").reindex(g["code"])
        wr = window_returns(rets, list(g["code"]), b)
        e = g.assign(**{s: sc[s].to_numpy() for s in ["comp_equal", *models]},
                     **{f"r{h}": wr[h].reindex(g["code"]).to_numpy() for h in HORIZONS})
        rows.append(e)
    ev = pd.concat(rows, ignore_index=True)
    if explain:                                       # 候選附樹模型的 SHAP 前三大貢獻
        tops = {}
        for b in batches:
            rows_b = ok & (snap["date"] == b)
            sv = shap_values(models["ml_gbt"][b.year], Xs[rows_b])
            tops |= {(b, c): top_contrib(sv.loc[i]) for i, c in zip(sv.index, snap.loc[rows_b, "code"])}
        ev["gbt_shap_top"] = [tops.get((pd.Timestamp(b), c)) for b, c in zip(ev["batch"], ev["code"])]
    ev.to_csv(scores_out, index=False)
    print(f"\nLLM 同候選比較（{', '.join(ev['model'].unique())}，回放 {len(batches)} 批，"
          f"候選 {len(ev)} 檔，有模型分數 {int(ev['ml_en'].notna().sum())} 檔）")
    for model, e in ev.groupby("model"):
        res = compare_picks(e, ["ml_en", "ml_gbt", "comp_equal"])
        for h, t in res.items():
            print(f"\n  {h} 日（{t['n_batch']} 批、{t['n']} 檔有完整報酬，每批挑和 LLM buy 一樣多檔，共 {t['k']} 檔）:"
                  f"\n    全部候選 {t['all']:+.2%}；隨機 {N_RANDOM} 次中位數 {t['rand_p50']:+.2%}"
                  f"（90% 區間 {t['rand_p5']:+.2%} ~ {t['rand_p95']:+.2%}）")
            for s, v in t["picks"].items():
                print(f"    {s:10s} {v['mean']:+.2%}　隨機分布第 {v['pctile'] * 100:.0f} 百分位"
                      f"　批內等級相關 {v['batch_ic']:+.3f}")
    print(f"\n明細：{scores_out}")


# ---------------------------------------------------------------- SHAP

def shap_values(predict, X: pd.DataFrame) -> pd.DataFrame:
    """樹模型對 X 每列的 SHAP（欄 = 特徵，另加 base）。背景分布用訓練時樹節點的樣本數（tree_path_dependent）。"""
    import shap
    e = shap.TreeExplainer(predict.model, feature_perturbation="tree_path_dependent")
    sv = pd.DataFrame(e.shap_values(X.to_numpy()), index=X.index, columns=X.columns)
    return sv.assign(base=float(np.ravel(e.expected_value)[0]))


def importance(sv: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """{年: SHAP 表} → 年 × 特徵 的平均 |SHAP| 佔比（%）。"""
    imp = pd.DataFrame({y: v.drop(columns="base").abs().mean() for y, v in sv.items()}).T
    return imp.div(imp.sum(axis=1), axis=0) * 100


def direction(sv: pd.DataFrame, X: pd.DataFrame) -> pd.Series:
    """特徵值和它的 SHAP 的相關：正 = 值越大越推高預測。"""
    return pd.Series({f: X[f].corr(sv[f]) for f in X.columns})


def stability(imp: pd.DataFrame) -> float:
    """相鄰年度重要度排序的等級相關平均。"""
    r = imp.rank(axis=1)
    return float(np.mean([r.iloc[i].corr(r.iloc[i + 1]) for i in range(len(r) - 1)])) if len(r) > 1 else np.nan


def top_contrib(row: pd.Series, k: int = 3) -> str:
    v = row.drop("base")
    return "、".join(f"{f} {x:+.4f}" for f, x in v.reindex(v.abs().sort_values(ascending=False).index[:k]).items())


def explain(models: dict, X: pd.DataFrame, out: Path, n_per_year: int = 3000) -> None:
    lvl = X.index.get_level_values(0)
    ok = X.notna().all(axis=1)
    sv, xs = {}, {}
    for year, predict in sorted(models.items()):
        rows = X[ok & (lvl.year == year)]
        if len(rows) > n_per_year:
            rows = rows.sample(n_per_year, random_state=0)
        sv[year], xs[year] = shap_values(predict, rows), rows
    imp = importance(sv)
    print(f"\n樹模型 SHAP：各年度預測的平均 |SHAP| 佔比（%，每年抽 {n_per_year} 列）")
    print(imp.round(1).to_string())
    last = max(sv)
    d = direction(sv[last], xs[last])
    print(f"\n{last} 年方向（特徵值和 SHAP 的相關，正 = 值越大越推高預測）:")
    print("  " + "、".join(f"{f} {v:+.2f}" for f, v in d.sort_values().items()))
    print(f"相鄰年度重要度排序的等級相關平均：{stability(imp):+.2f}（1 = 每年排序都一樣）")
    a, b = "from_high", "ma200_gap"
    allx, alls = pd.concat(xs.values()), pd.concat(sv.values())
    print(f"相關特徵：{a} 和 {b} 的特徵相關 {allx[a].corr(allx[b]):+.2f}，SHAP 相關 {alls[a].corr(alls[b]):+.2f}；"
          f"兩者 SHAP 相加後平均 |SHAP| {(alls[a] + alls[b]).abs().mean():.4f}，"
          f"分開相加 {alls[a].abs().mean() + alls[b].abs().mean():.4f}（前者小很多表示互相抵銷）")
    latest = lvl.max()
    rows = X[ok & (lvl == latest)]
    s = shap_values(models[latest.year], rows)
    s.assign(pred=s.drop(columns="base").sum(axis=1) + s["base"]).reset_index().to_parquet(out, index=False)
    imp.to_csv(out.with_name("ml_shap_importance.csv"))
    print(f"最新一期（{latest:%Y-%m-%d}）{len(s)} 檔的 SHAP：{out}")


# ---------------------------------------------------------------- 主程式

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2011-01")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--top-min-value", type=float, default=2e7)
    ap.add_argument("--llm", action="store_true", help="另做 LLM 同候選比較")
    ap.add_argument("--shap", action="store_true", help="另做樹模型的 SHAP 解釋")
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
    models = {"ml_en": {}, "ml_gbt": {}}
    preds = {"ml_en": walk_forward(X, y, elastic_net(meta["en"]), models=models["ml_en"]),
             "ml_gbt": walk_forward(X, y, gbt(meta["gbt"]), models=models["ml_gbt"])}
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
    if a.shap:
        explain(models["ml_gbt"], X, out / "ml_shap_latest.parquet")
    if a.llm:
        llm_compare(models, out / "ml_vs_llm.csv", explain=a.shap)


if __name__ == "__main__":
    main()
