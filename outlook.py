# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""短 / 中 / 長期看漲、看跌標籤與選股清單（學術研究用），每個標籤附歷史校準。

用法:
  uv run outlook.py               # 重算全部歷史標籤與校準，印出校準表與最新選股清單
  uv run outlook.py --code 2330   # 另外印出個股的三個期間標籤

方法（每個換股日 = composite_scores.parquet 的日期，即每月最後一個交易日 + 最新資料日）:
  1. 每個期間把幾個訊號各自轉成股票池內百分位（方向已轉正），平均成該期間分數（至少一半訊號有值）
  2. 分數百分位 >= 80 → 看漲，<= 20 → 看跌，其餘中性
  3. 校準：之後 H 個交易日的產業中性超額報酬（訊號日隔天收盤進場，減同日同產業等權平均），
     依標籤統計平均超額、跑贏同產業的比例；看漲 − 看跌 的 t 值用 Newey-West（持有期重疊）
  4. 可靠度看樣本外（2019 起）：看漲 − 看跌 > 0 且 t >= 2 → 有統計支持；> 0 但 t < 2 → 弱；<= 0 → 不可靠
選股清單：最新一期 20 日均成交值 >= 2,000 萬，在「有統計支持」的期間都看漲，依這些期間的平均百分位排序；
同樣條件在歷史上的表現也一起校準。

期間與訊號（訊號選擇見 HORIZONS 註解；部分是看過全期結果後挑的，樣本外仍有選擇偏誤）:
  短期 20 個交易日（約 1 個月）、中期 60 個交易日（約 3 個月）、長期 250 個交易日（約 1 年）

輸出 data/outlook.parquet（日期 × 股票：各期間百分位與標籤）、data/outlook_calib.json（校準表）。
標籤是歷史統計的分組，不是對個股的報酬預測；回測期間的關係未來不一定成立。
"""
import argparse
import json

import numpy as np
import pandas as pd

import technicals as tc
from build_panel import OUT

BULL, BEAR = 80, 20
MIN_VALUE = 2e7            # 選股清單的流動性門檻（同 composite.py 的 Top-N）
TOP_PICKS = 30
PERIODS = {"樣本內 2011–2018": ("2011-01-01", "2018-12-31"), "樣本外 2019–": ("2019-01-01", "2100-01-01"),
           "近期 2024–": ("2024-01-01", "2100-01-01")}
OOS = "樣本外 2019–"
# 訊號 -> 方向（+1 越大越看漲，−1 越小越看漲）
#   短期：technicals.py 在樣本內（2011–2018）H10 / H30 t 值都 > 2 的趨勢類訊號（均線排列、站上 21 日線、
#         MACD、距一年高點）；反轉類（ret_20d、RSI）樣本內不顯著，沒放
#   中期：綜合分數（composite.py，原本就是下月報酬的訊號）
#   長期：綜合分數中的品質 + 價值兩組（基本面，換手慢）
HORIZONS = {
    "short": {"name": "短期", "days": 20, "desc": "約 1 個月",
              "signals": {"triple_ma": 1, "px_ma21": 1, "macd": 1, "from_high": 1}},
    "mid":   {"name": "中期", "days": 60, "desc": "約 3 個月", "signals": {"comp_equal": 1}},
    "long":  {"name": "長期", "days": 250, "desc": "約 1 年", "signals": {"quality": 1, "value": 1}},
}
LABELS = ["看漲", "中性", "看跌"]


def label(p: pd.Series) -> pd.Series:
    return pd.Series(np.where(p >= BULL, "看漲", np.where(p <= BEAR, "看跌", "中性")), index=p.index).where(p.notna())


def build() -> tuple[pd.DataFrame, dict]:
    sc = pd.read_parquet(OUT / "bt" / "composite_scores.parquet")
    dates = pd.DatetimeIndex(sorted(sc["date"].unique()))
    w = tc.load()
    days = w["adj_close"].index
    dates = dates[dates.isin(days)]
    tech_names = {s for h in HORIZONS.values() for s in h["signals"] if s in tc.SIGNALS}
    tech = tc.signals(w, dates) if tech_names else {}
    fm = pd.read_parquet(OUT / "factors_monthly.parquet", columns=["date", "code", "industry"])
    d = sc.merge(fm, on=["date", "code"], how="left")
    d = d[d["date"].isin(dates)]
    for n in tech_names:
        d = d.merge(tech[n].stack().rename(n).reset_index().rename(columns={"level_0": "date"}),
                    on=["date", "code"], how="left")
    # 各期間分數與標籤
    for k, h in HORIZONS.items():
        sig = h["signals"]
        ranks = pd.concat({s: d.groupby("date")[s].rank(pct=True) * (1 if v > 0 else -1) + (0 if v > 0 else 1)
                           for s, v in sig.items()}, axis=1)
        score = ranks.mean(axis=1).where(ranks.notna().sum(axis=1) >= max(1, len(sig) / 2))
        d[f"{k}_pct"] = (score.groupby(d["date"]).rank(pct=True) * 100).round()
        d[f"{k}_label"] = label(d[f"{k}_pct"])
        # 之後 H 日產業中性超額
        fwd = tc.forward(w, h["days"]).loc[dates].stack().rename("fwd").reset_index().rename(columns={"level_0": "date"})
        x = d[["date", "code", "industry"]].merge(fwd, on=["date", "code"], how="left")
        d[f"{k}_ex"] = (x["fwd"] - x.groupby(["date", "industry"])["fwd"].transform("mean")).values
    return d, calibrate(d)


def _calib_group(d: pd.DataFrame, mask: pd.Series, ex: str, lag: int) -> dict:
    g = d[mask & d[ex].notna()]
    by = g.groupby("date")[ex]
    m = by.mean()
    return {"avg_n": round(float(by.size().mean()), 1) if len(m) else 0, "mean_ex_pct": round(float(m.mean() * 100), 2)
            if len(m) else None, "hit_pct": round(float((g[ex] > 0).mean() * 100), 1) if len(g) else None,
            "_series": m}


def calibrate(d: pd.DataFrame) -> dict:
    out = {}
    for k, h in HORIZONS.items():
        lag = max(1, round(h["days"] / 21))                 # 月取樣、持有 H 日 → 重疊約 H/21 期
        ex, lab = f"{k}_ex", f"{k}_label"
        res = {"name": h["name"], "days": h["days"], "desc": h["desc"], "signals": h["signals"], "periods": {}}
        for p, (a, b) in PERIODS.items():
            in_p = (d["date"] >= a) & (d["date"] <= b)
            groups = {l: _calib_group(d, in_p & (d[lab] == l), ex, lag) for l in LABELS}
            sp = (groups["看漲"]["_series"] - groups["看跌"]["_series"]).dropna()
            res["periods"][p] = {
                **{l: {kk: v for kk, v in g.items() if kk != "_series"} for l, g in groups.items()},
                "spread_pct": round(float(sp.mean() * 100), 2) if len(sp) else None,
                "t": round(float(tc.nw_t(sp, lag)), 2) if len(sp) >= 20 else None, "n_dates": int(len(sp))}
        o = res["periods"][OOS]
        res["reliability"] = ("不可靠" if o["spread_pct"] is None or o["spread_pct"] <= 0 else
                              "有統計支持" if (o["t"] or 0) >= 2 else "弱")
        out[k] = res
    # 選股清單條件的歷史表現：有統計支持的期間都看漲 + 流動性
    use = [k for k, r in out.items() if r["reliability"] == "有統計支持"]
    out["_picks"] = {"horizons": use}
    if use:
        mask = (d["avg_value"] >= MIN_VALUE)
        for k in use:
            mask &= d[f"{k}_label"] == "看漲"
        pk = {}
        for k in use:
            lag = max(1, round(HORIZONS[k]["days"] / 21))
            pk[k] = {p: {kk: v for kk, v in _calib_group(d, mask & (d["date"] >= a) & (d["date"] <= b),
                                                           f"{k}_ex", lag).items() if kk != "_series"}
                     for p, (a, b) in PERIODS.items()}
        out["_picks"]["history"] = pk
    return out


def picks(d: pd.DataFrame, calib: dict, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    use = calib["_picks"]["horizons"]
    asof = asof or d["date"].max()
    cur = d[(d["date"] == asof) & (d["avg_value"] >= MIN_VALUE)].copy()
    if not use:
        return cur.iloc[:0]
    for k in use:
        cur = cur[cur[f"{k}_label"] == "看漲"]
    cur["pick_score"] = cur[[f"{k}_pct" for k in use]].mean(axis=1)
    return cur.sort_values("pick_score", ascending=False).head(TOP_PICKS)


def load_outlook(code: str | None = None) -> pd.DataFrame:
    return pd.read_parquet(OUT / "outlook.parquet", filters=[("code", "==", code)] if code else None)


def load_calib() -> dict:
    f = OUT / "outlook_calib.json"
    return json.loads(f.read_text()) if f.exists() else {}


def stock_outlook(code: str, asof: pd.Timestamp) -> dict:
    """給 report.py / analyst.py：asof 當時最新一期的三個期間標籤與各自的歷史校準。"""
    f = OUT / "outlook.parquet"
    if not f.exists():
        return {}
    o = load_outlook(code)
    o = o[o["date"] <= asof].sort_values("date")
    if o.empty:
        return {}
    last, cal = o.iloc[-1], load_calib()
    sec = {"date": str(last["date"].date()),
           "method": "訊號百分位 >= 80 看漲、<= 20 看跌；歷史 = 樣本外（2019 起）之後 N 個交易日的產業中性超額報酬",
           "hit_pct_note": "勝率 = 跑贏同產業平均的比例；個股報酬右偏（少數大漲拉高平均），所以各組勝率都低於 50%，"
                           "要和其他標籤比較，不能單獨看",
           "horizons": {}}
    for k, h in HORIZONS.items():
        lab = last[f"{k}_label"]
        c = cal.get(k, {})
        oos = c.get("periods", {}).get(OOS, {})
        sec["horizons"][h["name"]] = {
            "label": None if pd.isna(lab) else lab, "pct": None if pd.isna(last[f"{k}_pct"]) else int(last[f"{k}_pct"]),
            "holding": f"{h['days']} 個交易日（{h['desc']}）", "reliability": c.get("reliability"),
            "history_same_label_oos": None if pd.isna(lab) else oos.get(lab),
            "history_by_label_oos": {l: oos.get(l, {}).get("mean_ex_pct") for l in LABELS},
            "bull_minus_bear_oos_pct": oos.get("spread_pct"), "t": oos.get("t")}
    return sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code")
    a = ap.parse_args()
    d, cal = build()
    keep = ["date", "code", "avg_value", *[f"{k}_{s}" for k in HORIZONS for s in ("pct", "label")]]
    d[keep].to_parquet(OUT / "outlook.parquet", index=False)
    (OUT / "outlook_calib.json").write_text(json.dumps(cal, ensure_ascii=False, indent=1))
    for k, c in ((k, cal[k]) for k in HORIZONS):
        print(f"\n== {c['name']}（{c['days']} 個交易日，{c['desc']}）訊號 {c['signals']}，可靠度：{c['reliability']}")
        rows = []
        for p, v in c["periods"].items():
            rows.append({"期間": p, **{f"{l}超額%": v[l]["mean_ex_pct"] for l in LABELS},
                         **{f"{l}勝率%": v[l]["hit_pct"] for l in LABELS},
                         "漲−跌%": v["spread_pct"], "t": v["t"], "期數": v["n_dates"]})
        print(pd.DataFrame(rows).set_index("期間").to_string())
    pk = picks(d, cal)
    names = pd.read_parquet(OUT / "prices_daily.parquet", columns=["code", "name"]).drop_duplicates("code", keep="last") \
        .set_index("code")["name"]
    print(f"\n選股清單（{d['date'].max():%Y-%m-%d}，條件：{'、'.join(HORIZONS[k]['name'] for k in cal['_picks']['horizons']) or '無'}"
          f"都看漲 + 成交值門檻）：{len(pk)} 檔")
    if len(pk):
        print(pk.assign(name=pk["code"].map(names))[["code", "name", *[f"{k}_pct" for k in HORIZONS]]].to_string(index=False))
        for k, v in cal["_picks"].get("history", {}).items():
            print(f"  同條件歷史（{HORIZONS[k]['name']}持有）:", {p: x for p, x in v.items()})
    if a.code:
        print(json.dumps(stock_outlook(a.code, d["date"].max()), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
