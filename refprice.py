# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""參考價與參考區間：全部用公式算，附歷史校準（學術研究用）。不是目標價。

用法:
  uv run refprice.py               # 校準 + 存最新一天的參考價
  uv run refprice.py --latest      # 只存最新參考價（沿用既有校準，給 update.py 用；要先跑 patterns.py）
  uv run refprice.py --code 2330   # 另外印出個股的參考價

三種參考價（用還原價計算，輸出時換回最新一天的價格基準）:
  支撐 / 壓力   21 / 60 / 200 日線、52 週收盤高低點、patterns.py 成立中的箱頂箱底與頸線中，
                低於現價最近的一個（支撐）、高於現價最近的一個（壓力）。技術面常用價位，不是預測
  停損參考      收盤 − 2 × ATR(14)（Wilder 平滑），風控用
  統計波動區間  收盤 × (1 ± 60 日日報酬標準差 × √H)，H = 20、60；等同 60 日年化波動 × √(H/244)。
                常態假設下約 68% 的機率範圍，不是目標價

校準（每 5 個交易日取樣、股票池同 technicals.py，分樣本內 2011–2018 / 樣本外 2019– / 近期 2024–）:
  統計區間：H 日後收盤落在區間內的比例（接近 68% 才算校準得好），以及超出上緣、跌破下緣的比例；
            樣本外再依當時波動分五組看
  停損參考：之後 H 日內最低價碰到停損價的比例
  區間評分（統計區間，看的是 H 日後的終點收盤，不是路徑上曾經碰到）:
    - 平均寬度與 interval score（Gneiting & Raftery 2007, JASA 102(477)）：
      S = 寬度 + (2/α)·跌破下緣的距離 + (2/α)·超出上緣的距離，α = 1 − 68.27%；越低越好，
      只看覆蓋率會鼓勵把區間放寬，這個分數同時懲罰寬度和漏出，之後比較 GARCH 等方法用它
    - Christoffersen (1998, International Economic Review 39(4)) 條件覆蓋檢定：每檔各自取不重疊的預測日
      （每 H 日一次，避免持有期重疊），檢定漏出率是否等於名目值（LR_uc）、漏出是否連續發生（LR_ind）、
      兩者合併（LR_cc）；報各期間 5% 水準下被拒絕的股票比例，校準正確時約 5%
    - 同日相依：不重疊預測日每天的漏出率，跨日標準差 ÷ 二項分布預期值；遠大於 1 表示漏出集中在同一天
      （大盤一起動），這時合併所有股票做檢定會過度拒絕，所以只做單檔檢定

輸出 data/refprice_calib.json（校準表）、data/refprice_latest.parquet（最新一天每檔的參考價）。
"""
import argparse
import json
import math

import numpy as np
import pandas as pd

import technicals as tc
from build_panel import OUT

H_BANDS = [20, 60]
ATR_N, ATR_MULT = 14, 2.0
VOL_N = 60
LATEST_ROWS = 320
PERIODS = {"樣本內 2011–2018": ("2011-01-01", "2018-12-31"), "樣本外 2019–": ("2019-01-01", "2100-01-01"),
           "近期 2024–": ("2024-01-01", "2100-01-01")}
OOS = "樣本外 2019–"
MA_LEVELS = {"21 日線": 21, "60 日線": 60, "200 日線": 200}
PATTERN_LEVELS = {"box": ["箱頂", "箱底"], "box_up": ["箱頂", "箱底"], "box_dn": ["箱頂", "箱底"],
                  "w_bottom": ["頸線"], "m_top": ["頸線"]}
NOMINAL_OUT = 0.3173                     # ±1 個標準差的常態區間之外的機率（α）
DISCLAIMER = ("參考價與參考區間是用公式算出的技術價位與統計範圍，不是目標價，也不是對未來股價的預測；"
              "僅供學術研究參考，不構成投資建議。")


def indicators(w: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """還原收盤、ATR(14)、60 日日報酬標準差。"""
    c = w["adj_close"].ffill(limit=5)
    h = w["adj_high"].ffill(limit=5)
    lo = w["adj_low"].ffill(limit=5)
    pc = c.shift(1)
    tr = pd.DataFrame(np.fmax(np.fmax((h - lo).values, (h - pc).abs().values), (lo - pc).abs().values),
                      index=c.index, columns=c.columns)
    atr = tr.where(h.notna()).ewm(alpha=1 / ATR_N, adjust=False, min_periods=ATR_N).mean()
    vol = w["ret"].where(~w["ret_suspect"]).rolling(VOL_N, min_periods=VOL_N).std()
    return c, atr, vol


# ---------------------------------------------------------------- 校準

def _pct(x: pd.DataFrame, valid: pd.DataFrame) -> float | None:
    n = int(valid.values.sum())
    return round(float(x.where(valid, False).values.sum() / n * 100), 1) if n else None


def _mean_pct(x: pd.DataFrame) -> float | None:
    v = x.to_numpy().ravel()
    v = v[~np.isnan(v)]
    return round(float(v.mean() * 100), 2) if len(v) else None


def interval_score(y: pd.DataFrame, half: pd.DataFrame, alpha: float = NOMINAL_OUT) -> pd.DataFrame:
    """中央區間 [−half, half] 對實現值 y 的 interval score（報酬單位，越低越好）。"""
    return 2 * half + 2 / alpha * ((-half - y).clip(lower=0) + (y - half).clip(lower=0))


def _chi2_p(x: float, df: int) -> float:
    """卡方分布右尾機率（df = 1 或 2）。"""
    return math.erfc(math.sqrt(x / 2)) if df == 1 else math.exp(-x / 2)


def _ll(n: int, p: float) -> float:
    return n * math.log(p) if n else 0.0


def christoffersen(hits: np.ndarray, p: float = NOMINAL_OUT) -> dict | None:
    """單一序列（1 = 漏出、0 = 區間內、NaN = 缺）的 LR_uc、LR_ind、LR_cc 與 p 值；轉移只算相鄰兩期都有值的。"""
    x = hits[~np.isnan(hits)]
    n1, n = int(x.sum()), len(x)
    if n < 20:
        return None
    pi = n1 / n
    lr_uc = max(0.0, -2 * (_ll(n - n1, 1 - p) + _ll(n1, p) - _ll(n - n1, 1 - pi) - _ll(n1, pi)))
    a, b = hits[:-1], hits[1:]
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[ok], b[ok]
    n00, n01 = int(((a == 0) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n10, n11 = int(((a == 1) & (b == 0)).sum()), int(((a == 1) & (b == 1)).sum())
    lr_ind = None
    if n00 + n01 and n10 + n11:
        p01, p11, p2 = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / (n00 + n01 + n10 + n11)
        lr_ind = max(0.0, -2 * (_ll(n00 + n10, 1 - p2) + _ll(n01 + n11, p2)
                                - _ll(n00, 1 - p01) - _ll(n01, p01) - _ll(n10, 1 - p11) - _ll(n11, p11)))
    return {"n": n, "out_rate": pi, "p_uc": _chi2_p(lr_uc, 1),
            "p_ind": None if lr_ind is None else _chi2_p(lr_ind, 1),
            "p_cc": None if lr_ind is None else _chi2_p(lr_uc + lr_ind, 2)}


def coverage_tests(out: pd.DataFrame, valid: pd.DataFrame) -> dict:
    """不重疊預測日的漏出表（列 = 日期）→ 單檔 Christoffersen 檢定的拒絕比例、同日相依比值。"""
    seq = out.astype(float).where(valid)
    res = [r for r in (christoffersen(seq[c].to_numpy()) for c in seq.columns) if r]
    rej = lambda k: round(float(np.mean([r[k] < 0.05 for r in res if r[k] is not None]) * 100), 1) \
        if any(r[k] is not None for r in res) else None
    rate, n = seq.mean(axis=1), seq.notna().sum(axis=1)
    m = n >= 30
    expect = np.sqrt(NOMINAL_OUT * (1 - NOMINAL_OUT) / n[m])
    return {"n_stocks": len(res), "reject_uc_pct": rej("p_uc"), "reject_ind_pct": rej("p_ind"),
            "reject_cc_pct": rej("p_cc"),
            "date_dispersion_ratio": round(float(rate[m].std() / expect.mean()), 2) if m.sum() > 2 else None}


def calibrate(w: dict, uni: pd.DataFrame) -> dict:
    c, atr, vol = indicators(w)
    lo = w["adj_low"].ffill(limit=5)
    idx = c.index
    sample = idx[idx >= "2011-01-01"][::tc.STEP]
    u = uni.loc[sample]
    bad = w["ret_suspect"].astype(int).cumsum()
    out = {"band": {}, "stop": {}}
    for hz in H_BANDS:
        clean = (bad.shift(-hz) - bad == 0)                      # t 到 t+H 之間沒有可疑報酬
        fut = (c.shift(-hz) / c - 1).where(clean).loc[sample].where(u)
        half = (vol * np.sqrt(hz)).loc[sample].where(u)
        valid = fut.notna() & half.notna()
        inside, above, below = fut.abs() <= half, fut > half, fut < -half
        score = interval_score(fut, half).where(valid)
        # 之後 1..H 日的最低價（反轉序列做 rolling，再位移一天）
        fmin = lo.iloc[::-1].rolling(hz, min_periods=hz).min().iloc[::-1].shift(-1).where(clean).loc[sample]
        stop = (c - ATR_MULT * atr).loc[sample].where(u)
        sv = fmin.notna() & stop.notna()
        band, stp = {}, {}
        for p, (a, b) in PERIODS.items():
            v = valid.loc[a:b]
            band[p] = {"inside_pct": _pct(inside.loc[a:b], v), "above_pct": _pct(above.loc[a:b], v),
                       "below_pct": _pct(below.loc[a:b], v), "n": int(v.values.sum()),
                       "width_pct": _mean_pct((2 * half).where(v).loc[a:b]),
                       "interval_score_pct": _mean_pct(score.loc[a:b])}
            # 不重疊預測日：每 H 個交易日一次
            nd = idx[(idx >= max(pd.Timestamp(a), pd.Timestamp("2011-01-01"))) & (idx <= pd.Timestamp(b))][::hz]
            fn = (c.shift(-hz) / c - 1).where(clean).loc[nd].where(uni.loc[nd])
            hn = (vol * np.sqrt(hz)).loc[nd].where(uni.loc[nd])
            band[p]["christoffersen"] = coverage_tests(fn.abs() > hn, fn.notna() & hn.notna())
            s = sv.loc[a:b]
            stp[p] = {"hit_pct": _pct((fmin <= stop).loc[a:b], s), "n": int(s.values.sum())}
        # 樣本外依當時波動（同日橫斷面）分五組
        q = np.ceil(half.rank(axis=1, pct=True) * 5).clip(1, 5)
        oa, ob = PERIODS[OOS]
        band["by_vol_quintile_oos"] = [
            {"quintile": k, "median_vol_ann_pct": round(float(np.nanmedian(
                (half / np.sqrt(hz) * np.sqrt(244)).where(q == k).loc[oa:ob].values) * 100), 1),
             "inside_pct": _pct(inside.loc[oa:ob], valid.loc[oa:ob] & (q == k).loc[oa:ob])}
            for k in range(1, 6)]
        out["band"][hz], out["stop"][hz] = band, stp
    return out


# ---------------------------------------------------------------- 最新參考價

def latest(w: dict, uni: pd.DataFrame, pats: pd.DataFrame | None = None) -> pd.DataFrame:
    c, atr, vol = indicators(w)
    last = c.index[-1]
    close = w["close"].loc[last]
    base = (w["adj_close"] / w["close"]).loc[last]               # 還原價 → 最新一天的價格基準
    cand = {k: tc.ma(c, n).loc[last] for k, n in MA_LEVELS.items()}
    cand["52 週高點"] = c.rolling("365D").max().loc[last]
    cand["52 週低點"] = c.rolling("365D").min().loc[last]
    cand = pd.DataFrame(cand).div(base, axis=0)
    extra: dict[str, dict[str, float]] = {}
    if pats is not None and len(pats) and pats["date"].max() == last:
        for r in pats[pats["pattern"].isin(PATTERN_LEVELS)].itertuples():
            lv = json.loads(r.levels)
            for k in PATTERN_LEVELS[r.pattern]:
                extra.setdefault(r.code, {})[f"{r.name}的{k}"] = lv[k]
    stop = (c - ATR_MULT * atr).loc[last] / base
    vq = np.ceil(vol.loc[last].where(uni.loc[last]).rank(pct=True) * 5).clip(1, 5)   # 同 calibrate 的波動分組
    rows = []
    for code in close.index[close.notna()]:
        px = float(close[code])
        lv = {k: round(float(v), 2) for k, v in cand.loc[code].items() if pd.notna(v)} | extra.get(code, {})
        below = {k: v for k, v in lv.items() if v < px}
        above = {k: v for k, v in lv.items() if v > px}
        sup = max(below, key=below.get) if below else None
        res = min(above, key=above.get) if above else None
        sd = vol.at[last, code]
        row = {"date": last, "code": code, "close": px,
               "support": below.get(sup), "support_src": sup, "resistance": above.get(res), "resistance_src": res,
               "atr14": round(float(atr.at[last, code] / base[code]), 2) if pd.notna(atr.at[last, code]) else None,
               "stop": round(float(stop[code]), 2) if pd.notna(stop[code]) else None,
               "vol_60d_ann": round(float(sd * np.sqrt(244)), 4) if pd.notna(sd) else None,
               "vol_quintile": int(vq[code]) if pd.notna(vq.get(code)) else None,
               "candidates": json.dumps(lv, ensure_ascii=False), "in_universe": bool(uni.at[last, code])}
        for hz in H_BANDS:
            ok = pd.notna(sd)
            row[f"band{hz}_lo"] = round(px * (1 - sd * np.sqrt(hz)), 2) if ok else None
            row[f"band{hz}_hi"] = round(px * (1 + sd * np.sqrt(hz)), 2) if ok else None
        rows.append(row)
    return pd.DataFrame(rows)


def stock_refprice(code: str) -> dict:
    """給事實表：個股最新一天的參考價、參考區間與歷史校準（樣本外）。"""
    f = OUT / "refprice_latest.parquet"
    if not f.exists():
        return {}
    d = pd.read_parquet(f, filters=[("code", "==", code)])
    if d.empty:
        return {}
    r = d.iloc[0]
    cf = OUT / "refprice_calib.json"
    cal = json.loads(cf.read_text()) if cf.exists() else {}
    band, stop = cal.get("band", {}), cal.get("stop", {})
    num = lambda x: None if pd.isna(x) else float(x)
    return {
        "date": str(r["date"].date()), "close": num(r["close"]), "disclaimer": DISCLAIMER,
        "support_resistance": {
            "support": {"price": num(r["support"]), "source": r["support_src"]},
            "resistance": {"price": num(r["resistance"]), "source": r["resistance_src"]},
            "candidates": json.loads(r["candidates"]),
            "note": "離現價最近的技術價位（均線、52 週高低、箱頂箱底、頸線），不是預測"},
        "stop_reference": {
            "price": num(r["stop"]), "atr14": num(r["atr14"]), "rule": "收盤 − 2 × ATR(14)",
            "history_oos_hit_pct": {f"{hz} 日內": stop.get(str(hz), {}).get(OOS, {}).get("hit_pct") for hz in H_BANDS},
            "note": "風控參考：股票池歷史上（樣本外）之後 H 日內最低價碰到停損價的比例"},
        "volatility_bands": {
            "vol_60d_ann": num(r["vol_60d_ann"]), "vol_quintile": None if pd.isna(q := r["vol_quintile"]) else int(q),
            **{f"{hz} 日": {"low": num(r[f"band{hz}_lo"]), "high": num(r[f"band{hz}_hi"]),
                           "history_oos_inside_pct": band.get(str(hz), {}).get(OOS, {}).get("inside_pct"),
                           "history_oos_inside_pct_same_vol": _quintile_inside(band.get(str(hz), {}), q)}
               for hz in H_BANDS},
            "note": "收盤 ×（1 ± 60 日波動 × √H）；常態假設下約 68% 的機率範圍，不是目標價。"
                    "歷史覆蓋率：全股票池，與同波動分組（1 最低 ~ 5 最高）"},
    }


def _quintile_inside(band: dict, q) -> float | None:
    if pd.isna(q):
        return None
    return next((x["inside_pct"] for x in band.get("by_vol_quintile_oos", []) if x["quintile"] == int(q)), None)


# ---------------------------------------------------------------- 主程式

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--code")
    a = ap.parse_args()
    w = tc.load()
    if a.latest:
        w = {k: v.iloc[-LATEST_ROWS:] for k, v in w.items()}
    uni = tc.universe(w)
    pf = OUT / "patterns_latest.parquet"
    lt = latest(w, uni, pd.read_parquet(pf) if pf.exists() else None)
    lt.to_parquet(OUT / "refprice_latest.parquet", index=False)
    if not a.latest:
        cal = calibrate(w, uni)
        (OUT / "refprice_calib.json").write_text(json.dumps(cal, ensure_ascii=False, indent=1))
        rows = []
        for hz in H_BANDS:
            for p in PERIODS:
                b, s = cal["band"][hz][p], cal["stop"][hz][p]
                ch = b["christoffersen"]
                rows.append({"H": hz, "期間": p, "區間內%": b["inside_pct"], "超出上緣%": b["above_pct"],
                             "跌破下緣%": b["below_pct"], "寬度%": b["width_pct"], "IS%": b["interval_score_pct"],
                             "拒絕uc%": ch["reject_uc_pct"], "拒絕ind%": ch["reject_ind_pct"],
                             "拒絕cc%": ch["reject_cc_pct"], "同日相依": ch["date_dispersion_ratio"],
                             "停損觸及%": s["hit_pct"], "樣本數": b["n"]})
        print("統計波動區間覆蓋率（目標約 68%）、寬度、interval score（IS，越低越好）、"
              "單檔 Christoffersen 檢定 5% 拒絕比例（校準正確約 5%）、同日相依比值（1 = 無相依），與停損參考觸及率")
        print(pd.DataFrame(rows).set_index(["H", "期間"]).to_string())
        for hz in H_BANDS:
            print(f"\nH{hz} 樣本外依波動分組:", [(x["median_vol_ann_pct"], x["inside_pct"])
                                           for x in cal["band"][hz]["by_vol_quintile_oos"]])
    print(f"\nrefprice_latest.parquet：{w['adj_close'].index[-1]:%Y-%m-%d}，{len(lt)} 檔")
    if a.code:
        print(json.dumps(stock_refprice(a.code), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
