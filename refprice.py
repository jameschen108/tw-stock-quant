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

輸出 data/refprice_calib.json（校準表）、data/refprice_latest.parquet（最新一天每檔的參考價）。
"""
import argparse
import json

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
        # 之後 1..H 日的最低價（反轉序列做 rolling，再位移一天）
        fmin = lo.iloc[::-1].rolling(hz, min_periods=hz).min().iloc[::-1].shift(-1).where(clean).loc[sample]
        stop = (c - ATR_MULT * atr).loc[sample].where(u)
        sv = fmin.notna() & stop.notna()
        band, stp = {}, {}
        for p, (a, b) in PERIODS.items():
            v = valid.loc[a:b]
            band[p] = {"inside_pct": _pct(inside.loc[a:b], v), "above_pct": _pct(above.loc[a:b], v),
                       "below_pct": _pct(below.loc[a:b], v), "n": int(v.values.sum())}
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
                rows.append({"H": hz, "期間": p, "區間內%": b["inside_pct"], "超出上緣%": b["above_pct"],
                             "跌破下緣%": b["below_pct"], "停損觸及%": s["hit_pct"], "樣本數": b["n"]})
        print("統計波動區間覆蓋率（目標約 68%）與停損參考觸及率")
        print(pd.DataFrame(rows).set_index(["H", "期間"]).to_string())
        for hz in H_BANDS:
            print(f"\nH{hz} 樣本外依波動分組:", [(x["median_vol_ann_pct"], x["inside_pct"])
                                           for x in cal["band"][hz]["by_vol_quintile_oos"]])
    print(f"\nrefprice_latest.parquet：{w['adj_close'].index[-1]:%Y-%m-%d}，{len(lt)} 檔")
    if a.code:
        print(json.dumps(stock_refprice(a.code), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
