"""refprice.py：ATR 停損、支撐壓力的挑選、統計區間在常態報酬下的覆蓋率。"""
import json

import numpy as np
import pandas as pd
import pytest

import refprice as R


def wide(close: dict[str, np.ndarray], start: str = "2024-01-01", spread: float = 1.0) -> dict:
    """收盤序列 → 寬表；最高 / 最低 = 收盤 ± spread（元）。"""
    n = len(next(iter(close.values())))
    c = pd.DataFrame(close, index=pd.bdate_range(start, periods=n))
    return {"adj_close": c, "adj_high": c + spread, "adj_low": c - spread, "close": c,
            "ret": c.pct_change(), "ret_suspect": pd.DataFrame(False, index=c.index, columns=c.columns),
            "value": pd.DataFrame(1e8, index=c.index, columns=c.columns)}


def test_atr_stop_and_support_resistance():
    path = 100 + 0.1 * np.arange(260)                      # 緩漲：收盤在所有均線之上、等於 52 週高點
    w = wide({"1101": path})
    uni = pd.DataFrame(True, index=w["close"].index, columns=w["close"].columns)
    pats = pd.DataFrame([{"date": w["close"].index[-1], "code": "1101", "pattern": "m_top", "name": "M 頭",
                          "levels": json.dumps({"頸線": 130.0, "頭部高點": 140.0})}])
    r = R.latest(w, uni, pats).iloc[0]
    # 每天高低差 2、跳動 0.1 → 真實區間 = 最高 − 最低 = 2（和昨收的差 1.1、0.9 都較小），ATR = 2
    assert r["atr14"] == pytest.approx(2.0, abs=0.01)
    assert r["stop"] == pytest.approx(path[-1] - 4.0, abs=0.02)
    assert r["support_src"] == "21 日線"
    assert r["support"] == pytest.approx(path[-21:].mean(), abs=0.01)
    assert (r["resistance_src"], r["resistance"]) == ("M 頭的頸線", 130.0)


def test_band_coverage_near_68_pct_for_normal_returns():
    rng = np.random.default_rng(1)
    n = 1300
    close = {f"{1101 + i}": 50 * np.cumprod(1 + rng.normal(0, 0.02, n)) for i in range(40)}
    w = wide(close, start="2017-01-02", spread=0.0)
    uni = pd.DataFrame(True, index=w["close"].index, columns=w["close"].columns)
    cal = R.calibrate(w, uni)
    for hz in R.H_BANDS:
        b = cal["band"][hz][R.OOS]
        assert 64 < b["inside_pct"] < 72
        assert b["inside_pct"] + b["above_pct"] + b["below_pct"] == pytest.approx(100, abs=0.2)
    ch = cal["band"][20][R.OOS]["christoffersen"]                            # H60 樣本外每檔只有 13 期，不夠檢定
    assert ch["n_stocks"] == 40 and ch["reject_uc_pct"] <= 25          # 獨立常態報酬：拒絕比例接近 5%
    assert 0.5 < ch["date_dispersion_ratio"] < 1.6                      # 各檔獨立：沒有同日相依
    assert 0 < cal["stop"][20][R.OOS]["hit_pct"] < 100


def test_interval_score():
    y = pd.DataFrame({"a": [0.0, 0.15, -0.30]})
    half = pd.DataFrame({"a": [0.10, 0.10, 0.10]})
    s = R.interval_score(y, half, alpha=0.5)["a"].tolist()
    assert s == pytest.approx([0.2, 0.2 + 4 * 0.05, 0.2 + 4 * 0.20])


def test_christoffersen():
    rng = np.random.default_rng(3)
    iid = (rng.random(400) < R.NOMINAL_OUT).astype(float)
    r = R.christoffersen(iid)
    assert r["p_uc"] > 0.01 and r["p_ind"] > 0.01
    assert R.christoffersen((rng.random(400) < 0.6).astype(float))["p_uc"] < 1e-6       # 漏出率太高
    clustered = np.repeat((rng.random(80) < R.NOMINAL_OUT).astype(float), 5)            # 漏出連續發生
    r = R.christoffersen(clustered)
    assert r["p_ind"] < 1e-6 and r["p_cc"] < 1e-6
    gaps = iid.copy()
    gaps[::7] = np.nan
    assert R.christoffersen(gaps)["n"] == int((~np.isnan(gaps)).sum())
    assert R.christoffersen(iid[:10]) is None
