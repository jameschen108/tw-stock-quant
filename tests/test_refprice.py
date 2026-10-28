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
    assert 0 < cal["stop"][20][R.OOS]["hit_pct"] < 100
