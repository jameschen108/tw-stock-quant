"""patterns.py 的形態偵測：W 底、跳空的價位，以及偵測不偷看未來。"""
import numpy as np
import pandas as pd
import pytest

import patterns as P


def wide(close: dict[str, np.ndarray], spread: float = 0.005, value: dict | None = None) -> dict:
    """收盤序列 → detect() 要的寬表；最高 / 最低 = 收盤 ± spread。"""
    n = len(next(iter(close.values())))
    idx = pd.bdate_range("2024-01-01", periods=n)
    c = pd.DataFrame(close, index=idx)
    v = pd.DataFrame(value, index=idx) if value else pd.DataFrame(1e8, index=idx, columns=c.columns)
    return {"adj_close": c, "adj_high": c * (1 + spread), "adj_low": c * (1 - spread), "close": c, "value": v}


def test_w_bottom_confirms_on_neckline_break():
    # 120 → 100（第 20 天）→ 112（第 35 天）→ 101（第 50 天）→ 一路漲
    path = np.r_[np.linspace(120, 100, 21), np.linspace(100, 112, 16)[1:], np.linspace(112, 101, 16)[1:],
                 101 + 0.7 * np.arange(1, 20)]
    w = wide({"1101": path})
    hits, levels = P.detect(w)
    neck = 112 * 1.005
    expect = 50 + int(np.flatnonzero(path[50:] > neck)[0])
    assert np.flatnonzero(hits["w_bottom"]["1101"].values).tolist() == [expect]
    d = w["adj_close"].index[expect]
    assert levels["w_bottom"]["頸線"].at[d, "1101"] == pytest.approx(neck)
    assert levels["w_bottom"]["底部低點"].at[d, "1101"] == pytest.approx(100 * 0.995)
    assert not hits["m_top"]["1101"].any()


def test_gap_levels():
    path = np.r_[np.full(30, 100.0), np.full(10, 110.0)]
    w = wide({"1101": path, "1102": path[::-1].copy()})
    hits, levels = P.detect(w, keep_levels=15)
    d = w["adj_close"].index[30]
    assert hits["gap_up"].loc[:, "1101"].sum() == 1 and hits["gap_up"].at[d, "1101"]
    assert levels["gap_up"]["缺口上緣"].at[d, "1101"] == pytest.approx(110 * 0.995)
    assert levels["gap_up"]["缺口下緣"].at[d, "1101"] == pytest.approx(100 * 1.005)
    assert hits["gap_dn"].at[w["adj_close"].index[10], "1102"] and not hits["gap_dn"]["1101"].any()


def test_detect_has_no_lookahead():
    rng = np.random.default_rng(0)
    n = 320
    close = {f"{1101 + i}": 50 * np.exp(np.cumsum(rng.normal(0, 0.025, n))) for i in range(6)}
    value = {k: rng.lognormal(18, 0.8, n) for k in close}
    full, _ = P.detect(wide(close, 0.015, value))
    for cut in (220, 260, 300):
        part, _ = P.detect(wide({k: v[:cut + 1] for k, v in close.items()}, 0.015,
                                {k: v[:cut + 1] for k, v in value.items()}))
        for name in P.PATTERNS:
            pd.testing.assert_frame_equal(part[name], full[name].iloc[:cut + 1], check_freq=False, obj=name)
    assert sum(int(x.values.sum()) for x in full.values()) > 0
