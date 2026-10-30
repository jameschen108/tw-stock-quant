"""lmw.py：核迴歸形態（Lo, Mamaysky & Wang 2000）的偵測、形態定義、KS 檢定與不偷看未來。"""
import numpy as np
import pandas as pd
import pytest

import lmw as M


def wide(close: dict[str, np.ndarray]) -> dict:
    n = len(next(iter(close.values())))
    c = pd.DataFrame(close, index=pd.bdate_range("2024-01-01", periods=n))
    return {"adj_close": c, "close": c}


def segs(*pts: tuple[int, float]) -> np.ndarray:
    """(日, 價) 折線。"""
    days, px = zip(*pts)
    return np.interp(np.arange(days[-1] + 1), days, px)


# 頭肩頂：E1 110（第 15 天）、E2 102、E3 118、E4 102、E5 110（第 43 天），之後下跌
HS = segs((0, 90), (15, 110), (22, 102), (29, 118), (36, 102), (43, 110), (60, 95))


def test_head_and_shoulders_detected_after_lag():
    hits, levels = M.detect(wide({"1101": HS, "1102": 200 - HS}), keep_levels=len(HS))
    day = 43 + M.D_LAG
    assert np.flatnonzero(hits["hs"]["1101"].values).tolist() == [day]
    assert np.flatnonzero(hits["ihs"]["1102"].values).tolist() == [day]
    assert not hits["ihs"]["1101"].any() and not hits["hs"]["1102"].any()
    d = hits["hs"].index[day]
    assert levels["hs"]["上緣"].at[d, "1101"] == pytest.approx((110 + 118 + 110) / 3)
    assert levels["hs"]["下緣"].at[d, "1101"] == pytest.approx(102)


def test_classify_definitions():
    t = np.array([[0, 7, 14, 21, 28]] * 2)
    first = np.array([1, -1])
    e = np.array([[110, 100, 110.5, 100.3, 110.2],          # 矩形頂：高點、低點各在 0.75% 內
                  [100, 110, 100.4, 110.5, 100.2]])         # 矩形底
    r = M.classify(e, t, first)
    assert r["rtop"].tolist() == [True, False] and r["rbot"].tolist() == [False, True]
    e = np.array([[100, 95, 105, 93, 110], [100, 105, 95, 107, 90]])   # 擴張頂、擴張底
    r = M.classify(e, t, first)
    assert r["btop"].tolist() == [True, False] and r["bbot"].tolist() == [False, True]
    e = np.array([[110, 90, 106, 95, 102], [90, 110, 94, 106, 98]])    # 三角頂、三角底
    r = M.classify(e, t, first)
    assert r["ttop"].tolist() == [True, False] and r["tbot"].tolist() == [False, True]
    e = np.array([[110, 100, 105, 98, 110.5]] * 2)                     # 雙重頂：E1、E5 相差 < 1.5%、隔 28 日
    r = M.classify(e, t, np.array([1, 1]))
    assert r["dtop"].all()
    r = M.classify(e, np.array([[0, 5, 10, 15, 20]] * 2), np.array([1, 1]))
    assert not r["dtop"].any()                                          # 只隔 20 日 ≤ 22


def test_ks_2samp():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    d, p = M.ks_2samp(a, a.copy())
    assert d == 0 and p == 1
    d, p = M.ks_2samp(a, rng.normal(0.3, 1, 2000))
    assert d > 0.08 and p < 1e-6
    _, p = M.ks_2samp(a, rng.normal(0, 1, 2000))
    assert p > 0.01


def test_detect_has_no_lookahead():
    rng = np.random.default_rng(1)
    n = 260
    close = {f"{1101 + i}": 50 * np.exp(np.cumsum(rng.normal(0, 0.02, n))) for i in range(8)}
    full, _ = M.detect(wide(close))
    for cut in (120, 180, 230):
        part, _ = M.detect(wide({k: v[:cut + 1] for k, v in close.items()}))
        for name in M.PATTERNS:
            pd.testing.assert_frame_equal(part[name], full[name].iloc[:cut + 1], check_freq=False, obj=name)
    assert sum(int(x.values.sum()) for x in full.values()) > 0


def test_fixed_bandwidth_variant():
    hits, _ = M.detect(wide({"1101": HS, "1102": 200 - HS}), h=1.0)
    assert np.flatnonzero(hits["hs"]["1101"].values).tolist() == [43 + M.D_LAG]
    assert np.flatnonzero(hits["ihs"]["1102"].values).tolist() == [43 + M.D_LAG]
    rng = np.random.default_rng(2)
    close = {f"{1101 + i}": 50 * np.exp(np.cumsum(rng.normal(0, 0.02, 400))) for i in range(5)}
    w = wide(close)
    ext = {h: M._n_extrema(M._sample_windows(w), h) for h in (0.5, 2.0, 4.0)}
    assert ext[0.5] >= ext[2.0] >= ext[4.0]                 # 頻寬越大、極值越少
    h, table = M.pick_fixed_h(w)
    assert table[h] <= M.TARGET_EXT and all(v > M.TARGET_EXT for k, v in table.items() if k < h)
