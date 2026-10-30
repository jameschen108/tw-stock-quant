"""factors.price_features：可指定任意快照日，預設仍是每月最後一個交易日。"""
import numpy as np
import pandas as pd

import factors as F


def wide(n: int = 400, k: int = 3) -> dict:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2024-01-01", periods=n)
    cols = [f"{1101 + i}" for i in range(k)]
    ret = pd.DataFrame(rng.normal(0, 0.01, (n, k)), index=idx, columns=cols)
    close = 50 * (1 + ret).cumprod()
    return {"close": close, "ret": ret, "ret_suspect": pd.DataFrame(False, index=idx, columns=cols),
            "value": pd.DataFrame(1e8, index=idx, columns=cols), "shares": pd.DataFrame(np.nan, index=idx, columns=cols)}


def test_default_equals_month_end_snapshot():
    w = wide()
    a = F.price_features(w)
    b = F.price_features(w, reb=F.rebalance_dates(w["close"].index))
    pd.testing.assert_frame_equal(a, b)


def test_custom_snapshot_dates():
    w = wide()
    days = w["close"].index[[100, 210, 330]]                  # 月中的交易日，少於 12 個也能算
    d = F.price_features(w, reb=days)
    assert sorted(d["date"].unique()) == list(days)
    got = d.set_index(["date", "code"])["close"]
    assert got[(days[1], "1102")] == w["close"].at[days[1], "1102"]
    assert d["mom_12_1"].isna().all()                          # 不足 12 期，動能給缺值
