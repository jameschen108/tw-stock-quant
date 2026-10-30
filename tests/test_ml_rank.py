"""ml_rank.py：特徵只用同期資料、滾動訓練只用已到期的標籤（不需要 scikit-learn）。"""
import numpy as np
import pandas as pd

import ml_rank as M


def panel(n_dates: int = 40, n_codes: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2011-01-31", periods=n_dates, freq="ME")
    idx = pd.MultiIndex.from_product([dates, [f"{1101 + i}" for i in range(n_codes)]], names=["date", "code"])
    df = pd.DataFrame(rng.normal(size=(len(idx), len(M.FEATURES))), index=idx, columns=list(M.FEATURES)).reset_index()
    df["industry"] = np.where(df["code"].str[-1].astype(int) % 2 == 0, "A", "B")
    df["size"] = rng.normal(size=len(df))
    df["fwd_ret"] = 0.02 * df["rev_accel"] + rng.normal(0, 0.05, len(df))      # 只有 rev_accel 有訊號
    return df


def ols(log: list) -> M.Fit:
    """測試用的最小平方法；記下每次訓練用到的最後一期。"""
    def fit(X, y):
        log.append(X.index.get_level_values(0).max())
        b, *_ = np.linalg.lstsq(np.c_[np.ones(len(X)), X.to_numpy()], y.to_numpy(), rcond=None)
        return lambda Xp: np.c_[np.ones(len(Xp)), Xp.to_numpy()] @ b
    return fit


def test_features_use_only_same_date():
    df = panel()
    full = M.features(df)
    d = df["date"].unique()[10]
    part = M.features(df[df["date"] <= d])
    pd.testing.assert_frame_equal(full[df["date"] <= d], part)
    assert full.notna().all().all()


def test_walk_forward_uses_only_matured_labels():
    df = panel()
    X = M.features(df).set_index([df["date"], df["code"]])
    y = M.label(df).set_axis(X.index)
    dates = np.sort(df["date"].unique())
    log: list = []
    pred = M.walk_forward(X, y, ols(log), start=str(pd.Timestamp(dates[14]).date()))
    pdates = np.sort(pred.index.get_level_values(0).unique())
    assert pdates[0] == dates[14]
    for t in pdates:                                        # 每個預測期用的模型，訓練資料都在 t−2 期以前
        i = int(np.searchsorted(dates, t))
        trained = [x for x in log if x <= dates[i - M.LABEL_LAG]]
        assert trained and max(trained) <= dates[i - M.LABEL_LAG]
    # 改掉 t−1 期以後的標籤，t 期的預測不能變
    t = dates[20]
    y2 = y.copy()
    y2[y2.index.get_level_values(0) >= dates[19]] = 0.5
    p2 = M.walk_forward(X, y2, ols([]), start=str(pd.Timestamp(dates[14]).date()))
    same = pred.index.get_level_values(0) <= t
    if pd.Timestamp(t).year == pd.Timestamp(dates[14]).year:
        pd.testing.assert_series_equal(pred[same], p2[same])
    assert len(log) == len({pd.Timestamp(x).year for x in pdates})    # 每年重訓一次


def test_walk_forward_learns_signal():
    df = panel(n_dates=48, n_codes=120, seed=1)
    X = M.features(df).set_index([df["date"], df["code"]])
    y = M.label(df).set_axis(X.index)
    pred = M.walk_forward(X, y, ols([]), start="2013-01-31")
    ic = M.date_ic(pred.to_numpy(), y.reindex(pred.index), pd.Series(pred.index.get_level_values(0)))
    assert ic > 0.1
