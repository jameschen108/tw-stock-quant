"""ff.py：2×3 排序、因子組合、持有期時點、迴歸與持股特徵（合成資料，不讀 data/）。"""
import numpy as np
import pandas as pd
import pytest

import factors as F
import ff


def cross_section(n: int = 20) -> pd.DataFrame:
    codes = [f"{1101 + i}" for i in range(n)]
    return pd.DataFrame({"code": codes, "mcap": np.arange(1, n + 1) * 100.0, "bm": np.arange(n)[::-1] / n,
                         "op": np.arange(n) / n, "inv": np.arange(n) / n, "mom": np.arange(n) / n})


def test_sort_2x3_breakpoints():
    lab = ff.sort_2x3(cross_section(), "bm")
    # 市值前 10 檔（≤ 中位數）是 S；B/M 由大到小排，所以最小的股票 B/M 最高
    assert lab["1101"] == "S_H" and lab["1120"] == "B_L"
    assert (lab.str[0] == "S").sum() == 10
    assert (lab.str[-1] == "H").sum() == 6 and (lab.str[-1] == "L").sum() == 6


def test_combine_long_short_legs():
    names = [f"{c}:{s}_{x}" for c in ff.SORTS for s in "SB" for x in "LMH"]
    p = pd.DataFrame({n: [0.0] for n in names})
    p["MKT"] = 0.01
    p["bm:S_H"] = p["bm:B_H"] = 0.02      # HML = 0.02
    p["inv:S_L"] = p["inv:B_L"] = 0.01    # CMA = 保守（低成長）− 積極 = 0.01
    p["bm:S_L"] = 0.03                    # SMB(bm) = (0.02 + 0.03) / 3 − 0.02 / 3
    f = ff.combine(p).iloc[0]
    assert f["HML"] == pytest.approx(0.02 - 0.015)
    assert f["CMA"] == pytest.approx(0.01)
    assert f["SMB"] == pytest.approx(0.05 / 3 - 0.02 / 3)
    assert f["MKT_RAW"] == pytest.approx(0.01)


def test_hold_returns_drift_weights():
    idx = pd.bdate_range("2024-01-01", periods=3)
    r = pd.DataFrame({"A": [0.0, 0.10, 0.0], "B": [0.0, 0.0, 0.10]}, index=idx)
    out = ff.hold_returns(r, 0, 2, {"p": pd.Series({"A": 1.0, "B": 1.0})})["p"]
    assert out.iloc[0] == pytest.approx(0.05)
    assert out.iloc[1] == pytest.approx(1.10 / 1.05 - 1)          # 期初等權，A 漲過後權重變大


def test_factor_period_starts_after_entry_day():
    days = pd.bdate_range("2024-01-01", "2024-03-29")
    reb = F.rebalance_dates(days)
    rets = pd.DataFrame(0.0, index=days, columns=["1101", "1102"])
    t = reb[0]
    entry = days[days.get_loc(t) + ff.LAG]
    rets.loc[entry, "1101"] = 0.5                                   # 成交當天的漲幅不屬於持有期
    after = days[days.get_loc(t) + ff.LAG + 1]
    rets.loc[after, "1102"] = 0.2
    ch = pd.concat([cross_section(2).assign(date=d) for d in reb], ignore_index=True)
    f = ff.build_factors(ch, rets)
    assert f["MKT_RAW"].get(entry, 0.0) == 0.0
    w = 200 / 300                                                   # 市值 100 與 200 的市值加權
    assert f.at[after, "MKT_RAW"] == pytest.approx(0.2 * w)


def test_attach_ignores_later_financials():
    base = pd.DataFrame({"date": pd.to_datetime(["2024-05-31", "2024-08-30"]), "code": "1101"})
    tbl = pd.DataFrame({"code": "1101", "period": pd.to_datetime(["2024-03-31"]),
                        "avail_date": pd.to_datetime(["2024-06-15"]), "op": [0.3]})
    out = F.attach(base, tbl, "fin", ["op"])
    assert np.isnan(out.loc[0, "op"]) and out.loc[1, "op"] == 0.3


def test_nw_ols_recovers_coefficients():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2010-01-31", periods=240, freq="ME")
    X = pd.DataFrame(rng.normal(0, 0.05, (240, 2)), index=idx, columns=["MKT", "SMB"])
    y = 0.01 + 1.2 * X["MKT"] + 0.5 * X["SMB"] + rng.normal(0, 0.01, 240)
    o = ff.nw_ols(y, X)
    assert o["coef"]["alpha"] == pytest.approx(0.01, abs=0.003)
    assert o["coef"]["MKT"] == pytest.approx(1.2, abs=0.05)
    assert o["coef"]["SMB"] == pytest.approx(0.5, abs=0.05)
    white = ff.nw_ols(y, X, lag=0)
    assert o["t"]["MKT"] == pytest.approx(white["t"]["MKT"], rel=0.2)   # 沒有自相關時兩種 t 接近


def test_exposure_percentiles_and_small_share():
    ch = cross_section(10).assign(date=pd.Timestamp("2024-01-31"), industry="M2300")
    pr = ff.pct_ranks(ch)
    e = ff.exposure(pr, ["1101", "1102"], pd.Timestamp("2024-02-15"))
    assert e["size"] == pytest.approx((0.1 + 0.2) / 2)
    assert e["small"] == 1.0 and e["n"] == 2
    assert np.isnan(ff.exposure(pr, ["1101"], pd.Timestamp("2023-12-31"))["size"])   # 換股日之前沒有資料


def test_perm_diff_detects_shift():
    rng = np.random.default_rng(2)
    d = pd.DataFrame({"batch": np.repeat([1, 2], 50), "buy": np.tile([True, False], 50)})
    d["x"] = rng.normal(0, 1, 100) + d["buy"] * 2
    obs, p = ff.perm_diff(d, "x", n=500)
    assert obs > 1.5 and p < 0.01
    d["x"] = rng.normal(0, 1, 100)
    assert ff.perm_diff(d, "x", n=500)[1] > 0.05


def test_rf_on_uses_month_rate_and_carries_forward():
    rf = pd.Series([1.2, 2.4], index=pd.PeriodIndex(["2024-01", "2024-02"], freq="M"))
    idx = pd.DatetimeIndex(["2024-01-15", "2024-02-29", "2024-04-30"])
    out = ff.rf_on(idx, 12, rf)
    assert list(out.round(6)) == [0.001, 0.002, 0.002]
