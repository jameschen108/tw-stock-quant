"""event_study.py：持有期重疊時的 t 值用 Newey-West，不把相鄰月份當成獨立樣本。"""
import numpy as np
import pandas as pd
import pytest

import event_study as E
from technicals import nw_t


def monthly_events(n_months: int, seed: int = 0, h: int = 60) -> pd.DataFrame:
    """每月一筆事件；報酬是連續 3 個月的移動平均，模擬 60 日持有期的重疊。"""
    rng = np.random.default_rng(seed)
    shock = rng.normal(0.02, 0.05, n_months + 2)
    r = np.convolve(shock, np.ones(3) / 3, mode="valid")
    return pd.DataFrame({"entry": pd.date_range("2013-01-15", periods=n_months, freq="MS"),
                         "ex5": r, "ex20": r, f"ex{h}": r})


@pytest.mark.parametrize("h, lag", [(5, 2), (20, 2), (21, 2), (22, 3), (60, 4)])
def test_overlap_lag(h, lag):
    assert E.overlap_lag(h) == lag


def test_t_matches_newey_west_on_monthly_means():
    ev = monthly_events(120)
    row = next(r for r in E.summarize(ev, "x", "全期") if r["持有"] == 60)
    by_m = ev.set_index(ev.entry.dt.to_period("M"))["ex60"]
    assert row["t"] == pytest.approx(nw_t(by_m, 4))


def test_overlapping_returns_lower_t_than_treating_months_as_independent():
    ev = monthly_events(120)
    row = next(r for r in E.summarize(ev, "x", "全期") if r["持有"] == 60)
    naive = ev.ex60.mean() / (ev.ex60.std(ddof=1) / np.sqrt(len(ev)))
    assert 0 < row["t"] < naive


def test_several_events_in_a_month_count_as_one_observation():
    one = monthly_events(120)
    two = pd.concat([one, one], ignore_index=True)
    t = lambda d: next(r for r in E.summarize(d, "x", "全期") if r["持有"] == 60)["t"]
    assert t(two) == pytest.approx(t(one))


def test_too_few_months_gives_no_t():
    row = next(r for r in E.summarize(monthly_events(40).iloc[:12].pipe(lambda d: pd.concat([d] * 3)), "x", "2020")
               if r["持有"] == 60)
    assert row["事件數"] == 36 and np.isnan(row["t"])
