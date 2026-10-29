"""surprise.py 的驚喜分數、分組、持有期，以及 mine.py 的事件組合與建立前回測（合成資料）。"""
import numpy as np
import pandas as pd
import pytest
from conftest import DAYS, make_prices

import custom
import mine
import strategy as S
import surprise as SU


@pytest.fixture(autouse=True)
def _clear_cache():
    SU.market.cache_clear()
    SU.events.cache_clear()
    yield
    SU.market.cache_clear()
    SU.events.cache_clear()


def test_sur_matches_formula(tmp_path, monkeypatch):
    monkeypatch.setattr(SU, "OUT", tmp_path)
    per = pd.date_range("2020-01-31", periods=40, freq="ME")
    rev = 100 + np.arange(40) * 2.0 + np.r_[np.zeros(39), 50.0]        # 穩定成長，最後一個月多 50
    rev[30] += 7                                                         # 讓過去的差額有變化（標準差不是 0）
    pd.DataFrame({"code": "1101", "period": per, "avail_date": per + pd.Timedelta(days=10), "avail_imputed": False,
                  "以合併為主單月營收(千元)": rev}).to_parquet(tmp_path / "tej_rev.parquet", index=False)
    sc = SU.scores("rev").set_index("period")
    d = pd.Series(rev - np.r_[np.full(12, np.nan), rev[:-12]], index=per)
    past = d.iloc[-25:-1]
    assert sc.loc[per[-1], "x"] == pytest.approx(min((d.iloc[-1] - past.mean()) / past.std(), 10))
    assert sc.index.min() >= per[12 + 12]                               # 差額至少要 12 個月才算分數


def test_imputed_dates_are_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(SU, "OUT", tmp_path)
    per = pd.date_range("2020-01-31", periods=40, freq="ME")
    pd.DataFrame({"code": "1101", "period": per, "avail_date": per + pd.Timedelta(days=10), "avail_imputed": True,
                  "以合併為主單月營收(千元)": np.arange(40.0) ** 1.5}).to_parquet(tmp_path / "tej_rev.parquet", index=False)
    assert SU.scores("rev").empty


def _fake_events(monkeypatch, ev: pd.DataFrame, days=DAYS):
    monkeypatch.setattr(SU, "market", lambda: (days, None))
    monkeypatch.setattr(SU, "events", lambda kind: ev)


def test_active_window_is_hold_days(monkeypatch):
    ev = pd.DataFrame({"code": ["1101", "1102", "1103"], "sig_i": [5, 5, 8], "dec": [10, 1, 10], "x": [3.0, -3.0, 5.0]})
    _fake_events(monkeypatch, ev)
    h = SU.KINDS["rev"]["hold"]
    assert SU.active_codes("rev", DAYS[4]) == []
    assert SU.active_codes("rev", DAYS[8]) == ["1103", "1101"]          # 分數高的在前
    assert SU.active_codes("rev", DAYS[5 + h - 1]) == ["1103", "1101"]
    assert SU.active_codes("rev", DAYS[5 + h]) == ["1103"]              # 1101 持有期滿
    assert SU.active("rev", DAYS[6], bad=True)["code"].tolist() == ["1102"]


def test_schedule_keeps_only_changes(monkeypatch):
    ev = pd.DataFrame({"code": ["1101", "1103"], "sig_i": [5, 8], "dec": [10, 10], "x": [3.0, 5.0]})
    _fake_events(monkeypatch, ev)
    h = SU.KINDS["rev"]["hold"]
    sch = SU.schedule("rev", DAYS[0], DAYS[-1])
    assert list(sch) == [DAYS[5], DAYS[8], DAYS[5 + h], DAYS[8 + h]]
    assert sch[DAYS[8 + h]] == []                                        # 名單空了也要記，才會賣掉


def test_deciles_use_only_earlier_months(monkeypatch):
    days = pd.bdate_range("2024-01-01", "2025-12-31")
    ok = pd.DataFrame(True, index=days, columns=[f"{1000 + i}" for i in range(100)])
    monkeypatch.setattr(SU, "market", lambda: (days, ok))
    rng = np.random.default_rng(0)
    rows = [{"code": f"{1000 + i}", "period": p, "pub": p + pd.Timedelta(days=40), "value": 1.0, "x": rng.normal()}
            for p in pd.date_range("2024-01-31", periods=20, freq="ME") for i in range(100)]
    sc = pd.DataFrame(rows)
    late = sc["pub"] == sc["pub"].max()
    sc.loc[late, "x"] = 100.0                                            # 最後一個月全部很高：門檻用前 12 個月，所以都在最高組
    monkeypatch.setattr(SU, "scores", lambda kind: sc)
    monkeypatch.setitem(SU.KINDS["rev"], "min_n", 1000)
    ev = SU.events("rev")
    assert (ev.loc[ev["pub"] == sc["pub"].max(), "dec"] == 10).all()
    assert ev["pub"].min() >= pd.Timestamp("2024-11-01")                 # 前 12 個月不到 1000 筆的月份不分組


@pytest.fixture
def mine_market(tmp_path, monkeypatch, market):
    """mine / custom 也讀合成行情，帳本寫到暫存資料夾。"""
    for mod in (custom, mine):
        monkeypatch.setattr(mod, "OUT", tmp_path)
    monkeypatch.setattr(mine, "PAPER", tmp_path / "paper")
    monkeypatch.setattr(mine, "LEDGER", tmp_path / "paper" / "mine_ledger.parquet")
    return market


def test_empty_event_signal_round_trips(mine_market):
    mine_market()
    mine._append("x", pd.Series({"1101": 1.0}), DAYS[3])
    mine._append("x", pd.Series(dtype=float), DAYS[6])
    sigs = mine._latest_signals(mine.load_ledger(), "x")
    assert list(sigs[DAYS[3]].index) == ["1101"] and sigs[DAYS[6]].empty


def test_events_sell_out_when_list_empties(mine_market):
    mine_market()
    sigs = {DAYS[3]: pd.Series({"1101": 0.5, "1102": 0.5}), DAYS[10]: pd.Series(dtype=float)}
    rets = S.clean_returns(make_prices())
    res = mine._run("x", sigs, 1_000_000, {}, "events", rets)
    assert res["hold"] == {}
    assert res["cash"] == pytest.approx(res["nav"].iloc[-1])


def test_events_band_skips_small_rebalances(mine_market):
    mine_market()
    a = pd.Series({"1101": 0.5, "1102": 0.5})
    sigs = {DAYS[3]: a, DAYS[10]: pd.Series({"1101": 0.45, "1102": 0.45, "1103": 0.1})}
    rets = S.clean_returns(make_prices())
    t = mine._run("x", sigs, 1_000_000, {}, "events", rets)["trades"]
    tn = mine._run("x", sigs, 1_000_000, {}, "none", rets)["trades"]
    assert t["tax"].iloc[1] == 0                                         # 事件組合：只差一點的不賣
    assert tn["tax"].iloc[1] > 0                                         # 一般組合照目標權重賣


def test_backtest_monthly_and_base(mine_market):
    days = pd.bdate_range("2024-11-01", periods=len(DAYS))              # 跨三個月底
    px = pd.concat([g.assign(date=days) for _, g in make_prices().groupby("code")])
    mine_market(px)
    w = pd.Series({"1101": 1.0, "1102": 1.0})
    res = mine.backtest(w, days[0], monthly=True, weighting="equal")
    assert res["rebalance"] == "monthly" and "base" not in res
    assert list(res["signals"])[1:] == [d for d in pd.Series(days, index=days).groupby(days.to_period("M")).max()
                                        if d > days[0]]
    res = mine.backtest(w, days[1], risk={"stop_loss": 0.1})            # 第一天還沒有報酬，算買不到
    assert res["base"]["perf"]["strategy"]["total"] == pytest.approx(res["perf"]["strategy"]["total"])   # 沒觸發停損就一樣
    with pytest.raises(ValueError):
        mine.backtest(None, days[0])
