"""strategy.py 的核心邏輯：設定檔、權重上限、報酬清理、模擬成交。"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import CODES, DAYS, ROOT, fixed_signals, make_prices, zero_cost_cfg

import strategy as S


@pytest.mark.parametrize("path", sorted((ROOT / "strategies").glob("*.toml")), ids=lambda p: p.stem)
def test_strategy_configs_load(path: Path):
    cfg = S.load_config(path)
    assert cfg["id"] == path.stem
    assert cfg["select"]["n"] > 0
    assert set(S.DEFAULTS) <= set(cfg)
    assert len(S.config_hash(cfg)) == 12


def test_target_weights_single_cap():
    pick = pd.DataFrame({"code": ["a", "b", "c", "d"], "vol_60d": [0.001, 0.02, 0.02, 0.02]})
    pos = {**S.DEFAULTS["position"], "weighting": "inv_vol", "max_weight": 0.3}
    w = S.target_weights(pick, pos, "vol_60d", True)
    assert w.max() <= 0.3 + 1e-9
    assert w.sum() == pytest.approx(1.0)


def test_target_weights_industry_cap():
    pick = pd.DataFrame({"code": list("abcde"), "industry": ["x", "x", "x", "y", "z"]})
    pos = {**S.DEFAULTS["position"], "max_industry": 0.4}
    w = S.target_weights(pick, pos, "code", True)
    assert w.groupby(pick.set_index("code")["industry"]).sum().max() <= 0.4 + 1e-9


def test_clean_returns_drops_suspect():
    px = make_prices()
    px.loc[(px["code"] == "1101") & (px["date"] == DAYS[5]), "ret_suspect"] = True
    rets = S.clean_returns(px)
    assert np.isnan(rets.at[DAYS[5], "1101"])
    assert rets.at[DAYS[5], "1102"] == pytest.approx(0.001)


def test_simulate_fills_next_day_and_compounds(market):
    rets = market()
    w = pd.Series(0.25, index=CODES[:4])
    res = S.simulate(zero_cost_cfg(), pd.DataFrame(), rets, fixed=fixed_signals({DAYS[10]: w}))
    assert res["trades"]["exe"].iloc[0] == DAYS[11]        # 訊號日隔一個交易日收盤成交
    held = len(DAYS) - 12                                  # 成交日之後的交易日數
    expect = 1.001 ** held
    assert res["nav"].iloc[-1] / 1_000_000 == pytest.approx(expect, rel=1e-3)


def test_simulate_returns_none_before_first_fill(market):
    rets = market()
    w = pd.Series(1.0, index=["1101"])
    assert S.simulate(zero_cost_cfg(), pd.DataFrame(), rets, fixed=fixed_signals({DAYS[-1]: w})) is None


def test_limit_up_blocks_buy(market):
    px = make_prices()
    up = (px["code"] == "1101") & (px["date"] >= DAYS[11])
    px.loc[up, "close"] = px.loc[up, "close"] * 1.1
    px.loc[(px["code"] == "1101") & (px["date"] == DAYS[11]), "ret"] = 0.1
    rets = market(px)
    w = pd.Series(0.5, index=["1101", "1102"])
    res = S.simulate(zero_cost_cfg(), pd.DataFrame(), rets, fixed=fixed_signals({DAYS[10]: w}))
    assert "1101 買不到" in res["trades"]["blocked"].iloc[0]
    assert "1101" not in res["hold"]


def test_costs_reduce_nav(market):
    rets = market()
    w = pd.Series(0.25, index=CODES[:4])
    sig = fixed_signals({DAYS[10]: w})
    free = S.simulate(zero_cost_cfg(), pd.DataFrame(), rets, fixed=sig)
    cfg = zero_cost_cfg()
    cfg["cost"] = dict(S.DEFAULTS["cost"])
    paid = S.simulate(cfg, pd.DataFrame(), rets, fixed=sig)
    assert paid["nav"].iloc[-1] < free["nav"].iloc[-1]
    assert paid["trades"]["fees"].iloc[0] > 0


def test_caps_hold_together_when_they_conflict():
    # 產業 x 有 4 檔、y 只有 1 檔：產業上限把 x 多出的權重分給 e，舊版會讓 e 超過單檔上限
    pick = pd.DataFrame({"code": list("abcde"), "industry": ["x", "x", "x", "x", "y"]})
    pos = {**S.DEFAULTS["position"], "max_weight": 0.3, "max_industry": 0.4}
    w = S.target_weights(pick, pos, "code", True)
    assert w.max() <= 0.3 + 1e-9
    assert w.groupby(pick.set_index("code")["industry"]).sum().max() <= 0.4 + 1e-9
    assert w.sum() <= 1.0 + 1e-9                           # 分不完的變現金


def test_lag_in_toml_is_used(tmp_path, market):
    base = (ROOT / "strategies" / "top30.toml").read_text(encoding="utf-8")
    plain, lagged = tmp_path / "plain.toml", tmp_path / "lagged.toml"
    plain.write_text(base, encoding="utf-8")
    lagged.write_text("lag = 2\n" + base, encoding="utf-8")
    cfg_plain, cfg_lag = S.load_config(plain), S.load_config(lagged)
    assert "lag" not in cfg_plain                          # 沒寫就不進設定，舊設定的雜湊不變
    assert cfg_lag["lag"] == 2
    assert S.config_hash(cfg_plain) != S.config_hash(cfg_lag)

    rets = market()
    cfg = zero_cost_cfg()
    cfg["lag"] = 2
    res = S.simulate(cfg, pd.DataFrame(), rets, fixed=fixed_signals({DAYS[10]: pd.Series(1.0, index=["1101"])}))
    assert res["trades"]["exe"].iloc[0] == DAYS[12]        # 訊號日後第 2 個交易日成交


def _delisted_market(market):
    """1101 在 DAYS[20] 之後就沒有行情（離資料尾端超過 DELIST_GAP 天）。"""
    px = make_prices()
    px = px[~((px["code"] == "1101") & (px["date"] > DAYS[20]))]
    return market(px)


def test_delisted_position_goes_to_cash(market):
    rets = _delisted_market(market)
    sig = fixed_signals({DAYS[10]: pd.Series(0.5, index=["1101", "1102"]),
                         DAYS[40]: pd.Series(1.0, index=["1102"])})
    res = S.simulate(zero_cost_cfg(), pd.DataFrame(), rets, fixed=sig)
    t = res["trades"]
    assert t["delisted"].iloc[1] == ["1101"]               # 第二次換股時轉成現金
    assert not any("1101" in b for b in t["blocked"].iloc[1])
    assert set(res["hold"]) == {"1102"}
    assert res["cash"] / res["nav"].iloc[-1] < 0.01        # 轉出來的現金有再投入


def test_delist_return_is_applied(market, monkeypatch):
    base = _delisted_market(market)
    monkeypatch.setattr(S, "DELIST_RET", -0.5)
    hit = _delisted_market(market)
    assert hit.at[DAYS[21], "1101"] == pytest.approx(-0.5)  # 最後一筆行情的隔一個交易日
    assert np.isnan(base.at[DAYS[21], "1101"])
    sig = fixed_signals({DAYS[10]: pd.Series(0.5, index=["1101", "1102"])})
    nav0 = S.simulate(zero_cost_cfg(), pd.DataFrame(), base, fixed=sig)["nav"].iloc[-1]
    nav1 = S.simulate(zero_cost_cfg(), pd.DataFrame(), hit, fixed=sig)["nav"].iloc[-1]
    assert nav1 < nav0 * 0.8


def test_simulate_with_no_prices_after_signal(market):
    # 模擬持股剛建立：訊號日在最後一筆行情之後，價格表是空的，不能出錯
    rets = market()
    sig = fixed_signals({DAYS[-1] + pd.Timedelta(days=3): pd.Series(1.0, index=["1101"])})
    assert S.simulate(zero_cost_cfg(), pd.DataFrame(), rets, fixed=sig) is None


def test_simulate_records_each_fill(market):
    rets = market()
    sig = fixed_signals({DAYS[10]: pd.Series(0.5, index=["1101", "1102"]),
                         DAYS[40]: pd.Series(1.0, index=["1102"])})
    f = S.simulate(zero_cost_cfg(), pd.DataFrame(), rets, fixed=sig)["fills"]
    assert list(f.columns) == S.FILL_COLS
    first = f[f["date"] == DAYS[11]]
    assert set(first["code"]) == {"1101", "1102"} and set(first["side"]) == {"買"}
    assert set(first["reason"]) == {"新買進"}
    second = f[f["date"] == DAYS[41]]
    sold = second[second["side"] == "賣"]
    assert list(sold["code"]) == ["1101"] and list(sold["reason"]) == ["換股出場"]
    assert (f["amount"] > 0).all()


def test_stop_loss_exit_is_recorded_as_fill(market):
    px = make_prices()
    px.loc[(px["code"] == "1101") & (px["date"] == DAYS[15]), "ret"] = -0.09   # 單日跌 9%（未達跌停），觸發停損
    rets = market(px)
    cfg = zero_cost_cfg()
    cfg["risk"] = {**cfg["risk"], "stop_loss": 0.05, "stop_type": "fixed"}
    res = S.simulate(cfg, pd.DataFrame(), rets, fixed=fixed_signals({DAYS[10]: pd.Series(1.0, index=["1101"])}))
    f = res["fills"]
    stop = f[f["reason"] == "停損"]
    assert len(stop) == 1 and stop["code"].iloc[0] == "1101" and stop["side"].iloc[0] == "賣"
