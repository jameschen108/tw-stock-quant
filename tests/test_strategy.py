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
