"""Benjamini-Hochberg 校正，以及形態可靠度要過 FDR 才算有統計支持。"""
import numpy as np
import pytest

import patterns as P
import technicals as tc


def test_bh_q_matches_hand_calculation():
    q = tc.bh_q([0.01, 0.04, 0.03, 0.20])
    assert q == pytest.approx([0.04, 0.0533333, 0.0533333, 0.20], rel=1e-5)


def test_bh_q_ignores_missing_tests():
    q = tc.bh_q([0.01, np.nan, 0.04])
    assert q[0] == pytest.approx(0.02) and np.isnan(q[1]) and q[2] == pytest.approx(0.04)


def test_bh_q_capped_at_one_and_handles_empty():
    assert tc.bh_q([0.9, 0.95]).max() <= 1
    assert len(tc.bh_q([])) == 0


def test_t_to_p_two_sided():
    assert tc.t_to_p(1.96) == pytest.approx(0.05, abs=1e-3)
    assert tc.t_to_p(-1.96) == tc.t_to_p(1.96)
    assert np.isnan(tc.t_to_p(None))


def test_pure_noise_rarely_passes_while_raw_p_does():
    p = np.random.default_rng(0).uniform(size=100)
    assert (p < 0.05).sum() >= 1 and (tc.bh_q(p) < tc.FDR_Q).sum() == 0


def cell(t):
    return {"t": t, "mean_ex_pct": 1.0}


def test_add_fdr_family_skips_neutral_patterns():
    hz = lambda t: {10: {p: cell(t) for p in P.PERIODS}}
    out = {"a": {"side": 1, "horizons": hz(3.5)}, "b": {"side": -1, "horizons": hz(0.1)},
           "c": {"side": 0, "horizons": hz(9.0)}}
    P.add_fdr(out)
    assert out["a"]["horizons"][10][P.OOS]["q"] == pytest.approx(tc.t_to_p(3.5) * 2, abs=1e-3)
    assert "q" not in out["c"]["horizons"][10][P.OOS]


@pytest.mark.parametrize("q, label", [(0.05, "有統計支持"), (0.20, "弱"), (None, "弱")])
def test_reliability_requires_fdr(q, label):
    res = {P.OOS: {"mean_ex_pct": 1.0, "t": 2.5, "q": q}, P.IS: {"mean_ex_pct": 0.5}}
    assert P.reliability(1, res) == label
