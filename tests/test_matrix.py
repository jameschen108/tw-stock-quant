"""matrix.py：四維判定規則、一致數與矛盾清單。"""
import numpy as np
import pandas as pd

import matrix as M


def test_dimension_rules():
    d = pd.DataFrame({
        "short_label": ["看漲", "看跌", "中性", None],
        "mid_label": ["看漲", "看跌", "看漲", None],
        "long_label": ["中性", "看漲", "看跌", None],
        "inst20_pct": [2.0, -2.5, 1.9, np.nan],
        "rev_yoy_3m": [10.0, -5.0, 3.0, np.nan],
        "gm_chg": [0.5, -1.0, -0.2, 1.0],
    })
    dm = M.dimensions(d)
    assert dm.iloc[0].tolist() == [1, 1, 1, 1]
    assert dm.iloc[1].tolist() == [-1, -1, -1, 0]          # 中期看跌 + 長期看漲 → 綜合中性
    assert dm.iloc[2].tolist() == [0, 0, 0, 0]             # 籌碼差一點沒到門檻、營收毛利一正一負
    assert dm.iloc[3].isna().all()
    assert M.net(dm, M.DIMS).tolist()[:3] == [4, -3, 0] and np.isnan(M.net(dm, M.DIMS).iloc[3])


def facts(short="看漲", foreign=-3.0, trust=0.5, streak=("賣超", 7), margin=(11000, 1000)):
    hz = {"短期": {"label": short}, "中期": {"label": "看漲"}, "長期": {"label": "中性"}}
    return {"asof": "2026-10-22", "outlook": {"horizons": hz},
            "institutional_flows": {"20d": {"foreign": {"pct_of_volume": foreign}, "trust": {"pct_of_volume": trust}},
                                    "foreign_streak": {"direction": streak[0], "days": streak[1]}},
            "revenue": {"yoy_3m_pct": 12.0}, "financials": {"yoy_same_quarter": {"gross_margin_chg_pt": 1.5}},
            "margin_foreign_holding": {"margin": {"margin_balance": margin[0], "margin_chg_20d": margin[1]}}}


def test_stock_matrix_contradictions():
    pats = {"date": "2026-10-22", "patterns": [
        {"name": "M 頭", "side": "看跌形態", "formed": "2026-10-20", "reliability": "有統計支持"},
        {"name": "向下跳空", "side": "看跌形態", "formed": "2026-10-21", "reliability": "弱"}]}
    m = M.stock_matrix(facts(), pats)
    assert {k: v["direction"] for k, v in m["dimensions"].items()} == {"技術": "多", "籌碼": "空", "基本面": "多", "綜合": "多"}
    assert m["agreement"]["net"] == 2
    con = m["contradictions"]
    assert sum("但籌碼偏空" in c for c in con) == 3              # 技術、基本面、綜合各和籌碼相反
    assert any("外資連 7 日賣超" in c for c in con)
    assert any("融資 20 日增加 10%" in c for c in con)          # 1000 / (11000 − 1000)
    assert any("M 頭" in c for c in con) and not any("向下跳空" in c for c in con)


def test_stock_matrix_quiet_when_aligned():
    m = M.stock_matrix(facts(foreign=3.0, streak=("買超", 7), margin=(10000, -500)),
                       {"date": "2026-10-01", "patterns": [
                           {"name": "M 頭", "side": "看跌形態", "formed": "2026-09-30", "reliability": "有統計支持"}]})
    assert m["agreement"]["net"] == 4
    assert m["contradictions"] == []                           # 形態不是 asof 當天的，不列
