"""restricted.py：處置、注意、全額交割股的解析與進場排除，以及接到 strategy / backtest / composite 的地方。"""
import pandas as pd
import pytest
from conftest import DAYS, zero_cost_cfg

import backtest as B
import composite as C
import restricted as R
import strategy as S

T = pd.Timestamp


def ev(*rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=R.COLS).astype({c: "datetime64[us]" for c in R.COLS[3:]})


def keys(date, codes) -> pd.DataFrame:
    return pd.DataFrame({"date": T(date), "code": codes})


# ---------------------------------------------------------------- 解析

def test_roc_dates_both_formats():
    assert R.roc_dates("101/01/16～101/01/31") == [T("2012-01-16"), T("2012-01-31")]
    assert R.roc_dates("110.01.21") == [T("2021-01-21")]
    assert R.roc_dates("99/1/4~99/1/8") == [T("2010-01-04"), T("2010-01-08")]
    assert R.roc_dates("") == [] and R.roc_dates(None) == []


def test_parse_twse_disposal():
    js = {"fields": ["編號", "公布日期", "證券代號", "證券名稱", "累計", "處置條件", "處置起迄時間"],
          "data": [[1, "101/01/13", "2314", "台揚", 1, "連續五次", "101/01/16～101/01/31"]]}
    df = R.parse_disposal(js, "TWSE")
    assert df.iloc[0].to_dict() == {"code": "2314", "market": "TWSE", "kind": "disposal",
                                    "announce_date": T("2012-01-13"), "start": T("2012-01-16"), "end": T("2012-01-31")}


def test_parse_tpex_disposal_drops_empty_rows():
    js = {"tables": [{"fields": ["編號", "公布日期", "證券代號", "證券名稱", "累計", "處置起訖時間"],
                      "data": [[1, "101/01/31", "", "", "", ""],
                               [2, "110/03/30", "3552", "同致(link)", 4, "110/03/31~110/04/15"]]}]}
    df = R.parse_disposal(js, "TPEx")
    assert df["code"].tolist() == ["3552"]
    assert df["end"].iloc[0] == T("2021-04-15")


def test_parse_attention_both_markets():
    tw = {"fields": ["編號", "證券代號", "證券名稱", "累計次數", "注意交易資訊", "日期"],
          "data": [[1, "3406", "玉晶光", "1", "週轉率", "101.01.03"]]}
    tp = {"tables": [{"fields": ["編號", "證券代號", "證券名稱", "累計", "注意交易資訊", "公告日期"],
                      "data": [[1, "", "", 1, "本日無公布注意交易資訊", "101/01/05"],
                               [2, "8082", "眾星", 1, "跌幅", "101/01/04"]]}]}
    assert R.parse_attention(tw, "TWSE")["announce_date"].tolist() == [T("2012-01-03")]
    assert R.parse_attention(tp, "TPEx")["code"].tolist() == ["8082"]


def test_parse_cmode():
    tw = R.parse_twse_cmode({"date": "20150105", "data": [["1529", "樂士", "**"]]})
    assert tw["start"].iloc[0] == T("2015-01-05") and tw["kind"].iloc[0] == "full_delivery"
    assert R.parse_twse_cmode({"date": "20110131", "data": []}).empty          # 空名單當缺資料
    tp = R.parse_tpex_cmode([{"Date": "1151030", "SecuritiesCompanyCode": "2067", "AlteredTrading": "Ｙ"},
                             {"Date": "1151030", "SecuritiesCompanyCode": "9999", "AlteredTrading": ""}])
    assert tp["code"].tolist() == ["2067"] and tp["start"].iloc[0] == T("2026-10-30")


# ---------------------------------------------------------------- 進場判斷

DAYS_IDX = pd.bdate_range("2025-01-01", "2025-03-31")


def test_disposal_announced_on_signal_day_effective_next_day():
    e = ev(("A", "TWSE", "disposal", T("2025-01-31"), T("2025-02-03"), T("2025-02-14")))
    assert R.flags(keys("2025-01-31", ["A", "B"]), DAYS_IDX, ev=e).tolist() == [True, False]


def test_disposal_ending_before_trade_day_not_flagged():
    e = ev(("A", "TWSE", "disposal", T("2025-01-15"), T("2025-01-16"), T("2025-01-31")))
    assert not R.flags(keys("2025-01-31", ["A"]), DAYS_IDX, ev=e).any()          # 成交日 2/3 已結束


def test_disposal_announced_after_signal_not_used():
    e = ev(("A", "TWSE", "disposal", T("2025-02-03"), T("2025-02-03"), T("2025-02-14")))
    assert not R.flags(keys("2025-01-31", ["A"]), DAYS_IDX, ev=e).any()          # 訊號日還不知道


def test_attention_only_signal_day():
    e = ev(("A", "TWSE", "attention", T("2025-01-31"), T("2025-01-31"), T("2025-01-31")),
           ("B", "TWSE", "attention", T("2025-01-30"), T("2025-01-30"), T("2025-01-30")))
    assert R.flags(keys("2025-01-31", ["A", "B"]), DAYS_IDX, ev=e).tolist() == [True, False]


def test_full_delivery_uses_latest_snapshot_per_market():
    e = ev(("A", "TWSE", "full_delivery", T("2024-12-31"), T("2024-12-31"), T("2024-12-31")),
           ("B", "TWSE", "full_delivery", T("2025-01-31"), T("2025-01-31"), T("2025-01-31")),
           ("C", "TPEx", "full_delivery", T("2024-12-31"), T("2024-12-31"), T("2024-12-31")))
    # 上市取 1/31 快照（A 已恢復、B 列入）；上櫃最近的 12/31 快照在 45 天內，C 仍列入
    assert R.flags(keys("2025-01-31", ["A", "B", "C"]), DAYS_IDX, ev=e).tolist() == [False, True, True]
    # 快照太舊就不用
    assert not R.flags(keys("2025-03-31", ["C"]), DAYS_IDX, ev=e).any()


def test_latest_signal_without_next_trading_day():
    e = ev(("A", "TWSE", "disposal", T("2025-03-31"), T("2025-04-01"), T("2025-04-14")))
    assert R.flags(keys("2025-03-31", ["A"]), DAYS_IDX, ev=e).all()


def test_missing_file_means_no_exclusion(tmp_path, capsys):
    out = R.flags(keys("2025-01-31", ["A"]), DAYS_IDX, ev=R.load(tmp_path / "none.parquet"))
    assert not out.any() and "警告" in capsys.readouterr().out


# ---------------------------------------------------------------- 接到選股

def _day(restricted: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({"date": DAYS[0], "code": ["1101", "1102", "1103"], "avg_value": 1e8,
                         "comp_equal": [3.0, 2.0, 1.0], "restricted": restricted})


def test_select_excludes_new_restricted_but_keeps_pool():
    cfg = zero_cost_cfg()
    cfg["select"] = {**cfg["select"], "n": 2}
    pool, pick = S.select(_day([True, False, False]), cfg)
    assert pick["code"].tolist() == ["1102", "1103"]
    assert len(pool) == 3


def test_select_keeps_held_restricted():
    cfg = zero_cost_cfg()
    cfg["select"] = {**cfg["select"], "n": 2}
    _, pick = S.select(_day([True, False, False]), cfg, held={"1101": 1.0})
    assert pick["code"].tolist() == ["1101", "1102"]


def test_simulate_does_not_buy_restricted(market):
    rets = market()
    s1, s2 = DAYS[10], DAYS[30]
    panel = pd.concat([_day([False, False, False]).assign(date=s1),
                       _day([True, False, True]).assign(date=s2, comp_equal=[2.0, 1.0, 3.0])])
    cfg = zero_cost_cfg()
    cfg["select"] = {**cfg["select"], "n": 2}
    res = S.simulate(cfg, panel, rets)
    # 第二期分數最高的 1103 被擋不買；1101 被擋但已持有，續抱不賣
    assert "1103" not in set(res["fills"]["code"])
    assert set(res["hold"]) == {"1101", "1102"}
    assert not (res["fills"]["side"] == "賣").any()


def test_backtest_universe_and_evaluate(monkeypatch):
    df = pd.DataFrame({"date": T("2025-01-31"), "code": ["A", "B"], "close": 10.0, "avg_value": 1e8, "n_obs": 200})
    monkeypatch.setattr(B, "trading_days", lambda: DAYS_IDX)
    monkeypatch.setattr(R, "load", lambda: ev(("A", "TWSE", "attention", T("2025-01-31"), T("2025-01-31"),
                                                    T("2025-01-31"))))
    assert B.universe(df, 5, 5e6, 120)["code"].tolist() == ["B"]
    kept = B.universe(df, 5, 5e6, 120, keep_restricted=True)
    assert kept["restricted"].tolist() == [True, False]


def test_composite_top_n_holds_but_does_not_buy_restricted():
    d1, d2 = T("2025-01-31"), T("2025-02-28")
    d = pd.DataFrame({"date": [d1, d1, d1, d2, d2, d2], "code": ["A", "B", "C"] * 2, "avg_value": 1e8,
                      "fwd_ret": 0.01, "score": [3, 2, 1, 3, 2, 1],
                      "restricted": [True, False, False, True, True, False]})
    top = C.top_n(d, 1, 0)
    # 第一期 A 被擋，買 B；第二期 B 被擋但已持有，續抱；A 仍不能新買
    assert top["turn"].tolist() == pytest.approx([1.0, 0.0])
