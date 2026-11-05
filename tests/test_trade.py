"""trade.py 的虛擬下單：成交價來源、成本、漲跌停、現金與持股檢查、平均成本與已實現損益（合成資料）。"""
from datetime import datetime

import pandas as pd
import pytest

import trade

OPEN_TIME = datetime(2026, 11, 5, 10, 0)      # 週四盤中
CLOSED_TIME = datetime(2026, 11, 7, 10, 0)    # 週六


def _q(price, up=False, down=False):
    return {"price": price, "src": "測試", "name": "測試股", "live": False, "up": up, "down": down}


@pytest.fixture
def ledger(tmp_path):
    p = tmp_path / "my_trades.jsonl"
    trade.open_account(100_000, path=p, now=OPEN_TIME, bench=100.0)
    return p


def test_costs_min_fee_and_etf_tax():
    assert trade.costs("2330", "buy", 1000) == (20.0, 0.0)                  # 最低 20 元
    assert trade.costs("2330", "buy", 1_000_000) == (1425.0, 0.0)
    assert trade.costs("2330", "sell", 1_000_000) == (1425.0, 3000.0)
    assert trade.costs("0050", "sell", 1_000_000) == (1425.0, 1000.0)      # ETF 證交稅 0.1%


def test_buy_sell_cash_avg_cost_and_realized(ledger):
    trade.place("1101", "buy", 1000, path=ledger, q=_q(50.0))               # 50,000 + 71
    trade.place("1101", "buy", 1000, path=ledger, q=_q(40.0))               # 40,000 + 57
    st = trade.state(trade.load(ledger))
    assert st["cash"] == pytest.approx(100_000 - 50_071 - 40_057)
    assert st["pos"]["1101"]["shares"] == 2000
    assert st["pos"]["1101"]["cost"] == pytest.approx(90_128)
    trade.place("1101", "sell", 500, path=ledger, q=_q(60.0))               # 30,000 − 43 − 90
    st = trade.state(trade.load(ledger))
    assert st["realized"] == pytest.approx(30_000 - 43 - 90 - 90_128 / 2000 * 500)
    assert st["pos"]["1101"]["shares"] == 1500
    assert st["cash"] == pytest.approx(100_000 - 50_071 - 40_057 + 30_000 - 43 - 90)
    h = trade.holdings(st, {"1101": 55.0})
    assert h.loc[0, "pnl"] == pytest.approx((55.0 - 90_128 / 2000) * 1500)


def test_sell_all_removes_position(ledger):
    trade.place("1101", "buy", 300, path=ledger, q=_q(10.0))
    trade.place("1101", "sell", 300, path=ledger, q=_q(11.0))
    assert trade.state(trade.load(ledger))["pos"] == {}


@pytest.mark.parametrize("side,shares,q,msg", [
    ("buy", 3000, _q(50.0), "現金不夠"),
    ("sell", 1, _q(50.0), "持股不夠"),
    ("buy", 0, _q(50.0), "正整數"),
    ("buy", 100, _q(50.0, up=True), "漲停"),
])
def test_rejected_orders_write_nothing(ledger, side, shares, q, msg):
    before = len(trade.load(ledger))
    with pytest.raises(ValueError, match=msg):
        trade.place("1101", side, shares, path=ledger, q=q)
    assert len(trade.load(ledger)) == before


def test_limit_down_blocks_sell_only(ledger):
    trade.place("1101", "buy", 100, path=ledger, q=_q(10.0))
    with pytest.raises(ValueError, match="跌停"):
        trade.place("1101", "sell", 100, path=ledger, q=_q(9.0, down=True))
    trade.place("1101", "buy", 100, path=ledger, q=_q(9.0, down=True))     # 跌停可以買


def test_new_account_starts_fresh_and_keeps_old(ledger):
    trade.place("1101", "buy", 100, path=ledger, q=_q(10.0))
    trade.open_account(50_000, path=ledger, now=OPEN_TIME, bench=100.0)
    rows = trade.load(ledger)
    assert trade.current(rows) == "帳戶 2"
    assert trade.state(rows)["cash"] == 50_000 and trade.state(rows)["pos"] == {}
    assert trade.state(rows, "帳戶 1")["pos"]["1101"]["shares"] == 100       # 舊帳戶還在


def _prices(tmp_path, monkeypatch, close=20.0, ret=0.01):
    monkeypatch.setattr(trade, "OUT", tmp_path)
    days = pd.bdate_range(pd.Timestamp.today().normalize() - pd.Timedelta(days=10), periods=3)
    pd.DataFrame({"date": days, "code": "1101", "name": "台泥", "close": [18.0, 19.0, close],
                  "ret": [0.0, 0.01, ret]}).to_parquet(tmp_path / "prices_daily.parquet", index=False)
    return days


def test_quote_uses_close_after_hours(tmp_path, monkeypatch):
    days = _prices(tmp_path, monkeypatch)
    q = trade.quote("1101", now=CLOSED_TIME, live={})
    assert q["price"] == 20.0 and not q["live"] and q["src"] == f"收盤價 {days[-1]:%Y-%m-%d}"
    assert trade.quote("1101", now=CLOSED_TIME, live={})["up"] is False
    _prices(tmp_path, monkeypatch, ret=0.099)
    assert trade.quote("1101", now=CLOSED_TIME, live={})["up"] is True       # 收盤漲停


def test_quote_uses_live_price_in_session(tmp_path, monkeypatch):
    _prices(tmp_path, monkeypatch)
    live = {"1101": {"last": 21.5, "indicative": False, "date": pd.Timestamp(OPEN_TIME.date()), "time": "10:00:00",
                     "name": "台泥", "limit_up": 22.0, "limit_down": 18.0, "bid": 21.45, "ask": 21.5}}
    q = trade.quote("1101", now=OPEN_TIME, live=live)
    assert q["price"] == 21.5 and q["live"] and not q["up"]
    live["1101"].update(last=21.4, indicative=True)                          # 還沒成交：買用賣價、賣用買價
    assert trade.quote("1101", "buy", now=OPEN_TIME, live=live)["price"] == 21.5
    assert trade.quote("1101", "sell", now=OPEN_TIME, live=live)["price"] == 21.45
    live["1101"]["date"] = pd.Timestamp("2026-11-04")                         # 報價不是今天 → 收盤價
    assert trade.quote("1101", now=OPEN_TIME, live=live)["price"] == 20.0
