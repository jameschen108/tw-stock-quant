"""trade.py 的虛擬下單：交易時段、預約單撮合、成交價來源、成本、漲跌停、現金與持股檢查、平均成本與已實現損益、配息（合成資料）。"""
from datetime import datetime

import pandas as pd
import pytest

import trade

OPEN_TIME = datetime(2026, 11, 5, 10, 0)      # 週四盤中
CLOSED_TIME = datetime(2026, 11, 7, 10, 0)    # 週六


def _q(price, up=False, down=False, phase=trade.LIVE):
    return {"price": price, "src": "測試", "name": "測試股", "phase": phase, "live": phase == trade.LIVE,
            "up": up, "down": down}


def _events(tmp_path, rows):
    pd.DataFrame(rows, columns=["date", "code", "name", "prev_close", "ref", "kind"]).assign(
        date=lambda d: pd.to_datetime(d["date"])).to_parquet(tmp_path / "price_events.parquet", index=False)


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(trade, "OUT", tmp_path)                             # 不讀真實的除權息資料
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
    ("buy", 1500, _q(10.0), "整股要是 1,000 股的倍數"),                    # 整股和零股要分開下
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


def test_dividend_only_if_held_before_ex_date(ledger, tmp_path):
    trade.place("1101", "buy", 1000, path=ledger, now=datetime(2026, 11, 5, 10, 0), q=_q(50.0))
    trade.place("1102", "buy", 1000, path=ledger, now=datetime(2026, 11, 9, 10, 0), q=_q(30.0))   # 除息當天才買
    _events(tmp_path, [("2026-11-09", "1101", "台泥", 50.0, 48.5, "div"), ("2026-11-09", "1102", "亞泥", 30.0, 29.0, "div"),
                       ("2026-11-04", "1101", "台泥", 51.0, 50.0, "div")])                      # 開帳戶前的不算
    rows = trade.load(ledger)
    d = trade.dividends(rows)
    assert d[["code", "shares", "amount"]].values.tolist() == [["1101", 1000, 1500.0]]
    st = trade.state(rows, divs=d)
    assert st["dividends"] == 1500.0
    assert st["cash"] == pytest.approx(100_000 - 50_071 - 30_043 + 1500)
    assert trade.state(rows)["cash"] == pytest.approx(100_000 - 50_071 - 30_043)    # 沒給 divs 就不算


def test_place_counts_dividend_cash(ledger, tmp_path):
    trade.place("1101", "buy", 1000, path=ledger, now=datetime(2026, 11, 5, 10, 0), q=_q(95.0))   # 剩 4,865
    _events(tmp_path, [("2026-11-09", "1101", "台泥", 95.0, 85.5, "div")])                       # 配 9,500
    trade.place("1102", "buy", 400, path=ledger, now=datetime(2026, 11, 10, 10, 0), q=_q(30.0))  # 12,017 要靠配息
    assert trade.state(trade.load(ledger))["pos"]["1102"]["shares"] == 400


def test_quote_in_session_without_counter_order_raises(tmp_path, monkeypatch):
    _prices(tmp_path, monkeypatch)
    live = {"1101": {"last": None, "indicative": True, "date": pd.Timestamp(OPEN_TIME.date()), "time": "10:00:00",
                     "name": "台泥", "limit_up": 22.0, "limit_down": 18.0, "bid": 22.0, "ask": None}}
    with pytest.raises(ValueError, match="沒有賣單"):                        # 漲停鎖死沒有賣單：不能退回用昨天收盤價
        trade.quote("1101", "buy", now=OPEN_TIME, live=live)


def test_load_skips_broken_line(ledger):
    with ledger.open("a", encoding="utf-8") as f:
        f.write('{"type": "trade", "acc')                                  # 寫到一半中斷
    assert trade.current(trade.load(ledger)) == "帳戶 1"


def test_max_shares_covers_fee():
    n = trade.max_shares(100_000, 50.0, "1101")
    amt = round(n * 50.0)
    assert amt + trade.costs("1101", "buy", amt)[0] <= 100_000
    amt = round((n + 1) * 50.0)
    assert amt + trade.costs("1101", "buy", amt)[0] > 100_000
    assert trade.max_shares(10, 50.0, "1101") == 0


def test_nav_marks_to_close_and_bench(tmp_path, monkeypatch):
    monkeypatch.setattr(trade, "OUT", tmp_path)
    days = pd.bdate_range("2026-11-02", periods=4)                       # 週一到週四
    px = pd.concat([pd.DataFrame({"date": days, "code": "1101", "close": [10.0, 11.0, 12.0, 9.0]}),
                    pd.DataFrame({"date": days, "code": trade.BENCH, "close": [100.0, 101.0, 102.0, 103.0]})])
    px.to_parquet(tmp_path / "prices_daily.parquet", index=False)
    led = tmp_path / "t.jsonl"
    trade.open_account(100_000, path=led, now=datetime(2026, 11, 2, 9, 30), bench=100.0)
    trade.place("1101", "buy", 1000, path=led, now=datetime(2026, 11, 2, 10, 0), q=_q(10.0))        # 10,000 + 20
    trade.place("1101", "sell", 400, path=led, now=datetime(2026, 11, 3, 18, 0), q=_q(11.0))        # 盤後，算當天
    n = trade.nav(trade.load(led))
    cash1 = 100_000 - 10_020
    cash2 = cash1 + 4_400 - 20 - 13
    assert list(n.index) == list(days)
    assert n["nav"].tolist() == pytest.approx([cash1 + 10_000, cash2 + 600 * 11, cash2 + 600 * 12, cash2 + 600 * 9])
    assert n["bench"].tolist() == pytest.approx([100_000, 101_000, 102_000, 103_000])
    _events(tmp_path, [(days[2], "1101", "台泥", 11.0, 10.5, "div"), (days[3], trade.BENCH, "元大台灣50", 102.0, 100.0, "div")])
    n = trade.nav(trade.load(led))
    assert n["nav"].tolist() == pytest.approx([cash1 + 10_000, cash2 + 600 * 11, cash2 + 300 + 600 * 12, cash2 + 300 + 600 * 9])
    assert n["bench"].iloc[-1] == pytest.approx(103_000 * 102 / 100)                 # 0050 含息


def _prices(tmp_path, monkeypatch, close=20.0, ret=0.01):
    monkeypatch.setattr(trade, "OUT", tmp_path)
    days = pd.bdate_range(pd.Timestamp.today().normalize() - pd.Timedelta(days=10), periods=3)
    pd.DataFrame({"date": days, "code": "1101", "name": "台泥", "close": [18.0, 19.0, close],
                  "ret": [0.0, 0.01, ret]}).to_parquet(tmp_path / "prices_daily.parquet", index=False)
    return days


def test_quote_on_weekend_is_reserve_with_reference_price(tmp_path, monkeypatch):
    days = _prices(tmp_path, monkeypatch, ret=0.099)
    q = trade.quote("1101", now=CLOSED_TIME, live={})
    assert q["phase"] == trade.RESERVE and q["price"] == 20.0 and not q["live"]
    assert q["src"] == f"參考價：{days[-1]:%Y-%m-%d} 收盤"
    assert q["up"] is False                                                   # 漲跌停到開盤撮合時才看


@pytest.mark.parametrize("hm,kind,traded,want", [
    ((10, 0), "整股", True, trade.LIVE), ((10, 0), "零股", True, trade.LIVE),
    ((8, 50), "整股", False, trade.RESERVE),                                 # 盤前
    ((13, 35), "整股", True, trade.RESERVE), ((13, 35), "零股", True, trade.RESERVE),
    ((13, 50), "零股", True, trade.AFTER_ODD), ((13, 50), "整股", True, trade.RESERVE),
    ((14, 10), "整股", True, trade.AFTER_FIXED), ((14, 10), "零股", True, trade.AFTER_ODD),
    ((14, 40), "整股", True, trade.RESERVE),
    ((10, 0), "整股", False, trade.RESERVE),                                 # 平日但證交所今天沒開（假日）
])
def test_phase_windows(hm, kind, traded, want):
    assert trade.phase(datetime(2026, 11, 5, *hm), kind, traded) == want
    assert trade.phase(datetime(2026, 11, 7, *hm), kind, True) == trade.RESERVE   # 週六


def test_after_hours_trades_use_today_close(tmp_path, monkeypatch):
    _prices(tmp_path, monkeypatch)
    live = {"1101": {"last": 21.0, "indicative": False, "date": pd.Timestamp(OPEN_TIME.date()), "time": "13:30:00",
                     "name": "台泥", "limit_up": 22.0, "limit_down": 18.0}}
    q = trade.quote("1101", now=OPEN_TIME.replace(hour=14, minute=10), live=live)
    assert (q["phase"], q["price"]) == (trade.AFTER_FIXED, 21.0)
    q = trade.quote("1101", now=OPEN_TIME.replace(hour=13, minute=50), live=live, kind="零股")
    assert (q["phase"], q["price"]) == (trade.AFTER_ODD, 21.0)
    assert trade.quote("1101", now=OPEN_TIME.replace(hour=13, minute=50), live=live)["phase"] == trade.RESERVE


def test_quote_uses_live_price_in_session(tmp_path, monkeypatch):
    _prices(tmp_path, monkeypatch)
    live = {"1101": {"last": 21.5, "indicative": False, "date": pd.Timestamp(OPEN_TIME.date()), "time": "10:00:00",
                     "name": "台泥", "limit_up": 22.0, "limit_down": 18.0, "bid": 21.45, "ask": 21.5}}
    q = trade.quote("1101", now=OPEN_TIME, live=live)
    assert q["price"] == 21.5 and q["live"] and not q["up"]
    live["1101"].update(last=21.4, indicative=True)                          # 還沒成交：買用賣價、賣用買價
    assert trade.quote("1101", "buy", now=OPEN_TIME, live=live)["price"] == 21.5
    assert trade.quote("1101", "sell", now=OPEN_TIME, live=live)["price"] == 21.45
    live["1101"]["date"] = pd.Timestamp("2026-11-04")                         # 報價不是今天（假日）→ 預約單，參考收盤價
    q = trade.quote("1101", now=OPEN_TIME, live=live)
    assert (q["phase"], q["price"]) == (trade.RESERVE, 20.0)


# ---------------------------------------------------------------- 預約單

SAT, MON = datetime(2026, 11, 7, 10, 0), datetime(2026, 11, 9, 10, 0)


def _open_prices(tmp_path, mon_open=None):
    """週五 11/06 收 50；mon_open 有給就加週一 11/09 的開盤。"""
    rows = [{"date": pd.Timestamp("2026-11-06"), "code": "1101", "open": 49.0, "close": 50.0}]
    if mon_open is not None:
        rows.append({"date": pd.Timestamp("2026-11-09"), "code": "1101", "open": mon_open, "close": 51.0})
    pd.DataFrame(rows).to_parquet(tmp_path / "prices_daily.parquet", index=False)


def test_reserve_order_holds_cash_then_fills_at_next_open(ledger, tmp_path):
    r = trade.place("1101", "buy", 1000, path=ledger, now=SAT, q=_q(50.0, phase=trade.RESERVE))
    assert r["type"] == "order" and r["id"] == "O1"
    st = trade.state(trade.load(ledger))
    assert st["reserved"] == 50_071 and st["pos"] == {} and st["cash"] == 100_000
    with pytest.raises(ValueError, match="現金不夠"):                         # 保留的現金不能再用
        trade.place("1102", "buy", 1000, path=ledger, now=SAT, q=_q(50.0))
    _open_prices(tmp_path)
    assert trade.settle(ledger, now=SAT.replace(day=8), live={}) == []        # 週日：還沒開盤
    _open_prices(tmp_path, mon_open=52.0)
    out = trade.settle(ledger, now=MON, live={})
    assert [(x["type"], x["price"], x["ts"]) for x in out] == [("trade", 52.0, "2026-11-09T09:00:00")]
    st = trade.state(trade.load(ledger))
    assert st["orders"] == {} and st["reserved"] == 0 and st["pos"]["1101"]["shares"] == 1000
    assert st["cash"] == pytest.approx(100_000 - 52_000 - 74)
    assert trade.settle(ledger, now=MON, live={}) == []                       # 不會重複撮合


def test_reserve_buy_cancelled_on_limit_up_open(ledger, tmp_path):
    trade.place("1101", "buy", 100, path=ledger, now=SAT, q=_q(50.0, phase=trade.RESERVE))
    _open_prices(tmp_path, mon_open=55.0)                                     # 開盤漲停（前收 50）
    out = trade.settle(ledger, now=MON, live={})
    assert out[0]["type"] == "cancel" and "漲停" in out[0]["reason"]
    st = trade.state(trade.load(ledger))
    assert st["reserved"] == 0 and st["cash"] == 100_000 and len(st["cancels"]) == 1


def test_reserve_fills_from_live_quote_before_data_update(ledger, tmp_path):
    trade.place("1101", "buy", 100, path=ledger, now=SAT, q=_q(50.0, phase=trade.RESERVE))
    _open_prices(tmp_path)                                                    # 資料表還沒有週一
    live = {"1101": {"open": 51.0, "date": pd.Timestamp(MON.date()), "limit_up": 55.0, "limit_down": 45.0}}
    out = trade.settle(ledger, now=MON.replace(hour=9, minute=5), live=live)
    assert out[0]["type"] == "trade" and out[0]["price"] == 51.0


def test_cancel_frees_reserve_and_sell_order_holds_shares(ledger):
    trade.place("1101", "buy", 1000, path=ledger, q=_q(10.0))
    trade.place("1101", "sell", 1000, path=ledger, now=SAT, q=_q(10.0, phase=trade.RESERVE))
    with pytest.raises(ValueError, match="持股不夠"):                         # 已掛預約賣單的股數不能再賣
        trade.place("1101", "sell", 1000, path=ledger, q=_q(10.0))
    trade.cancel("O1", ledger, now=SAT)
    assert trade.state(trade.load(ledger))["pending_sell"] == {}
    trade.place("1101", "sell", 1000, path=ledger, q=_q(10.0))
    with pytest.raises(ValueError, match="沒有還沒成交"):
        trade.cancel("O1", ledger)


def test_quote_after_close_prefers_newer_exchange_close(tmp_path, monkeypatch):
    days = _prices(tmp_path, monkeypatch)
    after = datetime.combine(days[-1].date() + pd.Timedelta(days=1), datetime.min.time()).replace(hour=15)
    live = {"1101": {"last": 21.0, "indicative": False, "date": pd.Timestamp(after.date()), "time": "13:30:00",
                     "name": "台泥", "limit_up": 22.0, "limit_down": 18.0}}
    q = trade.quote("1101", now=after, live=live)                             # 收盤後、晚上更新資料前
    assert q["price"] == 21.0 and not q["live"] and "證交所" in q["src"]
    live["1101"]["date"] = days[-1]                                           # 報價沒有比資料新 → 用資料
    assert trade.quote("1101", now=after, live=live)["price"] == 20.0
