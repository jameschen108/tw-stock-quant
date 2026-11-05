"""模擬持股頁的「虛擬下單」區塊：使用者用虛擬資金自己買賣，成交價用現在的股價。規則與帳本在 trade.py。"""
from datetime import date

import pandas as pd
import streamlit as st

import common
import trade

SIDES = {"買進": "buy", "賣出": "sell"}
UNITS = {"張（1,000 股）": 1000, "股（零股）": 1}
STALE_DAYS = 4       # 收盤價比今天舊超過這幾天就提醒資料可能沒更新


def _labels(sig: str) -> tuple[list[str], dict[str, str]]:
    nm = common.names_frame(sig).reset_index()
    nm = nm[nm["active"]].sort_values("code")
    labels = (nm["code"].astype(str) + " " + nm["name"].fillna("").astype(str)).tolist()
    return labels, {lab.split()[0]: lab for lab in labels}


def _prices(codes: list[str]) -> dict[str, dict]:
    """持股現價：盤中用即時報價，其他時候用最後收盤價（收盤後到晚上更新資料前用證交所報價的收盤價）。"""
    live = common.live_quotes(tuple(sorted(codes))) if codes else {}
    out = {}
    for c in codes:
        try:
            out[c] = trade.quote(c, "sell", live=live)
        except ValueError:
            pass
    return out


def _open_form(first: bool) -> None:
    c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
    cap = c1.number_input("虛擬資金（元）", min_value=10_000, max_value=100_000_000, value=trade.DEFAULT_CAPITAL,
                          step=100_000, key="trade_capital")
    if c2.button("開帳戶" if first else "開新帳戶（舊帳戶保留）", type="primary" if first else "secondary"):
        r = trade.open_account(cap)
        st.toast(f"已開「{r['account']}」，資金 {r['capital']:,.0f} 元", icon=":material/check:")
        st.rerun()


def _sell_from(code: str, shares: int, sig: str) -> None:
    """點持股列：下單區切到賣出這檔、零股、數量填全部持股。"""
    _, by_code = _labels(sig)
    st.session_state.update(trade_side="賣出", _trade_side_last="賣出", trade_pick_sell=by_code.get(code),
                            trade_unit=list(UNITS)[1], trade_qty=int(shares))


def _account(stt: dict, sig: str) -> None:
    """帳戶總覽與持股；盤中由 fragment 每 10 秒重跑。"""
    px = _prices(list(stt["pos"]) + [trade.BENCH])
    h = trade.holdings(stt, {c: q["price"] for c, q in px.items()})
    mv = float(h["value"].sum())
    total = stt["cash"] + mv
    bench = px.get(trade.BENCH, {}).get("price")
    b_ret = bench / stt["bench_price"] - 1 if bench and stt["bench_price"] else None
    common.kpis([
        ("總資產", f"{total:,.0f}", f"初始 {stt['capital']:,.0f}", True),
        ("總報酬", common.signed_html(total / stt["capital"] - 1, 2), "含已實現與未實現，已扣成本"),
        (f"同期 {trade.BENCH}", common.signed_html(b_ret, 2), "開帳戶後的漲跌，不含息"),
        ("現金", f"{stt['cash']:,.0f}", f"占總資產 {stt['cash'] / total:.0%}" if total else ""),
        ("持股市值", f"{mv:,.0f}", f"{len(h)} 檔"),
        ("已實現損益", common.signed_html(stt["realized"], 0, 1, " 元"), ""),
        ("交易成本", f"{stt['fees']:,.0f}", "手續費 + 證交稅"),
    ])
    st.subheader("持股")
    if h.empty:
        common.empty("還沒有持股。")
    else:
        h = h.assign(weight=h["value"] / total, src=h["code"].map({c: q["src"] for c, q in px.items()}))
        show = h.rename(columns={"code": "代號", "name": "名稱", "shares": "股數", "avg_cost": "平均成本", "price": "現價",
                                 "value": "市值", "weight": "占資產", "pnl": "未實現損益", "ret": "報酬率", "src": "現價來源"})
        ev = common.table(common.styled(show, fmt={"股數": "{:,.0f}", "平均成本": "{:,.2f}", "現價": "{:,.2f}",
                                                   "市值": "{:,.0f}", "占資產": "{:.1%}", "未實現損益": "{:+,.0f}",
                                                   "報酬率": "{:+.2%}"}, signed=["未實現損益", "報酬率"]),
                          hide_index=True, on_select="rerun", selection_mode="single-row",
                          key=f"trade_hold_tbl_{st.session_state.get('trade_tbl_n', 0)}")
        sel = ev.selection.rows if ev and ev.selection else []
        if sel:
            r = h.iloc[sel[0]]
            _sell_from(r["code"], r["shares"], sig)
            st.session_state["trade_tbl_n"] = st.session_state.get("trade_tbl_n", 0) + 1   # 換 key 清掉選取
            st.rerun(scope="app")
    st.caption("點一列，下單區會帶入這檔的全部持股賣出。平均成本含買進手續費。"
               + ("盤中現價是證交所即時報價，延遲約 5 秒，每 10 秒自動更新。" if common.live_every()
                  else "現在不是盤中，現價用最後收盤價。"))


def _nav(rows: list[dict]) -> None:
    n = trade.nav(rows)
    if len(n) < 2:
        st.caption("帳戶淨值曲線要有兩個交易日以上的收盤資料才畫（每天晚上更新資料後多一點）。")
        return
    series = {"我的帳戶": n["nav"]}
    if n["bench"].notna().any():
        series[f"同期 {trade.BENCH}"] = n["bench"]
    common.chart(common.line_fig(n.index, series, "帳戶總資產（每日收盤）",
                                 {"我的帳戶": common.C_STRAT, f"同期 {trade.BENCH}": common.C_ETF}, 300))
    st.caption(f"到 {n.index[-1]:%Y-%m-%d} 收盤；之後的成交等資料更新後才會畫進來。{trade.BENCH} 是開帳戶時用同樣資金買進持有。")


def _order(stt: dict, sig: str) -> None:
    labels, by_code = _labels(sig)
    held = list(stt["pos"])
    side = SIDES[common.segmented("買賣", list(SIDES), "買進", "trade_side")]
    opts = [by_code[c] for c in held if c in by_code] if side == "sell" else labels
    if not opts:
        common.empty("沒有持股可以賣。")
        return
    key = f"trade_pick_{side}"
    goto = st.session_state.pop("goto_trade_code", None)
    if goto in by_code and by_code[goto] in opts:
        st.session_state[key] = by_code[goto]
    if st.session_state.get(key) not in opts:
        st.session_state[key] = by_code.get("2330") if by_code.get("2330") in opts else opts[0]
    c1, c2, c3, c4 = st.columns([2.4, 1, 1.3, 1], vertical_alignment="bottom")
    chosen = c1.selectbox("股票（打代號或名稱篩選）", opts, key=key)
    code = chosen.split()[0]
    unit = UNITS[c2.radio("單位", list(UNITS), key="trade_unit")]
    st.session_state.setdefault("trade_qty", 1)
    qty = c3.number_input("數量", min_value=1, max_value=1_000_000, step=1, key="trade_qty")
    shares = int(qty) * unit
    try:
        q = trade.quote(code, side, live=common.live_quotes((code,)))
    except ValueError as err:
        st.error(str(err))
        return
    try:
        c, err = trade.check(stt, code, side, shares, q), None
    except ValueError as e:
        c, err = None, str(e)
    amount = q["price"] * shares
    fee, tax = trade.costs(code, side, round(amount))
    pay = amount + fee if side == "buy" else amount - fee - tax
    common.kpis([
        ("成交價", f"{q['price']:,.2f}", q["src"], True),
        ("股數", f"{shares:,}", f"{shares / 1000:g} 張" if shares >= 1000 else "零股"),
        ("成交金額", f"{amount:,.0f}", ""),
        ("手續費", f"{fee:,.0f}", "0.1425%，最低 20 元"),
        ("證交稅", f"{tax:,.0f}", "賣出才收：股票 0.3%、ETF 0.1%"),
        ("應付" if side == "buy" else "應收", f"{pay:,.0f}", f"下單後現金 {stt['cash'] + c['net']:,.0f}" if c else ""),
    ])
    hint = (f"現金最多可買 {trade.max_shares(stt['cash'], q['price'], code):,} 股" if side == "buy"
            else f"持有 {stt['pos'].get(code, {}).get('shares', 0):,} 股")
    st.caption(hint + ("。現在不是盤中（或查不到今天的即時報價），以最後收盤價成交，類似盤後定價交易。" if not q["live"] else "。"))
    if not q["live"] and (date.today() - pd.Timestamp(q["date"]).date()).days > STALE_DAYS:
        st.warning(f"這檔最後的收盤價是 {pd.Timestamp(q['date']):%Y-%m-%d}，資料可能還沒更新，或這檔最近沒有交易。")
    if err:
        st.warning(f"不能下單：{err}")
    label = f"{'買進' if side == 'buy' else '賣出'} {code} {shares:,} 股"
    if c4.button(label, type="primary", disabled=err is not None, width="stretch"):
        try:
            r = trade.place(code, side, shares, q=q)
        except ValueError as e:
            st.error(f"下單失敗：{e}")
            return
        st.toast(f"成交：{label} @ {r['price']:,.2f}，現金剩 {r['cash_after']:,.0f}", icon=":material/check:")
        st.rerun()


def _history(stt: dict) -> None:
    if not stt["trades"]:
        common.empty("還沒有成交紀錄。")
        return
    t = pd.DataFrame(stt["trades"]).iloc[::-1]
    t["side"] = t["side"].map({"buy": "買進", "sell": "賣出"})
    t["ts"] = t["ts"].str.replace("T", " ")
    show = t[["ts", "side", "code", "name", "shares", "price", "src", "amount", "fee", "tax", "cash_after"]].rename(columns={
        "ts": "時間", "side": "買賣", "code": "代號", "name": "名稱", "shares": "股數", "price": "成交價", "src": "價格來源",
        "amount": "金額", "fee": "手續費", "tax": "證交稅", "cash_after": "成交後現金"})
    common.table(common.styled(show, fmt={"股數": "{:,.0f}", "成交價": "{:,.2f}", "金額": "{:,.0f}", "手續費": "{:,.0f}",
                                          "證交稅": "{:,.0f}", "成交後現金": "{:,.0f}"}), hide_index=True)


def render(sig: str) -> None:
    st.caption("用虛擬資金自己買賣：按下去就成交。盤中用證交所即時報價（延遲約 5 秒），盤後用最後收盤價；手續費、證交稅照實扣；"
               "漲停買不到、跌停賣不掉；不能融資融券或放空；配息與除權不入帳。和「自選組合」不同，這裡不用設定權重，"
               "也不用等隔天收盤。不是真實下單，也不代表真的買得到這個價格。")
    rows = trade.load()
    stt = trade.state(rows)
    if stt["account"] is None:
        common.empty("還沒有帳戶。先決定虛擬資金，開一個帳戶。")
        _open_form(first=True)
        return
    st.markdown(f"**{stt['account']}**　開戶時間 {stt['opened'].replace('T', ' ')}")
    st.fragment(run_every=common.live_every())(_account)(stt, sig)
    _nav(rows)
    st.subheader("下單")
    _order(stt, sig)
    st.subheader("成交紀錄")
    _history(stt)
    with st.expander("開新帳戶"):
        st.caption("舊帳戶的紀錄會保留在 data/paper/my_trades.jsonl，畫面改顯示新帳戶。")
        _open_form(first=False)
