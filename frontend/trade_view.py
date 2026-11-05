"""模擬持股頁的「虛擬下單」區塊：使用者用虛擬資金自己買賣，照台股交易時段成交。規則與帳本在 trade.py。"""
from datetime import date

import pandas as pd
import streamlit as st

import common
import trade

SIDES = {"買進": "buy", "賣出": "sell"}
UNITS = {"整股（張）": 1000, "零股（1–999 股）": 1}
STALE_DAYS = 4       # 參考價比今天舊超過這幾天就提醒資料可能沒更新
PHASE_NOTE = {
    trade.LIVE: "盤中（09:00–13:30）：用即時成交價馬上成交。",
    trade.AFTER_FIXED: "盤後定價交易（14:00–14:30，整股）：用今天的收盤價成交。",
    trade.AFTER_ODD: "盤後零股交易（13:40–14:30）：用今天的收盤價成交。",
    trade.RESERVE: "現在不在交易時段，這筆會變成預約單：下一個交易日用開盤價成交。下面的參考價只用來先檢查現金和股數。",
}


def _labels(sig: str) -> tuple[list[str], dict[str, str]]:
    nm = common.names_frame(sig).reset_index()
    nm = nm[nm["active"]].sort_values("code")
    labels = (nm["code"].astype(str) + " " + nm["name"].fillna("").astype(str)).tolist()
    return labels, {lab.split()[0]: lab for lab in labels}


def _prices(codes: list[str]) -> dict[str, dict]:
    """持股現價：證交所報價（盤中即時、收盤後今天收盤）不比資料舊就用報價，否則用資料表最後收盤。"""
    live = common.live_quotes(tuple(sorted(codes))) if codes else {}
    return {c: m for c in codes if (m := trade.mark(c, live))}


def _open_form(first: bool) -> None:
    c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
    cap = c1.number_input("虛擬資金（元）", min_value=10_000, max_value=100_000_000, value=trade.DEFAULT_CAPITAL,
                          step=100_000, key="trade_capital")
    if c2.button("開帳戶" if first else "開新帳戶（舊帳戶保留）", type="primary" if first else "secondary"):
        r = trade.open_account(cap)
        st.toast(f"已開「{r['account']}」，資金 {r['capital']:,.0f} 元", icon=":material/check:")
        st.rerun()


def _sell_from(code: str, shares: int, sig: str) -> None:
    """點持股列：下單區切到賣出這檔，數量填可賣的股數（整股和零股要分開下，有零頭時先帶整股）。"""
    _, by_code = _labels(sig)
    units = list(UNITS)
    unit, qty = (units[1], shares) if shares < trade.LOT else (units[0], shares // trade.LOT)
    st.session_state.update(trade_side="賣出", _trade_side_last="賣出", trade_pick_sell=by_code.get(code),
                            trade_unit=unit, trade_qty=int(qty))
    if shares >= trade.LOT and shares % trade.LOT:
        st.session_state["trade_note"] = f"{code} 有 {shares % trade.LOT} 股零頭，整股賣完後再用零股賣。"


def _account(stt: dict, sig: str) -> None:
    """帳戶總覽與持股；盤中由 fragment 每 10 秒重跑。"""
    px = _prices(list(stt["pos"]) + [trade.BENCH])
    h = trade.holdings(stt, {c: q["price"] for c, q in px.items()})
    mv = float(h["value"].sum())
    total = stt["cash"] + mv
    bench = px.get(trade.BENCH, {}).get("price")
    fac = trade.bench_factor(stt["opened"])
    b_ret = bench / stt["bench_price"] * (fac.iloc[-1] if len(fac) else 1.0) - 1 if bench and stt["bench_price"] else None
    common.kpis([
        ("總資產", f"{total:,.0f}", f"初始 {stt['capital']:,.0f}", True),
        ("總報酬", common.signed_html(total / stt["capital"] - 1, 2), "含已實現、未實現與配息，已扣成本"),
        (f"同期 {trade.BENCH}", common.signed_html(b_ret, 2), "開帳戶後買進持有，含息"),
        ("現金", f"{stt['cash']:,.0f}", f"預約單保留 {stt['reserved']:,.0f}" if stt["reserved"] else
         (f"占總資產 {stt['cash'] / total:.0%}" if total else "")),
        ("持股市值", f"{mv:,.0f}", f"{len(h)} 檔"),
        ("已實現損益", common.signed_html(stt["realized"], 0, 1, " 元"), f"另有配息 {stt['dividends']:,.0f} 元"),
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
            _sell_from(r["code"], int(r["shares"]), sig)
            st.session_state["trade_tbl_n"] = st.session_state.get("trade_tbl_n", 0) + 1   # 換 key 清掉選取
            st.rerun(scope="app")
    st.caption("點一列，下單區會帶入這檔的持股賣出。平均成本含買進手續費。"
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
    st.caption(f"到 {n.index[-1]:%Y-%m-%d} 收盤；之後的成交等資料更新後才會畫進來。{trade.BENCH} 是開帳戶時用同樣資金買進持有（含息）。")


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
    c1, c2, c3, c4 = st.columns([2.4, 1.1, 1.2, 1.1], vertical_alignment="bottom")
    chosen = c1.selectbox("股票（打代號或名稱篩選）", opts, key=key)
    code = chosen.split()[0]
    unit = UNITS[c2.radio("單位", list(UNITS), key="trade_unit")]
    top = 999 if unit == 1 else 1000
    st.session_state["trade_qty"] = min(max(int(st.session_state.get("trade_qty", 1)), 1), top)
    qty = c3.number_input("張數" if unit == trade.LOT else "股數", min_value=1, max_value=top, step=1, key="trade_qty")
    shares = int(qty) * unit
    kind = "零股" if unit == 1 else "整股"
    try:
        q = trade.quote(code, side, live=common.live_quotes((code,)), kind=kind)
    except ValueError as err:
        st.error(str(err))
        return
    reserve = q["phase"] == trade.RESERVE
    (st.info if reserve else st.success)(PHASE_NOTE[q["phase"]], icon=":material/schedule:")
    if note := st.session_state.pop("trade_note", None):
        st.caption(note)
    try:
        c, err = trade.check(stt, code, side, shares, q), None
    except ValueError as e:
        c, err = None, str(e)
    amount = q["price"] * shares
    fee, tax = trade.costs(code, side, round(amount))
    pay = amount + fee if side == "buy" else amount - fee - tax
    est = "預估" if reserve else ""
    common.kpis([
        ("參考價" if reserve else "成交價", f"{q['price']:,.2f}", q["src"], True),
        ("股數", f"{shares:,}", f"{shares // trade.LOT:,} 張" if kind == "整股" else "零股"),
        (f"{est}成交金額", f"{amount:,.0f}", ""),
        (f"{est}手續費", f"{fee:,.0f}", "0.1425%，最低 20 元"),
        (f"{est}證交稅", f"{tax:,.0f}", "賣出才收：股票 0.3%、ETF 0.1%"),
        (est + ("應付" if side == "buy" else "應收"), f"{pay:,.0f}",
         (f"下單後可用現金 {stt['cash'] - stt['reserved'] + c['net']:,.0f}" if c else "")),
    ])
    avail = stt["cash"] - stt["reserved"]
    hint = (f"可用現金最多可買 {trade.max_shares(avail, q['price'], code):,} 股" if side == "buy"
            else f"可賣 {stt['pos'].get(code, {}).get('shares', 0) - stt['pending_sell'].get(code, 0):,} 股")
    st.caption(hint + "。整股是 1,000 股的倍數，零股 1–999 股，要分開下。")
    if not q["live"] and q.get("date") is not None and (date.today() - pd.Timestamp(q["date"]).date()).days > STALE_DAYS:
        st.warning(f"這檔最後的價格是 {pd.Timestamp(q['date']):%Y-%m-%d}，資料可能還沒更新，或這檔最近沒有交易。")
    if err:
        st.warning(f"不能下單：{err}")
    act = "買進" if side == "buy" else "賣出"
    label = f"{'預約' if reserve else ''}{act} {code} {shares:,} 股"
    if c4.button(label, type="primary", disabled=err is not None, width="stretch"):
        try:
            r = trade.place(code, side, shares, q=q)
        except ValueError as e:
            st.error(f"下單失敗：{e}")
            return
        if r["type"] == "order":
            st.toast(f"已掛預約單 {r['id']}：{act} {code} {shares:,} 股，下一個交易日開盤價成交", icon=":material/schedule:")
        else:
            st.toast(f"成交：{act} {code} {shares:,} 股 @ {r['price']:,.2f}，現金剩 {r['cash_after']:,.0f}", icon=":material/check:")
        st.rerun()


def _orders(stt: dict) -> None:
    if stt["orders"]:
        st.subheader("預約單（還沒成交）")
        o = pd.DataFrame(list(stt["orders"].values()))
        o["side"] = o["side"].map({"buy": "買進", "sell": "賣出"})
        o["ts"] = o["ts"].str.replace("T", " ")
        show = o[["id", "ts", "side", "code", "name", "shares", "ref_price", "est_amount"]].rename(columns={
            "id": "編號", "ts": "下單時間", "side": "買賣", "code": "代號", "name": "名稱", "shares": "股數",
            "ref_price": "參考價", "est_amount": "預估金額"})
        ev = common.table(common.styled(show, fmt={"股數": "{:,.0f}", "參考價": "{:,.2f}", "預估金額": "{:,.0f}"}),
                          hide_index=True, on_select="rerun", selection_mode="single-row",
                          key=f"trade_orders_tbl_{st.session_state.get('trade_ord_n', 0)}")
        sel = ev.selection.rows if ev and ev.selection else []
        if sel:
            oid = show.iloc[sel[0]]["編號"]
            if st.button(f"取消預約單 {oid}", icon=":material/cancel:"):
                trade.cancel(oid)
                st.toast(f"已取消預約單 {oid}", icon=":material/check:")
                st.session_state["trade_ord_n"] = st.session_state.get("trade_ord_n", 0) + 1   # 換 key 清掉選取
                st.rerun()
        st.caption("下一個交易日開盤後撮合（打開這頁時自動處理）；開盤漲停買不到、跌停賣不掉、開盤沒有成交或現金不夠就記成未成交。"
                   "選一列可以取消。")
    if stt["cancels"]:
        with st.expander(f"取消與未成交的預約單（{len(stt['cancels'])} 筆）"):
            x = pd.DataFrame(stt["cancels"]).iloc[::-1]
            x["side"] = x["side"].map({"buy": "買進", "sell": "賣出"})
            common.table(x[["order_id", "side", "code", "name", "shares", "reason"]].rename(columns={
                "order_id": "編號", "side": "買賣", "code": "代號", "name": "名稱", "shares": "股數", "reason": "原因"}),
                hide_index=True)


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


def _dividends(divs: pd.DataFrame) -> None:
    if divs.empty:
        return
    st.subheader("配息入帳")
    show = divs.iloc[::-1].assign(date=divs["date"].dt.strftime("%Y-%m-%d")).rename(columns={
        "date": "除權息日", "code": "代號", "name": "名稱", "shares": "持有股數", "per_share": "每股", "amount": "入帳金額"})
    common.table(common.styled(show, fmt={"持有股數": "{:,.0f}", "每股": "{:,.2f}", "入帳金額": "{:,.0f}"}), hide_index=True)
    st.caption("除權息日前一天收盤還持有的，每股（前收盤 − 參考價）當成現金入帳；配股也折成現金。晚上更新資料後才會出現。")


def _settle() -> None:
    """打開頁面時撮合到期的預約單。"""
    rows = trade.load()
    codes = tuple(sorted({r["code"] for r in rows if r["type"] == "order"}))
    if not codes:
        return
    for r in trade.settle(live=common.live_quotes(codes)):
        if r["type"] == "trade":
            st.toast(f"預約單 {r['order_id']} 成交：{r['code']} {r['shares']:,} 股 @ {r['price']:,.2f}", icon=":material/check:")
        else:
            st.toast(f"預約單 {r['order_id']} {r['reason']}", icon=":material/info:")


def render(sig: str) -> None:
    st.caption("用虛擬資金自己買賣，照台股交易時段：盤中 09:00–13:30 用即時成交價（延遲約 5 秒）；盤後定價 14:00–14:30（整股）、"
               "盤後零股 13:40–14:30 用今天收盤價；其他時間下的單變成預約單，下一個交易日開盤價成交。手續費、證交稅照實扣；"
               "漲停買不到、跌停賣不掉；不能融資融券或放空；除權息折成現金入帳。不是真實下單，也不代表真的買得到這個價格。")
    _settle()
    rows = trade.load()
    divs = trade.dividends(rows)
    stt = trade.state(rows, divs=divs)
    if stt["account"] is None:
        common.empty("還沒有帳戶。先決定虛擬資金，開一個帳戶。")
        _open_form(first=True)
        return
    st.markdown(f"**{stt['account']}**　開戶時間 {stt['opened'].replace('T', ' ')}")
    st.fragment(run_every=common.live_every())(_account)(stt, sig)
    _nav(rows)
    st.subheader("下單")
    _order(stt, sig)
    _orders(stt)
    st.subheader("成交紀錄")
    _history(stt)
    _dividends(divs)
    with st.expander("開新帳戶"):
        st.caption("舊帳戶的紀錄會保留在 data/paper/my_trades.jsonl，畫面改顯示新帳戶。")
        _open_form(first=False)
