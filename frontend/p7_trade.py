"""虛擬下單：使用者用虛擬資金自己買賣，成交價用現在的股價。規則與帳本在 trade.py。"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common
import trade

SIDES = {"買進": "buy", "賣出": "sell"}
UNITS = {"張（1,000 股）": 1000, "股（零股）": 1}


def _labels(sig: str) -> tuple[list[str], dict[str, str]]:
    nm = common.names_frame(sig).reset_index()
    nm = nm[nm["active"]].sort_values("code")
    labels = (nm["code"].astype(str) + " " + nm["name"].fillna("").astype(str)).tolist()
    return labels, {lab.split()[0]: lab for lab in labels}


def _prices(codes: list[str]) -> dict[str, dict]:
    """持股現價：盤中用即時報價，其他時候用最後收盤價。"""
    live = common.live_quotes(tuple(sorted(codes))) if codes and common.live_every() else {}
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


def _account(stt: dict) -> None:
    """帳戶總覽與持股；盤中由 fragment 每 10 秒重跑。"""
    px = _prices(list(stt["pos"]) + [trade.BENCH])
    h = trade.holdings(stt, {c: q["price"] for c, q in px.items()})
    mv = float(h["value"].sum())
    total = stt["cash"] + mv
    ret = total / stt["capital"] - 1
    bench = px.get(trade.BENCH, {}).get("price")
    b_ret = bench / stt["bench_price"] - 1 if bench and stt["bench_price"] else None
    common.kpis([
        ("總資產", f"{total:,.0f}", f"初始 {stt['capital']:,.0f}", True),
        ("總報酬", common.signed_html(ret, 2), "含已實現與未實現，已扣成本"),
        (f"同期 {trade.BENCH}", common.signed_html(b_ret, 2), "開帳戶後的漲跌，不含息"),
        ("現金", f"{stt['cash']:,.0f}", ""),
        ("持股市值", f"{mv:,.0f}", f"{len(h)} 檔"),
        ("已實現損益", common.signed_html(stt["realized"], 0, 1, " 元"), ""),
        ("交易成本", f"{stt['fees']:,.0f}", "手續費 + 證交稅"),
    ])
    st.subheader("持股")
    if h.empty:
        common.empty("還沒有持股。")
    else:
        src = {c: q["src"] for c, q in px.items()}
        show = h.assign(src=h["code"].map(src)).rename(columns={
            "code": "代號", "name": "名稱", "shares": "股數", "avg_cost": "平均成本", "price": "現價",
            "value": "市值", "pnl": "未實現損益", "ret": "報酬率", "src": "現價來源"})
        common.table(common.styled(show, fmt={"股數": "{:,.0f}", "平均成本": "{:,.2f}", "現價": "{:,.2f}",
                                              "市值": "{:,.0f}", "未實現損益": "{:+,.0f}", "報酬率": "{:+.2%}"},
                                   signed=["未實現損益", "報酬率"]), hide_index=True)
    st.caption("平均成本含買進手續費。" + ("盤中現價是證交所即時報價，延遲約 5 秒，每 10 秒自動更新。" if common.live_every()
                                     else "現在不是盤中，現價用最後收盤價。"))


def _order(stt: dict, sig: str) -> None:
    labels, by_code = _labels(sig)
    held = list(stt["pos"])
    side = SIDES[common.segmented("買賣", list(SIDES), "買進", "trade_side")]
    c1, c2, c3, c4 = st.columns([2.4, 1, 1.3, 1], vertical_alignment="bottom")
    opts = [by_code[c] for c in held if c in by_code] if side == "sell" else labels
    if not opts:
        common.empty("沒有持股可以賣。")
        return
    default = by_code.get(st.query_params.get("code", ""), by_code.get("2330", opts[0]))
    chosen = c1.selectbox("股票（打代號或名稱篩選）", opts, index=opts.index(default) if default in opts else 0,
                          key=f"trade_pick_{side}")
    code = chosen.split()[0]
    unit = UNITS[c2.radio("單位", list(UNITS), key="trade_unit", label_visibility="visible")]
    qty = c3.number_input("數量", min_value=1, max_value=1_000_000, value=1, step=1, key="trade_qty")
    shares = int(qty) * unit
    try:
        q = trade.quote(code, side, live=common.live_quotes((code,)) if common.live_every() else {})
    except ValueError as err:
        st.error(str(err))
        return
    try:
        c = trade.check(stt, code, side, shares, q)
        err = None
    except ValueError as e:
        c, err = None, str(e)
    amount = q["price"] * shares
    fee, tax = trade.costs(code, side, round(amount))
    common.kpis([
        ("成交價", f"{q['price']:,.2f}", q["src"], True),
        ("股數", f"{shares:,}", f"{shares / 1000:g} 張" if shares >= 1000 else "零股"),
        ("成交金額", f"{amount:,.0f}", ""),
        ("手續費", f"{fee:,.0f}", "0.1425%，最低 20 元"),
        ("證交稅", f"{tax:,.0f}", "賣出才收：股票 0.3%、ETF 0.1%"),
        ("應付" if side == "buy" else "應收", f"{abs(c['net']) if c else amount + fee if side == 'buy' else amount - fee - tax:,.0f}",
         f"下單後現金 {stt['cash'] + c['net']:,.0f}" if c else ""),
    ])
    if err:
        st.warning(f"不能下單：{err}")
    label = f"{'買進' if side == 'buy' else '賣出'} {code} {shares:,} 股"
    if c4.button(label, type="primary", disabled=err is not None, use_container_width=True):
        try:
            r = trade.place(code, side, shares, q=q)
        except ValueError as e:
            st.error(f"下單失敗：{e}")
            return
        st.toast(f"成交：{label} @ {r['price']:,.2f}，現金剩 {r['cash_after']:,.0f}", icon=":material/check:")
        st.rerun()
    if not q["live"]:
        st.caption("現在不是盤中（或查不到今天的即時報價），以最後收盤價成交，類似盤後定價交易。")


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


common.page_title("虛擬下單", "用虛擬資金自己買賣，成交價用現在的股價")
st.caption("盤中用證交所即時報價成交（延遲約 5 秒），盤後用最後收盤價。手續費、證交稅照實扣；漲停買不到、跌停賣不掉；"
           "不能融資融券或放空；配息與除權不入帳。不是真實下單，也不代表真的買得到這個價格。")
sig = common.data_sig()
rows = trade.load()
stt = trade.state(rows)
if stt["account"] is None:
    common.empty("還沒有帳戶。先決定虛擬資金，開一個帳戶。")
    _open_form(first=True)
    st.stop()

st.markdown(f"**{stt['account']}**　開戶時間 {stt['opened'].replace('T', ' ')}")
st.fragment(run_every=common.live_every())(_account)(stt)
st.subheader("下單")
_order(stt, sig)
st.subheader("成交紀錄")
_history(stt)
with st.expander("開新帳戶"):
    st.caption("舊帳戶的紀錄會保留在 data/paper/my_trades.jsonl，畫面改顯示新帳戶。")
    _open_form(first=False)
