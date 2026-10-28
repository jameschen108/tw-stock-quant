"""總覽：今天要看什麼。觀察名單、提醒、模型選股、財報季、多策略模擬摘要。"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common
import daily
import paper

WATCH = daily.WATCHLIST
OL_NAME = {"short": "短期", "mid": "中期", "long": "長期"}


def _reload_watchlist():
    st.session_state.wl_text = WATCH.read_text(encoding="utf-8") if WATCH.exists() else ""


def _watch_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "代號": r.get("code"), "名稱": r.get("name") or "",
        "收盤": r.get("close"), "1日": r.get("ret_1d"), "5日": r.get("ret_5d"),
        "短期": r.get("ol_short") or "—", "中期": r.get("ol_mid") or "—", "長期": r.get("ol_long") or "—",
        "綜合分數": r.get("comp"), "分數變化": r.get("comp_chg"),
        "外資持股%": r.get("foreign_pct"), "外資5日(點)": r.get("foreign_5d"),
        "融資(張)": r.get("margin_bal"), "融資5日": r.get("margin_5d"),
        "法人5日(張)": r.get("insti_5d"),
        "提醒": len(r.get("alerts") or []), "新公布": len(r.get("events") or []),
    } for r in rows])


def _attention(rows: list[dict]) -> None:
    hits = [r for r in rows if r.get("alerts") or r.get("events")]
    if not hits:
        common.empty("觀察名單這次沒有超過門檻的提醒，也沒有新公布的營收、財報或重大訊息。")
        return
    cards = []
    for r in hits:
        items = [f'<li class="alert">{common.esc(a)}</li>' for a in r.get("alerts") or []]
        items += [f"<li>{common.esc(e)}</li>" for e in r.get("events") or []]
        labels = "".join(common.badge(f"{OL_NAME[k]} {r.get(f'ol_{k}')}", common.LABEL_CLASS.get(r.get(f"ol_{k}")))
                         for k in OL_NAME if r.get(f"ol_{k}"))
        cards.append(
            f'<div class="card"><div class="hd"><b>{common.esc(r.get("code"))} {common.esc(r.get("name"))}</b>'
            f'<span>{common.signed_html(r.get("ret_1d"))}</span></div>'
            f'<div>{labels}</div><ul>{"".join(items)}</ul></div>')
    st.markdown(f'<div class="cgrid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _watch_table(rows: list[dict]) -> None:
    df = _watch_frame(rows)
    sty = common.styled(
        df,
        fmt={"收盤": "{:,.2f}", "1日": "{:+.2%}", "5日": "{:+.2%}", "綜合分數": "{:.0f}", "分數變化": "{:+.0f}",
             "外資持股%": "{:.2f}", "外資5日(點)": "{:+.2f}", "融資(張)": "{:,.0f}", "融資5日": "{:+.1%}",
             "法人5日(張)": "{:+,.0f}"},
        signed=["1日", "5日", "分數變化", "外資5日(點)", "融資5日", "法人5日(張)"],
        labels=["短期", "中期", "長期"])
    ev = common.table(sty, hide_index=True, on_select="rerun", selection_mode="single-row", key="watch_tbl",
                      column_config={"綜合分數": st.column_config.ProgressColumn(
                          "綜合分數", min_value=0, max_value=100, format="%d")})
    st.caption("點一列打開個股頁。綜合分數是全市場百分位；分數變化、提醒、新公布都是和上次 daily.py 執行時比。")
    sel = ev.selection.rows if ev and ev.selection else []
    if sel:
        common.goto_stock(str(df.iloc[sel[0]]["代號"]))


def _picks(got) -> None:
    if not got:
        common.empty("還沒有 outlook.parquet，先跑 uv run outlook.py。")
        return
    cal, tab = got
    rels = "".join(common.badge(f"{v.get('name', k)}：{v.get('reliability', '—')}",
                                "info" if v.get("reliability") == "有統計支持" else "muted")
                   for k, v in (cal or {}).items() if k != "_picks" and isinstance(v, dict))
    use = (cal or {}).get("_picks", {}).get("horizons") or []
    cond = "、".join(cal[k].get("name", k) for k in use if isinstance(cal.get(k), dict))
    st.markdown(f'<div>樣本外可靠度 {rels}</div>', unsafe_allow_html=True)
    st.caption(f"條件：{cond + ' 都看漲' if cond else '目前沒有有統計支持的期間'}，"
               "20 日均成交值 2,000 萬以上，依這些期間的平均百分位排序。")
    if tab is None or tab.empty:
        common.empty("這一期沒有同時符合的股票。")
        return
    cols = ["code", "name", "pick_score"] + [f"{k}_pct" for k in OL_NAME if f"{k}_pct" in tab.columns]
    show = tab[cols].rename(columns={"code": "代號", "name": "名稱", "pick_score": "平均百分位",
                                     **{f"{k}_pct": f"{v}百分位" for k, v in OL_NAME.items()}}).reset_index(drop=True)
    ev = common.table(common.styled(show, fmt={c: "{:.0f}" for c in show.columns if "百分位" in c}),
                      hide_index=True, on_select="rerun", selection_mode="single-row", key="picks_tbl")
    sel = ev.selection.rows if ev and ev.selection else []
    if sel:
        common.goto_stock(str(show.iloc[sel[0]]["代號"]))


def _season(ss, cur, ind) -> None:
    if not ss:
        common.empty("現在不在財報季窗口（季底到申報期限後 30 天）。")
        return
    rep, exp = ss.get("reported") or 0, ss.get("expected") or 0
    st.progress(min(rep / exp, 1.0) if exp else 0.0,
                text=f"{ss.get('quarter')}：已公布 {rep:,} / {exp:,} 家，申報期限 {ss.get('deadline')}"
                     f"（還有 {ss.get('days_to_deadline', '—')} 天）")
    if cur is None or cur.empty:
        st.caption("這一季還沒有公司公布。")
        return
    view = cur.sort_values(["watch", "mcap_億"], ascending=[False, False])
    cols = [x for x in ["watch", "code", "name", "產業", "avail_date", "rev_yoy", "營業毛利率", "gm_chg", "gm_rank",
                        "每股盈餘", "eps_prev", "mcap_億"] if x in view.columns]
    df = view[cols].rename(columns={
        "watch": "觀察", "code": "代號", "name": "名稱", "avail_date": "公布日", "rev_yoy": "營收年增%",
        "營業毛利率": "毛利率%", "gm_chg": "毛利率年增(點)", "gm_rank": "毛利率名次(近20季)",
        "每股盈餘": "EPS", "eps_prev": "去年同季EPS", "mcap_億": "市值(億)"})
    if "觀察" in df:
        df["觀察"] = df["觀察"].map(lambda w: "★" if w else "")
    if "公布日" in df:
        df["公布日"] = pd.to_datetime(df["公布日"]).dt.strftime("%m/%d")
    common.table(common.styled(df, fmt={"營收年增%": "{:+.1f}", "毛利率%": "{:.1f}", "毛利率年增(點)": "{:+.1f}",
                                        "毛利率名次(近20季)": "{:.0f}", "EPS": "{:.2f}", "去年同季EPS": "{:.2f}",
                                        "市值(億)": "{:,.0f}"},
                                signed=["營收年增%", "毛利率年增(點)"]), hide_index=True)
    if ind is not None and not ind.empty:
        st.markdown("**依產業**")
        show = ind.reset_index()
        show.columns = ["產業", "已公布", "營收年增中位數%", "毛利率年增中位數(點)", "營收成長家數比例%"]
        common.table(common.styled(show, fmt={"營收年增中位數%": "{:+.1f}", "毛利率年增中位數(點)": "{:+.1f}",
                                              "營收成長家數比例%": "{:.0f}"},
                                   signed=["營收年增中位數%", "毛利率年增中位數(點)"]), hide_index=True)


def _strategies(rows: list[dict]) -> None:
    if not rows:
        common.empty("還沒有 data/paper/strat_summary.json。")
        return
    cards = []
    for x in rows:
        rn = x.get("regime_now")
        reg = "" if not rn else common.badge(
            f"{'均線上' if rn.get('above') else '均線下'} {common.pct(rn.get('gap'))}",
            "bull" if rn.get("above") else "bear")
        has = "ret" in x
        body = (f"<li>累積 {common.signed_html(x.get('ret'), 2)}，0050 {common.signed_html(x.get('etf'), 2)}，"
                f"同池 {common.signed_html(x.get('pool'), 2)}</li>"
                f"<li>最大回落 {common.pct(x.get('mdd'), 2)}，現金 {common.pct(x.get('cash_pct'), 0)}，"
                f"出場 {x.get('exits', 0)} 次</li>") if has else \
            f"<li>本期目標 {len(x.get('targets') or [])} 檔，訊號日 {common.esc(x.get('signal_date'))}，等進場</li>"
        expo = common.pct(x.get("target_exposure"), 0).lstrip("+")
        tags = reg + common.badge("起點 " + str(x.get("start")), "muted") + common.badge(f"{x.get('n_periods')} 期", "muted")
        cards.append(
            f'<div class="card"><div class="hd"><b>{common.esc(x.get("name"))}</b>'
            f'<span class="badge muted">曝險 {expo}</span></div>'
            f'<div>{tags}</div><ul>{body}</ul></div>')
    st.markdown(f'<div class="cgrid">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.page_link("p4_paper.py", label="持股明細與每期紀錄", icon=":material/arrow_forward:")


def _editor() -> None:
    st.caption("寫入 watchlist.txt（一行一個代號，# 之後是註解）。這是介面上唯一會改的設定檔，"
               "存檔不會動到 data/daily_state.json。")
    if "wl_text" not in st.session_state:
        _reload_watchlist()
    st.text_area("watchlist.txt", key="wl_text", height=200, label_visibility="collapsed")
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("儲存", type="primary", icon=":material/save:"):
        text = st.session_state.wl_text
        with st.spinner("核對代號…"):
            known = set(common.names_frame(common.data_sig()).index.astype(str))
        bad = [c for c in (ln.split("#")[0].strip() for ln in text.splitlines()) if c and c not in known]
        if bad:
            st.error("這些代號不在行情裡，沒有寫入：" + "、".join(bad))
        else:
            WATCH.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
            st.toast("已寫入 watchlist.txt", icon=":material/check:")
            st.rerun()
    c2.button("重新讀檔", on_click=_reload_watchlist, icon=":material/refresh:")


sig = common.data_sig()
codes = tuple(daily.read_watchlist())
prev_json = daily.STATE.read_text(encoding="utf-8") if daily.STATE.exists() else ""
with st.spinner("整理觀察名單與財報季…"):
    rows = common.cached_watch(sig, codes, prev_json) if codes else []
    ss, cur, ind = common.cached_season(sig, codes)
    got = common.cached_picks(sig)
strats = paper.load_strategy_summaries()

status = common.load_status()
prev_asof = (json.loads(prev_json).get("asof") if prev_json else None) or "—"
common.page_title("總覽", f"資料日 {status.get('last_price_date', '—')}　·　和上次 daily.py（{prev_asof}）比較")

n_alert = sum(len(r.get("alerts") or []) for r in rows)
n_event = sum(len(r.get("events") or []) for r in rows)
n_bull = sum(1 for r in rows for k in OL_NAME if r.get(f"ol_{k}") == "看漲")
n_picks = 0 if not got or got[1] is None else len(got[1])
common.kpis([
    ("觀察名單", f"{len(rows)} 檔", f"看漲標籤 {n_bull} 個（短中長期合計）"),
    ("提醒", f"{n_alert} 則", "漲跌 7%、分數 15 分、外資 1 點、融資 15%、標籤改變", n_alert > 0),
    ("新公布", f"{n_event} 則", "月營收、季財報、重大訊息"),
    ("財報季", f"{ss.get('reported', 0):,} / {ss.get('expected', 0):,}" if ss else "—",
     f"{ss.get('quarter')}，距期限 {ss.get('days_to_deadline')} 天" if ss else "不在財報季"),
    ("模型選股", f"{n_picks} 檔", "短中長期都看漲"),
])

st.subheader("需要注意")
_attention(rows)

st.subheader("觀察名單")
if rows:
    _watch_table(rows)
else:
    common.empty("觀察名單是空的，到下方「編輯觀察名單」加入代號。")

section = common.segmented("區塊", ["模型選股", "財報季", "多策略模擬", "編輯觀察名單"], "模型選股",
                           key="home_section", label_visibility="collapsed")
if section == "模型選股":
    _picks(got)
elif section == "財報季":
    _season(ss, cur, ind)
elif section == "多策略模擬":
    _strategies(strats)
elif section == "編輯觀察名單":
    _editor()
