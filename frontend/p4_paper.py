"""模擬持股：綜合分數前 30、多策略摘要、紀錄、淨值（淨值序列等後端）。"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common
import paper

STRAT_NAV = paper.PAPER / "strat_nav.parquet"


def _top30(s: dict) -> None:
    if not s:
        common.empty("還沒有模擬持股紀錄。")
        return
    has = "port" in s
    common.kpis([
        ("訊號起點", common.esc(s.get("start") or "—"), f"共 {s.get('n_periods', '—')} 期"),
        ("最新訊號日", common.esc(s.get("signal_date") or "—"), f"進場 {s.get('entry_date') or '尚未進場'}"),
        ("累積報酬", common.signed_html(s.get("port"), 2) if has else "等進場", f"淨值日 {s.get('nav_date') or '—'}"),
        ("同池等權", common.signed_html(s.get("pool"), 2) if has else "—", "不扣成本"),
        (paper.ETF, common.signed_html(s.get("etf"), 2) if has else "—", "不扣成本"),
        ("最大回落", common.pct(s.get("mdd"), 2) if has else "—", ""),
    ])
    left, right = st.columns([3, 2])
    with left:
        if paper.NAV.exists():
            nav = pd.read_parquet(paper.NAV)
            fig = common.line_fig(nav["date"], {"模擬": nav["port"], "同池等權": nav["pool"], paper.ETF: nav["etf"]},
                                  "累積淨值（起點 = 1）", {"模擬": common.C_STRAT, "同池等權": common.C_POOL,
                                                       paper.ETF: common.C_ETF}, 360)
            common.chart(fig)
        else:
            common.empty("累積淨值圖要等 10/26 進場之後，data/paper/nav.parquet 才會有資料。")
    with right:
        holds = pd.DataFrame(s.get("holdings") or [])
        if holds.empty:
            common.empty("沒有本期持股。")
        else:
            show = holds.rename(columns={"rank": "名次", "code": "代號", "name": "名稱",
                                         "ret_since_entry": "進場以來"})[["名次", "代號", "名稱", "進場以來"]]
            show["進場以來"] = pd.to_numeric(show["進場以來"], errors="coerce")
            ev = st.dataframe(common.styled(show, fmt={"進場以來": "{:+.2%}"}, signed=["進場以來"]), hide_index=True,
                              height=360, on_select="rerun", selection_mode="single-row", key="paper_holds")
            sel = ev.selection.rows if ev and ev.selection else []
            if sel:
                common.goto_stock(str(show.iloc[sel[0]]["代號"]))
    st.caption("規則：每月記下綜合分數前 30 名（20 日均成交值 2,000 萬以上），訊號日隔天收盤進場、等權、扣成本。"
               "點持股可打開個股頁。")


def _multi(rows: list[dict]) -> None:
    if not rows:
        common.empty("還沒有 strat_summary.json。")
        return
    for x in rows:
        has = "ret" in x and x.get("holdings")
        rn = x.get("regime_now") or {}
        with st.container(border=True):
            tags = common.badge(f"曝險 {common.pct(x.get('target_exposure'), 0).lstrip('+')}", "muted")
            if rn:
                tags += common.badge(f"{'均線上' if rn.get('above') else '均線下'} {common.pct(rn.get('gap'))}"
                                     f"（{'每日' if rn.get('freq') == 'daily' else '每月'}）",
                                     "bull" if rn.get("above") else "bear")
            st.markdown(f"**{common.esc(x.get('name'))}**　{tags}", unsafe_allow_html=True)
            st.caption(x.get("description") or "")
            common.kpis([
                ("起點", common.esc(x.get("start")), f"{x.get('n_periods')} 期，資金 {x.get('capital', 0):,} 元"),
                ("累積報酬", common.signed_html(x.get("ret"), 2) if "ret" in x else "等進場", ""),
                (paper.ETF, common.signed_html(x.get("etf"), 2) if "ret" in x else "—", ""),
                ("同池", common.signed_html(x.get("pool"), 2) if "ret" in x else "—", ""),
                ("最大回落", common.pct(x.get("mdd"), 2) if "ret" in x else "—", ""),
                ("現金 / 出場", f"{common.pct(x.get('cash_pct'), 0).lstrip('+')} / {x.get('exits', 0)}"
                 if "ret" in x else "—", ""),
            ])
            if has:
                df = pd.DataFrame(x["holdings"]).rename(columns={"code": "代號", "name": "名稱", "value": "市值(元)",
                                                                 "weight": "佔資產"})
                st.dataframe(common.styled(df[["代號", "名稱", "市值(元)", "佔資產"]],
                                           fmt={"市值(元)": "{:,.0f}", "佔資產": "{:.1%}"}), hide_index=True)
            else:
                df = pd.DataFrame(x.get("targets") or [])
                if not df.empty:
                    st.markdown(f"本期目標權重（訊號日 {x.get('signal_date')}）")
                    df = df.rename(columns={"code": "代號", "name": "名稱", "weight": "目標權重"})
                    st.dataframe(common.styled(df[["代號", "名稱", "目標權重"]], fmt={"目標權重": "{:.1%}"}),
                                 hide_index=True, height=240)


def _ledger() -> None:
    if not paper.STRAT_LEDGER.exists():
        common.empty("還沒有 strat_ledger.parquet。")
        return
    led = pd.read_parquet(paper.STRAT_LEDGER)
    c1, c2, c3 = st.columns([2, 2, 1])
    names = sorted(led["strategy"].dropna().unique())
    pick = c1.multiselect("策略", names, default=names)
    date_labels = [str(pd.Timestamp(d).date()) for d in sorted(led["signal_date"].dropna().unique())]
    chosen = c2.multiselect("訊號日", date_labels, default=date_labels[-1:])
    only_pick = c3.toggle("只看選進組合的", value=True)
    view = led[led["strategy"].isin(pick)]
    if chosen:
        view = view[view["signal_date"].isin({pd.Timestamp(d) for d in chosen})]
    if only_pick and "in_pick" in view.columns:
        view = view[view["in_pick"].astype(bool)]
    cols = [c for c in ["signal_date", "strategy", "hash", "code", "weight", "in_pick", "pick_order", "regime_on"]
            if c in view.columns]
    show = view[cols].sort_values([c for c in ["strategy", "signal_date", "pick_order"] if c in cols]).rename(columns={
        "signal_date": "訊號日", "strategy": "策略", "hash": "設定雜湊", "code": "代號", "weight": "權重",
        "in_pick": "選進組合", "pick_order": "順序", "regime_on": "濾網開"})
    if "訊號日" in show:
        show["訊號日"] = pd.to_datetime(show["訊號日"]).dt.strftime("%Y-%m-%d")
    st.dataframe(common.styled(show, fmt={"權重": "{:.2%}", "順序": "{:.0f}"}), hide_index=True)
    st.caption(f"{len(show):,} 列。紀錄只追加不改寫；設定改過會用新的雜湊重新起算。")


def _strat_nav() -> None:
    if not STRAT_NAV.exists():
        common.empty("各策略淨值曲線要等後端把序列存成 data/paper/strat_nav.parquet（strategy、date、nav、pool、etf）。"
                     "目前只有摘要，畫不出曲線。")
        return
    df = pd.read_parquet(STRAT_NAV)
    if not {"date", "strategy", "nav"} <= set(df.columns):
        st.warning("strat_nav.parquet 缺 strategy / date / nav 欄，先不畫。")
        return
    names = sorted(df["strategy"].dropna().unique())
    pick = st.multiselect("要畫的策略", names, default=names, key="strat_nav_pick")
    view = df[df["strategy"].isin(pick)].sort_values("date")
    fig = go.Figure()
    for i, (name, g) in enumerate(view.groupby("strategy")):
        fig.add_trace(go.Scatter(x=g["date"], y=g["nav"], name=str(name), mode="lines", line=dict(width=1.8)))
    if len(pick) == 1:
        for col, label, color, dash in (("pool", "同池", common.C_POOL, "dot"), ("etf", paper.ETF, common.C_ETF, "dash")):
            if col in view:
                fig.add_trace(go.Scatter(x=view["date"], y=view[col], name=label, mode="lines",
                                         line=dict(color=color, dash=dash, width=1.2)))
    common.chart(common.fig_layout(fig, "策略淨值", 380))
    if len(pick) != 1:
        st.caption("只選一個策略時才疊上同池與 0050（各策略的池子不一定相同）。")


sig = common.data_sig()
summary = common.cached_paper_summary(sig)
strats = paper.load_strategy_summaries()
common.page_title("模擬持股", f"即時紀錄，起點之前沒有資料・多策略 {len(strats)} 個")

section = st.segmented_control("區塊", ["綜合分數前 30", "多策略", "各策略淨值", "每期紀錄", "追蹤設定"],
                               default="綜合分數前 30", key="paper_section", label_visibility="collapsed")
section = section or "綜合分數前 30"
if section == "綜合分數前 30":
    _top30(summary)
elif section == "多策略":
    _multi(strats)
elif section == "各策略淨值":
    _strat_nav()
elif section == "每期紀錄":
    _ledger()
elif section == "追蹤設定":
    st.caption("paper_strategies.txt。這裡只顯示；改了要等下個月才會記新策略。")
    if paper.STRATS_FILE.exists():
        st.code(paper.STRATS_FILE.read_text(encoding="utf-8"), language="text")
    else:
        common.empty("沒有 paper_strategies.txt。")
