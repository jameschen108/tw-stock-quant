"""模擬持股：綜合分數前 30、多策略摘要、紀錄、淨值（淨值序列等後端）。"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common
import custom
import mine
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
    if df.empty:
        common.empty("data/paper/strat_nav.parquet 已在，目前 0 列。10/26 進場之後才有淨值曲線。")
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


def _custom() -> None:
    st.caption("自己挑股票和起訖日回頭模擬：成本、零股取整、漲跌停規則同回測。代號用逗號分隔，可加 :權重（例 2330:50,2317:30）。")
    c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1.2])
    spec = c1.text_input("股票代號", "2330,2317,2454", key="cu_codes")
    start = c2.date_input("開始日", pd.Timestamp("2023-01-03"), key="cu_start")
    end = c3.date_input("結束日", pd.Timestamp.today().normalize(), key="cu_end")
    cap = c4.number_input("本金（元）", 10_000, 100_000_000, 1_000_000, 10_000, key="cu_cap")
    mode = st.radio("調整方式", ["買進持有", "每月調回目標權重"], horizontal=True, key="cu_mode")
    try:
        w = custom.parse_codes(spec)
        res = custom.run(w, start, end, "none" if mode == "買進持有" else "monthly", cap)
    except ValueError as e:
        st.warning(str(e))
        return
    if res["missing"]:
        st.warning("期間內沒有行情，已略過：" + "、".join(res["missing"]))
    nv, daily, p = res["nav"], res["daily"], res["perf"]
    cols = st.columns(3)
    for col, (k, label) in zip(cols, (("strategy", "自選（扣成本）"), ("pool", "自選（不扣成本）"), ("etf", paper.ETF))):
        col.metric(label, f"{p[k]['total']:+.1%}" if p[k] else "—",
                   f"年化 {p[k]['cagr']:+.1%}・回落 {p[k]['mdd']:.1%}" if p[k] else "期間太短", delta_color="off")
    fig = go.Figure()
    for k, label, color in (("strategy", "自選（扣成本）", common.C_STRAT), ("pool", "自選（不扣成本）", common.C_POOL),
                            ("etf", paper.ETF, common.C_ETF)):
        fig.add_trace(go.Scatter(x=daily.index, y=(1 + daily[k]).cumprod(), name=label, mode="lines",
                                 line=dict(color=color, width=1.8 if k == "strategy" else 1.2)))
    common.chart(common.fig_layout(fig, "累積淨值（起點 = 1）", 380))
    names = custom.names_of(list(res["weights"].index))
    st.dataframe(pd.DataFrame({"代號": res["weights"].index, "名稱": [names.get(c, "") for c in res["weights"].index],
                               "目標權重": res["weights"].values}), hide_index=True, width="stretch",
                 column_config={"目標權重": st.column_config.NumberColumn(format="percent")})
    t = res["trades"]
    st.caption(f"期末資產 {nv.iloc[-1]:,.0f} 元・手續費 {t['fees'].sum():,.0f}・證交稅 {t['tax'].sum():,.0f}・"
               f"成交 {len(t)} 次。" + "".join(f"\n{r['exe']:%Y-%m-%d} 未成交：{'、'.join(r['blocked'])}。"
                                               for _, r in t.iterrows() if r["blocked"]))


def _mine() -> None:
    st.caption("自己挑股票，從最新資料日起往後記錄（不補過去，要看過去用「自選股模擬」）。隔一個交易日收盤進場，成本與回測相同。"
               "月調組合在新月份自動調回目標權重，其餘只在你改持股時調整。")
    meta = {k: v for k, v in mine.load_meta().items() if not v["archived"]}
    with st.expander("建立新組合"):
        c1, c2, c3 = st.columns([1.5, 3, 1.2])
        name = c1.text_input("組合名稱", key="mn_name")
        spec = c2.text_input("股票代號（可加 :權重）", "2330,2317,2454", key="mn_spec")
        cap = c3.number_input("本金（元）", 10_000, 100_000_000, 1_000_000, 10_000, key="mn_cap")
        monthly = st.checkbox("每月調回目標權重", key="mn_monthly")
        if st.button("建立", key="mn_create"):
            try:
                sig = mine.create(name.strip(), custom.parse_codes(spec), monthly, cap)
                st.success(f"已建立，訊號日 {sig:%Y-%m-%d}，隔一個交易日收盤進場。")
                st.rerun()
            except ValueError as e:
                st.warning(str(e))
    if not meta:
        common.empty("還沒有自選組合。")
        return
    name = st.selectbox("組合", list(meta), key="mn_pick")
    m = meta[name]
    sigs = mine._latest_signals(mine.load_ledger(), name)
    cur = sigs[max(sigs)]
    names = custom.names_of(list(cur.index))
    st.caption(f"建立於 {m['created']}・本金 {m['capital']:,.0f}・{'每月調回目標權重' if m['rebalance'] == 'monthly' else '只在改動時調整'}"
               f"・{len(sigs)} 次訊號（最近 {max(sigs):%Y-%m-%d}）")
    st.dataframe(pd.DataFrame({"代號": cur.index, "名稱": [names.get(c, "") for c in cur.index], "目標權重": cur.values}),
                 hide_index=True, width="stretch", column_config={"目標權重": st.column_config.NumberColumn(format="percent")})
    res = mine.replay(name)
    if res is None:
        common.empty("訊號已記下，等下一個交易日收盤進場後才有淨值。")
    else:
        daily, p = res["daily"], res["perf"]
        tot = lambda k: (1 + daily[k]).prod() - 1
        cols = st.columns(3)
        for col, (k, label) in zip(cols, (("strategy", "我的組合（扣成本）"), ("pool", "同組等權（不扣成本）"), ("etf", paper.ETF))):
            col.metric(label, f"{tot(k):+.1%}", f"回落 {p[k]['mdd']:.1%}" if p[k] else "期間太短", delta_color="off")
        fig = go.Figure()
        for k, label, color in (("strategy", "我的組合", common.C_STRAT), ("pool", "同組等權", common.C_POOL),
                                ("etf", paper.ETF, common.C_ETF)):
            fig.add_trace(go.Scatter(x=daily.index, y=(1 + daily[k]).cumprod(), name=label, mode="lines",
                                     line=dict(color=color, width=1.8 if k == "strategy" else 1.2)))
        common.chart(common.fig_layout(fig, "累積淨值（起點 = 1）", 340))
        t = res["trades"]
        st.caption(f"資產 {res['nav'].iloc[-1]:,.0f} 元・手續費 {t['fees'].sum():,.0f}・證交稅 {t['tax'].sum():,.0f}"
                   + "".join(f"\n{r['exe']:%Y-%m-%d} 未成交：{'、'.join(r['blocked'])}。" for _, r in t.iterrows() if r["blocked"]))
    with st.expander("改持股 / 封存"):
        spec = st.text_input("新的持股（代號:權重）", ",".join(f"{c}:{w * 100:.0f}" for c, w in cur.items()), key=f"mn_edit_{name}")
        c1, c2 = st.columns(2)
        if c1.button("記錄新持股", key=f"mn_save_{name}"):
            try:
                sig = mine.edit(name, custom.parse_codes(spec))
                st.success(f"已記錄，訊號日 {sig:%Y-%m-%d}。")
                st.rerun()
            except ValueError as e:
                st.warning(str(e))
        if c2.button("封存這個組合", key=f"mn_arch_{name}"):
            mine.archive(name)
            st.rerun()


sig = common.data_sig()
summary = common.cached_paper_summary(sig)
strats = paper.load_strategy_summaries()
common.page_title("模擬持股", f"即時紀錄，起點之前沒有資料・多策略 {len(strats)} 個")

section = st.segmented_control("區塊", ["綜合分數前 30", "多策略", "各策略淨值", "每期紀錄", "自選股模擬", "我的組合", "追蹤設定"],
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
elif section == "自選股模擬":
    _custom()
elif section == "我的組合":
    _mine()
elif section == "追蹤設定":
    st.caption("paper_strategies.txt。這裡只顯示；改了要等下個月才會記新策略。")
    if paper.STRATS_FILE.exists():
        st.code(paper.STRATS_FILE.read_text(encoding="utf-8"), language="text")
    else:
        common.empty("沒有 paper_strategies.txt。")
