"""個股：事實表、圖、同業、區間報酬、LLM 報告。"""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common
import daily
import report

HAIKU = "claude-haiku-5-5"
SECTIONS = ["價格", "模型看法", "訊號與形態", "營收財報", "籌碼", "重大訊息", "同業", "報告"]


# ---------------------------------------------------------------- 選股

def _labels(sig: str) -> tuple[list[str], dict[str, str]]:
    nm = common.names_frame(sig).reset_index().sort_values(["active", "code"], ascending=[False, True])
    labels = (nm["code"].astype(str) + " " + nm["name"].fillna("").astype(str)
              + nm["active"].map({True: "", False: "（近 10 天無交易）"})).tolist()
    return labels, {lab.split()[0]: lab for lab in labels}


def _toggle_watch(code: str, name: str) -> None:
    """加入或移出 watchlist.txt（前端唯一會寫的設定檔），保留其他行和註解。"""
    path = daily.WATCHLIST
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    if code in daily.read_watchlist():
        lines = [ln for ln in lines if ln.split("#")[0].strip() != code]
        st.toast(f"已把 {code} {name} 移出觀察名單", icon=":material/check:")
    else:
        lines.append(f"{code}  # {name}" if name else code)
        st.toast(f"已把 {code} {name} 加入觀察名單", icon=":material/check:")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _picker(sig: str, default_asof: str):
    labels, by_code = _labels(sig)
    goto = st.session_state.pop("goto_code", None)
    if goto and goto in by_code:
        st.session_state["stock_pick"] = by_code[goto]
    elif "stock_pick" not in st.session_state:
        q = st.query_params.get("code")
        st.session_state["stock_pick"] = by_code.get(q) or by_code.get("2330") or labels[0]
    c1, c2, c3 = st.columns([3, 1, 0.7], vertical_alignment="bottom")
    chosen = c1.selectbox("股票（打代號或名稱篩選）", labels, key="stock_pick")
    asof = c2.date_input("資料時點", value=date.fromisoformat(default_asof), max_value=date.fromisoformat(default_asof),
                         help="選過去的日期做時點回溯：只用當時已公布的資料")
    code = chosen.split()[0]
    name = chosen[len(code):].replace("（近 10 天無交易）", "").strip()
    watched = code in daily.read_watchlist()
    c3.button("移出觀察" if watched else "加入觀察", icon=":material/star:" if watched else ":material/star_border:",
              on_click=_toggle_watch, args=(code, name), width="stretch", key="watch_toggle")
    if st.query_params.get("code") != code:
        st.query_params["code"] = code
    return code, asof


# ---------------------------------------------------------------- 頂部

def _hero(facts: dict, px: pd.DataFrame) -> None:
    p, ol, sc = facts.get("price") or {}, facts.get("outlook") or {}, facts.get("scores") or {}
    chg = None
    if px is not None and len(px) >= 2:
        chg = px["adj_close"].iloc[-1] / px["adj_close"].iloc[-2] - 1
    hz = ol.get("horizons") or {}
    lab_html = "".join(
        f'<div class="row"><span class="k">{common.esc(name)}</span>{common.badge(h.get("label") or "—")}'
        f'<span style="opacity:.6;font-size:.78rem">百分位 {h.get("pct") if h.get("pct") is not None else "—"}'
        f'・{common.esc(h.get("reliability") or "—")}</span></div>'
        for name, h in hz.items())
    comp = sc.get("composite_pct")
    comp_txt = "—" if comp is None else f"第 {comp * 100:.0f} 百分位"
    lo, hi, close = p.get("low_52w_adj"), p.get("high_52w_adj"), p.get("close")
    rng = ""
    if None not in (lo, hi, close) and hi > lo:
        pos = max(0.0, min(1.0, (close - lo) / (hi - lo))) * 100
        rng = (f'<div class="row"><span class="k">52 週</span>{common.num(lo)} – {common.num(hi)}'
               f'<div class="range"><i style="left:calc({pos:.0f}% - 1px)"></i></div></div>')
    st.markdown(
        f'<div class="hero"><div>'
        f'<div><span class="nm">{common.esc(p.get("name"))}</span><span class="cd">{common.esc(facts.get("code"))}'
        f' · {common.esc(p.get("market"))}</span></div>'
        f'<div><span class="px">{common.num(close)}</span><span class="chg">{common.signed_html(chg, 2)}</span></div>'
        f'<div class="meta">資料日 {common.esc(facts.get("asof"))}・漲跌為含息調整後日報酬</div>{rng}</div>'
        f'<div class="right"><div class="row"><span class="k">綜合分數</span><b>{comp_txt}</b></div>{lab_html}</div>'
        f'</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- 各區塊

def _price(facts: dict, px: pd.DataFrame):
    p = facts.get("price") or {}
    common.kpis([
        ("近 1 月", common.signed_html(p.get("ret_1m")), ""),
        ("近 3 月", common.signed_html(p.get("ret_3m")), ""),
        ("近 6 月", common.signed_html(p.get("ret_6m")), ""),
        ("近 12 月", common.signed_html(p.get("ret_12m")), ""),
        ("距 52 週高點", common.signed_html(p.get("from_52w_high")), ""),
        ("相對 200 日均線", common.signed_html(p.get("ma200_gap")), ""),
        ("60 日年化波動", common.pct(p.get("vol_60d_ann")).lstrip("+"), ""),
        ("近 1 年最大回落", common.pct(p.get("max_drawdown_1y")), ""),
        ("20 日均成交值", common.num(p.get("avg_value_20d_億")) + " 億", ""),
    ])
    st.caption("報酬都含息（用調整後價格）。")
    if px is None or px.empty:
        return
    s = px.set_index("date")["adj_close"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s, name="調整收盤", line=dict(color=common.C_STRAT, width=1.8)))
    fig.add_trace(go.Scatter(x=s.index, y=s.rolling(200).mean(), name="200 日均線",
                             line=dict(color=common.C_ETF, width=1.2, dash="dot")))
    fig.update_xaxes(rangeselector=dict(x=0.5, xanchor="center", y=1.02, yanchor="bottom", buttons=[
        dict(count=3, label="3 月", step="month", stepmode="backward"),
        dict(count=6, label="6 月", step="month", stepmode="backward"),
        dict(count=1, label="1 年", step="year", stepmode="backward"),
        dict(step="all", label="全部")]))
    common.chart(common.fig_layout(fig, "調整收盤（近 3 年）", 380))


def _outlook(facts: dict, hist: pd.DataFrame):
    ol = facts.get("outlook") or {}
    horizons = ol.get("horizons") or {}
    if not horizons:
        common.empty("沒有這個時點的模型標籤。")
        return
    cols = st.columns(len(horizons))
    for col, (name, h) in zip(cols, horizons.items()):
        same = h.get("history_same_label_oos") or {}
        by = h.get("history_by_label_oos") or {}
        with col.container(border=True):
            st.markdown(f"**{name}**　{common.badge(h.get('label') or '—')}", unsafe_allow_html=True)
            st.caption(f"{h.get('holding') or ''}・百分位 {h.get('pct') if h.get('pct') is not None else '—'}"
                       f"・可靠度 {h.get('reliability') or '—'}")
            if isinstance(same, dict) and same.get("mean_ex_pct") is not None:
                st.markdown(f"樣本外同標籤平均產業中性超額 **{same['mean_ex_pct']:+.2f}%**"
                            + (f"，跑贏同業比例 {same['hit_pct']:.0f}%" if same.get("hit_pct") is not None else ""))
            if by:
                df = pd.DataFrame({"標籤": list(by), "平均超額%": [v for v in by.values()]})
                common.table(common.styled(df, fmt={"平均超額%": "{:+.2f}"}, signed=["平均超額%"], labels=["標籤"]),
                             hide_index=True)
    notes = [x for x in (ol.get("method"), ol.get("hit_pct_note")) if x]
    if notes:
        st.caption("；".join(notes))
    if hist is not None and not hist.empty:
        series = {n: hist[k] for k, n in (("short_pct", "短期"), ("mid_pct", "中期"), ("long_pct", "長期")) if k in hist}
        fig = common.line_fig(hist["date"], series, "標籤用的百分位（80 以上看漲、20 以下看跌）", height=320)
        fig.add_hrect(y0=80, y1=100, fillcolor=common.UP, opacity=0.06, line_width=0)
        fig.add_hrect(y0=0, y1=20, fillcolor=common.DOWN, opacity=0.06, line_width=0)
        common.chart(fig)


def _technical(facts: dict):
    mx, pt, rp = facts.get("signal_matrix"), facts.get("patterns") or {}, facts.get("reference_prices") or {}
    if not mx:
        common.empty("沒有訊號矩陣。")
        return
    ag = mx["agreement"]
    hist = mx.get("history_oos_same_net_3dims") or {}
    h20 = hist.get("20 日") or {}
    common.kpis([
        ("一致數", f"{ag['net']:+d}", f"偏多 {ag['多']}・偏空 {ag['空']}・中性 {ag['中性']}", True),
        ("三維一致數", "—" if hist.get("net_3dims") is None else f"{hist['net_3dims']:+d}", "技術、基本面、綜合"),
        ("同三維一致數的歷史", common.signed_html(h20.get("mean_ex_pct"), 2, 1),
         "樣本外 20 日平均產業中性超額" + (f"・勝率 {h20['hit_pct']:.0f}%" if h20.get("hit_pct") is not None else "")),
    ])
    for col, (k, v) in zip(st.columns(len(mx["dimensions"])), mx["dimensions"].items()):
        with col.container(border=True):
            st.markdown(f"**{k}**　{common.badge(v['direction'] or '資料不足')}", unsafe_allow_html=True)
            st.caption(v["basis"])
    st.markdown("**矛盾清單**")
    st.markdown("\n".join(f"- {c}" for c in mx["contradictions"]) or "沒有")
    st.caption(mx["method"])

    st.markdown("**成立中的形態**")
    if pt.get("patterns"):
        df = pd.DataFrame([{"形態": x["name"], "方向": x["side"], "成立日": x["formed"],
                            "關鍵價位": "、".join(f"{k} {v:,.2f}" for k, v in x["levels"].items()),
                            "可靠度": x["reliability"] or "—",
                            "樣本外 20 日平均超額%": (x["history_oos"] or {}).get("mean_ex_pct"),
                            "勝率%": (x["history_oos"] or {}).get("hit_pct")} for x in pt["patterns"]])
        common.table(common.styled(df, fmt={"樣本外 20 日平均超額%": "{:+.2f}", "勝率%": "{:.0f}"},
                                   signed=["樣本外 20 日平均超額%"]), hide_index=True)
        st.caption(pt["note"])
    elif pt:
        st.caption(f"{pt['date']} 沒有成立中的形態。")
    if not pt and not rp:
        st.caption("形態與參考價只存最新資料日，這個時點沒有。")
    if rp:
        sr, sp, vb = rp["support_resistance"], rp["stop_reference"], rp["volatility_bands"]
        st.markdown(f"**參考價**（{rp['date']}，收盤 {rp['close']:,.2f}）")
        lv = lambda x: "—" if x["price"] is None else f"{x['price']:,.2f}"
        band = lambda b: "—" if b.get("low") is None else f"{b['low']:,.2f} – {b['high']:,.2f}"
        cov = lambda b: f"歷史覆蓋率 {common.plain(b.get('history_oos_inside_pct'), 1, '%')}" + (
            f"（同波動 {b['history_oos_inside_pct_same_vol']:.1f}%）"
            if b.get("history_oos_inside_pct_same_vol") is not None else "")
        hit = "、".join(f"{k} {v:.0f}%" for k, v in sp["history_oos_hit_pct"].items() if v is not None)
        common.kpis([
            ("支撐", lv(sr["support"]), sr["support"]["source"] or ""),
            ("壓力", lv(sr["resistance"]), sr["resistance"]["source"] or ""),
            ("停損參考", "—" if sp["price"] is None else f"{sp['price']:,.2f}", f"{sp['rule']}・歷史碰到 {hit}"),
            ("統計區間 20 日", band(vb["20 日"]), cov(vb["20 日"])),
            ("統計區間 60 日", band(vb["60 日"]), cov(vb["60 日"])),
        ])
        st.caption(f"{vb['note']}　{rp['disclaimer']}")


def _scores(facts: dict, trend: pd.Series):
    sc = facts.get("scores") or {}
    groups = sc.get("groups_pct") or {}
    left, right = st.columns([2, 3])
    with left:
        comp = sc.get("composite_pct")
        common.kpis([("綜合分數百分位", "—" if comp is None else f"{comp * 100:.0f}", "全市場排名，0–100，越高越好", True)])
        if groups:
            names, vals = list(groups), [None if v is None else v * 100 for v in groups.values()]
            fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=common.C_STRAT,
                                   text=[common.plain(v, 0) for v in vals], textposition="outside"))
            fig.update_xaxes(range=[0, 105])
            common.chart(common.fig_layout(fig, "四組百分位（產業與市值中性）", 260, legend=False))
        st.caption(f"分數日 {sc.get('score_date') or '—'}。百分位 0–100，越高越好。"
                   + ("" if sc.get("in_universe", True) else "這檔不在因子股票池內。"))
    with right:
        if trend is not None and len(trend):
            fig = common.line_fig(trend.index, {"綜合分數百分位": trend * 100}, "綜合分數百分位（近 36 期）",
                                  {"綜合分數百分位": common.C_STRAT}, 300)
            fig.update_yaxes(range=[0, 100])
            common.chart(fig)
        fac = sc.get("factors") or {}
        if fac:
            df = pd.DataFrame([{"因子": v.get("desc"), "組別": v.get("group"), "數值": v.get("value"),
                                "全市場百分位": None if v.get("pct_good") is None else v["pct_good"] * 100}
                               for v in fac.values()])
            common.table(df, hide_index=True, column_config={
                "全市場百分位": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")})


def _revenue(facts: dict, rv: pd.DataFrame):
    rev = facts.get("revenue") or {}
    if not rev:
        common.empty("這個時點之前沒有月營收。")
        return
    common.kpis([
        ("最新月份", common.esc(rev.get("latest_period") or "—"), f"{rev.get('published') or ''} 公布"),
        ("單月營收", common.num(rev.get("revenue_億")) + " 億", ""),
        ("年增", common.signed_html(rev.get("yoy_pct"), scale=1), ""),
        ("月增", common.signed_html(rev.get("mom_pct"), scale=1), ""),
        ("近 3 月年增", common.signed_html(rev.get("yoy_3m_pct"), scale=1), ""),
        ("近 12 月年增", common.signed_html(rev.get("yoy_12m_pct"), scale=1), ""),
        ("與歷史最高單月比", common.pct(rev.get("vs_record_high_pct"), scale=1), "0 = 歷史新高"),
    ])
    if rv is not None and not rv.empty:
        y = rv["以合併為主單月營收成長率％"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=rv["period"], y=rv["以合併為主單月營收(千元)"] / 1e5, name="營收（億）",
                             marker_color=common.C_POOL), secondary_y=False)
        fig.add_trace(go.Scatter(x=rv["period"], y=y, name="年增 %", mode="lines+markers",
                                 line=dict(color=common.C_STRAT, width=1.8)), secondary_y=True)
        fig.add_hline(y=0, line_width=1, line_dash="dot", opacity=.4, secondary_y=True)
        fig.update_yaxes(title_text="億", secondary_y=False)
        fig.update_yaxes(title_text="年增 %", tickmode="auto", showgrid=False, secondary_y=True)
        common.chart(common.fig_layout(fig, "月營收與年增率", 360))
    for n in rev.get("company_notes") or []:
        st.caption(f"{n.get('period')} 公司說明：{n.get('note')}")


def _financials(facts: dict, fn: pd.DataFrame):
    fin = facts.get("financials") or {}
    if not fin:
        common.empty("這個時點之前沒有季財報。")
        return
    yoy = fin.get("yoy_same_quarter") or {}
    common.kpis([
        ("最新季", common.esc(fin.get("latest_quarter") or "—"), f"{fin.get('published') or ''} 公布"),
        ("營收", common.num(fin.get("revenue_億")) + " 億", "年增 " + common.pct(yoy.get("revenue_pct"), scale=1)),
        ("毛利率", common.plain(fin.get("gross_margin_pct"), 1, "%"),
         "年增 " + common.plain(yoy.get("gross_margin_chg_pt"), 1, " 點") if yoy else ""),
        ("營益率", common.plain(fin.get("op_margin_pct"), 1, "%"), ""),
        ("EPS", common.num(fin.get("eps")), "去年同季 " + common.num(yoy.get("eps_prev")) if yoy else ""),
        ("近四季 EPS", common.num(fin.get("eps_ttm")), ""),
        ("單季 ROE", common.plain(fin.get("roe_quarter_pct"), 1, "%"), ""),
        ("負債比", common.plain(fin.get("debt_ratio_pct"), 1, "%"), ""),
    ])
    if fin.get("note"):
        st.caption(fin["note"])
    if fn is None or fn.empty:
        return
    q = fn["period"].dt.year.astype(str) + "Q" + fn["period"].dt.quarter.astype(str)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=q, y=fn["每股盈餘"], name="EPS", marker_color=common.C_POOL), secondary_y=False)
    for col, color in (("營業毛利率", common.C_STRAT), ("營業利益率", common.C_ETF)):
        fig.add_trace(go.Scatter(x=q, y=fn[col], name=col.replace("營業", "") + " %", mode="lines+markers",
                                 line=dict(color=color, width=1.8)), secondary_y=True)
    fig.update_yaxes(title_text="EPS（元）", secondary_y=False)
    fig.update_yaxes(title_text="%", tickmode="auto", showgrid=False, secondary_y=True)
    common.chart(common.fig_layout(fig, "EPS 與利潤率（近 8 季）", 340))
    cols = [c for c in ["period", "avail_date", "營業收入淨額", "營業毛利率", "營業利益率", "稅後淨利率", "每股盈餘",
                        "ROE(A)－稅後", "負債比率"] if c in fn.columns]
    show = fn[cols].copy()
    show.insert(0, "季", q.values)
    show = show.drop(columns=["period"]).rename(columns={"avail_date": "公布日", "營業收入淨額": "營收(億)"})
    show["營收(億)"] = show["營收(億)"] / 1e5
    show["公布日"] = pd.to_datetime(show["公布日"]).dt.strftime("%Y-%m-%d")
    common.table(common.styled(show.iloc[::-1], fmt={c: "{:,.2f}" for c in show.columns if c not in ("季", "公布日")}),
                 hide_index=True)


def _chips(facts: dict, ch: pd.DataFrame):
    flows = facts.get("institutional_flows") or {}
    if not flows:
        common.empty("沒有法人買賣資料。")
        return
    streak = flows.get("foreign_streak") or {}
    if streak:
        st.markdown(f"外資連續 **{streak.get('direction')} {streak.get('days')}** 個交易日"
                    "（只描述資金流向，不代表價格方向）")
    if ch is not None and not ch.empty and "total" in ch.columns:
        v = ch["total"] / 1000
        fig = go.Figure(go.Bar(x=ch["date"], y=v, marker_color=[common.UP if x > 0 else common.DOWN for x in v],
                               name="三大法人（張）"))
        common.chart(common.fig_layout(fig, "近 60 日三大法人買賣超（張）", 300, legend=False))
    rows = []
    for n in (5, 20, 60):
        block = flows.get(f"{n}d")
        if block:
            rows.append({"窗口": f"{n} 日", **{f"{lab}(張)": (block.get(k) or {}).get("net_lots") for k, lab in
                                                (("foreign", "外資"), ("trust", "投信"), ("dealer", "自營"), ("total", "合計"))},
                         "合計佔成交量%": (block.get("total") or {}).get("pct_of_volume")})
    if rows:
        df = pd.DataFrame(rows)
        num_cols = [c for c in df.columns if c.endswith("(張)")]
        common.table(common.styled(df, fmt={**{c: "{:+,.0f}" for c in num_cols}, "合計佔成交量%": "{:+.2f}"},
                                   signed=num_cols + ["合計佔成交量%"]), hide_index=True)


def _credit(facts: dict):
    cr = facts.get("margin_foreign_holding") or {}
    mg, qf = cr.get("margin") or {}, cr.get("foreign_holding") or {}
    if not mg and not qf:
        common.empty("沒有融資融券或外資持股資料。")
        return
    if mg:
        st.markdown(f"**融資融券**（{mg.get('last_date')}，單位：張）")
        common.kpis([
            ("融資餘額", common.num(mg.get("margin_balance"), 0), f"使用率 {common.plain(mg.get('margin_usage_pct'), 1, '%')}"),
            ("融資 5 日", common.signed_html(mg.get("margin_chg_5d"), 0, 1, ""), ""),
            ("融資 20 日", common.signed_html(mg.get("margin_chg_20d"), 0, 1, ""), ""),
            ("融券餘額", common.num(mg.get("short_balance"), 0), "20 日 " + common.num(mg.get("short_chg_20d"), 0)),
            ("券資比", common.plain(mg.get("short_to_margin_pct"), 1, "%"), ""),
        ])
    if qf:
        st.markdown(f"**外資持股**（{qf.get('last_date')}）")
        common.kpis([
            ("持股比率", common.plain(qf.get("foreign_pct"), 2, "%"), f"上限 {common.plain(qf.get('limit_pct'), 0, '%')}"),
            ("5 日", common.signed_html(qf.get("chg_5d_pt"), 2, 1, " 點"), ""),
            ("20 日", common.signed_html(qf.get("chg_20d_pt"), 2, 1, " 點"), ""),
            ("60 日", common.signed_html(qf.get("chg_60d_pt"), 2, 1, " 點"), ""),
            ("近 1 年區間", f"{common.plain(qf.get('low_1y_pct'), 1)} – {common.plain(qf.get('high_1y_pct'), 1)}%", ""),
        ])


def _announcements(facts: dict):
    ann = facts.get("announcements") or {}
    items = ann.get("items") or []
    st.caption(f"近 {ann.get('window_days', 90)} 天共 {ann.get('count', 0)} 則，下面是最近 {len(items)} 則（內文節錄）。")
    if not items:
        common.empty("這個窗口沒有重大訊息。")
        return
    for it in items:
        with st.expander(f"{it.get('time')}　{it.get('subject')}"):
            st.write(it.get("content_excerpt") or "（沒有內文）")


def _actions(facts: dict):
    rows = facts.get("corporate_actions") or []
    if not rows:
        st.caption("近期沒有除權息、減資或 ETF 分割。")
        return
    df = pd.DataFrame(rows).sort_values("date", ascending=False)
    common.table(df.rename(columns={"date": "日期", "kind": "類型", "prev_close": "前收盤", "ref": "參考價"}),
                 hide_index=True)


def _peers(sig: str, code: str):
    with st.spinner("同業比較…"):
        raw = json.loads(common.cached_peers(sig, code))
    if isinstance(raw, dict) and raw.get("error"):
        common.empty(raw["error"])
        return
    n_peers = raw.get("n_peers")
    tags = common.badge(raw.get("peer_level") or "", "muted") + (common.badge(f"{n_peers} 家", "muted") if n_peers else "")
    st.markdown(f"**{common.esc(raw.get('industry_name') or '')}**　{tags}", unsafe_allow_html=True)
    rows = raw.get("rows") or []
    rename = {"code": "代號", "name": "名稱", "產業名稱": "產業", "mcap_億": "市值(億)", "close": "收盤",
              "rev_yoy": "營收年增%", "rev_yoy_3m": "近3月年增%", "gross_margin": "毛利率%", "roe": "ROE%",
              "pe": "本益比", "pb": "股價淨值比", "composite_pct": "綜合分數"}
    if rows:
        df = pd.DataFrame(rows)
        keep = [c for c in rename if c in df.columns]
        show = df[keep].rename(columns=rename)
        sty = common.styled(show, fmt={"市值(億)": "{:,.0f}", "收盤": "{:,.2f}", "營收年增%": "{:+.1f}",
                                       "近3月年增%": "{:+.1f}", "毛利率%": "{:.1f}", "ROE%": "{:.1f}", "本益比": "{:.1f}",
                                       "股價淨值比": "{:.2f}", "綜合分數": "{:.0f}"},
                            signed=["營收年增%", "近3月年增%"])
        sty = sty.apply(lambda r: ["font-weight: 700; background: rgba(59,130,246,.10)" if r["代號"] == code else ""
                                   for _ in r], axis=1)
        # key 帶代號：換股後是新的表，不會沿用上一檔的選取又跳走
        ev = common.table(sty, hide_index=True, on_select="rerun", selection_mode="single-row", key=f"peers_{code}",
                          column_config={"綜合分數": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                                                  format="%d")})
        sel = ev.selection.rows if ev and ev.selection else []
        if sel and str(show.iloc[sel[0]]["代號"]) != code:
            common.goto_stock(str(show.iloc[sel[0]]["代號"]))
        st.caption("點一列打開那檔個股。")
    med = raw.get("median") or {}
    if med:
        st.caption("同業中位數：" + "、".join(f"{rename.get(k, k)} {common.plain(v, 1)}"
                                         for k, v in med.items() if v is not None))


def _performance(sig: str, code: str, asof: date):
    c1, c2, c3 = st.columns([1, 1, 2])
    with c3:
        preset = common.segmented("快速區間", ["1 月", "3 月", "6 月", "1 年", "3 年", "今年"], "1 年", key="perf_preset")
    days = {"1 月": 30, "3 月": 91, "6 月": 182, "1 年": 365, "3 年": 1095}
    start0 = date(asof.year, 1, 1) if preset == "今年" else asof - timedelta(days=days.get(preset, 365))
    start = c1.date_input("起日", value=start0, key=f"perf_start_{preset}")
    end = c2.date_input("迄日", value=asof, key="perf_end")
    if start > end:
        st.error("起日晚於迄日。")
        return
    raw = json.loads(common.cached_performance(sig, code, start.isoformat(), end.isoformat()))
    if raw.get("error"):
        common.empty(raw["error"])
        return
    common.kpis([
        ("區間", f"{raw.get('from')}<br>→ {raw.get('to')}", ""),
        ("含息報酬", common.signed_html(raw.get("total_return_incl_div")), ""),
        ("同期全市場等權", common.signed_html(raw.get("market_equal_weight_return")), "普通股平均"),
        ("最大回落", common.pct(raw.get("max_drawdown")), ""),
        ("收盤", f"{common.num(raw.get('close_start'))} → {common.num(raw.get('close_end'))}", "未調整"),
    ])


def _reports(code: str, facts: dict, series: dict):
    est = common.estimate_usd(HAIKU, len(json.dumps(facts, ensure_ascii=False, default=str)), common.REPORT_OUT_TOKENS)
    name = f"{code}_{facts.get('asof')}"
    exists = (report.REPORTS / f"{name}.html").exists()
    with st.container(border=True):
        st.markdown("**產生 LLM 文字報告**")
        st.caption(f"用 Haiku 依上面的事實表寫摘要，這份大約 {common.cents(est)}。輸出寫進 reports/{name}.html"
                   " 與 .json。不顯示金鑰。")
        if exists:
            st.warning(f"這個時點已經有報告（{name}.html，下面可以直接看）。再產生會花錢，並覆蓋舊的。",
                       icon=":material/info:")
        go_btn = st.button("重新產生並覆蓋" if exists else "產生報告", type="secondary" if exists else "primary",
                           icon=":material/auto_awesome:")
    if go_btn:
        with st.spinner("呼叫 Haiku…"):
            try:
                narrative = report.llm_narrative(facts)
            except Exception as e:
                st.error("沒有產生報告：" + common.public_error(e))
                narrative = None
        if not narrative:
            st.warning("沒有文字摘要（憑證、連線或被拒絕）。原因印在終端機，畫面上不印金鑰或回應內容。")
        else:
            report.REPORTS.mkdir(exist_ok=True)
            stem = report.REPORTS / f"{code}_{facts['asof']}"
            stem.with_suffix(".json").write_text(json.dumps({**facts, "narrative": narrative}, ensure_ascii=False,
                                                            indent=1), encoding="utf-8")
            stem.with_suffix(".html").write_text(report.render(facts, series, narrative), encoding="utf-8")
            st.toast(f"已寫入 {stem.name}.html", icon=":material/check:")
            with st.container(border=True):
                st.markdown(narrative.get("summary") or "")
                c1, c2, c3 = st.columns(3)
                for col, title, key in ((c1, "正面", "positives"), (c2, "負面", "negatives"), (c3, "下次可核對", "watch_items")):
                    col.markdown(f"**{title}**")
                    for it in narrative.get(key) or []:
                        col.markdown(f"- {it}")
                if narrative.get("outlook"):
                    st.markdown("**模型標籤**　" + narrative["outlook"])
                if narrative.get("technical"):
                    st.markdown("**技術面**　" + narrative["technical"])
    files = sorted(report.REPORTS.glob(f"{code}_*.html"))
    if not files:
        st.caption("這個代號還沒有報告。")
        return
    pick = st.selectbox("既有報告", [p.name for p in files][::-1])
    common.preview_html(report.REPORTS / pick)


# ---------------------------------------------------------------- 頁面

sig = common.data_sig()
status = common.load_status()
default_asof = status.get("last_price_date") or date.today().isoformat()
with st.spinner("讀股票清單…"):
    code, asof = _picker(sig, default_asof)

with st.spinner("整理事實表…"):
    facts, series = common.cached_facts(sig, code, asof.isoformat())
if isinstance(facts, dict) and facts.get("_error"):
    st.error(facts["_error"])
    st.stop()

_hero(facts, series.get("px"))
section = common.segmented("區塊", SECTIONS, "價格", key="stock_section", label_visibility="collapsed")
if section == "價格":
    _price(facts, series.get("px"))
    st.subheader("自訂區間報酬")
    _performance(sig, code, asof)
    st.subheader("除權息與減資")
    _actions(facts)
elif section == "模型看法":
    st.subheader("短中長期標籤")
    _outlook(facts, common.cached_outlook(sig, code))
    st.subheader("綜合分數")
    _scores(facts, series.get("trend"))
elif section == "訊號與形態":
    _technical(facts)
elif section == "營收財報":
    st.subheader("月營收")
    _revenue(facts, series.get("rv"))
    st.subheader("季財報")
    _financials(facts, series.get("fn"))
elif section == "籌碼":
    st.subheader("三大法人")
    _chips(facts, series.get("chips"))
    st.subheader("融資融券與外資持股")
    _credit(facts)
elif section == "重大訊息":
    _announcements(facts)
elif section == "同業":
    _peers(sig, code)
elif section == "報告":
    _reports(code, facts, series)
