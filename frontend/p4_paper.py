"""模擬持股：綜合分數前 30、多策略摘要、紀錄、淨值（淨值序列等後端）。"""
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common
import custom
import llm_replay
import llm_trader
import mine
import paper

STRAT_NAV = paper.PAPER / "strat_nav.parquet"


def _wait_entry(what: str, signal_date: str | None = None) -> None:
    """還沒進場時的說明。進場日看價格資料，不寫死日期。"""
    last = common.load_status().get("last_price_date")
    msg = f"{what}要等進場後才有：訊號日{f' {signal_date} ' if signal_date else ''}的下一個交易日收盤進場"
    if last and signal_date and str(last) <= str(signal_date):
        msg += f"，價格資料目前只到 {last}，先更新資料。"
    else:
        msg += f"（價格資料到 {last or '—'}）。"
    common.empty(msg)


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
            _wait_entry("累積淨值圖", s.get("signal_date"))
    with right:
        holds = pd.DataFrame(s.get("holdings") or [])
        if holds.empty:
            common.empty("沒有本期持股。")
        else:
            show = holds.rename(columns={"rank": "名次", "code": "代號", "name": "名稱",
                                         "ret_since_entry": "進場以來"})[["名次", "代號", "名稱", "進場以來"]]
            show["進場以來"] = pd.to_numeric(show["進場以來"], errors="coerce")
            ev = common.table(common.styled(show, fmt={"進場以來": "{:+.2%}"}, signed=["進場以來"]), hide_index=True,
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
                common.table(common.styled(df[["代號", "名稱", "市值(元)", "佔資產"]],
                                           fmt={"市值(元)": "{:,.0f}", "佔資產": "{:.1%}"}), hide_index=True)
            else:
                df = pd.DataFrame(x.get("targets") or [])
                if not df.empty:
                    st.markdown(f"本期目標權重（訊號日 {x.get('signal_date')}）")
                    df = df.rename(columns={"code": "代號", "name": "名稱", "weight": "目標權重"})
                    common.table(common.styled(df[["代號", "名稱", "目標權重"]], fmt={"目標權重": "{:.1%}"}),
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
    common.table(common.styled(show, fmt={"權重": "{:.2%}", "順序": "{:.0f}"}), hide_index=True)
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
        _wait_entry("淨值曲線")
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


_NAV_COLOR = {"自選（扣成本）": common.C_STRAT, "自選（不扣成本）": common.C_POOL,
              "我的組合": common.C_STRAT, "同組等權": common.C_POOL, paper.ETF: common.C_ETF}


def _weight_table(weights: pd.Series) -> None:
    names = custom.names_of(list(weights.index))
    df = pd.DataFrame({"代號": list(weights.index), "名稱": [names.get(c, "") for c in weights.index],
                       "目標權重": weights.values})
    common.table(common.styled(df, fmt={"目標權重": "{:.1%}"}), hide_index=True)


def _cmp_table(daily: pd.DataFrame, perf: dict, labels: list[tuple[str, str]]) -> None:
    rows = []
    for key, label in labels:
        p = perf.get(key) or {}
        rows.append({"組合": label, "總報酬": (1 + daily[key]).prod() - 1, "年化": p.get("cagr"),
                     "波動": p.get("vol"), "Sharpe": p.get("sharpe"), "最大回落": p.get("mdd")})
    df = pd.DataFrame(rows)
    common.table(common.styled(df, fmt={"總報酬": "{:+.1%}", "年化": "{:+.1%}", "波動": "{:.1%}",
                                        "Sharpe": "{:.2f}", "最大回落": "{:.1%}"},
                               signed=["總報酬", "年化"]), hide_index=True)


def _nav_chart(daily: pd.DataFrame, labels: list[tuple[str, str]], title: str, height: int = 380) -> None:
    nav = (1 + daily).cumprod()
    named = {label: nav[key] for key, label in labels}
    common.chart(common.line_fig(nav.index, named, title, {k: _NAV_COLOR[k] for k in named}, height))


_LLM_ROLES = ("LLM篩選", "量化全買", "隨機對照")
_LLM_LABEL = {"LLM篩選": "LLM 篩選", "量化全買": "量化全買", "隨機對照": "隨機對照"}
_LLM_COLOR = {"LLM 篩選": "#3b82f6", "量化全買": "#64748b", "隨機對照": "#a78bfa", paper.ETF: common.C_ETF}
_DEC_LABEL = {"buy": "買", "skip": "不買", "wait": "等待"}


def _is_llm_book(name: str) -> bool:
    return name == "量化全買" or name.startswith("LLM篩選-") or name.startswith("隨機對照-")


def _llm_groups(meta: dict) -> dict[str, dict[str, str]]:
    """{模型簡稱: {角色: 組合名稱}}。量化全買不分模型，每一組比較都帶上。"""
    groups: dict[str, dict[str, str]] = {}
    for name in meta:
        if name.startswith("LLM篩選-") or name.startswith("隨機對照-"):
            role, short = name.split("-", 1)
            groups.setdefault(short, {})[role] = name
    if "量化全買" in meta:
        if not groups:
            groups["量化"] = {"量化全買": "量化全買"}
        else:
            for g in groups.values():
                g["量化全買"] = "量化全買"
    return groups


def _model_short(model: str) -> str:
    parts = str(model).split("-")
    return parts[1] if len(parts) >= 2 else str(model)


def _book_sig() -> str:
    parts = [common.data_sig()]
    for p in (mine.LEDGER, mine.META, mine.PAPER / "llm_decisions.jsonl"):
        parts.append(str(p.stat().st_mtime_ns) if p.exists() else "0")
    return "|".join(parts)


@st.cache_data(show_spinner=False, max_entries=12)
def _cached_replay(name: str, sig: str):
    return mine.replay(name)


def _load_decisions() -> pd.DataFrame:
    path = mine.PAPER / "llm_decisions.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _book_weights(name: str) -> pd.Series:
    sigs = mine._latest_signals(mine.load_ledger(), name)
    if not sigs:
        return pd.Series(dtype=float)
    w = sigs[max(sigs)]
    return w[w > 0]


def _nav_of(ret: pd.Series) -> pd.Series:
    r = ret.dropna()
    return (1 + r).cumprod() if len(r) else r


def _text(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v)


def _lines(v) -> list:
    return list(v) if isinstance(v, (list, tuple)) else []


def _llm_compare(group: dict[str, str], sig: str) -> None:
    st.markdown("**三組對照**")
    ordered = [(role, group[role]) for role in _LLM_ROLES if role in group]
    if not ordered:
        common.empty("這三組模擬持股還沒建立。決策跑完、不加 --limit 時，llm_trader.py 會自己寫進帳本。")
        return
    results: dict[str, dict | None] = {}
    with st.spinner("重算三組淨值…"):
        for role, name in ordered:
            results[role] = _cached_replay(name, sig)
    filled = [(role, results[role]) for role, _ in ordered if results[role] is not None]
    waiting = [_LLM_LABEL[role] for role, res in results.items() if res is None]
    if not filled:
        common.empty("訊號已記下，等下一個交易日收盤進場後才有淨值。三組會和 0050 畫在同一張圖。")
    else:
        ref_role = next((r for r, _ in filled if r == "LLM篩選"), filled[0][0])
        ref = results[ref_role]
        navs = {_LLM_LABEL[role]: _nav_of(res["daily"]["strategy"]) for role, res in filled}
        navs[paper.ETF] = _nav_of(ref["daily"]["etf"])
        idx = None
        for s in navs.values():
            idx = s.index if idx is None else idx.union(s.index)
        idx = idx.sort_values()
        aligned = {k: v.reindex(idx) for k, v in navs.items()}
        common.chart(common.line_fig(idx, aligned, "累積淨值（起點 = 1）", _LLM_COLOR, 380))
        rows = []
        for role, res in filled:
            p = res["perf"].get("strategy") or {}
            rows.append({"組合": _LLM_LABEL[role], "總報酬": (1 + res["daily"]["strategy"]).prod() - 1,
                         "年化": p.get("cagr"), "波動": p.get("vol"), "Sharpe": p.get("sharpe"),
                         "最大回落": p.get("mdd"), "持股": int((_book_weights(group[role]) > 0).sum())})
        ep = ref["perf"].get("etf") or {}
        rows.append({"組合": paper.ETF, "總報酬": (1 + ref["daily"]["etf"]).prod() - 1,
                     "年化": ep.get("cagr"), "波動": ep.get("vol"), "Sharpe": ep.get("sharpe"),
                     "最大回落": ep.get("mdd"), "持股": pd.NA})
        common.table(common.styled(pd.DataFrame(rows), fmt={"總報酬": "{:+.1%}", "年化": "{:+.1%}", "波動": "{:.1%}",
                                                            "Sharpe": "{:.2f}", "最大回落": "{:.1%}", "持股": "{:.0f}"},
                                   signed=["總報酬", "年化"]), hide_index=True)
        note = f"0050 跟「{_LLM_LABEL[ref_role]}」同一段期間。不到 20 個交易日不計算年化、波動和 Sharpe。"
        if waiting:
            note += "還沒進場：" + "、".join(waiting) + "。"
        st.caption(note)
    held = {role: _book_weights(name) for role, name in ordered}
    codes = sorted({c for w in held.values() for c in w.index})
    if not codes:
        return
    names = custom.names_of(codes)
    show = pd.DataFrame({"代號": codes, "名稱": [names.get(c, "") for c in codes]})
    for role, w in held.items():
        show[_LLM_LABEL[role]] = [w.get(c, float("nan")) for c in codes]
    st.markdown("**目前目標權重**")
    ev = common.table(common.styled(show, fmt={_LLM_LABEL[r]: "{:.1%}" for r, _ in ordered}), hide_index=True,
                      height=320, on_select="rerun", selection_mode="single-row",
                      key=f"llm_holds_{'|'.join(group.values())}")
    st.caption("有數字的是這組目前持有、等權。點一列打開個股頁。")
    sel = ev.selection.rows if ev and ev.selection else []
    if sel:
        code = str(show.iloc[sel[0]]["代號"])
        if st.session_state.get("llm_hold_nav") != code:
            st.session_state["llm_hold_nav"] = code
            common.goto_stock(code)


def _llm_decisions(dec: pd.DataFrame, short: str) -> None:
    st.markdown("**決策紀錄**")
    if dec.empty or "model" not in dec.columns:
        common.empty("還沒有決策紀錄。")
        return
    view = dec[dec["model"].map(_model_short) == short].copy()
    if view.empty:
        common.empty(f"{short} 還沒有決策。")
        return
    view["決策"] = view["decision"].map(_DEC_LABEL).fillna(view["decision"])
    n = len(view)
    counts = view["決策"].value_counts()
    cost = pd.to_numeric(view["cost_usd"], errors="coerce").sum() if "cost_usd" in view else float("nan")
    common.kpis([
        ("決策", f"{n}", short),
        ("買", f"{int(counts.get('買', 0))}", ""),
        ("不買", f"{int(counts.get('不買', 0))}", ""),
        ("等待", f"{int(counts.get('等待', 0))}", ""),
        ("費用", f"US${cost:.2f}" if pd.notna(cost) else "—", "累計"),
    ])
    pick = common.segmented("決策篩選", ["全部", "買", "不買", "等待"], "全部",
                            key=f"llm_dec_filter_{short}", label_visibility="collapsed")
    if pick != "全部":
        view = view[view["決策"] == pick]
    view = view.sort_values("time", ascending=False).reset_index(drop=True)
    if view.empty:
        common.empty(f"沒有「{pick}」。")
        return
    show = pd.DataFrame({
        "資料日": view["asof"],
        "代號": view["code"],
        "名稱": view["name"],
        "營收月": view["period"],
        "年增": pd.to_numeric(view["yoy"], errors="coerce") / 100,
        "決策": view["決策"],
        "信心": pd.to_numeric(view["conviction"], errors="coerce"),
        "摘要": view["summary"].fillna("").map(lambda s: s if len(str(s)) <= 72 else str(s)[:72] + "…"),
    })
    ev = common.table(common.styled(show, fmt={"年增": "{:.1%}", "信心": "{:.0f}"}), hide_index=True, height=420,
                      on_select="rerun", selection_mode="single-row", key=f"llm_dec_{short}_{pick}")
    st.caption("點一列看理由與風險。紀錄只追加，不在這裡重跑。")
    sel = ev.selection.rows if ev and ev.selection else []
    if not sel:
        return
    row = view.iloc[sel[0]]
    with st.container(border=True):
        st.markdown(f"<b>{common.esc(row['code'])} {common.esc(row['name'])}</b>　{common.esc(row['決策'])}"
                    f"　信心 {common.esc(row['conviction'])}　{common.esc(row['period'])} 營收",
                    unsafe_allow_html=True)
        st.markdown(f"<p>{common.esc(_text(row.get('summary')))}</p>", unsafe_allow_html=True)
        left, right = st.columns(2)
        for col, title, key in ((left, "理由", "reasons"), (right, "風險", "risks")):
            body = "".join(f"<li>{common.esc(x)}</li>" for x in _lines(row.get(key))) or "<li>—</li>"
            col.markdown(f"<b>{title}</b><ul>{body}</ul>", unsafe_allow_html=True)
        if row["決策"] == "等待" and _text(row.get("wait_for")):
            st.caption("要等：" + _text(row.get("wait_for")))
        if st.button(f"打開 {row['code']} {row['name']}", icon=":material/arrow_forward:", key="llm_goto"):
            common.goto_stock(str(row["code"]))


def _llm() -> None:
    st.caption("量化規則先挑候選（月營收創高且年增 30% 以上），LLM 再決定買、不買或等待。"
               "三組都是等權、持有約 60 個交易日、扣成本。只有 LLM 組穩定贏過量化全買和隨機對照，才代表有加分。"
               "由 llm_trader.py 寫入，這裡不能改。")
    meta = {k: v for k, v in mine.load_meta().items() if not v.get("archived")}
    groups = _llm_groups(meta)
    dec = _load_decisions()
    shorts = list(groups)
    if not dec.empty and "model" in dec.columns:
        for s in dec["model"].map(_model_short):
            if s not in shorts:
                shorts.append(s)
    if not shorts:
        common.empty("還沒有 LLM 決策。命令列先看候選，再用 run --dry-run 估價。")
        return
    short = shorts[0]
    if len(shorts) > 1:
        short = common.segmented("模型", shorts, short, key="llm_model")
    _llm_compare(groups.get(short, {}), _book_sig())
    models = sorted({m for m in dec.get("model", pd.Series(dtype=str)) if _model_short(m) == short})
    if models:
        _llm_selection(models[0])
    _llm_replay(next(iter(models), llm_trader.A.MODELS.get(short, short)))
    _llm_decisions(dec, short)


@st.cache_data(show_spinner="計算選股檢定…", max_entries=6)
def _cached_selection(model: str, sig: str) -> dict | None:
    dec = _load_decisions()
    return llm_trader.selection_test(dec[dec["model"] == model].reset_index(drop=True), model)


@st.cache_data(show_spinner="計算回放結果…", max_entries=6)
def _cached_replay_eval(model: str, sig: str) -> dict | None:
    return llm_replay.evaluate(model)


def _replay_sig() -> str:
    return common.data_sig() + "|" + (str(llm_replay.LOG.stat().st_mtime_ns) if llm_replay.LOG.exists() else "0")


def _pctile_kpi(t: dict) -> tuple:
    return ("LLM 在隨機分布的百分位", f"{t['pctile'] * 100:.0f}", f"單尾 p = {t['p_one_sided']:.2f}", t["p_one_sided"] < 0.05)


def _llm_selection(model: str) -> None:
    st.markdown("**選股檢定**")
    t = _cached_selection(model, _book_sig())
    if t is None:
        common.empty("還沒有成交後的報酬。成交日之後更新資料，這裡就會出現。")
        return
    common.kpis([("LLM buy 平均報酬", common.signed_html(t["llm"]), f"{t['n_buy']} 檔"),
                 ("全部候選", common.signed_html(t["quant"]), f"{t['n_all']} 檔"),
                 ("隨機挑同樣檔數（中位數）", common.signed_html(t["rand_p50"]),
                  f"90% 區間 {t['rand_p5']:+.1%} ~ {t['rand_p95']:+.1%}"),
                 _pctile_kpi(t)])
    st.caption(f"決策後第一個交易日收盤進場，持有中位數 {t['held_days']} 個交易日（未滿 60 日的算到最新資料日）；"
               f"不扣成本、等權。隨機分布 = 每批從候選隨機挑和 LLM 一樣多檔，重複 {llm_trader.N_RANDOM} 次。"
               "百分位 95 以上（p < 0.05）才算 LLM 挑得比隨機好；持有天數太短時結果主要是雜訊。")


def _llm_replay(model: str) -> None:
    st.markdown("**準前瞻回放（2026-07～09）**")
    res = _cached_replay_eval(model, _replay_sig())
    if res is None:
        common.empty("還沒有回放紀錄。命令列跑 uv run llm_replay.py run --dry-run 估價，再跑 run。")
        return
    st.caption("模型知識截止 2026-06，拿之後三個月的事件、用當時的事實表重跑同一套決策。"
               f"共 {len(res['ev'])} 檔，花費 US${res['cost']:.2f}。報酬從批次資料時點隔一個交易日收盤起算，不扣成本、等權。")
    hs = [h for h in llm_replay.HORIZONS if h in res["tests"]]
    if not hs:
        common.empty("回放的報酬期間還不夠長。")
        return
    h = common.segmented("天期", [f"{x} 日" for x in hs], f"{hs[0]} 日", key=f"replay_h_{model}")
    t = res["tests"][int(h.split()[0])]
    common.kpis([("LLM buy（選股）", common.signed_html(t["llm"]), f"{t['n_buy']} 檔，含等待時點 {t['llm_timed']:+.1%}"),
                 ("全部候選", common.signed_html(t["all"]), f"{t['n']} 檔，對 0050 {t['x_all']:+.1%}"),
                 ("隨機挑同樣檔數（中位數）", common.signed_html(t["rand_p50"]),
                  f"90% 區間 {t['rand_p5']:+.1%} ~ {t['rand_p95']:+.1%}"),
                 _pctile_kpi(t)])
    st.caption(f"{t['n_batch']} 批有完整的 {h}報酬。LLM 沒買的平均 {t['not_buy']:+.1%}；"
               f"全部候選贏 0050 的比例 {t['win_etf']:.0%}。")
    b = res["batches"].rename(columns={"batch": "批次資料時點", "n": "候選", "n_buy": "LLM buy", "all": "全部",
                                       "buy": "LLM buy 報酬", "not_buy": "沒買的", "x_all": "全部對 0050"})
    common.table(common.styled(b, fmt={c: "{:+.1%}" for c in ("全部", "LLM buy 報酬", "沒買的", "全部對 0050")},
                               signed=["全部", "LLM buy 報酬", "沒買的", "全部對 0050"]), hide_index=True)
    st.caption("各批都是 20 日報酬。三批方向不一致時，代表 LLM 的篩選沒有穩定的效果。")


def _cost_line(res: dict) -> str:
    t = res["trades"]
    blocked = [f"{r['exe']:%Y-%m-%d} 未成交：{'、'.join(r['blocked'])}" for _, r in t.iterrows() if r["blocked"]]
    line = (f"期末資產 {res['nav'].iloc[-1]:,.0f} 元・手續費 {t['fees'].sum():,.0f}・證交稅 {t['tax'].sum():,.0f}・"
            f"成交 {len(t)} 次")
    return line + ("・" + "；".join(blocked) if blocked else "")


def _custom() -> None:
    st.caption("自己挑股票和起訖日回頭重播。進場是開始日的下一個交易日收盤，成本、零股取整、漲跌停同回測。"
               "代號用逗號分隔，可加 :權重（例 2330:50,2317:30）。不寫檔。")
    last = common.load_status().get("last_price_date")
    end0 = pd.Timestamp(last) if last else pd.Timestamp.today().normalize()
    c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1.2])
    spec = c1.text_input("股票代號", "2330,2317,2454", key="cu_codes")
    start = c2.date_input("開始日", pd.Timestamp("2023-01-03"), key="cu_start")
    end = c3.date_input("結束日", end0, max_value=end0, key="cu_end")
    cap = c4.number_input("本金（元）", 10_000, 100_000_000, 1_000_000, 10_000, key="cu_cap")
    mode = common.segmented("調整方式", ["買進持有", "每月調回目標權重"], "買進持有", key="cu_mode")
    if st.button("開始回測", icon=":material/play_arrow:", key="cu_run"):
        try:
            with st.spinner("回測中…"):
                st.session_state["cu_res"] = custom.run(
                    custom.parse_codes(spec), start, end, "none" if mode == "買進持有" else "monthly", float(cap))
                st.session_state["cu_key"] = (spec, str(start), str(end), mode, float(cap))
        except ValueError as e:
            st.warning(str(e))
            return
        except Exception as e:
            st.error("回測失敗：" + common.scrub(e))
            return
    res = st.session_state.get("cu_res")
    if not res:
        common.empty("填好代號和日期後按開始回測。")
        return
    key = (spec, str(start), str(end), mode, float(cap))
    if st.session_state.get("cu_key") != key:
        st.caption("上面的條件已改，下面還是上一次的結果。")
    if res["missing"]:
        st.warning("期間內沒有行情，已略過：" + "、".join(res["missing"]))
    labels = [("strategy", "自選（扣成本）"), ("pool", "自選（不扣成本）"), ("etf", paper.ETF)]
    p = res["perf"].get("strategy") or {}
    common.kpis([
        ("總報酬", common.signed_html((1 + res["daily"]["strategy"]).prod() - 1, 2), "扣成本"),
        ("年化", common.signed_html(p.get("cagr")), "不到 20 個交易日不算" if not p else ""),
        ("Sharpe", common.plain(p.get("sharpe"), 2), ""),
        ("最大回落", common.pct(p.get("mdd")), f"{p.get('mdd_days', '—')} 個交易日沒回到高點" if p else ""),
        ("期末資產", f"{res['nav'].iloc[-1]:,.0f}", "元"),
    ])
    _nav_chart(res["daily"], labels, "累積淨值（起點 = 1）")
    _cmp_table(res["daily"], res["perf"], labels)
    st.markdown("**目標權重**")
    _weight_table(res["weights"])
    st.caption(_cost_line(res) + "。不扣成本的那條是同一組目標權重買進持有。")


def _mine() -> None:
    st.caption("自己挑股票，從最新資料日起往後記錄（不補過去；要看過去用「自選股回測」）。"
               "隔一個交易日收盤進場。月調組合在新月份由 update.py 調回目標權重，其餘只在你改持股時調整。"
               "紀錄寫進 data/paper/mine_ledger.parquet，只追加。")
    meta = {k: v for k, v in mine.load_meta().items() if not v["archived"]}
    with st.expander("建立新組合"):
        c1, c2, c3 = st.columns([1.5, 3, 1.2])
        name = c1.text_input("組合名稱", key="mn_name")
        spec = c2.text_input("股票代號（可加 :權重）", "2330,2317,2454", key="mn_spec")
        cap = c3.number_input("本金（元）", 10_000, 100_000_000, 1_000_000, 10_000, key="mn_cap")
        monthly = st.checkbox("每月調回目標權重", key="mn_monthly")
        if st.button("建立", icon=":material/add:", key="mn_create"):
            if _is_llm_book(name.strip()):
                st.warning("這個名稱是 LLM 實驗用的，請換一個。")
            else:
                try:
                    sig = mine.create(name.strip(), custom.parse_codes(spec), monthly, float(cap))
                    st.success(f"已建立，訊號日 {sig:%Y-%m-%d}，隔一個交易日收盤進場。")
                    st.rerun()
                except ValueError as e:
                    st.warning(str(e))
                except Exception as e:
                    st.error("建立失敗：" + common.scrub(e))
    if not meta:
        common.empty("還沒有自選組合。")
        return
    names = [k for k in meta if not _is_llm_book(k)] + [k for k in meta if _is_llm_book(k)]
    name = st.selectbox("組合", names, key="mn_pick")
    m = meta[name]
    sigs = mine._latest_signals(mine.load_ledger(), name)
    cur = sigs[max(sigs)]
    common.kpis([
        ("建立日", common.esc(m["created"]), f"本金 {m['capital']:,.0f} 元"),
        ("調整", "每月調回" if m["rebalance"] == "monthly" else "只在改動時", f"{len(sigs)} 次訊號"),
        ("最近訊號", f"{max(sigs):%Y-%m-%d}", "隔一個交易日收盤成交"),
    ])
    st.markdown("**目前目標權重**")
    _weight_table(cur)
    with st.spinner("重算淨值…"):
        res = mine.replay(name)
    if res is None:
        common.empty("訊號已記下，等下一個交易日收盤進場後才有淨值。")
    else:
        labels = [("strategy", "我的組合"), ("pool", "同組等權"), ("etf", paper.ETF)]
        p = res["perf"].get("strategy") or {}
        common.kpis([
            ("累積報酬", common.signed_html((1 + res["daily"]["strategy"]).prod() - 1, 2), "扣成本"),
            ("同組等權", common.signed_html((1 + res["daily"]["pool"]).prod() - 1, 2), "不扣成本"),
            (paper.ETF, common.signed_html((1 + res["daily"]["etf"]).prod() - 1, 2), "不扣成本"),
            ("最大回落", common.pct(p.get("mdd")), "" if p else "不到 20 個交易日不算"),
            ("期末資產", f"{res['nav'].iloc[-1]:,.0f}", "元"),
        ])
        _nav_chart(res["daily"], labels, "累積淨值（起點 = 1）", 340)
        _cmp_table(res["daily"], res["perf"], labels)
        st.caption(_cost_line(res) + "。同組等權不扣成本。")
    if len(sigs) > 1:
        with st.expander(f"歷次訊號（{len(sigs)}）"):
            rows = [{"訊號日": f"{s:%Y-%m-%d}", "代號": c, "目標權重": float(w)}
                    for s, weights in sigs.items() for c, w in weights.items()]
            common.table(common.styled(pd.DataFrame(rows), fmt={"目標權重": "{:.1%}"}), hide_index=True)
    if _is_llm_book(name):
        st.caption("這組由 llm_trader.py 管理，不能改持股，也不能封存。三組比較和每檔理由在「LLM 決策」。")
    else:
        _mine_edit(name, cur)


def _mine_edit(name: str, cur: pd.Series) -> None:
    with st.expander("改持股 / 封存"):
        spec = st.text_input("新的持股（代號:權重）", ",".join(f"{c}:{w * 100:.4g}" for c, w in cur.items()),
                             key=f"mn_edit_{name}")
        c1, c2 = st.columns(2)
        if c1.button("記錄新持股", key=f"mn_save_{name}"):
            try:
                sig = mine.edit(name, custom.parse_codes(spec))
                st.success(f"已記錄，訊號日 {sig:%Y-%m-%d}。")
                st.rerun()
            except ValueError as e:
                st.warning(str(e))
            except Exception as e:
                st.error("記錄失敗：" + common.scrub(e))
        if c2.button("封存這個組合", key=f"mn_arch_{name}"):
            st.session_state["mn_arch_ask"] = name
        if st.session_state.get("mn_arch_ask") == name:
            st.warning("封存後不再更新，已有紀錄保留。")
            if st.button("確定封存", key=f"mn_arch_ok_{name}"):
                mine.archive(name)
                st.session_state.pop("mn_arch_ask", None)
                st.rerun()


sig = common.data_sig()
summary = common.cached_paper_summary(sig)
strats = paper.load_strategy_summaries()
common.page_title("模擬持股", f"即時紀錄，起點之前沒有資料・多策略 {len(strats)} 個")

section = common.segmented(
    "區塊", ["綜合分數前 30", "多策略", "各策略淨值", "每期紀錄", "自選股回測", "自選組合", "LLM 決策", "追蹤設定"],
    "綜合分數前 30", key="paper_section", label_visibility="collapsed")
if section == "綜合分數前 30":
    _top30(summary)
elif section == "多策略":
    _multi(strats)
elif section == "各策略淨值":
    _strat_nav()
elif section == "每期紀錄":
    _ledger()
elif section == "自選股回測":
    _custom()
elif section == "自選組合":
    _mine()
elif section == "LLM 決策":
    _llm()
elif section == "追蹤設定":
    st.caption("paper_strategies.txt。這裡只顯示；改了要等下個月才會記新策略。")
    if paper.STRATS_FILE.exists():
        st.code(paper.STRATS_FILE.read_text(encoding="utf-8"), language="text")
    else:
        common.empty("沒有 paper_strategies.txt。")
