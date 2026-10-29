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
                with st.expander(f"目前持股 {len(df)} 檔"):
                    common.table(common.styled(df[["代號", "名稱", "市值(元)", "佔資產"]],
                                               fmt={"市值(元)": "{:,.0f}", "佔資產": "{:.1%}"}), hide_index=True)
            else:
                df = pd.DataFrame(x.get("targets") or [])
                if not df.empty:
                    df = df.rename(columns={"code": "代號", "name": "名稱", "weight": "目標權重"})
                    with st.expander(f"本期目標權重 {len(df)} 檔（訊號日 {x.get('signal_date')}）"):
                        common.table(common.styled(df[["代號", "名稱", "目標權重"]], fmt={"目標權重": "{:.1%}"}),
                                     hide_index=True, height=240)


def _ledger() -> None:
    if not paper.STRAT_LEDGER.exists():
        common.empty("還沒有每期紀錄。")
        return
    led = pd.read_parquet(paper.STRAT_LEDGER)
    title = {x["id"]: x.get("name") or x["id"] for x in paper.load_strategy_summaries()}
    c1, c2, c3 = st.columns([2, 2, 1])
    names = sorted(led["strategy"].dropna().unique())
    pick = c1.multiselect("策略", names, default=names, format_func=lambda k: title.get(k, k))
    date_labels = [str(pd.Timestamp(d).date()) for d in sorted(led["signal_date"].dropna().unique())]
    chosen = c2.multiselect("訊號日", date_labels, default=date_labels[-1:])
    only_pick = c3.toggle("只看選進組合的", value=True)
    view = led[led["strategy"].isin(pick)]
    if chosen:
        view = view[view["signal_date"].isin({pd.Timestamp(d) for d in chosen})]
    if only_pick and "in_pick" in view.columns:
        view = view[view["in_pick"].astype(bool)]
    view = view.sort_values([c for c in ["strategy", "signal_date", "pick_order"] if c in view.columns])
    names_of = custom.names_of(sorted(view["code"].astype(str).unique()))
    show = pd.DataFrame({"訊號日": pd.to_datetime(view["signal_date"]).dt.strftime("%Y-%m-%d"),
                         "策略": view["strategy"].map(lambda k: title.get(k, k)),
                         "代號": view["code"].astype(str), "名稱": view["code"].astype(str).map(names_of),
                         "權重": view["weight"]})
    if "pick_order" in view:
        show["順序"] = view["pick_order"] + 1  # 存的是 0 起算
    if not only_pick and "in_pick" in view:
        show["選進組合"] = view["in_pick"].map({True: "是", False: ""})
    if "regime_on" in view and not view["regime_on"].all():  # 沒有濾網的策略永遠是 True，全 True 就不列
        show["大盤濾網"] = view["regime_on"].map({True: "通過", False: "未通過，降倉"})
    common.table(common.styled(show, fmt={"權重": "{:.2%}", "順序": "{:.0f}"}), hide_index=True)
    st.caption(f"{len(show):,} 列。紀錄只追加不改寫；策略設定改過會重新起算。")


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
        return
    st.subheader("各策略淨值")
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


_RISK_INDEX = {"none": "不用", "0050": "0050 跌破均線", "ew": "全市場等權指數跌破均線"}
_STOP_TYPE = {"trailing": "從進場後高點算", "fixed": "從進場價算"}


@st.cache_data(show_spinner=False, max_entries=16)
def _source_codes(_sig: str, source: str) -> list[str]:
    return mine.source_codes(source)


def _mine_create() -> None:
    """建立自選組合：名單（自己輸入或從來源帶入）、加權方式、選用的風險規則。"""
    sources = [None, *mine.SOURCES, *(f"strategies/{p.name}" for p in sorted((mine.ROOT / "strategies").glob("*.toml")))]
    c1, c2, c3 = st.columns([1.5, 3, 1.2])
    name = c1.text_input("組合名稱", key="mn_name")
    source = c2.selectbox("名單來源", sources, format_func=mine.source_label, key="mn_source",
                          help="從策略帶入只用它的選股條件；倉位與風險規則用下面的設定")
    cap = c3.number_input("本金（元）", 10_000, 100_000_000, 1_000_000, 10_000, key="mn_cap")
    if source is None:
        spec = st.text_input("股票代號（可加 :權重）", "2330,2317,2454", key="mn_spec")
    else:
        try:
            codes = _source_codes(common.data_sig(), source)
            names = custom.names_of(codes) if codes else {}
            st.caption(f"目前 {len(codes)} 檔：" + "、".join(f"{c} {names.get(c, '')}" for c in codes[:30])
                       + ("…" if len(codes) > 30 else "") if codes else "這個來源目前沒有股票。")
        except Exception as e:
            st.warning("讀不到名單：" + common.scrub(e))
    event = source in mine.EVENT_SOURCES
    if event:
        st.caption("財報公布後追價：驚喜分數在最近 12 個月公布事件的最高 10%、上市且成交量不在最低 20% 的股票，"
                   "公布後隔一個交易日收盤買進，持有期滿賣出；每天收盤後名單有變就調整。"
                   "2018–2025 回測與 2026 年樣本外都贏同條件的一般股票，但沒有贏 0050（規則見 surprise.py）。")
    weightings = list(mine.WEIGHTINGS) if source is None else [k for k in mine.WEIGHTINGS if k != "manual"]
    c1, c2, c3 = st.columns([2, 1, 2], vertical_alignment="bottom")
    weighting = c1.selectbox("加權方式", weightings, format_func=mine.WEIGHTINGS.get, key=f"mn_w_{source is None}")
    max_weight = c2.number_input("單檔上限", 0.05, 1.0, 1.0, 0.05, format="%.2f", key="mn_mw",
                                 help="0–1；1 = 不限，超過的分給其他股票，都到上限就留現金")
    monthly = False if event else c3.checkbox("每月照規則重選並調回權重" if source else "每月調回目標權重", key="mn_monthly")

    risk = {}
    if st.toggle("加上風險規則（停損、停利、大盤濾網…）", key="mn_risk_on"):
        frac = "0–1，例 0.10 = 10%；0 = 不用"
        c1, c2, c3, c4 = st.columns(4)
        risk["stop_loss"] = c1.number_input("個股停損", 0.0, 1.0, 0.0, 0.01, format="%.2f", key="mn_sl", help=frac)
        risk["stop_type"] = c2.selectbox("停損怎麼算", list(_STOP_TYPE), format_func=_STOP_TYPE.get, key="mn_st")
        risk["take_profit"] = c3.number_input("個股停利", 0.0, 10.0, 0.0, 0.05, format="%.2f", key="mn_tp",
                                              help="從進場價算；" + frac)
        risk["max_hold_days"] = int(c4.number_input("持有滿幾個交易日出場", 0, 1000, 0, 5, key="mn_hd",
                                                    help="0 = 不用；賣掉的到下次調整才會再買"))
        c1, c2, c3, c4 = st.columns(4)
        idx = c1.selectbox("大盤濾網", list(_RISK_INDEX), format_func=_RISK_INDEX.get, key="mn_rg")
        ma = int(c2.number_input("均線天數", 5, 400, 200, 10, key="mn_ma"))
        daily = c3.selectbox("多久判斷一次", ["daily", "monthly"], key="mn_rf",
                             format_func={"daily": "每天", "monthly": "每次調整時"}.get)
        off = c4.number_input("跌破時留多少部位", 0.0, 1.0, 0.0, 0.1, key="mn_off", help="0–1；站回均線再買回")
        if idx != "none":
            risk.update(regime="ma", regime_index=idx, regime_ma=ma, regime_freq=daily, regime_off=float(off))
        c1, c2, c3, _ = st.columns(4)
        risk["dd_limit"] = c1.number_input("資產回落門檻", 0.0, 1.0, 0.0, 0.01, format="%.2f", key="mn_dd",
                                           help="組合從高點回落超過就降部位；" + frac)
        risk["dd_exposure"] = c2.number_input("回落後留多少部位", 0.0, 1.0, 0.0, 0.1, key="mn_de",
                                              help="0–1；到下次調整才恢復")
        risk["target_vol"] = c3.number_input("目標年化波動", 0.0, 1.0, 0.0, 0.01, format="%.2f", key="mn_tv",
                                             help="持股過去 60 日波動太大就降部位；" + frac)
        st.caption("每天收盤判斷，隔一個交易日收盤成交（跌停賣不掉就再等一天）；規則建立後不能改，要換就建新組合。")

    settings = dict(monthly=monthly, capital=float(cap), weighting=weighting, max_weight=float(max_weight),
                    risk=risk, source=source)
    c1, c2, c3, _ = st.columns([1.2, 1.2, 1.2, 2], vertical_alignment="bottom")
    start = c1.date_input("回測起日", pd.Timestamp.today() - pd.DateOffset(years=3), key="mn_bt_start",
                          min_value=pd.Timestamp("2012-01-01"), max_value=pd.Timestamp.today())
    if c2.button("先回測過去", icon=":material/history:", key="mn_bt"):
        try:
            w = custom.parse_codes(spec) if source is None else None
            with st.spinner("用同一套規則回頭重播…"):
                st.session_state["mn_bt_res"] = (repr((spec if source is None else None, start, settings)),
                                                 mine.backtest(w, start, **settings))
        except ValueError as e:
            st.warning(str(e))
        except Exception as e:
            st.error("回測失敗：" + common.scrub(e))
    if c3.button("建立", icon=":material/add:", key="mn_create"):
        if _is_llm_book(name.strip()):
            st.warning("這個名稱是 LLM 實驗用的，請換一個。")
            return
        try:
            w = custom.parse_codes(spec) if source is None else None
            sig = mine.create(name.strip(), w, **settings)
            st.success(f"已建立，訊號日 {sig:%Y-%m-%d}，隔一個交易日收盤進場。")
            st.session_state.pop("mn_bt_res", None)
            st.rerun()
        except ValueError as e:
            st.warning(str(e))
        except Exception as e:
            st.error("建立失敗：" + common.scrub(e))
    if "mn_bt_res" in st.session_state:
        key, res = st.session_state["mn_bt_res"]
        if key != repr((spec if source is None else None, start, settings)):
            st.caption("設定改過了，下面是上一次設定的回測；按「先回測過去」重算。")
        _mine_backtest(res, source)


def _mine_backtest(res: dict, source: str | None) -> None:
    """建立前回測的結果：同一套規則、不加風險規則、同組等權、0050。"""
    d, perf = res["daily"].copy(), dict(res["perf"])
    labels = [("strategy", "這套規則")]
    if res.get("base"):
        d["base"], perf["base"] = res["base"]["daily"]["strategy"].reindex(d.index).fillna(0), res["base"]["perf"]["strategy"]
        labels.append(("base", "不加風險規則"))
    labels += [("pool", "同組等權"), ("etf", paper.ETF)]
    nv = res["nav"]
    st.markdown(f"**回測 {nv.index[0]:%Y-%m-%d} → {nv.index[-1]:%Y-%m-%d}**（{len(res['signals'])} 次訊號・"
                f"{mine.REBALANCE[res['rebalance']]}）")
    common.nav_chart(d, labels, "累積淨值（起點 = 1）", 320)
    common.cmp_table(d, perf, labels)
    t = res["trades"]
    note = (f"這套規則與不加風險規則都扣成本（手續費 {t['fees'].sum():,.0f}、證交稅 {t['tax'].sum():,.0f} 元）；"
            "同組等權是每個訊號日的名單等權、不扣成本、漲停也買得到。每個訊號日只用當時的資料選股和算權重。")
    if not source or source == "watchlist":
        note += "注意：名單是現在挑的，回測過去有後見之明，只能參考。"
    st.caption(note)


def _mine() -> None:
    st.caption("自己挑股票，從最新資料日起往後記錄（不補過去；建立前可按「先回測過去」看同一套規則以前的表現）。"
               "隔一個交易日收盤進場。月調組合在新月份、財報驚喜組合在名單有變時由 update.py 記新一期，其餘只在你改持股時調整。"
               "紀錄只追加，不改寫。")
    meta = {k: v for k, v in mine.load_meta().items() if not v["archived"]}
    own, llm = [k for k in meta if not _is_llm_book(k)], [k for k in meta if _is_llm_book(k)]
    with st.expander("建立新組合", expanded=not own):
        _mine_create()
    if not own:
        common.empty("還沒有自己的組合，用上面的「建立新組合」建一個。")
    # LLM 實驗帳本主要在「LLM 決策」看，這裡預設不列，免得一進來就選到別人的組合
    show_llm = bool(llm) and st.toggle(f"也列出 LLM 實驗帳本（{len(llm)} 組，只能看）", key="mn_show_llm")
    names = own + (llm if show_llm else [])
    if not names:
        return
    name = st.selectbox("組合", names, key="mn_pick")
    m = meta[name]
    sigs = mine._latest_signals(mine.load_ledger(), name)
    cur = sigs[max(sigs)]
    common.kpis([
        ("建立日", common.esc(m["created"]), f"本金 {m['capital']:,.0f} 元"),
        ("調整", mine.REBALANCE[m["rebalance"]], f"{len(sigs)} 次訊號"),
        ("最近訊號", f"{max(sigs):%Y-%m-%d}", "隔一個交易日收盤成交"),
    ])
    st.caption("規則：" + "；".join(mine.describe(m)))
    st.markdown("**目前目標權重**")
    common.weight_table(cur)
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
        common.nav_chart(res["daily"], labels, "累積淨值（起點 = 1）", 340)
        common.cmp_table(res["daily"], res["perf"], labels)
        st.caption(common.cost_line(res) + "。同組等權不扣成本。")
        st.fragment(run_every=common.live_every())(_mine_live)(res)
    if len(sigs) > 1:
        with st.expander(f"歷次訊號（{len(sigs)}）"):
            rows = [{"訊號日": f"{s:%Y-%m-%d}", "代號": c, "目標權重": float(w)}
                    for s, weights in sigs.items() for c, w in weights.items()]
            common.table(common.styled(pd.DataFrame(rows), fmt={"目標權重": "{:.1%}"}), hide_index=True)
    if _is_llm_book(name):
        st.caption("這組由 llm_trader.py 管理，不能改持股，也不能封存。三組比較和每檔理由在「LLM 決策」。")
    else:
        _mine_edit(name, cur)


def _mine_live(res: dict) -> None:
    """持股的即時估值：資料日的部位市值 × 即時價 ÷ 資料日收盤。只顯示，不寫進紀錄；盤中由 fragment 每 10 秒重跑。"""
    hold = {c: v for c, v in res["hold"].items() if v > 0}
    if not hold:
        return
    last = res["nav"].index[-1]
    px = pd.read_parquet(mine.OUT / "prices_daily.parquet", columns=["date", "code", "close"],
                         filters=[("date", "==", last), ("code", "in", list(hold))]).set_index("code")["close"]
    q = common.live_quotes(tuple(sorted(hold)))
    names = custom.names_of(list(hold))
    rows = []
    for c, v in sorted(hold.items(), key=lambda x: -x[1]):
        x = q.get(c) or {}
        newer = x.get("last") and x.get("date") is not None and x["date"] > last and px.get(c)
        rows.append({"代號": c, "名稱": names.get(c, ""), "資料日市值": v, "資料日收盤": px.get(c),
                     "即時價": x.get("last"), "今日漲跌": x.get("chg") if newer else None,
                     "估計市值": v * x["last"] / px[c] if newer else v})
    df = pd.DataFrame(rows)
    base = res["nav"].iloc[-1]
    total = res["cash"] + df["估計市值"].sum()
    st.markdown("**即時估值**")
    if not q:
        st.caption("查不到即時報價（證交所連不上），下面是資料日的市值。")
    common.kpis([
        ("估計資產", f"{total:,.0f}", f"資料日 {last:%m-%d} 為 {base:,.0f} 元", True),
        ("今日估計損益", common.signed_html(total / base - 1, 2), f"{total - base:+,.0f} 元"),
        ("現金", f"{res['cash']:,.0f}", "元"),
    ])
    common.table(common.styled(df, fmt={"資料日市值": "{:,.0f}", "資料日收盤": "{:,.2f}", "即時價": "{:,.2f}",
                                        "今日漲跌": "{:+.2%}", "估計市值": "{:,.0f}"}, signed=["今日漲跌"]),
                 hide_index=True)
    st.caption("只是估計：用證交所即時價（延遲約 5 秒）換算資料日的持股，不含今天可能觸發的停損等規則與交易成本；"
               "報價日期沒有比資料日新（例如晚上已更新資料）時不加漲跌。正式淨值晚上更新資料後重算。")


def _mine_edit(name: str, cur: pd.Series) -> None:
    with st.expander("改持股 / 封存"):
        spec = st.text_input("新的持股（代號:權重）", ",".join(f"{c}:{w * 100:.4g}" for c, w in cur.items()),
                             key=f"mn_edit_{name}")
        m = mine.load_meta().get(name, {})
        if m.get("weighting", "manual") != "manual":
            st.caption(f"這個組合用「{mine.WEIGHTINGS[m['weighting']]}」，只看代號，權重會重算。")
        if m.get("source") and m["rebalance"] == "monthly":
            st.caption("下個月會照名單來源重選，這裡的修改只維持到那時。")
        if m["rebalance"] == "events":
            st.caption("事件名單下次有變動時會照規則重記，這裡的修改只維持到那時。")
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

section = common.segmented("區塊", ["策略追蹤", "自選組合", "LLM 決策"], "策略追蹤",
                           key="paper_section", label_visibility="collapsed")
if section == "策略追蹤":
    st.subheader("綜合分數前 30")
    _top30(summary)
    _strat_nav()
    st.subheader("多策略")
    _multi(strats)
    with st.expander("每期紀錄"):
        _ledger()
    with st.expander("追蹤哪些策略"):
        st.caption("paper_strategies.txt。這裡只顯示；改了要等下個月才會記新策略。")
        if paper.STRATS_FILE.exists():
            st.code(paper.STRATS_FILE.read_text(encoding="utf-8"), language="text")
        else:
            common.empty("沒有 paper_strategies.txt。")
elif section == "自選組合":
    _mine()
elif section == "LLM 決策":
    _llm()
