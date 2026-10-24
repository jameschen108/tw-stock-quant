"""策略：淨值、成績單、分段、參數微調、嘗試紀錄、線上試跑。"""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common
import robust
import strategy

RESULTS = common.ROOT / "data" / "strategy" / "results"


def _configs() -> list[Path]:
    return sorted((common.ROOT / "strategies").glob("*.toml"))


def _as_returns(df: pd.DataFrame) -> pd.DataFrame:
    """後端 parquet 可能是日報酬，也可能是淨值。絕對值中位數 < 0.2 視為日報酬。"""
    out = df.copy()
    if "date" in out.columns:
        out = out.set_index("date")
    out.index = pd.to_datetime(out.index)
    cols = [c for c in ("strategy", "pool", "etf") if c in out.columns]
    out = out[cols].sort_index().astype(float)
    if float(out.abs().median().median()) < 0.2:
        return out
    ret = out.pct_change()
    ret.iloc[0] = 0.0
    return ret


def _saved(cfg_id: str):
    pq = RESULTS / f"{cfg_id}.parquet"
    js = pq.with_suffix(".json")
    if not (pq.exists() and js.exists()):
        return None
    daily = _as_returns(pd.read_parquet(pq))
    score = json.loads(js.read_text(encoding="utf-8"))
    return daily, score


def _trials() -> list[dict]:
    f = strategy.TRIALS
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


def _trial_stats(cfg_id: str) -> tuple[int, int, float]:
    """跟 strategy.py --no-log 一樣：看的時候不寫入嘗試紀錄。"""
    rows = _trials()
    srs = [r["sharpe_m"] for r in rows if r.get("sharpe_m") is not None]
    n_same = sum(r.get("id") == cfg_id for r in rows)
    n_all = max(len(rows), 1)
    var = float(np.var(srs)) if len(srs) > 1 else 0.0
    return n_same, n_all, var


def _run(sig: str, cfg: dict):
    saved = _saved(cfg["id"])
    if saved is not None:
        return saved[0], saved[1], "後端快取"
    res = common.cached_simulate(sig, cfg)
    if res is None:
        return None, None, "現場"
    n_same, n_all, var = _trial_stats(cfg["id"])
    sc = strategy.scorecard(cfg, res, n_same, n_all, var)
    return res["daily"], sc, "現場（已快取，未寫入嘗試紀錄）"


def _score_table(sc: dict) -> None:
    if not sc or "periods" not in sc:
        st.json(sc)
        return
    full = (sc.get("periods") or {}).get("全期", {}).get("strategy") or {}
    oos_b = (sc.get("periods") or {}).get("樣本外 2019–", {})
    oos = oos_b.get("strategy") or {}
    dsr = sc.get("dsr")
    common.kpis([
        ("期末資產", f"{sc['final']:,.0f}" if sc.get("final") is not None else "—", "元"),
        ("全期年化", common.signed_html(full.get("cagr")), f"Sharpe {common.plain(full.get('sharpe'), 2)}"),
        ("樣本外年化", common.signed_html(oos.get("cagr")),
         f"同池 {common.pct((oos_b.get('pool') or {}).get('cagr'))}・0050 {common.pct((oos_b.get('etf') or {}).get('cagr'))}"),
        ("最大回落", common.pct(full.get("mdd")), f"{full.get('mdd_days', '—')} 個交易日沒回到高點"),
        ("成本 / 年", common.plain(sc.get("cost_per_year_pct"), 2, "%"), f"月換手 {common.pct(sc.get('turnover_month')).lstrip('+')}"),
        ("Deflated Sharpe", "—" if dsr is None else f"{dsr * 100:.0f}%",
         f"試過 {sc.get('n_all', '—')} 個設定；檢定月 Sharpe > 0"),
    ])
    rows = []
    for period, block in sc["periods"].items():
        s = block.get("strategy") or {}
        if not s:
            continue
        rows.append({"期間": period, "年化": s.get("cagr"), "同池年化": (block.get("pool") or {}).get("cagr"),
                     "0050 年化": (block.get("etf") or {}).get("cagr"), "波動": s.get("vol"),
                     "Sharpe": s.get("sharpe"), "最大回落": s.get("mdd")})
    df = pd.DataFrame(rows)
    st.dataframe(common.styled(df, fmt={"年化": "{:+.1%}", "同池年化": "{:+.1%}", "0050 年化": "{:+.1%}",
                                        "波動": "{:.1%}", "Sharpe": "{:.2f}", "最大回落": "{:.1%}"},
                               signed=["年化", "同池年化", "0050 年化"]), hide_index=True)
    st.caption(f"平均持股 {common.plain(sc.get('avg_hold'), 1)} 檔，平均現金 {common.pct(sc.get('avg_cash_pct')).lstrip('+')}，"
               f"這個名稱試過 {sc.get('n_same', '—')} 個設定。")


def _yearly(daily: pd.DataFrame) -> None:
    yr = (1 + daily).groupby(daily.index.year).prod() - 1
    show = pd.DataFrame({"年": yr.index.astype(str), "策略": yr["strategy"].values, "同池": yr["pool"].values,
                         "0050": yr["etf"].values, "超額（對同池）": (yr["strategy"] - yr["pool"]).values}).iloc[::-1]
    cols = ["策略", "同池", "0050", "超額（對同池）"]
    st.dataframe(common.styled(show, fmt={c: "{:+.1%}" for c in cols}, signed=cols), hide_index=True)


def _segments(daily: pd.DataFrame) -> None:
    rows = []
    for name, (a, b) in robust.SEGMENTS.items():
        o = robust.vs_etf(daily, a, b)
        if not o:
            continue
        rows.append({"分段": name, "年化": o.get("cagr"), "0050 年化": o.get("etf_cagr"), "超額 / 年": o.get("excess"),
                     "超額 t": o.get("t_excess"), "α / 年": o.get("alpha"), "β": o.get("beta"),
                     "α t": o.get("t_alpha"), "最大回落": o.get("mdd")})
    if not rows:
        common.empty("每一段都不足 12 個月。")
        return
    df = pd.DataFrame(rows)
    st.dataframe(common.styled(df, fmt={"年化": "{:+.1%}", "0050 年化": "{:+.1%}", "超額 / 年": "{:+.1%}",
                                        "超額 t": "{:+.2f}", "α / 年": "{:+.1%}", "β": "{:.2f}", "α t": "{:+.2f}",
                                        "最大回落": "{:.1%}"},
                               signed=["年化", "0050 年化", "超額 / 年", "α / 年"]), hide_index=True)
    st.caption("用月報酬：超額 = 策略 − 0050，t 為 Newey-West（lag 3）；α、β 來自 策略 = α + β × 0050。")


PALETTE = ["#3b82f6", "#e5484d", "#8b5cf6", "#10b981", "#ec4899", "#0ea5e9", "#f97316"]


def _charts(series: list[tuple[str, pd.DataFrame]]) -> None:
    import plotly.graph_objects as go
    nav_fig, dd_fig = go.Figure(), go.Figure()
    for i, (name, daily) in enumerate(series):
        color = PALETTE[i % len(PALETTE)]
        nav = (1 + daily).cumprod()
        nav_fig.add_trace(go.Scatter(x=nav.index, y=nav["strategy"], name=name, mode="lines",
                                     line=dict(color=color, width=1.8)))
        nav_fig.add_trace(go.Scatter(x=nav.index, y=nav["pool"], name=f"{name}・同池", mode="lines",
                                     line=dict(color=color if len(series) > 1 else common.C_POOL, width=1, dash="dot")))
        if i == 0:
            nav_fig.add_trace(go.Scatter(x=nav.index, y=nav["etf"], name="0050", mode="lines",
                                         line=dict(color=common.C_ETF, width=1.4, dash="dash")))
        dd = nav["strategy"] / nav["strategy"].cummax() - 1
        dd_fig.add_trace(go.Scatter(x=dd.index, y=dd, name=name, mode="lines", fill="tozeroy",
                                    line=dict(color=color, width=1)))
    nav_fig.update_yaxes(type="log", tickvals=[0.5, 1, 2, 3, 5, 10, 20, 30, 50, 100],
                         ticktext=["0.5", "1", "2", "3", "5", "10", "20", "30", "50", "100"])
    dd_fig.update_yaxes(tickformat=".0%")
    common.chart(common.fig_layout(nav_fig, "累積淨值（起點 = 1，對數刻度）", 400))
    common.chart(common.fig_layout(dd_fig, "從高點回落", 260))


def _robust_file(cfg_id: str):
    for path in (RESULTS / f"{cfg_id}_robust.json",
                 common.ROOT / "data" / "strategy" / "robust" / f"{cfg_id}.json"):
        if path.exists():
            return path, json.loads(path.read_text(encoding="utf-8"))
    return None, None


def _show_robust_json(payload) -> None:
    if isinstance(payload, dict) and "variants" in payload:
        st.markdown("**變體**")
        st.dataframe(pd.DataFrame(payload["variants"]), hide_index=True)
        if payload.get("segments"):
            st.markdown("**分段**")
            st.dataframe(pd.DataFrame(payload["segments"]), hide_index=True)
    elif isinstance(payload, list):
        st.dataframe(pd.DataFrame(payload), hide_index=True)
    else:
        st.json(payload)


def _variant_rows(sig: str, base: dict) -> list[dict]:
    rows = []
    specs = robust.variants(base)
    bar = st.progress(0.0, text="準備中")
    for i, (tag, cfg) in enumerate(specs, start=1):
        bar.progress(i / len(specs), text=f"{tag}（{i}/{len(specs)}）")
        res = common.cached_simulate(sig, cfg)
        if res is None:
            continue
        strategy.log_trial(cfg, res)
        o = robust.vs_etf(res["daily"], *robust.OOS) or {}
        tr, yrs = res["trades"], len(res["daily"]) / 252
        cost = float(((tr["fees"] + tr["tax"]) / tr["total"]).sum() / yrs) if yrs else None
        rows.append({"變體": tag, "樣本外年化": o.get("cagr"), "0050": o.get("etf_cagr"), "超額/年": o.get("excess"),
                     "超額 t": o.get("t_excess"), "α/年": o.get("alpha"), "β": o.get("beta"), "α t": o.get("t_alpha"),
                     "最大回落": o.get("mdd"), "成本/年": cost})
    bar.empty()
    return rows


def _variant_table(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    st.dataframe(common.styled(df, fmt={"樣本外年化": "{:+.1%}", "0050": "{:+.1%}", "超額/年": "{:+.1%}",
                                        "超額 t": "{:+.2f}", "α/年": "{:+.1%}", "β": "{:.2f}", "α t": "{:+.2f}",
                                        "最大回落": "{:.1%}", "成本/年": "{:.2%}"},
                               signed=["樣本外年化", "超額/年", "α/年"]), hide_index=True)
    st.caption("樣本外 = 2019 起。變體之間差很多代表結果對參數敏感。")


def _trial_form(sig: str, paths: list[Path]) -> None:
    st.caption("改參數後回測。每次有結果都會寫入 data/strategy/trials.jsonl（同一設定不重複記），"
               "讓 Deflated Sharpe 的「試過幾次」反映真實次數。股票池沿用起點設定檔，約 3–5 秒，有每日出場規則更久。")
    base_path = st.selectbox("起點設定", paths, format_func=lambda p: p.stem, key="trial_base")
    cfg0 = strategy.load_config(base_path)
    pre = base_path.stem
    sel, pos, risk, cost = cfg0["select"], cfg0["position"], cfg0["risk"], cfg0["cost"]
    with st.form("trial"):
        c1, c2, c3 = st.columns(3)
        sort = c1.text_input("選股排序欄位", value=sel["sort"], key=f"tr_{pre}_sort")
        n = c2.number_input("持股檔數", min_value=1, max_value=200, value=int(sel["n"]), step=1, key=f"tr_{pre}_n")
        ascending = c3.checkbox("排序由小到大", value=bool(sel["ascending"]), key=f"tr_{pre}_asc")
        filters = st.text_area("選股條件（一行一條，pandas query）", value="\n".join(sel["filters"]), key=f"tr_{pre}_filters")
        c1, c2, c3, c4 = st.columns(4)
        weighting = c1.selectbox("權重", ["equal", "inv_vol", "score"],
                                 index=["equal", "inv_vol", "score"].index(pos["weighting"]), key=f"tr_{pre}_w")
        max_weight = c2.number_input("單檔上限", min_value=0.0, max_value=1.0, value=float(pos["max_weight"]),
                                     step=0.05, key=f"tr_{pre}_mw")
        max_industry = c3.number_input("單一產業上限", min_value=0.0, max_value=1.0,
                                       value=float(pos["max_industry"]), step=0.05, key=f"tr_{pre}_mi")
        exposure = c4.number_input("股票部位比例", min_value=0.0, max_value=1.0, value=float(pos["exposure"]),
                                   step=0.05, key=f"tr_{pre}_ex")
        min_trade = st.number_input("續抱股票小額不調（元）", min_value=0, value=int(pos["min_trade"]), step=1000,
                                    key=f"tr_{pre}_mt")
        c1, c2, c3, c4 = st.columns(4)
        regime = c1.selectbox("大盤濾網", ["none", "ma"], index=0 if risk["regime"] != "ma" else 1, key=f"tr_{pre}_rg")
        regime_ma = c2.number_input("均線天數", min_value=5, value=int(risk["regime_ma"]), step=10, key=f"tr_{pre}_ma")
        regime_freq = c3.selectbox("濾網頻率", ["monthly", "daily"],
                                   index=0 if risk["regime_freq"] != "daily" else 1, key=f"tr_{pre}_rf")
        regime_off = c4.number_input("均線下留倉", min_value=0.0, max_value=1.0, value=float(risk["regime_off"]),
                                     step=0.1, key=f"tr_{pre}_off")
        c1, c2, c3, c4 = st.columns(4)
        target_vol = c1.number_input("目標波動（0 = 關）", min_value=0.0, max_value=1.0,
                                     value=float(risk["target_vol"]), step=0.01, format="%.2f", key=f"tr_{pre}_tv")
        stop_loss = c2.number_input("停損（0 = 關）", min_value=0.0, max_value=1.0,
                                    value=float(risk["stop_loss"]), step=0.01, format="%.2f", key=f"tr_{pre}_sl")
        stop_type = c3.selectbox("停損型態", ["trailing", "fixed"],
                                 index=0 if risk["stop_type"] != "fixed" else 1, key=f"tr_{pre}_st")
        dd_limit = c4.number_input("回落門檻（0 = 關）", min_value=0.0, max_value=1.0,
                                   value=float(risk["dd_limit"]), step=0.01, format="%.2f", key=f"tr_{pre}_dd")
        dd_exposure = st.number_input("觸及回落後的曝險", min_value=0.0, max_value=1.0,
                                      value=float(risk["dd_exposure"]), step=0.1, key=f"tr_{pre}_de")
        c1, c2, c3, c4 = st.columns(4)
        fee_rate = c1.number_input("手續費率", min_value=0.0, value=float(cost["fee_rate"]), step=0.000001,
                                   format="%.6f", key=f"tr_{pre}_fee")
        fee_discount = c2.number_input("手續費折數", min_value=0.0, value=float(cost["fee_discount"]), step=0.01,
                                       format="%.2f", key=f"tr_{pre}_disc")
        min_fee = c3.number_input("最低手續費（元）", min_value=0, value=int(cost["min_fee"]), step=1, key=f"tr_{pre}_minfee")
        tax = c4.number_input("證交稅率", min_value=0.0, value=float(cost["tax"]), step=0.000001, format="%.6f",
                              key=f"tr_{pre}_tax")
        lag = st.number_input("訊號日後第幾個交易日成交", min_value=1, max_value=20, value=int(cfg0.get("lag", 1)),
                              step=1, key=f"tr_{pre}_lag")
        go = st.form_submit_button("試跑並寫入紀錄")
    if not go:
        return
    cfg = copy.deepcopy(cfg0)
    cfg["select"].update(sort=sort.strip(), ascending=bool(ascending), n=int(n),
                         filters=[ln.strip() for ln in filters.splitlines() if ln.strip()])
    cfg["position"].update(weighting=weighting, max_weight=float(max_weight), max_industry=float(max_industry),
                           exposure=float(exposure), min_trade=int(min_trade))
    cfg["risk"].update(regime=regime, regime_ma=int(regime_ma), regime_freq=regime_freq,
                       regime_off=float(regime_off), target_vol=float(target_vol), stop_loss=float(stop_loss),
                       stop_type=stop_type, dd_limit=float(dd_limit), dd_exposure=float(dd_exposure))
    cfg["cost"].update(fee_rate=float(fee_rate), fee_discount=float(fee_discount), min_fee=int(min_fee), tax=float(tax))
    cfg["lag"] = int(lag)
    with st.spinner("回測中…"):
        try:
            res = common.cached_simulate(sig, cfg)
        except Exception as e:
            st.error("回測失敗：" + common.scrub(e))
            return
    if res is None:
        st.warning("這個設定沒有任何成交日，沒有寫入紀錄。")
        return
    n_same, n_all, var = strategy.log_trial(cfg, res)
    sc = strategy.scorecard(cfg, res, n_same, n_all, var)
    st.success(f"已寫入嘗試紀錄。雜湊 {strategy.config_hash(cfg)}。這個名稱 {n_same} 個設定，全部 {n_all} 個。")
    _score_table(sc)
    _charts([(cfg["name"], res["daily"])])


def _compare(sig: str, paths: list[Path], labels: dict) -> None:
    st.caption("每個策略第一次回測約 3–5 秒（第一次還要載入整段月資料與日報酬），之後用快取，也不寫入嘗試紀錄。")
    chosen = st.multiselect("要比較的策略", paths, default=paths[:1], format_func=lambda p: labels[p],
                            key="cmp_pick", placeholder="選一個或多個策略")
    if not chosen:
        common.empty("選策略後顯示淨值疊圖（含同池等權與 0050）、回落圖、成績單、每年報酬與分段。")
        return
    got = []
    try:
        with st.spinner("回測中…"):
            for path in chosen:
                cfg = strategy.load_config(path)
                daily, sc, how = _run(sig, cfg)
                got.append((cfg, daily, sc, how))
    except Exception as e:
        st.error("回測失敗：" + common.scrub(e))
        return
    plotable = [(cfg["name"], daily) for cfg, daily, sc, how in got if daily is not None]
    if plotable:
        _charts(plotable)
    if len(got) > 1:
        rows = []
        for cfg, daily, sc, how in got:
            if not sc:
                continue
            f = sc["periods"]["全期"]["strategy"]
            o = sc["periods"]["樣本外 2019–"]["strategy"]
            rows.append({"策略": cfg["name"], "全期年化": f.get("cagr"), "樣本外年化": o.get("cagr"),
                         "Sharpe": f.get("sharpe"), "最大回落": f.get("mdd"), "成本/年%": sc.get("cost_per_year_pct"),
                         "DSR": sc.get("dsr")})
        st.markdown("**並排比較**")
        st.dataframe(common.styled(pd.DataFrame(rows), fmt={"全期年化": "{:+.1%}", "樣本外年化": "{:+.1%}",
                                                            "Sharpe": "{:.2f}", "最大回落": "{:.1%}",
                                                            "成本/年%": "{:.2f}", "DSR": "{:.0%}"},
                                   signed=["全期年化", "樣本外年化"]), hide_index=True)
    for cfg, daily, sc, how in got:
        with st.expander(f"{cfg['name']}　成績單", expanded=len(got) == 1):
            st.caption(f"{cfg.get('description') or ''}　來源：{how}")
            if daily is None:
                st.warning("沒有成交日。")
                continue
            _score_table(sc)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**每年報酬**")
                _yearly(daily)
            with c2:
                st.markdown("**分段，相對 0050**")
                _segments(daily)


def _variants(sig: str, paths: list[Path], labels: dict) -> None:
    one = st.selectbox("策略", paths, format_func=lambda p: labels[p], key="variant_base")
    base = strategy.load_config(one)
    path_json, payload = _robust_file(base["id"])
    if payload is not None:
        st.caption(f"後端 JSON：{path_json.relative_to(common.ROOT)}")
        _show_robust_json(payload)
    else:
        st.caption("還沒有這個策略的穩健性 JSON（data/strategy/results/<id>_robust.json），可以現場重算。")
    n = len(robust.variants(base))
    if st.button(f"現場重算 {n} 個變體", icon=":material/tune:", help="每個幾秒鐘，都會寫入嘗試紀錄"):
        try:
            with st.spinner("跑變體…"):
                st.session_state["variant_rows_" + base["id"]] = _variant_rows(sig, base)
        except Exception as e:
            st.error("變體失敗：" + common.scrub(e))
    cached_rows = st.session_state.get("variant_rows_" + base["id"])
    if cached_rows:
        _variant_table(cached_rows)


def _trial_log() -> None:
    rows = _trials()
    if not rows:
        common.empty("還沒有 trials.jsonl。")
        return
    df = pd.DataFrame([{"時間": r.get("time"), "代號": r.get("id"), "名稱": r.get("name"), "雜湊": r.get("hash"),
                        "月 Sharpe": r.get("sharpe_m"), "年化": r.get("cagr")} for r in rows]).iloc[::-1]
    srs = df["月 Sharpe"].dropna()
    common.kpis([
        ("嘗試次數", f"{len(df)}", f"{df['代號'].nunique()} 個不同代號"),
        ("月 Sharpe 中位數", common.plain(srs.median(), 3), ""),
        ("月 Sharpe 最高", common.plain(srs.max(), 3), "挑最好的那個會高估實際表現"),
    ])
    st.dataframe(common.styled(df, fmt={"月 Sharpe": "{:.3f}", "年化": "{:+.1%}"}, signed=["年化"]), hide_index=True)


def _reports() -> None:
    files = sorted(strategy.REPORTS.glob("strategy_*.html")) + sorted(strategy.REPORTS.glob("robust*.html"))
    if not files:
        common.empty("reports/ 裡還沒有策略報告。")
        return
    pick = st.selectbox("HTML", [p.name for p in files])
    common.preview_html(strategy.REPORTS / pick)


sig = common.data_sig()
paths = _configs()
if not paths:
    common.empty("strategies/ 裡沒有 toml。")
    st.stop()
cfgs = {p: strategy.load_config(p) for p in paths}
labels = {p: f"{c['name']}（{p.stem}）" for p, c in cfgs.items()}
trials = _trials()
common.page_title("策略", f"{len(paths)} 個設定檔・嘗試紀錄 {len(trials)} 筆")

section = st.segmented_control("區塊", ["回測比較", "參數微調", "線上試跑", "嘗試紀錄", "策略清單", "現有報告"],
                               default="回測比較", key="strat_section", label_visibility="collapsed") or "回測比較"
if section == "回測比較":
    _compare(sig, paths, labels)
elif section == "參數微調":
    _variants(sig, paths, labels)
elif section == "線上試跑":
    _trial_form(sig, paths)
elif section == "嘗試紀錄":
    _trial_log()
elif section == "策略清單":
    st.dataframe(pd.DataFrame([{
        "代號": p.stem, "名稱": c["name"], "說明": c.get("description") or "", "持股": c["select"]["n"],
        "排序": c["select"]["sort"], "權重": c["position"]["weighting"],
        "大盤濾網": c["risk"]["regime"] if c["risk"]["regime"] != "none" else "",
        "停損": c["risk"]["stop_loss"] or None, "目標波動": c["risk"]["target_vol"] or None}
        for p, c in cfgs.items()]), hide_index=True,
        column_config={"停損": st.column_config.NumberColumn(format="percent"),
                       "目標波動": st.column_config.NumberColumn(format="percent")})
    st.caption("欄位說明在 strategy.py 檔頭的 docstring。")
elif section == "現有報告":
    _reports()
