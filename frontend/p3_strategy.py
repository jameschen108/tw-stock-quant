"""策略：淨值、成績單、分段、自選股回測、參數微調、嘗試紀錄、線上試跑。"""
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common
import custom
import paper
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


def _cache_status(cfg: dict, score: dict) -> str | None:
    """快取和現在的設定、價格日不一致時回傳原因。雜湊不符代表不能用。"""
    if score.get("hash") != strategy.config_hash(cfg):
        return "雜湊不符"
    last = common.load_status().get("last_price_date")
    if score.get("asof") and last and str(score["asof"]) < str(last):
        return f"資料日 {score['asof']}，價格已到 {last}"
    return None


def _run(sig: str, cfg: dict):
    saved = _saved(cfg["id"])
    skipped = None
    if saved is not None:
        stale = _cache_status(cfg, saved[1])
        if stale is None:
            return saved[0], saved[1], "已存結果"
        if stale != "雜湊不符":
            return saved[0], saved[1], f"已存結果（{stale}）"
        skipped = "策略檔改過，已存結果不能用。"
    res = common.cached_simulate(sig, cfg)
    if res is None:
        return None, None, "現場"
    n_same, n_all, var = _trial_stats(cfg["id"])
    sc = strategy.scorecard(cfg, res, n_same, n_all, var)
    return res["daily"], sc, f"{skipped or ''}現場重算（不寫入嘗試紀錄）"


def _score_table(sc: dict) -> None:
    if not sc or "periods" not in sc:
        st.json(sc)
        return
    full = (sc.get("periods") or {}).get("全期", {}).get("strategy") or {}
    oos_b = (sc.get("periods") or {}).get("樣本外 2019–", {})
    oos = oos_b.get("strategy") or {}
    def _dsr(key: str) -> str:
        v = sc.get(key)
        return "—" if v is None else f"{v * 100:.0f}%"

    n_all, n_now = sc.get("n_all"), len(_trials())
    tried = f"算的時候試過 {n_all} 個設定（現在 {n_now}）" if n_all is not None and n_all != n_now \
        else f"試過 {n_all if n_all is not None else '—'} 個設定"
    common.kpis([
        ("期末資產", f"{sc['final']:,.0f}" if sc.get("final") is not None else "—", "元"),
        ("全期年化", common.signed_html(full.get("cagr")), f"Sharpe {common.plain(full.get('sharpe'), 2)}"),
        ("樣本外年化", common.signed_html(oos.get("cagr")),
         f"同池 {common.pct((oos_b.get('pool') or {}).get('cagr'))}・0050 {common.pct((oos_b.get('etf') or {}).get('cagr'))}"),
        ("最大回落", common.pct(full.get("mdd")), f"{full.get('mdd_days', '—')} 個交易日沒回到高點"),
        ("成本 / 年", common.plain(sc.get("cost_per_year_pct"), 2, "%"), f"月換手 {common.pct(sc.get('turnover_month')).lstrip('+')}"),
        ("Deflated Sharpe", _dsr("dsr"), f"{tried}；檢定月 Sharpe > 0"),
        ("對 0050 的 DSR", _dsr("dsr_excess"), "檢定月超額 Sharpe > 0"),
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
    common.table(common.styled(df, fmt={"年化": "{:+.1%}", "同池年化": "{:+.1%}", "0050 年化": "{:+.1%}",
                                        "波動": "{:.1%}", "Sharpe": "{:.2f}", "最大回落": "{:.1%}"},
                               signed=["年化", "同池年化", "0050 年化"]), hide_index=True)
    st.caption(f"平均持股 {common.plain(sc.get('avg_hold'), 1)} 檔，平均現金 {common.pct(sc.get('avg_cash_pct')).lstrip('+')}，"
               f"這個名稱試過 {sc.get('n_same', '—')} 個設定。")


def _yearly(daily: pd.DataFrame) -> None:
    yr = (1 + daily).groupby(daily.index.year).prod() - 1
    show = pd.DataFrame({"年": yr.index.astype(str), "策略": yr["strategy"].values, "同池": yr["pool"].values,
                         "0050": yr["etf"].values, "超額（對同池）": (yr["strategy"] - yr["pool"]).values}).iloc[::-1]
    cols = ["策略", "同池", "0050", "超額（對同池）"]
    common.table(common.styled(show, fmt={c: "{:+.1%}" for c in cols}, signed=cols), hide_index=True)


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
    common.table(common.styled(df, fmt={"年化": "{:+.1%}", "0050 年化": "{:+.1%}", "超額 / 年": "{:+.1%}",
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


_ROBUST_FMT = {"樣本外年化": "{:+.1%}", "0050": "{:+.1%}", "超額/年": "{:+.1%}", "α/年": "{:+.1%}",
               "最大回落": "{:.1%}", "成本/年": "{:.2%}", "年化": "{:+.1%}", "0050 年化": "{:+.1%}",
               "超額 / 年": "{:+.1%}", "α / 年": "{:+.1%}", "贏 0050 月份": "{:.0%}",
               "超額 t": "{:+.2f}", "α t": "{:+.2f}", "β": "{:.2f}"}
_ROBUST_SIGNED = ["樣本外年化", "超額/年", "α/年", "年化", "0050 年化", "超額 / 年", "α / 年"]


_VARIANT_RULES = [  # robust.variants 的標籤 → 中文；{v} 是變體的值，{b} 是原設定的值
    (r"n(\d+)", "持股 {v} 檔", lambda c: c["select"]["n"], "（原 {b}）"),
    (r"lag(\d+)", "訊號後第 {v} 個交易日成交", lambda c: c.get("lag", 1), "（原第 {b} 個）"),
    (r"ma(\d+)", "大盤均線 {v} 日", lambda c: c["risk"]["regime_ma"], "（原 {b}）"),
    (r"off(\d+)", "跌破均線留 {v}% 部位", lambda c: round(c["risk"]["regime_off"] * 100), "（原 {b}%）"),
    (r"stop(\d+)", "停損 {v}%", lambda c: round(c["risk"]["stop_loss"] * 100), "（原 {b}%）"),
    (r"dd(\d+)", "回落門檻 {v}%", lambda c: round(c["risk"]["dd_limit"] * 100), "（原 {b}%）"),
    (r"tvol(\d+)", "目標波動 {v}%", lambda c: round(c["risk"]["target_vol"] * 100), "（原 {b}%）"),
]
_VARIANT_FIXED = {"cost2x": "手續費與證交稅加倍", "disc28": "手續費 2.8 折", "cap1m": "本金 100 萬",
                  "fixed": "改成固定停損", "trailing": "改成移動停損"}


def _variant_label(tag: str, base: dict) -> str:
    if tag in _VARIANT_FIXED:
        return _VARIANT_FIXED[tag]
    for pat, text, cur, orig in _VARIANT_RULES:
        m = re.fullmatch(pat, tag)
        if m:
            try:
                b = cur(base)
            except (KeyError, TypeError):
                b = None
            return text.format(v=m.group(1)) + ("" if b is None else orig.format(b=b))
    return tag


def _readable_variants(rows: list, base: dict) -> tuple[pd.DataFrame, str]:
    """變體名換成中文；0050 每列都一樣（同一段期間），拿掉這欄改寫在說明。"""
    df = pd.DataFrame(rows)
    if "變體" in df:
        df["變體"] = [_variant_label(str(t), base) for t in df["變體"]]
    note = ""
    if "0050" in df and df["0050"].nunique(dropna=True) <= 1:
        etf = df["0050"].dropna()
        note = f"同期 0050 樣本外年化 {etf.iloc[0]:+.1%}。" if len(etf) else ""
        df = df.drop(columns="0050")
    return df, note


def _show_robust_json(payload, base: dict) -> None:
    def _table(rows: list) -> None:
        df = pd.DataFrame(rows)
        common.table(common.styled(df, fmt=_ROBUST_FMT, signed=_ROBUST_SIGNED), hide_index=True)

    if isinstance(payload, dict) and "variants" in payload:
        st.markdown("**變體**")
        df, note = _readable_variants(payload["variants"], base)
        common.table(common.styled(df, fmt=_ROBUST_FMT, signed=_ROBUST_SIGNED), hide_index=True)
        if note:
            st.caption(note)
        if payload.get("segments"):
            st.markdown("**分段**")
            _table(payload["segments"])
    elif isinstance(payload, list):
        _table(payload)
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


def _variant_table(rows: list[dict], base: dict) -> None:
    df, note = _readable_variants(rows, base)
    common.table(common.styled(df, fmt={"樣本外年化": "{:+.1%}", "0050": "{:+.1%}", "超額/年": "{:+.1%}",
                                        "超額 t": "{:+.2f}", "α/年": "{:+.1%}", "β": "{:.2f}", "α t": "{:+.2f}",
                                        "最大回落": "{:.1%}", "成本/年": "{:.2%}"},
                               signed=["樣本外年化", "超額/年", "α/年"]), hide_index=True)
    st.caption(note + "樣本外 = 2019 起。變體之間差很多代表結果對參數敏感。")


# 線上試跑的選項：值照設定檔原樣（雜湊不變），畫面上顯示中文
_SORT_LABEL = {"comp_equal": "綜合分數", "short_pct": "短期百分位", "mid_pct": "中期百分位", "long_pct": "長期百分位",
               "rev_mom": "營收動能分數", "quality": "品質分數", "value": "價值分數", "trend": "趨勢分數",
               "rev_yoy": "營收年增", "roe": "ROE", "ep": "盈餘殖利率 E/P", "bp": "淨值股價比 B/P", "mcap": "市值",
               "avg_value": "20 日均成交值", "vol_60d": "60 日波動"}
_WEIGHT_LABEL = {"equal": "等權", "inv_vol": "波動倒數（波動小的多買）", "score": "依排序分數"}
_REGIME_LABEL = {"none": "不用", "ma": "均線濾網"}
_INDEX_LABEL = {"0050": "0050 含息指數", "ew": "全市場等權指數"}
_FREQ_LABEL = {"monthly": "每月換股時判斷", "daily": "每天判斷"}
_STOP_LABEL = {"trailing": "從進場後高點算（移動）", "fixed": "從進場價算（固定）"}
_FILTER_HELP = ("一行一條 pandas query，全部成立才選。可用欄位：短中長期 short / mid / long 的 _pct（0–100）與 _label"
                "（'看漲'、'中性'、'看跌'），以及上面排序清單括號裡的欄位。例：mid_label == '看漲'")


def _choice(col, label: str, mapping: dict, current: str, key: str, **kw) -> str:
    """下拉選單顯示中文、回傳原值；設定檔用了清單外的值也保留。"""
    opts = list(mapping) + ([current] if current not in mapping else [])
    return col.selectbox(label, opts, index=opts.index(current), key=key,
                         format_func=lambda v: mapping.get(v, v), **kw)


def _trial_form(sig: str, paths: list[Path]) -> None:
    st.caption("改參數後回測。每次有結果都會記進嘗試紀錄（同一設定只記一次），"
               "讓 Deflated Sharpe 的「試過幾次」反映真實次數。股票池沿用起點設定檔，約 3–5 秒，有每日出場規則更久。")
    base_path = st.selectbox("起點設定", paths, format_func=lambda p: f"{strategy.load_config(p)['name']}（{p.stem}）",
                             key="trial_base")
    cfg0 = strategy.load_config(base_path)
    pre = base_path.stem
    sel, pos, risk, cost = cfg0["select"], cfg0["position"], cfg0["risk"], cfg0["cost"]
    frac = "0–1，例 0.20 = 20%"
    with st.form("trial"):
        st.markdown("**選股**")
        c1, c2, c3 = st.columns([2, 1, 1], vertical_alignment="bottom")
        sort = _choice(c1, "排序依據", {k: f"{v}（{k}）" for k, v in _SORT_LABEL.items()}, sel["sort"],
                       f"tr_{pre}_sort")
        n = c2.number_input("持股檔數", min_value=1, max_value=200, value=int(sel["n"]), step=1, key=f"tr_{pre}_n")
        ascending = c3.checkbox("由小到大取（預設取最大的）", value=bool(sel["ascending"]), key=f"tr_{pre}_asc")
        filters = st.text_area("選股條件", value="\n".join(sel["filters"]), key=f"tr_{pre}_filters", help=_FILTER_HELP,
                               placeholder="沒有條件就留白")

        st.markdown("**倉位**")
        c1, c2, c3, c4 = st.columns(4)
        weighting = _choice(c1, "權重", _WEIGHT_LABEL, pos["weighting"], f"tr_{pre}_w")
        max_weight = c2.number_input("單檔上限", min_value=0.0, max_value=1.0, value=float(pos["max_weight"]),
                                     step=0.05, key=f"tr_{pre}_mw", help=frac + "；1 = 不限")
        max_industry = c3.number_input("單一產業上限", min_value=0.0, max_value=1.0,
                                       value=float(pos["max_industry"]), step=0.05, key=f"tr_{pre}_mi",
                                       help=frac + "；1 = 不限")
        exposure = c4.number_input("股票部位比例", min_value=0.0, max_value=1.0, value=float(pos["exposure"]),
                                   step=0.05, key=f"tr_{pre}_ex", help=frac + "；其餘放現金")
        min_trade = st.number_input("續抱股票調整金額小於這個數就不調（元，0 = 都調）", min_value=0,
                                    value=int(pos["min_trade"]), step=1000, key=f"tr_{pre}_mt",
                                    help="避免每筆最低手續費吃掉小額調整；新買進、全部賣出不受影響")

        st.markdown("**大盤濾網**")
        c1, c2, c3, c4, c5 = st.columns(5)
        regime = _choice(c1, "濾網", _REGIME_LABEL, risk["regime"], f"tr_{pre}_rg")
        regime_index = _choice(c2, "大盤指標", _INDEX_LABEL, risk["regime_index"], f"tr_{pre}_ri")
        regime_ma = c3.number_input("均線天數", min_value=5, value=int(risk["regime_ma"]), step=10, key=f"tr_{pre}_ma")
        regime_freq = _choice(c4, "多久判斷一次", _FREQ_LABEL, risk["regime_freq"], f"tr_{pre}_rf")
        regime_off = c5.number_input("跌破均線時留多少部位", min_value=0.0, max_value=1.0,
                                     value=float(risk["regime_off"]), step=0.1, key=f"tr_{pre}_off", help=frac)
        st.caption("濾網選「不用」時，右邊幾格不會生效。")

        st.markdown("**風險控制**（0 = 關）")
        c1, c2, c3, c4, c5 = st.columns(5)
        target_vol = c1.number_input("目標年化波動", min_value=0.0, max_value=1.0, value=float(risk["target_vol"]),
                                     step=0.01, format="%.2f", key=f"tr_{pre}_tv", help=frac + "；用持股過去 60 日估計")
        stop_loss = c2.number_input("個股停損", min_value=0.0, max_value=1.0, value=float(risk["stop_loss"]),
                                    step=0.01, format="%.2f", key=f"tr_{pre}_sl", help=frac + "；每天檢查")
        stop_type = _choice(c3, "停損怎麼算", _STOP_LABEL, risk["stop_type"], f"tr_{pre}_st")
        dd_limit = c4.number_input("資產回落門檻", min_value=0.0, max_value=1.0, value=float(risk["dd_limit"]),
                                   step=0.01, format="%.2f", key=f"tr_{pre}_dd", help=frac + "；從高點回落超過就降曝險")
        dd_exposure = c5.number_input("回落後留多少部位", min_value=0.0, max_value=1.0,
                                      value=float(risk["dd_exposure"]), step=0.1, key=f"tr_{pre}_de",
                                      help=frac + "；到下次換股才恢復")
        c1, c2, _, _, _ = st.columns(5)
        take_profit = c1.number_input("個股停利", min_value=0.0, max_value=10.0, value=float(risk["take_profit"]),
                                      step=0.05, format="%.2f", key=f"tr_{pre}_tp", help=frac + "；從進場價算，每天檢查")
        max_hold_days = c2.number_input("持有滿幾個交易日出場", min_value=0, value=int(risk["max_hold_days"]), step=5,
                                        key=f"tr_{pre}_hd", help="從第一次買進算；這期不再買回")

        with st.expander("交易成本與成交時點（通常不用改）"):
            c1, c2, c3, c4, c5 = st.columns(5)
            fee_rate = c1.number_input("手續費率", min_value=0.0, value=float(cost["fee_rate"]), step=0.000001,
                                       format="%.6f", key=f"tr_{pre}_fee", help="法定 0.1425%")
            fee_discount = c2.number_input("手續費折數", min_value=0.0, value=float(cost["fee_discount"]), step=0.01,
                                           format="%.2f", key=f"tr_{pre}_disc", help="0.28 = 2.8 折，1 = 不打折")
            min_fee = c3.number_input("最低手續費（元）", min_value=0, value=int(cost["min_fee"]), step=1,
                                      key=f"tr_{pre}_minfee")
            tax = c4.number_input("證交稅率", min_value=0.0, value=float(cost["tax"]), step=0.000001, format="%.6f",
                                  key=f"tr_{pre}_tax", help="賣出時扣，股票 0.3%")
            lag = c5.number_input("訊號後第幾個交易日成交", min_value=1, max_value=20, value=int(cfg0.get("lag", 1)),
                                  step=1, key=f"tr_{pre}_lag")
        go = st.form_submit_button("試跑並寫入紀錄")
    if not go:
        return
    cfg = copy.deepcopy(cfg0)
    cfg["select"].update(sort=sort.strip(), ascending=bool(ascending), n=int(n),
                         filters=[ln.strip() for ln in filters.splitlines() if ln.strip()])
    cfg["position"].update(weighting=weighting, max_weight=float(max_weight), max_industry=float(max_industry),
                           exposure=float(exposure), min_trade=int(min_trade))
    cfg["risk"].update(regime=regime, regime_index=regime_index, regime_ma=int(regime_ma), regime_freq=regime_freq,
                       regime_off=float(regime_off), target_vol=float(target_vol), stop_loss=float(stop_loss),
                       stop_type=stop_type, dd_limit=float(dd_limit), dd_exposure=float(dd_exposure),
                       take_profit=float(take_profit), max_hold_days=int(max_hold_days))
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


def _fills(sig: str, cfg: dict, how: str) -> None:
    """買賣明細：優先讀存檔（成績單也是存檔時），沒有就用快取的回測現場補。"""
    f = None
    path = RESULTS / f"{cfg['id']}_fills.parquet"
    if how.startswith("已存結果") and path.exists():
        f = pd.read_parquet(path)
    if f is None:
        res = common.cached_simulate(sig, cfg)
        f = strategy.with_names(res["fills"]) if res is not None else pd.DataFrame()
    if f.empty:
        st.caption("沒有任何成交。")
        return
    f["date"] = pd.to_datetime(f["date"])
    days = sorted(f["date"].dt.date.unique(), reverse=True)
    c1, c2 = st.columns([2, 1])
    day = c1.selectbox("成交日（有買賣的日子）", ["全部"] + days, key=f"fills_day_{cfg['id']}",
                       format_func=lambda d: d if d == "全部" else f"{d:%Y-%m-%d}")
    side = c2.selectbox("買賣", ["全部", "買", "賣"], key=f"fills_side_{cfg['id']}")
    show = f if day == "全部" else f[f["date"].dt.date == day]
    if side != "全部":
        show = show[show["side"] == side]
    show = show.sort_values(["date", "side", "amount"], ascending=[False, True, False])
    out = show.rename(columns={"date": "成交日", "code": "代號", "name": "名稱", "side": "買賣", "price": "成交價",
                               "shares": "股數", "amount": "金額", "reason": "原因"})
    out["成交日"] = out["成交日"].dt.strftime("%Y-%m-%d")
    common.table(common.styled(out, fmt={"成交價": "{:,.2f}", "股數": "{:,.0f}", "金額": "{:,.0f}"}), hide_index=True)
    buy, sell = show.loc[show["side"] == "買", "amount"].sum(), show.loc[show["side"] == "賣", "amount"].sum()
    st.caption(f"共 {len(show)} 筆，買進 {buy:,.0f} 元、賣出 {sell:,.0f} 元。成交價是當天收盤價；全部賣出的股數是當天市值換算，"
               "可能和實際整股數差一點。")
    st.download_button("下載全部明細 CSV", f.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{cfg['id']}_fills.csv", mime="text/csv", key=f"fills_dl_{cfg['id']}")


def _compare(sig: str, paths: list[Path], labels: dict) -> None:
    st.caption("用上次「重算回測快取」存下的結果，看的時候不寫入嘗試紀錄；策略檔改過的會現場重算。"
               "資料更新後，按右上「更新 / 重算」→「重算回測快取」。")
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
                         "DSR": sc.get("dsr"), "對 0050 的 DSR": sc.get("dsr_excess")})
        st.markdown("**並排比較**")
        common.table(common.styled(pd.DataFrame(rows), fmt={"全期年化": "{:+.1%}", "樣本外年化": "{:+.1%}",
                                                            "Sharpe": "{:.2f}", "最大回落": "{:.1%}",
                                                            "成本/年%": "{:.2f}", "DSR": "{:.0%}",
                                                            "對 0050 的 DSR": "{:.0%}"},
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
            st.markdown("**買賣明細**")
            _fills(sig, cfg, how)


def _variants(sig: str, paths: list[Path], labels: dict) -> None:
    one = st.selectbox("策略", paths, format_func=lambda p: labels[p], key="variant_base")
    base = strategy.load_config(one)
    path_json, payload = _robust_file(base["id"])
    if payload is not None:
        stale = _cache_status(base, payload)
        st.caption("上次存下的穩健性結果" + (f"，已經舊了（{stale}），可以現場重算" if stale else "") + "。")
        _show_robust_json(payload, base)
    else:
        st.caption("這個策略還沒有存好的穩健性結果，可以現場重算。")
    n = len(robust.variants(base))
    if st.button(f"現場重算 {n} 個變體", icon=":material/tune:", help="每個幾秒鐘，都會寫入嘗試紀錄"):
        try:
            with st.spinner("跑變體…"):
                st.session_state["variant_rows_" + base["id"]] = _variant_rows(sig, base)
        except Exception as e:
            st.error("變體失敗：" + common.scrub(e))
    cached_rows = st.session_state.get("variant_rows_" + base["id"])
    if cached_rows:
        _variant_table(cached_rows, base)


def _trial_log() -> None:
    rows = _trials()
    if not rows:
        common.empty("還沒有嘗試紀錄。")
        return
    def _variant(rid) -> str:  # robust.py 的變體代號是「策略__標籤」
        tag = str(rid or "").partition("__")[2]
        return _variant_label(tag, {}) if tag else ""

    df = pd.DataFrame([{"時間": pd.to_datetime(r.get("time")).strftime("%Y-%m-%d %H:%M") if r.get("time") else None,
                        "名稱": r.get("name"), "變體": _variant(r.get("id")), "代號": r.get("id"),
                        "月 Sharpe": r.get("sharpe_m"), "超額 Sharpe": r.get("sharpe_ex"),
                        "年化": r.get("cagr")} for r in rows]).iloc[::-1]
    srs = df["月 Sharpe"].dropna()
    common.kpis([
        ("嘗試次數", f"{len(df)}", f"同設定只記一次；有月 Sharpe 的 {len(srs)} 筆"),
        ("月 Sharpe 中位數", common.plain(srs.median(), 3), ""),
        ("月 Sharpe 最高", common.plain(srs.max(), 3), "挑最好的那個會高估實際表現"),
    ])
    common.table(common.styled(df, fmt={"月 Sharpe": "{:.3f}", "超額 Sharpe": "{:.3f}", "年化": "{:+.1%}"},
                               signed=["年化"]), hide_index=True)
    st.caption("訊號矩陣那幾筆沒有 Sharpe，只是讓 Deflated Sharpe 的「試過幾次」算進去。")


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
        ("總報酬", common.signed_html((1 + res["daily"]["strategy"]).prod() - 1), "扣成本"),
        ("年化", common.signed_html(p.get("cagr")), "不到 20 個交易日不算" if not p else ""),
        ("Sharpe", common.plain(p.get("sharpe"), 2), ""),
        ("最大回落", common.pct(p.get("mdd")), f"{p.get('mdd_days', '—')} 個交易日沒回到高點" if p else ""),
        ("期末資產", f"{res['nav'].iloc[-1]:,.0f}", "元"),
    ])
    hold = st.session_state.get("cu_key", (None,) * 4)[3] == "買進持有"
    # 買進持有只成交一次，扣不扣成本的兩條線幾乎重疊，圖上只畫扣成本的
    common.nav_chart(res["daily"], [x for x in labels if not (hold and x[0] == "pool")], "累積淨值（起點 = 1）")
    common.cmp_table(res["daily"], res["perf"], labels)
    st.markdown("**目標權重**")
    common.weight_table(res["weights"])
    st.caption(common.cost_line(res) + "。「不扣成本」是同一組目標權重買進持有"
               + ("，買進持有時和扣成本幾乎一樣，所以圖上沒畫。" if hold else "。"))


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

section = common.segmented("區塊", ["回測比較", "自選股回測", "參數微調", "線上試跑", "嘗試紀錄", "策略清單"],
                           "回測比較", key="strat_section", label_visibility="collapsed")
if section == "回測比較":
    _compare(sig, paths, labels)
    with st.expander("存檔的 HTML 報告"):
        _reports()
elif section == "自選股回測":
    _custom()
elif section == "參數微調":
    _variants(sig, paths, labels)
elif section == "線上試跑":
    _trial_form(sig, paths)
elif section == "嘗試紀錄":
    _trial_log()
elif section == "策略清單":
    common.table(pd.DataFrame([{
        "代號": p.stem, "名稱": c["name"], "持股": c["select"]["n"],
        "排序": _SORT_LABEL.get(c["select"]["sort"], c["select"]["sort"]),
        "權重": _WEIGHT_LABEL.get(c["position"]["weighting"], c["position"]["weighting"]).split("（")[0],
        "大盤濾網": "" if c["risk"]["regime"] == "none" else
        f"{_INDEX_LABEL.get(c['risk']['regime_index'], c['risk']['regime_index'])} {c['risk']['regime_ma']} 日均線"
        + ("，每天判斷" if c["risk"]["regime_freq"] == "daily" else ""),
        "停損": c["risk"]["stop_loss"] or None, "目標波動": c["risk"]["target_vol"] or None,
        "說明": c.get("description") or ""}  # 說明很長，放最後才不會把參數擠出畫面
        for p, c in cfgs.items()]), hide_index=True,
        column_config={"大盤濾網": st.column_config.TextColumn(width="medium"),
                       "停損": st.column_config.NumberColumn(format="percent"),
                       "目標波動": st.column_config.NumberColumn(format="percent")})
    st.caption("欄位說明在 strategy.py 檔頭的 docstring。")
