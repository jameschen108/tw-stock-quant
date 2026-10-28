"""本機介面的共用部分：路徑、樣式、資料狀態列、快取、花費估計、圖表。"""
import copy
import html
import json
import math
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analyst
import custom
import daily
import outlook
import paper
import report
import strategy
import update

DISCLAIMER = "學術研究用。看漲 / 看跌是模型依歷史統計的分組，不保證未來報酬。"
# 每百萬 token（輸入, 輸出），同 compare_llm.py。快取 token 依輸入價估計，畫面上的數字是上限。
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}
REPORT_OUT_TOKENS = 1200
CHAT_OUT_TOKENS = 2000

# 台股慣例：漲紅跌綠
UP, DOWN = "#e5484d", "#18a058"
C_STRAT, C_POOL, C_ETF = "#3b82f6", "#94a3b8", "#f59e0b"
LABEL_CLASS = {"看漲": "bull", "看跌": "bear", "中性": "neutral", "多": "bull", "空": "bear",
               "看漲形態": "bull", "看跌形態": "bear"}

STATUS_FILE = ROOT / "data" / "update_status.json"

CSS = """
<style>
.block-container {padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1440px;}
[data-testid="stBottomBlockContainer"] {max-width: 1440px; margin: 0 auto;}
h1, h2, h3 {letter-spacing: .01em;}
div[data-testid="stMetricValue"] {font-size: 1.45rem;}
.sbar {display:flex; flex-wrap:wrap; gap:.4rem; align-items:center; font-size:.78rem; margin:0; opacity:.75;}
.sbar b {font-weight:600;}
.sbar i {font-style:normal; opacity:.45;}
.sbar + .disc {margin:.1rem 0 0;}
[data-testid="stHorizontalBlock"]:has(.sbar) {align-items:center; margin-bottom:.4rem;}
.fresh {padding:.4rem .8rem; border-radius:.5rem; font-size:.85rem; background:rgba(234,179,8,.12);
        border:1px solid rgba(234,179,8,.45);}
[data-testid="stHorizontalBlock"]:has(.fresh) {align-items:center; margin-bottom:.4rem;}
[data-testid="stHorizontalBlock"]:has(.sbar) [data-testid="stButton"] button,
[data-testid="stHorizontalBlock"]:has(.fresh) [data-testid="stButton"] button {
  border: 1px solid rgba(128,128,128,.22); background: transparent; min-height: 1.7rem;
  padding: .05rem .7rem; font-size: .8rem; border-radius: 999px; white-space: nowrap;}
.chip {padding:.12rem .6rem; border-radius:999px; background:rgba(128,128,128,.10);
       border:1px solid rgba(128,128,128,.22); white-space:nowrap;}
.chip b {font-weight:600;}
.chip.warn {background:rgba(234,179,8,.14); border-color:rgba(234,179,8,.55);}
.chip.bad {background:rgba(229,72,77,.14); border-color:rgba(229,72,77,.55);}
.disc {font-size:.76rem; opacity:.62; margin:.15rem 0 .6rem;}
.ptitle {display:flex; align-items:baseline; gap:.8rem; flex-wrap:wrap; margin:.4rem 0 .6rem;}
.ptitle h1 {font-size:1.9rem; margin:0; padding:0;}
.ptitle .sub {opacity:.65; font-size:.9rem;}
.kgrid {display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:.7rem; margin:.3rem 0 1rem;}
@media (min-width: 1200px) {.kgrid {grid-template-columns:repeat(var(--cols, 6), minmax(0, 1fr));}}
.kpi {border:1px solid rgba(128,128,128,.22); border-radius:.8rem; padding:.7rem .95rem; background:rgba(128,128,128,.05);}
.kpi .t {font-size:.78rem; opacity:.68;}
.kpi .v {font-size:1.5rem; font-weight:650; line-height:1.35; font-variant-numeric:tabular-nums;}
.kpi .s {font-size:.76rem; opacity:.62;}
.kpi.accent {border-color:rgba(59,130,246,.5); background:rgba(59,130,246,.06);}  /* 紅綠留給漲跌 */
.badge {display:inline-block; padding:.05rem .5rem; border-radius:.4rem; font-size:.78rem; font-weight:600;
        margin:0 .25rem .2rem 0; white-space:nowrap;}
.badge.bull {background:rgba(229,72,77,.15); color:#e5484d;}
.badge.bear {background:rgba(24,160,88,.15); color:#18a058;}
.badge.neutral, .badge.muted {background:rgba(128,128,128,.14);}
.badge.info {background:rgba(59,130,246,.14); color:#3b82f6;}
.up {color:#e5484d;} .down {color:#18a058;}
.cgrid {display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:.7rem; margin:.3rem 0 1rem;}
.card {border:1px solid rgba(128,128,128,.22); border-radius:.8rem; padding:.7rem .95rem;}
.card .hd {display:flex; justify-content:space-between; align-items:baseline; gap:.5rem; margin-bottom:.3rem;}
.card .hd b {font-size:1rem;}
.card ul {margin:0; padding-left:1.1rem; font-size:.86rem; line-height:1.55;}
.card li.alert::marker {color:#e5484d;}
.hero {display:flex; justify-content:space-between; gap:1.5rem; flex-wrap:wrap; border:1px solid rgba(128,128,128,.22);
       border-radius:.9rem; padding:1rem 1.2rem; margin:.4rem 0 .9rem; background:rgba(128,128,128,.04);}
.hero .nm {font-size:1.5rem; font-weight:650;}
.hero .cd {opacity:.6; font-size:.95rem; margin-left:.4rem;}
.hero .px {font-size:2.1rem; font-weight:700; font-variant-numeric:tabular-nums; line-height:1.2;}
.hero .chg {font-size:1.05rem; font-weight:600; margin-left:.5rem;}
.hero .meta {opacity:.6; font-size:.8rem;}
.hero .right {display:flex; flex-direction:column; gap:.35rem; align-items:flex-start; min-width:280px;}
.hero .row {font-size:.85rem;}
.hero .row .k {opacity:.6; display:inline-block; width:4.2rem;}
.range {position:relative; height:6px; border-radius:3px; background:rgba(128,128,128,.25); width:220px; margin-top:.3rem;}
.range i {position:absolute; top:-4px; width:3px; height:14px; background:currentColor; border-radius:2px;}
.empty {border:1px dashed rgba(128,128,128,.35); border-radius:.8rem; padding:1rem 1.2rem; opacity:.8; font-size:.9rem;}
</style>
"""


# ---------------------------------------------------------------- 版面

def setup() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    report.load_dotenv()


def esc(x) -> str:
    return html.escape("" if x is None else str(x))


def badge(label, kind: str | None = None) -> str:
    kind = kind or LABEL_CLASS.get(str(label), "muted")
    return f'<span class="badge {kind}">{esc(label)}</span>'


def signed_html(x, nd=1, scale=100.0, suffix="%") -> str:
    v = _num(x)
    if v is None:
        return "—"
    cls = "up" if v > 0 else ("down" if v < 0 else "")
    return f'<span class="{cls}">{v * scale:+.{nd}f}{suffix}</span>'


def page_title(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="ptitle"><h1>{esc(title)}</h1><span class="sub">{esc(sub)}</span></div>',
                unsafe_allow_html=True)


def segmented(label: str, options: list[str], default: str, key: str, **kw) -> str:
    """單選的 segmented_control。原生的再點一次已選的會變成 None，這裡改回上一個選項。"""
    last = f"_{key}_last"
    if st.session_state.get(key) not in options:
        st.session_state[key] = st.session_state.get(last) if st.session_state.get(last) in options else default

    def keep():
        if st.session_state.get(key) is None:
            st.session_state[key] = st.session_state.get(last, default)
        st.session_state[last] = st.session_state[key]

    st.session_state.setdefault(last, st.session_state[key])
    return st.segmented_control(label, options, key=key, on_change=keep, **kw) or default


def kpis(items: list[tuple], ) -> None:
    """items: (標題, 值 HTML, 小字, accent?)。值已經是 HTML，呼叫端負責 escape。"""
    cells = []
    for it in items:
        t, v, s = it[0], it[1], it[2] if len(it) > 2 else ""
        acc = " accent" if len(it) > 3 and it[3] else ""
        cells.append(f'<div class="kpi{acc}"><div class="t">{esc(t)}</div><div class="v">{v}</div>'
                     f'<div class="s">{esc(s)}</div></div>')
    # 寬螢幕一列最多 7 張，張數多時平均分列（8 張排 4+4，不要 7+1）
    cols = len(cells) if len(cells) <= 7 else math.ceil(len(cells) / math.ceil(len(cells) / 7))
    st.markdown(f'<div class="kgrid" style="--cols:{max(cols, 1)}">{"".join(cells)}</div>', unsafe_allow_html=True)


def table(data, **kw):
    """st.dataframe，缺值顯示「—」。原生預設顯示 None，Styler 的 na_rep 管不到。"""
    return st.dataframe(data, placeholder="—", **kw)


def empty(msg: str) -> None:
    st.markdown(f'<div class="empty">{esc(msg)}</div>', unsafe_allow_html=True)


def load_status() -> dict:
    return json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}


def _job_commands() -> list[tuple[str, str, list[str]]]:
    tomls = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "strategies").glob("*.toml"))
    jobs = [
        ("更新資料", "抓行情到每日速覽，含模擬持股和自選組合", ["uv", "run", "update.py"]),
        ("重算模擬持股", "只跑 paper.py，不抓新資料", ["uv", "run", "paper.py"]),
        ("自選組合月調", "只為月調組合記新的一期", ["uv", "run", "mine.py"]),
    ]
    if tomls:
        jobs.insert(1, ("重算回測快取", "策略檔或資料改過之後。不寫入嘗試紀錄",
                        ["uv", "run", "strategy.py", *tomls, "--no-log"]))
    return jobs


def _run_job(title: str, args: list[str]) -> None:
    import fcntl
    import os
    import subprocess
    lock_path = ROOT / "logs" / ".frontend_job.lock"
    lock_path.parent.mkdir(exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            st.warning("已經有一個指令在跑。")
            return
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        key = env.get("ANTHROPIC_API_KEY", "")
        with st.status(f"{title}…", expanded=True) as box:
            proc = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, env=env)
            lines: list[str] = []
            slot = st.empty()
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                if key and key in line:
                    line = line.replace(key, "***")
                lines.append(line)
                slot.code("\n".join(lines[-24:]), language="text")
            code = proc.wait()
            text = "\n".join(lines[-80:]) or "（沒有輸出）"
            st.session_state["job_last"] = {"title": title, "code": code, "text": text}
            if code == 0:
                box.update(label=f"{title} 完成", state="complete", expanded=False)
                st.cache_data.clear()
            else:
                box.update(label=f"{title} 結束碼 {code}", state="error")


def _open_jobs() -> None:
    st.session_state.jobs_open = True


def _close_jobs() -> None:
    st.session_state.jobs_open = False


@st.dialog("執行", width="small", on_dismiss=_close_jobs)
def _jobs_dialog() -> None:
    import shutil
    st.caption("只跑固定指令。滑過按鈕可看它做什麼。關掉視窗後，頁面會讀到新結果。")
    if shutil.which("uv") is None:
        st.error("找不到 uv，沒辦法從這裡執行。")
        return
    for i, (title, note, args) in enumerate(_job_commands()):
        if st.button(title, key=f"job_{i}", help=note, width="stretch"):
            _run_job(title, args)
    last = st.session_state.get("job_last")
    if last:
        st.caption(f"上次：{last['title']}　結束碼 {last['code']}")
        st.code(last["text"], language="text")


def header() -> None:
    """每頁頂端：一列資料狀態、過舊或更新問題的提醒、免責。"""
    stt = load_status()
    problems = stt.get("problems") or []
    ok = stt.get("fetch_ok")
    run_at = (stt.get("run_at") or "—").replace("T", " ")[5:16] if stt.get("run_at") else "—"

    def share(k):
        v = stt.get(k)
        return "—" if v is None else f"{v * 100:.1f}%"

    items = [f'<b>價格</b> {esc(stt.get("last_price_date") or "—")}',
             f'<b>因子</b> {esc(stt.get("factor_date") or "—")}',
             f'<b>更新</b> {esc(run_at)}',
             f'<b>營收</b> {share("rev_fresh_share")}',
             f'<b>財報</b> {share("fin_fresh_share")}']
    # 正常時只有一行小字；有狀況才加標籤
    if ok is None:
        items.append('<span class="chip warn">沒有更新紀錄</span>')
    elif problems or not ok:
        items.append(f'<span class="chip bad">更新有問題{f"（{len(problems)}）" if problems else ""}</span>')
    last = stt.get("last_price_date") or ""
    for d, why in sorted(update.CLOSED.items()):
        if d > last:  # 價格日之後的休市才解釋得了為什麼沒有新行情
            items.append(f'<span class="chip">休市 {esc(d[5:])} {esc(why)}</span>')
    bar, act = st.columns([1, 0.16], gap="small", vertical_alignment="center")
    bar.markdown(f'<div class="sbar">{"<i>·</i>".join(items)}</div>'
                 f'<div class="disc">{DISCLAIMER}</div>', unsafe_allow_html=True)
    act.button("更新 / 重算", type="tertiary", key="open_jobs_btn", on_click=_open_jobs, width="stretch",
               help="更新資料或重算。只在這個視窗裡跑固定指令")
    if st.session_state.get("jobs_open"):
        _jobs_dialog()
    if problems or ok is False:
        st.error("更新問題：" + ("；".join(problems) if problems else "fetch_ok 為 false"))
    fresh = report.freshness_warning()
    if fresh:
        msg, btn = st.columns([1, 0.16], gap="small", vertical_alignment="center")
        msg.markdown(f'<div class="fresh">{esc(fresh.removesuffix("。分析前建議先跑 uv run update.py"))}。</div>',
                     unsafe_allow_html=True)
        btn.button("更新資料", key="fresh_update_btn", on_click=_open_jobs, width="stretch")


# ---------------------------------------------------------------- 格式

def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def pct(x, nd=1, scale=100.0) -> str:
    v = _num(x)
    return "—" if v is None else f"{v * scale:+.{nd}f}%"


def num(x, nd=2) -> str:
    v = _num(x)
    return "—" if v is None else f"{v:,.{nd}f}"


def plain(x, nd=1, suffix="") -> str:
    v = _num(x)
    return "—" if v is None else f"{v:.{nd}f}{suffix}"


def color_signed(v) -> str:
    v = _num(v)
    if v is None or v == 0:
        return ""
    return f"color: {UP}" if v > 0 else f"color: {DOWN}"


def color_label(v) -> str:
    return {"看漲": f"color: {UP}; font-weight: 600", "看跌": f"color: {DOWN}; font-weight: 600"}.get(v, "")


def styled(df: pd.DataFrame, fmt: dict | None = None, signed: list | None = None, labels: list | None = None):
    s = df.style
    if signed:
        s = s.map(color_signed, subset=[c for c in signed if c in df.columns])
    if labels:
        s = s.map(color_label, subset=[c for c in labels if c in df.columns])
    if fmt:
        s = s.format({k: v for k, v in fmt.items() if k in df.columns}, na_rep="—")
    return s


def public_error(exc: BaseException) -> str:
    """只留例外類型與 HTTP 狀態碼，避免把金鑰或請求內容印出來。"""
    code = getattr(exc, "status_code", None)
    return type(exc).__name__ + (f" ({code})" if code else "")


def scrub(exc: BaseException) -> str:
    """給本機回測用：留錯誤內容，但把金鑰挖掉。"""
    import os
    text = str(exc)
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key and key in text:
        text = text.replace(key, "***")
    return type(exc).__name__ + (f": {text[:400]}" if text else "")


def estimate_usd(model: str, input_chars: int, output_tokens: int) -> float:
    """中文大約 2 字 1 token。回傳美元。"""
    inp, out = PRICES[model]
    return (max(input_chars, 0) / 2) / 1e6 * inp + output_tokens / 1e6 * out


def usage_usd(model: str, usage: dict) -> float:
    inp, out = PRICES[model]
    in_tok = sum(usage.get(k, 0) or 0 for k in (
        "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    return in_tok / 1e6 * inp + (usage.get("output_tokens", 0) or 0) / 1e6 * out


def cents(usd: float) -> str:
    return f"{usd * 100:.2f} 美分"


# ---------------------------------------------------------------- 快取
# 大表只跟著 update.py 的結果失效；觀察名單和每日狀態另外當參數傳，存檔時不會讓大表重讀。

def data_sig() -> str:
    return str(STATUS_FILE.stat().st_mtime_ns) if STATUS_FILE.exists() else "0"


@st.cache_data(show_spinner=False, max_entries=8)
def cached_watch(_sig: str, codes: tuple[str, ...], prev_json: str) -> list[dict]:
    prev = json.loads(prev_json) if prev_json else {}
    rows, _state = daily.watch_rows(list(codes), prev, first_run=not prev)
    return rows  # 狀態丟掉：只有 daily.py 能寫 daily_state.json


@st.cache_data(show_spinner=False, max_entries=8)
def cached_season(_sig: str, codes: tuple[str, ...]):
    return daily.season_rows(list(codes))


@st.cache_data(show_spinner=False, max_entries=2)
def cached_picks(_sig: str):
    return daily.model_picks()


@st.cache_data(show_spinner=False, max_entries=64)
def cached_facts(_sig: str, code: str, asof: str):
    try:
        return report.build_facts(code, pd.Timestamp(asof), extended=True)   # 個股頁和報告同一份事實表
    except SystemExit as e:
        return {"_error": str(e)}, {}


@st.cache_data(show_spinner=False, max_entries=64)
def cached_outlook(_sig: str, code: str) -> pd.DataFrame:
    return outlook.load_outlook(code)


@st.cache_resource(show_spinner=False, max_entries=1)
def names_frame(_sig: str) -> pd.DataFrame:
    analyst.names.cache_clear()
    return analyst.names()


@st.cache_resource(show_spinner=False, max_entries=1)
def panel_and_rets(_sig: str):
    return strategy.load_data()


@st.cache_data(show_spinner=False, max_entries=48)
def cached_simulate(_sig: str, cfg: dict):
    panel, rets = panel_and_rets(_sig)
    return strategy.simulate(copy.deepcopy(cfg), panel, rets)


@st.cache_resource(show_spinner=False, max_entries=1)
def _warm_universe(_sig: str):
    analyst.names.cache_clear()
    analyst.universe.cache_clear()
    return analyst.universe()


@st.cache_data(show_spinner=False, max_entries=64)
def cached_peers(_sig: str, code: str) -> str:
    _warm_universe(_sig)
    return analyst.compare_peers(code)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_performance(_sig: str, code: str, start: str, end: str) -> str:
    return analyst.price_performance(code, start, end or None)


@st.cache_data(show_spinner=False, max_entries=2)
def cached_paper_summary(_sig: str) -> dict:
    return paper.summary()


def goto_stock(code: str) -> None:
    st.session_state["goto_code"] = code
    st.switch_page("p2_stock.py")


# ---------------------------------------------------------------- 圖表

def fig_layout(fig, title: str = "", height: int = 340, legend: bool = True):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)) if title else None,
        height=height, margin=dict(l=8, r=8, t=40 if title else 10, b=8),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1) if legend else None,
        showlegend=legend, hovermode="x unified",
    )
    # 滑鼠提示的數字跟座標軸一樣格式化，不然會出現 2.256441 這種
    fig.for_each_yaxis(lambda ax: ax.update(hoverformat=".1%" if (ax.tickformat or "").endswith("%") else ",.3~f")
                       if ax.hoverformat is None else None)
    return fig


def line_fig(xs, series: dict[str, pd.Series], title: str, colors: dict | None = None, height: int = 340):
    import plotly.graph_objects as go
    fig = go.Figure()
    for name, y in series.items():
        line = dict(width=1.8)
        if colors and name in colors:
            line["color"] = colors[name]
        fig.add_trace(go.Scatter(x=list(xs), y=list(y), name=name, mode="lines", line=line))
    return fig_layout(fig, title, height)


def chart(fig) -> None:
    st.plotly_chart(fig, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def preview_html(path: Path) -> None:
    st.iframe(path, height=760)


# ---------------------------------------------------------------- 回測結果

NAV_COLOR = {"自選（扣成本）": C_STRAT, "自選（不扣成本）": C_POOL,
             "我的組合": C_STRAT, "同組等權": C_POOL, paper.ETF: C_ETF}


def weight_table(weights: pd.Series) -> None:
    names = custom.names_of(list(weights.index))
    df = pd.DataFrame({"代號": list(weights.index), "名稱": [names.get(c, "") for c in weights.index],
                       "目標權重": weights.values})
    table(styled(df, fmt={"目標權重": "{:.1%}"}), hide_index=True)


def cmp_table(daily: pd.DataFrame, perf: dict, labels: list[tuple[str, str]]) -> None:
    rows = []
    for key, label in labels:
        p = perf.get(key) or {}
        rows.append({"組合": label, "總報酬": (1 + daily[key]).prod() - 1, "年化": p.get("cagr"),
                     "波動": p.get("vol"), "Sharpe": p.get("sharpe"), "最大回落": p.get("mdd")})
    df = pd.DataFrame(rows)
    table(styled(df, fmt={"總報酬": "{:+.1%}", "年化": "{:+.1%}", "波動": "{:.1%}", "Sharpe": "{:.2f}",
                          "最大回落": "{:.1%}"}, signed=["總報酬", "年化"]), hide_index=True)


def nav_chart(daily: pd.DataFrame, labels: list[tuple[str, str]], title: str, height: int = 380) -> None:
    nav = (1 + daily).cumprod()
    named = {label: nav[key] for key, label in labels}
    chart(line_fig(nav.index, named, title, {k: NAV_COLOR[k] for k in named}, height))


def cost_line(res: dict) -> str:
    t = res["trades"]
    blocked = [f"{r['exe']:%Y-%m-%d} 未成交：{'、'.join(r['blocked'])}" for _, r in t.iterrows() if r["blocked"]]
    line = (f"期末資產 {res['nav'].iloc[-1]:,.0f} 元・手續費 {t['fees'].sum():,.0f}・證交稅 {t['tax'].sum():,.0f}・"
            f"成交 {len(t)} 次")
    return line + ("・" + "；".join(blocked) if blocked else "")
