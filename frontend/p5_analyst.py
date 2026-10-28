"""分析師問答。預設 Sonnet，不提供 Opus。每輪先標大約花費，回答後標實際花費。"""
import sys
from datetime import date, datetime
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import analyst
import common
import report

CHOICES = {"Sonnet（預設）": "sonnet", "Haiku（便宜）": "haiku"}
EXAMPLES = [
    "台積電最近一季毛利率和去年同季比如何？",
    "半導體業裡營收年增最高、綜合分數也高的公司有哪些？",
    "長榮近 90 天有哪些重大訊息？",
    "模型選股清單現在有哪些股票？歷史表現如何？",
]


def _new_chat():
    st.session_state.analyst_messages = []
    st.session_state.analyst_transcript = []
    st.session_state.analyst_log = None
    st.session_state.analyst_spent = 0.0


def _fill(q: str) -> None:
    """範例只填進輸入框，不直接送出：每次送出都要花錢。"""
    st.session_state.analyst_input = q


def _log_path() -> Path:
    if not st.session_state.analyst_log:
        analyst.LOGS.mkdir(parents=True, exist_ok=True)
        st.session_state.analyst_log = str(analyst.LOGS / f"{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    return Path(st.session_state.analyst_log)


def _chars(prompt: str) -> int:
    hist = sum(len(text) for _, text in st.session_state.analyst_transcript)
    return len(analyst.SYSTEM) + hist + len(prompt)


if "analyst_messages" not in st.session_state:
    _new_chat()

turns = len(st.session_state.analyst_transcript) // 2
common.page_title("分析師問答", f"本機資料・{turns} 輪・本次對話約 {common.cents(st.session_state.analyst_spent)}")

status = common.load_status()
default_asof = status.get("last_price_date") or date.today().isoformat()
c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
with c1:
    label = common.segmented("模型", list(CHOICES), "Sonnet（預設）", key="analyst_model")
asof = c2.date_input("資料時點", value=date.fromisoformat(str(default_asof)),
                     max_value=date.fromisoformat(str(default_asof)))
c3.button("新對話", on_click=_new_chat, icon=":material/add_comment:", width="stretch")

model_id = analyst.MODELS[CHOICES[label]]
st.caption(f"下一則大約 {common.cents(common.estimate_usd(model_id, _chars(''), common.CHAT_OUT_TOKENS))} 起"
           "（系統提示加目前對話；工具查到的資料會再增加，送出後顯示這一輪實際花費）。"
           "不提供 Opus。對話寫進 logs/analyst/。")

for role, text in st.session_state.analyst_transcript:
    with st.chat_message(role):
        st.markdown(text)

prompt = st.chat_input("問一個台股問題，例如：鴻海近 12 個月營收趨勢", key="analyst_input")
if not st.session_state.analyst_transcript and not prompt:
    st.markdown("**可以這樣問**")
    st.caption("按一下只會填進下方輸入框，按 Enter 才送出（會花錢）。")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        cols[i % 2].button(q, key=f"ex_{i}", width="stretch", on_click=_fill, args=(q,))
if not prompt:
    st.stop()

est = common.estimate_usd(model_id, _chars(prompt), common.CHAT_OUT_TOKENS)
with st.chat_message("user"):
    st.markdown(prompt)
with st.chat_message("assistant"):
    stats: dict = {}
    messages = st.session_state.analyst_messages
    messages.append({"role": "user", "content": prompt})
    turn_start = len(messages)
    analyst.MODEL = model_id
    analyst.ASOF = pd.Timestamp(asof)
    analyst.names.cache_clear()
    analyst.universe.cache_clear()
    try:
        report.load_dotenv()
        client = anthropic.Anthropic()
        with st.spinner(f"{label.split('（')[0]} 查資料中，這一輪大約 {common.cents(est)} 起…"):
            answer = analyst.ask(client, messages, verbose=False, stats=stats)
    except Exception as e:
        messages.pop()
        st.error("沒有回答：" + common.public_error(e))
        st.stop()
    answer = answer or f"（沒有文字回答。stop_reason={stats.get('stop_reason') or '—'}）"
    spent = common.usage_usd(model_id, stats.get("usage") or {})
    tools = [b.name for m in messages[turn_start:] if m["role"] == "assistant" and isinstance(m["content"], list)
             for b in m["content"] if getattr(b, "type", None) == "tool_use"]
    note = (f"這一輪約 {common.cents(spent)}，{stats.get('api_calls', 0)} 次呼叫"
            + (f"，用了 {len(tools)} 次工具" if tools else "") + "（快取 token 依輸入價估算）。")
    st.markdown(answer)
    st.caption(note)

st.session_state.analyst_spent += spent
st.session_state.analyst_transcript.append(("user", prompt))
st.session_state.analyst_transcript.append(("assistant", answer + "\n\n" + f"*{note}*"))
try:
    analyst.log_turn(_log_path(), prompt, answer, messages)
except Exception as e:
    st.caption("紀錄沒寫成：" + common.public_error(e))
st.rerun()
