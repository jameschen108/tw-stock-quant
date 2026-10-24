"""本機研究介面。只綁 127.0.0.1，見 .streamlit/config.toml。"""
import sys
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common

st.set_page_config(page_title="台股研究（本機）", page_icon=":material/monitoring:", layout="wide",
                   initial_sidebar_state="collapsed")
common.setup()

page = st.navigation([
    st.Page("p1_overview.py", title="總覽", icon=":material/dashboard:", default=True),
    st.Page("p2_stock.py", title="個股", icon=":material/query_stats:", url_path="stock"),
    st.Page("p3_strategy.py", title="策略", icon=":material/science:", url_path="strategy"),
    st.Page("p4_paper.py", title="模擬持股", icon=":material/account_balance_wallet:", url_path="paper"),
    st.Page("p5_analyst.py", title="分析師問答", icon=":material/forum:", url_path="analyst"),
], position="top")
common.header()
page.run()
