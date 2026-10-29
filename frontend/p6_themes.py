"""主題：各主題的強弱排行、主題指數走勢、成分股。主題成分在 themes.toml。"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent)]

import common
import themes as T

SINCE = "2025-06-01"          # 只讀近一年多的行情，主題頁最長看 120 日
WINDOWS = {"20 日": 20, "60 日": 60, "120 日": 120}
PALETTE = ["#3b82f6", "#e5484d", "#18a058", "#f59e0b", "#8b5cf6", "#06b6d4", "#ec4899", "#64748b",
           "#84cc16", "#f97316", "#14b8a6"]


@st.cache_resource(show_spinner="讀取行情…", max_entries=1)
def _data(_sig: str):
    rets, val = T.load_prices(since=SINCE)
    return T.load_themes(), rets, val


def _summary(themes, rets, val) -> pd.DataFrame:
    s = T.summarize(themes, rets, val)
    return s.rename(columns={"name": "主題", "n": "成分數", "r1": "1日", "r5": "5日", "r20": "20日", "r60": "60日",
                             "ex5": "超額5日", "ex20": "超額20日", "ex60": "超額60日", "up_ratio": "20日上漲比例",
                             "value_share_5d": "成交值占比", "value_mult": "成交值倍數"})


common.page_title("主題", "依 themes.toml 手動分類的主題等權指數")
themes, rets, val = _data(common.data_sig())
st.caption(f"行情到 {rets.index.max().date()}。超額 = 主題報酬 − {T.ETF}；成交值倍數 = 近 5 日 / 近 60 日平均占全市場比重，"
           "大於 1 代表資金在流入。")

s = _summary(themes, rets, val)
cols = ["主題", "成分數", "1日", "5日", "20日", "60日", "超額20日", "超額60日", "20日上漲比例", "成交值占比", "成交值倍數"]
sort_by = common.segmented("排序", ["20日", "5日", "60日", "超額20日", "成交值倍數"], "20日", "theme_sort",
                           label_visibility="collapsed")
s = s.sort_values(sort_by, ascending=False).reset_index(drop=True)
ret_cols = ["1日", "5日", "20日", "60日", "超額20日", "超額60日"]
show = s[cols]
ev = common.table(common.styled(show, fmt={**{c: "{:+.1%}" for c in ret_cols}, "20日上漲比例": "{:.0%}",
                                           "成交值占比": "{:.1%}", "成交值倍數": "{:.2f}x"}, signed=ret_cols),
                  hide_index=True, height=min(450, 38 + 35 * len(show)), on_select="rerun",
                  selection_mode="single-row", key="theme_tbl")
sel = ev.selection.rows if ev and ev.selection else []
if not sel:
    st.caption("點一列看成分股。")

win_label = common.segmented("走勢區間", list(WINDOWS), "60 日", "theme_win")
n = WINDOWS[win_label]
keys = dict(zip(s["主題"], s["theme"]))
default = list(s["主題"].head(3))
chosen = st.multiselect("走勢圖主題", list(s["主題"]), default=default, key="theme_pick")
series = {}
for name in chosen:
    d = T.theme_returns(rets, themes[keys[name]]["codes"])
    series[name] = T.index_levels(d, n)
if T.ETF in rets.columns:
    series[T.ETF] = T.index_levels(rets[T.ETF], n)
if chosen:
    xs = series[chosen[0]].index
    colors = {nm: PALETTE[i % len(PALETTE)] for i, nm in enumerate(chosen)} | {T.ETF: common.C_ETF}
    common.chart(common.line_fig(xs, {k: v.reindex(xs) for k, v in series.items()}, f"累積指數（起點 = 1，近 {n} 日）",
                                 colors, 380))

pick = s.iloc[sel[0]]["主題"] if sel else None
if pick:
    t = themes[keys[pick]]
    st.subheader(pick)
    st.caption(t.get("desc", ""))
    names = common.names_frame(common.data_sig())["name"]
    m = T.members(t, rets, names).rename(columns={"code": "代號", "name": "名稱", "r1": "1日", "r5": "5日",
                                                  "r20": "20日", "r60": "60日"})
    ev2 = common.table(common.styled(m, fmt={c: "{:+.1%}" for c in ["1日", "5日", "20日", "60日"]},
                                     signed=["1日", "5日", "20日", "60日"]),
                       hide_index=True, on_select="rerun", selection_mode="single-row", key="theme_members")
    sel2 = ev2.selection.rows if ev2 and ev2.selection else []
    if sel2:
        common.goto_stock(str(m.iloc[sel2[0]]["代號"]))

st.markdown(f'<div class="disc">{common.esc(common.DISCLAIMER)} 成分股是現在手動挑的，有事後偏差，'
            "只看當下相對強弱，不拿來回測。</div>", unsafe_allow_html=True)
