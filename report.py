# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15", "anthropic>=0.70"]
# ///
"""單一個股分析報告 MVP：整理事實表（JSON）→ 選擇性 LLM 文字摘要 → 自足的 HTML 報告。

用法:
  uv run report.py 2330                    # 以最新資料日為準
  uv run report.py 2603 --asof 2024-06-30  # 時點回溯：只用當時已公布的資料
  uv run report.py 2330 --no-llm           # 不呼叫 LLM

輸出 reports/{code}_{asof}.json / .html。
LLM 摘要需要 Anthropic 憑證（ANTHROPIC_API_KEY 或 `ant auth login`）；沒有就只產生數據報告。
報告附 outlook.py 的短 / 中 / 長期模型標籤（看漲 / 看跌，附歷史校準）；不給目標價。學術研究用。
"""
import argparse
import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from build_panel import OUT
from composite import GROUPS
from factors import FACTORS
from fetch_chips import load_chips, load_margin, load_qfii
from matrix import stock_matrix
from outlook import stock_outlook
from patterns import stock_patterns
from public_data import snapshots

REPORTS = Path(__file__).parent / "reports"
MODEL = "claude-haiku-5-5"
UNIV = dict(min_price=5, min_value=5e6, min_obs=120)
LOWER_IS_BETTER = {"accruals", "asset_growth", "debt_ratio", "rev_1m"}
GROUP_NAMES = {"rev_mom": "營收動能", "quality": "品質", "value": "價值", "trend": "趨勢"}


# ---------------------------------------------------------------- 事實表

def _f(x, nd=2):
    """轉成可 JSON 化的數字；NaN -> None。"""
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return None
    return round(float(x), nd)


def _pct_rank(s: pd.Series, v) -> float | None:
    s = s.dropna()
    if pd.isna(v) or s.empty:
        return None
    return _f((s < v).mean() + 0.5 * (s == v).mean(), 3)


def price_section(code: str, asof: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    px = pd.read_parquet(OUT / "prices_daily.parquet",
                         columns=["date", "code", "name", "market", "close", "adj_close", "value", "ret"],
                         filters=[("code", "==", code)])
    px = px[px["date"] <= asof].dropna(subset=["close"]).sort_values("date")
    if px.empty:
        raise SystemExit(f"{code} 在 {asof.date()} 前沒有行情")
    last = px.iloc[-1]
    a = px.set_index("date")["adj_close"]

    def ret(days):
        past = a[a.index <= last["date"] - pd.Timedelta(days=days)]
        return _f(a.iloc[-1] / past.iloc[-1] - 1, 4) if len(past) else None

    y1 = a[a.index > last["date"] - pd.Timedelta(days=365)]
    # adj_close 以整份資料最後一天為基準；換成 asof 當天的收盤基準，才能和 close 直接比
    y1_asof = y1 * last["close"] / a.iloc[-1]
    r = px.set_index("date")["ret"].dropna()
    sec = {
        "date": str(last["date"].date()), "name": last["name"], "market": str(last["market"]),
        "close": _f(last["close"]), "ret_1m": ret(30), "ret_3m": ret(91), "ret_6m": ret(182), "ret_12m": ret(365),
        "high_52w_adj": _f(y1_asof.max()), "low_52w_adj": _f(y1_asof.min()),
        "from_52w_high": _f(a.iloc[-1] / y1.max() - 1, 4),
        "ma200_gap": _f(a.iloc[-1] / a.tail(200).mean() - 1, 4) if len(a) >= 200 else None,
        "vol_60d_ann": _f(r.tail(60).std() * np.sqrt(244), 4),
        "max_drawdown_1y": _f((y1 / y1.cummax() - 1).min(), 4),
        "avg_value_20d_億": _f(px["value"].tail(20).mean() / 1e8),
    }
    return sec, px[px["date"] > last["date"] - pd.Timedelta(days=3 * 365)][["date", "adj_close"]]


def revenue_section(code: str, asof: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    cols = ["code", "period", "avail_date", "以合併為主單月營收(千元)", "以合併為主單月營收成長率％",
            "單月營收與上月比％", "近3月累計營收成長率", "近12月累計營收成長率", "與歷史最高單月營收比%",
            "營收變化說明"]
    rv = pd.read_parquet(OUT / "tej_rev.parquet", columns=cols, filters=[("code", "==", code)])
    rv = rv[rv["avail_date"] < asof].sort_values("period").tail(24)
    if rv.empty:
        return {}, rv
    last = rv.iloc[-1]
    notes = [{"period": f"{p:%Y-%m}", "note": n} for p, n in
             zip(rv["period"].tail(3), rv["營收變化說明"].tail(3)) if isinstance(n, str) and n.strip()]
    sec = {
        "latest_period": f"{last['period']:%Y-%m}", "published": str(last["avail_date"].date()),
        "revenue_億": _f(last["以合併為主單月營收(千元)"] / 1e5),
        "yoy_pct": _f(last["以合併為主單月營收成長率％"]), "mom_pct": _f(last["單月營收與上月比％"]),
        "yoy_3m_pct": _f(last["近3月累計營收成長率"]), "yoy_12m_pct": _f(last["近12月累計營收成長率"]),
        "vs_record_high_pct": _f(last["與歷史最高單月營收比%"]),
        "yoy_last_6m_pct": [_f(x) for x in rv["以合併為主單月營收成長率％"].tail(6)],
        "company_notes": notes,
    }
    return sec, rv


def fin_section(code: str, asof: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    cols = ["code", "period", "avail_date", "單季(Q)/單半年(H)", "營業收入淨額", "營業毛利率", "營業利益率",
            "稅後淨利率", "每股盈餘", "ROE(A)－稅後", "來自營運之現金流量", "歸屬母公司淨利（損）", "合併總損益",
            "負債比率", "每股淨值(B)", "TEJ主產業代碼"]
    fn = pd.read_parquet(OUT / "tej_fin.parquet", columns=cols, filters=[("code", "==", code)])
    fn = fn[fn["avail_date"] < asof].sort_values("period").tail(8)
    if fn.empty:
        return {}, fn
    last = fn.iloc[-1]
    q4 = fn.tail(4)
    ttm_ok = len(q4) == 4 and q4["單季(Q)/單半年(H)"].eq("Q").all() and \
        (q4["period"].iloc[-1].to_period("Q") - q4["period"].iloc[0].to_period("Q")).n == 3
    prev_y = fn[fn["period"] == last["period"] - pd.DateOffset(years=1) + pd.offsets.MonthEnd(0)]
    sec = {
        "latest_quarter": f"{last['period'].year}Q{last['period'].quarter}", "published": str(last["avail_date"].date()),
        "revenue_億": _f(last["營業收入淨額"] / 1e5), "gross_margin_pct": _f(last["營業毛利率"]),
        "op_margin_pct": _f(last["營業利益率"]), "net_margin_pct": _f(last["稅後淨利率"]),
        "eps": _f(last["每股盈餘"]), "roe_quarter_pct": _f(last["ROE(A)－稅後"]),
        "debt_ratio_pct": _f(last["負債比率"]), "bvps": _f(last["每股淨值(B)"]),
        "net_income_total_億": _f(last["合併總損益"] / 1e5), "net_income_parent_億": _f(last["歸屬母公司淨利（損）"] / 1e5),
        "eps_ttm": _f(q4["每股盈餘"].sum()) if ttm_ok else None,
        "cfo_to_ni_ttm": _f(q4["來自營運之現金流量"].sum() / q4["歸屬母公司淨利（損）"].sum())
        if ttm_ok and q4["歸屬母公司淨利（損）"].sum() > 0 else None,
        "yoy_same_quarter": None if prev_y.empty else {
            "revenue_pct": _f((last["營業收入淨額"] / prev_y["營業收入淨額"].iloc[0] - 1) * 100),
            "gross_margin_chg_pt": _f(last["營業毛利率"] - prev_y["營業毛利率"].iloc[0]),
            "eps_prev": _f(prev_y["每股盈餘"].iloc[0]),
        },
    }
    tot, par = last["合併總損益"], last["歸屬母公司淨利（損）"]
    if pd.notna(tot) and pd.notna(par) and tot * par < 0:
        sec["note"] = (f"稅後淨利率用合併總損益（{tot / 1e5:,.1f} 億，含非控制權益），EPS 用歸屬母公司淨利"
                       f"（{par / 1e5:,.1f} 億）；兩者正負號不同是因為非控制權益承擔了 {(tot - par) / 1e5:,.1f} 億，"
                       "不是資料錯誤")
    return sec, fn


def industry_info(asof: pd.Timestamp) -> pd.DataFrame:
    """每檔股票 asof 前最新財報的 TEJ 主 / 子產業，加上交易所產業中文名（OpenAPI 月營收快照）。"""
    f = pd.read_parquet(OUT / "tej_fin.parquet", columns=["code", "period", "avail_date", "TEJ主產業代碼", "TEJ子產業代碼"])
    f = f[f["avail_date"] < asof].sort_values("period").groupby("code")[["TEJ主產業代碼", "TEJ子產業代碼"]].last()
    rv = snapshots("rev_")
    if not rv.empty:
        f["產業名稱"] = rv.sort_values("snap_date").groupby("code")["產業別"].last()
    return f


def score_section(code: str, asof: pd.Timestamp) -> tuple[dict, pd.DataFrame, pd.Series]:
    fm = pd.read_parquet(OUT / "factors_monthly.parquet")
    fm = fm[fm["date"] <= asof]
    date = fm["date"].max()
    cur = fm[fm["date"] == date]
    me = cur[cur["code"] == code]
    univ = cur[(cur["close"] >= UNIV["min_price"]) & (cur["avg_value"] >= UNIV["min_value"])
               & (cur["n_obs"] >= UNIV["min_obs"])]

    sc = pd.read_parquet(OUT / "bt" / "composite_scores.parquet")
    sc = sc[sc["date"] <= asof]
    s_now = sc[sc["date"] == sc["date"].max()]
    mine = s_now[s_now["code"] == code]
    sec = {"score_date": str(date.date()), "in_universe": bool(code in set(univ["code"]))}
    if not mine.empty and pd.notna(mine["comp_equal"].iloc[0]):
        groups = [g for g in GROUP_NAMES if g in s_now]       # 舊版 composite_scores 沒有 trend
        rk = s_now.set_index("code")[["comp_equal", *groups]].rank(pct=True).loc[code]
        sec["composite_pct"] = _f(rk["comp_equal"], 3)
        sec["groups_pct"] = {GROUP_NAMES[g]: _f(rk[g], 3) for g in groups}
    # 個別因子：股票池內百分位（值越大越好的方向已轉正）
    fac = {}
    if not me.empty:
        for f, desc in FACTORS.items():
            v = me[f].iloc[0]
            p = _pct_rank(univ[f], v)
            if p is not None and f in LOWER_IS_BETTER:
                p = _f(1 - p, 3)
            grp = next((GROUP_NAMES[g] for g, m in GROUPS.items() if f in m), "不在綜合分數內")
            fac[f] = {"desc": desc, "group": grp, "value": _f(v, 4), "pct_good": p}
    sec["factors"] = fac

    # 歷史脈絡：2019 起各綜合分數五分位的下月平均超額（全池等權為基準）
    h = sc[(sc["date"] >= "2019-01-01")].dropna(subset=["comp_equal", "fwd_ret"]).copy()
    h["q"] = np.ceil(h.groupby("date")["comp_equal"].rank(pct=True) * 5).astype(int)
    h["ex"] = h["fwd_ret"] - h.groupby("date")["fwd_ret"].transform("mean")
    qx = h.groupby("q")["ex"].mean() * 100
    sec["quintile_next_month_excess_pct_since_2019"] = {f"Q{k}": _f(v) for k, v in qx.items()}

    # 綜合分數百分位走勢（近 36 期）
    sc = sc[sc["date"] > date - pd.DateOffset(months=36)]
    hist = sc.assign(p=sc.groupby("date")["comp_equal"].rank(pct=True))
    trend = hist[hist["code"] == code].set_index("date")["p"]

    # 同業：同 TEJ 子產業至少 3 檔就用子產業，否則用主產業
    info = industry_info(asof)
    names = pd.read_parquet(OUT / "prices_daily.parquet", columns=["code", "name"]).drop_duplicates("code", keep="last")
    cur = cur.merge(info, left_on="code", right_index=True, how="left").merge(names, on="code", how="left")
    mine_info = info.loc[code] if code in info.index else pd.Series(dtype=object)
    sub, main = mine_info.get("TEJ子產業代碼"), mine_info.get("TEJ主產業代碼")
    if pd.notna(sub) and (cur["TEJ子產業代碼"] == sub).sum() >= 3:
        level, peers = "子產業", cur[cur["TEJ子產業代碼"] == sub].copy()
    elif pd.notna(main):
        level, peers = "主產業", cur[cur["TEJ主產業代碼"] == main].copy()
    else:
        level, peers = None, cur.iloc[:0].copy()
    peers["pe"] = np.where(peers["ep"] > 0, 1 / peers["ep"], np.nan)
    peers["pb"] = np.where(peers["bp"] > 0, 1 / peers["bp"], np.nan)
    leaders = peers.sort_values("mcap", ascending=False)["name"].dropna().head(3)
    sec["industry"] = {"name": mine_info.get("產業名稱"), "tej_main": main, "tej_sub": sub,
                       "peer_level": level, "peer_group": "、".join(leaders), "n_stocks": int(len(peers)),
                       "median": {k: _f(peers[c].median()) for k, c in
                                  [("rev_yoy_pct", "rev_yoy"), ("roe_quarter_pct", "roe"),
                                   ("gross_margin_pct", "gross_margin"), ("pe_ttm", "pe"), ("pb", "pb")]}}
    if not me.empty:
        ep, bp = me["ep"].iloc[0], me["bp"].iloc[0]
        own = fm[(fm["code"] == code) & (fm["date"] > date - pd.DateOffset(years=5))]
        sec["valuation"] = {
            "pe_ttm": _f(1 / ep) if ep > 0 else None, "pb": _f(1 / bp) if bp > 0 else None,
            "mcap_億": _f(me["mcap"].iloc[0] / 1e5, 0),
            # 便宜程度：目前 E/P、B/P 在自己 5 年歷史的百分位（越高 = 比過去越便宜）
            "pe_cheapness_vs_own_5y_pct": _pct_rank(own["ep"], ep),
            "pb_cheapness_vs_own_5y_pct": _pct_rank(own["bp"], bp),
        }
    pmap = s_now.set_index("code")["comp_equal"].rank(pct=True)
    peers = peers.sort_values("mcap", ascending=False).head(8)
    sec["peers"] = [{"code": r.code, "name": r.name, "mcap_億": _f(r.mcap / 1e5, 0), "rev_yoy_pct": _f(r.rev_yoy),
                     "roe_quarter_pct": _f(r.roe), "pe_ttm": _f(r.pe, 1), "composite_pct": _f(pmap.get(r.code), 3)}
                    for r in peers.itertuples()]
    return sec, peers, trend


def chips_section(code: str, asof: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    """三大法人買賣超：近 5 / 20 / 60 個交易日合計（張）與佔成交量比例、外資連續買賣超天數。"""
    ch = load_chips(code)
    ch = ch[ch["date"] <= asof].sort_values("date")
    if ch.empty:
        return {}, ch
    vol = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "volume"],
                          filters=[("code", "==", code)]).set_index("date")["volume"]
    ch = ch.set_index("date")
    ch["volume"] = vol.reindex(ch.index)
    sec = {"last_date": str(ch.index[-1].date()), "days_available": int(len(ch))}
    for n in (5, 20, 60):
        if len(ch) < n:
            continue
        w = ch.tail(n)
        sec[f"{n}d"] = {k: {"net_lots": _f(w[k].sum() / 1000, 0),
                            "pct_of_volume": _f(w[k].sum() / w["volume"].sum() * 100) if w["volume"].sum() else None}
                        for k in ("foreign", "trust", "dealer", "total")}
    sign = np.sign(ch["foreign"])
    streak = (sign != sign.shift()).cumsum()
    run = int((streak == streak.iloc[-1]).sum())
    sec["foreign_streak"] = {"direction": "買超" if sign.iloc[-1] > 0 else ("賣超" if sign.iloc[-1] < 0 else "持平"),
                             "days": run}
    return sec, ch.tail(60).reset_index()


def credit_section(code: str, asof: pd.Timestamp) -> dict:
    """融資融券餘額與外資持股比率：最新值與近 5 / 20 / 60 個交易日變化。"""
    sec = {}
    mg = load_margin(code)
    mg = mg[mg["date"] <= asof].sort_values("date").reset_index(drop=True)
    if len(mg):
        last = mg.iloc[-1]
        chg = lambda col, n: _f(last[col] - mg[col].iloc[-1 - n], 0) if len(mg) > n else None
        sec["margin"] = {
            "last_date": str(last["date"].date()), "unit": "張",
            "margin_balance": _f(last["margin_bal"], 0), "margin_chg_5d": chg("margin_bal", 5),
            "margin_chg_20d": chg("margin_bal", 20), "margin_chg_60d": chg("margin_bal", 60),
            "margin_usage_pct": _f(last["margin_bal"] / last["margin_limit"] * 100) if last["margin_limit"] else None,
            "short_balance": _f(last["short_bal"], 0), "short_chg_20d": chg("short_bal", 20),
            # 券資比 = 融券餘額 / 融資餘額
            "short_to_margin_pct": _f(last["short_bal"] / last["margin_bal"] * 100) if last["margin_bal"] else None,
        }
    qf = load_qfii(code)
    qf = qf[qf["date"] <= asof].sort_values("date").reset_index(drop=True)
    if len(qf):
        last = qf.iloc[-1]
        chg = lambda n: _f(last["foreign_pct"] - qf["foreign_pct"].iloc[-1 - n]) if len(qf) > n else None
        hi = qf[qf["date"] > last["date"] - pd.DateOffset(years=1)]["foreign_pct"]
        sec["foreign_holding"] = {
            "last_date": str(last["date"].date()), "foreign_pct": _f(last["foreign_pct"]),
            "chg_5d_pt": chg(5), "chg_20d_pt": chg(20), "chg_60d_pt": chg(60),
            "high_1y_pct": _f(hi.max()), "low_1y_pct": _f(hi.min()), "limit_pct": _f(last["limit_pct"]),
            "days_available": int(len(qf)),
        }
    return sec


def announcements_section(code: str, asof: pd.Timestamp, days: int = 90, n_llm: int = 12) -> tuple[dict, pd.DataFrame]:
    """近 days 天的重大訊息（fetch_mops.py）。事實表只放最近 n_llm 則、內文截到 400 字。"""
    f = OUT / "announcements.parquet"
    if not f.exists():
        return {}, pd.DataFrame()
    a = pd.read_parquet(f, filters=[("code", "==", code)])
    end = asof + pd.Timedelta(days=1)                      # asof 當天收盤後發布的也算
    a = a[(a["time"] < end) & (a["time"] >= end - pd.Timedelta(days=days))].sort_values("time", ascending=False)
    sec = {"window_days": days, "count": int(len(a)),
           "items": [{"time": f"{r.time:%Y-%m-%d %H:%M}", "subject": r.subject,
                      "content_excerpt": (r.content or "")[:400]} for r in a.head(n_llm).itertuples()]}
    return sec, a


def events_section(code: str, asof: pd.Timestamp) -> list:
    ev = pd.read_parquet(OUT / "price_events.parquet", filters=[("code", "==", code)])
    ev = ev[ev["date"] <= asof].sort_values("date").tail(4)
    kinds = {"div": "除權息", "reduction": "減資", "par_change": "變更面額", "split": "ETF 分割 / 反分割"}
    return [{"date": str(r.date.date()), "kind": kinds.get(r.kind, r.kind), "prev_close": _f(r.prev_close), "ref": _f(r.ref)}
            for r in ev.itertuples()]


def build_facts(code: str, asof: pd.Timestamp, extended: bool = False) -> tuple[dict, dict]:
    """extended 只給報告用：outlook 附各標籤勝率、加訊號矩陣章節。analyst.stock_facts 也是 llm_trader 的輸入，
    預設不加，避免前向實驗前後條件不一致。"""
    price, px = price_section(code, asof)
    rev, rv = revenue_section(code, asof)
    fin, fn = fin_section(code, asof)
    score, _, trend = score_section(code, asof)
    chips, ch = chips_section(code, asof)
    credit = credit_section(code, asof)
    ann, an = announcements_section(code, asof)
    actions = events_section(code, asof)
    notes = ["營收、財報只用 asof 前已公布的資料", "百分位 0~1，越高越好（估值類越高越便宜）",
             "綜合分數 = 營收動能 + 品質 + 價值 + 趨勢四組等權，產業與市值中性；groups_pct 是中性化後的百分位",
             "趨勢組 = price.from_52w_high（距 52 週高點）與 price.ma200_gap（收盤相對 200 日均線）",
             "不要推測組別百分位或綜合分數高低的原因（例如同業成長也很快、某項指標拉低）；資料沒有拆解中性化的過程，只能陳述百分位本身",
             "factors.pct_good 是原始因子值在全股票池的百分位，未中性化，size / mom_12_1 / rev_1m 不在綜合分數內"]
    if not fn.empty:
        q_end = fn["period"].iloc[-1]
        changed = [x for x in actions if x["kind"] in ("減資", "變更面額") and pd.Timestamp(x["date"]) > q_end]
        if changed:
            notes.append(f"{'、'.join(x['date'] + ' ' + x['kind'] for x in changed)} 在最新財報季底之後："
                         "financials 的 eps、eps_ttm、bvps 是變動前的股數基準，不能直接和 close 相除；"
                         "估值請用 valuation 的 pe_ttm、pb（以市值計算）")
    facts = {"code": code, "asof": price["date"], "price": price, "revenue": rev, "financials": fin,
             "scores": score, "institutional_flows": chips,
             "margin_foreign_holding": credit, "outlook": stock_outlook(code, asof, extended), "announcements": ann,
             "corporate_actions": actions, "notes": notes}
    if extended:
        facts["signal_matrix"] = stock_matrix(facts, stock_patterns(code))
    return facts, {"px": px, "rv": rv, "fn": fn, "trend": trend, "chips": ch, "ann": an}


# ---------------------------------------------------------------- LLM 摘要

NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "positives": {"type": "array", "items": {"type": "string"}},
        "negatives": {"type": "array", "items": {"type": "string"}},
        "watch_items": {"type": "array", "items": {"type": "string"}},
        "outlook": {"type": "string"},
    },
    "required": ["summary", "outlook", "positives", "negatives", "watch_items"],
    "additionalProperties": False,
}

SYSTEM = """你是台股研究助理，替一位個人投資者整理單一個股的數據摘要。
只能使用使用者提供的 JSON 事實表，不得引用表外的新聞、產業知識或數字；表裡沒有的就說資料不足。
每個論點都要帶出具體數字與期別（例如「2026-09 營收年增 54.6%」）。
不要給目標價或具體的價格、報酬數字預測。看漲 / 看跌只能用 outlook 的模型標籤（見下方規則），不要自己另外判斷方向。

寫法規則：
- 只用中文名稱，不要寫 JSON 欄位名（rev_accel、ep、pe_ttm 這類）或英文單字（median 寫成中位數）；因子名稱用 factors 裡的 desc。
- 同一個數字只提一次；價值類因子（股東權益 / 市值、營收 / 市值）不要改稱為股價淨值比、股價營收比。
- 百分位寫成「第 N 百分位」（0.388 寫成第 39 百分位），不要寫小數；其他數字最多保留一位小數。
- 百分位越高越好。估值的 pe_cheapness_vs_own_5y_pct、pb_cheapness_vs_own_5y_pct 是「便宜程度」百分位：
  第 24 百分位要寫成「本益比比自己過去 5 年約 76% 的時間都高（偏貴）」，不要寫成「本益比在第 24 百分位」。
- 提到因子屬於哪一組時，只能照 factors.group 寫；group 是「不在綜合分數內」的不要說成任何一組。
- groups_pct 與 composite_pct 是與同產業、同市值等級公司比較的百分位；factors.pct_good 是未中性化的全市場百分位，兩者可能差很多，引用時要說清楚是哪一種。
- 「與歷史最高單月營收比」接近 0 代表營收接近歷史新高，不是負面訊息。
- 同業（peers、industry.median）只用來對照本股；其他公司本身的好壞不要列為本股的正面或負面。
- announcements 是公司在公開資訊觀測站發布的重大訊息（標題與內文節錄）：可以引用事件與日期，
  但只能用文中寫的內容；不要推測事件對股價或獲利的影響，例行公告（股東會、法說會通知）不必逐一列出。
- institutional_flows 是三大法人買賣超（張）與佔成交量比例，只描述資金流向，不要推論成價格會漲或跌。
- margin_foreign_holding 是融資融券餘額（張）、券資比與外資持股比率（%、變化單位是百分點），一樣只描述，不要推論價格方向。

- outlook 是模型依歷史統計給的短 / 中 / 長期標籤。outlook 欄位 2~4 句：依序寫三個期間的標籤（例如「短期模型標籤：看漲」），
  每個附上持有期與樣本外歷史：同標籤平均產業中性超額、和中性組比較；可靠度是「弱」或「不可靠」的要明說。
  若某期間看漲組的歷史平均超額和中性組差不到 0.5 個百分點，要說明這個看漲標籤的區分力有限（主要是看跌有區分力）。
  勝率要用 hit_pct_by_label_oos 和其他標籤比較，不能單獨說成「勝率不到一半所以不好」。這是模型的統計分組，不要寫成你自己的看法或保證。
- signal_matrix 是技術、籌碼、基本面、綜合四個維度依固定規則判斷的多 / 空 / 中性、一致數與矛盾清單。
  提到維度方向、一致或矛盾時只能照 dimensions、agreement、contradictions 寫；不要自己另外比對出新的矛盾，
  也不要從一致數推論股價會漲或跌。

summary 3~4 句；positives、negatives 各 2~4 點；watch_items 2~3 點，寫成下次財報或月營收可以檢查的具體項目。用繁體中文。"""


def freshness_warning() -> str | None:
    """update.py 留下的狀態：太久沒更新、行情落後或上次有問題時回傳提醒文字。"""
    f = OUT / "update_status.json"
    if not f.exists():
        return "找不到 data/update_status.json：還沒跑過 uv run update.py，資料可能不是最新"
    st = json.loads(f.read_text())
    msgs = []
    run_at = pd.Timestamp(st["run_at"])
    if pd.Timestamp.now() - run_at > pd.Timedelta(hours=24):
        msgs.append(f"資料上次更新是 {run_at:%Y-%m-%d %H:%M}")
    lag = len(pd.bdate_range(st["last_price_date"], pd.Timestamp.today().normalize())) - 1
    if lag > 1:
        msgs.append(f"最新行情是 {st['last_price_date']}（落後 {lag} 個工作天）")
    if st.get("problems"):
        msgs.append("上次更新有警告：" + "；".join(st["problems"]))
    return ("；".join(msgs) + "。分析前建議先跑 uv run update.py") if msgs else None


def load_dotenv(path: Path = Path(__file__).parent / ".env"):
    """讀專案 .env（KEY=VALUE），已存在的環境變數不覆蓋。"""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def llm_narrative(facts: dict) -> dict | None:
    import anthropic
    load_dotenv()
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(  # Haiku 5.5 沒有伺服器端備援，被拒絕時直接略過摘要
            model=MODEL,
            max_tokens=16000,
            output_config={"effort": "medium",
                           "format": {"type": "json_schema", "schema": NARRATIVE_SCHEMA}},
            system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps(facts, ensure_ascii=False)}],
        )
    except anthropic.AuthenticationError:
        print("LLM: 沒有可用的 Anthropic 憑證（設定 ANTHROPIC_API_KEY 或執行 ant auth login），略過文字摘要")
        return None
    except anthropic.APIStatusError as e:
        print(f"LLM: API 錯誤 {e.status_code} {e.message}，略過文字摘要")
        return None
    except anthropic.APIConnectionError:
        print("LLM: 連線失敗，略過文字摘要")
        return None
    except TypeError as e:  # 沒有任何憑證時 SDK 在建立請求前就會丟出
        print(f"LLM: {e}，略過文字摘要")
        return None
    if resp.stop_reason == "refusal":
        print("LLM: 請求被拒絕，略過文字摘要")
        return None
    text = next((b.text for b in resp.content if b.type == "text"), "")
    return json.loads(text)


# ---------------------------------------------------------------- 圖表（inline SVG）

W, H, PAD_L, PAD_R, PAD_T, PAD_B = 560, 180, 48, 56, 12, 24


def _scale(vals, lo=None, hi=None):
    """回傳對齊整數刻度的 (lo, hi, ticks)；lo / hi 給定時照用不留白。"""
    v = [x for x in vals if x is not None and np.isfinite(x)]
    fixed = lo is not None and hi is not None
    lo = min(v) if lo is None else lo
    hi = max(v) if hi is None else hi
    if hi == lo:
        hi = lo + 1
    raw = (hi - lo) / 3
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    if not fixed:
        lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
    ticks = np.arange(np.ceil(lo / step) * step, hi + step / 2, step)
    return lo, hi, ticks


def _y(v, lo, hi):
    return PAD_T + (H - PAD_T - PAD_B) * (1 - (v - lo) / (hi - lo))


def _axes(lo, hi, ticks, xlabels, fmt="{:,.0f}", zero=False):
    out = []
    for t in ticks:
        y = _y(t, lo, hi)
        out.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                   f'<text x="{PAD_L - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{fmt.format(t)}</text>')
    if zero and lo < 0 < hi:
        y = _y(0, lo, hi)
        out.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" class="zero"/>')
    n = len(xlabels)
    for i, lab in enumerate(xlabels):
        if lab:
            x = PAD_L + (W - PAD_L - PAD_R) * (i + 0.5) / n
            out.append(f'<text x="{x:.1f}" y="{H - 6}" class="tick" text-anchor="middle">{lab}</text>')
    return "".join(out)


def svg_bars(labels, vals, fmt, title, tip_fmt):
    lo, hi, ticks = _scale(vals, lo=0)
    n = len(vals)
    bw = (W - PAD_L - PAD_R) / n
    bars = []
    for i, (lab, v) in enumerate(zip(labels, vals)):
        if v is None or not np.isfinite(v):
            continue
        x = PAD_L + i * bw + 1
        y = _y(v, lo, hi)
        h = _y(0, lo, hi) - y
        bars.append(f'<g class="hit"><rect x="{x - 1:.1f}" y="{PAD_T}" width="{bw:.1f}" height="{H - PAD_T - PAD_B}" '
                    f'fill="transparent"/><path d="M{x:.1f},{y + h:.1f} v{-h + 2:.1f} q0,-2 2,-2 h{bw - 6:.1f} '
                    f'q2,0 2,2 v{h - 2:.1f} z" class="bar"/><title>{lab}：{tip_fmt.format(v)}</title></g>')
    xl = [lab if i % max(1, n // 6) == 0 else "" for i, lab in enumerate(labels)]
    return _wrap(title, _axes(lo, hi, ticks, xl, fmt) + "".join(bars))


def svg_lines(labels, series: dict, fmt, title, tip_fmt, zero=False, lo=None, hi=None, ref=None):
    allv = [v for s in series.values() for v in s]
    lo, hi, ticks = _scale(allv + ([ref] if ref is not None else []), lo, hi)
    n = len(labels)
    x = [PAD_L + (W - PAD_L - PAD_R) * (i + 0.5) / n for i in range(n)]
    out = []
    if ref is not None:
        y = _y(ref, lo, hi)
        out.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y:.1f}" y2="{y:.1f}" class="ref"/>')
    for k, (name, vals) in enumerate(series.items()):
        pts = [(x[i], _y(v, lo, hi)) for i, v in enumerate(vals) if v is not None and np.isfinite(v)]
        if not pts:
            continue
        d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        out.append(f'<path d="{d}" class="line s{k + 1}"/>')
        if len(series) > 1:  # 多條線：線尾直接標名
            out.append(f'<text x="{pts[-1][0] + 6:.1f}" y="{pts[-1][1] + 4:.1f}" class="lab">{name}</text>')
    # 每個 x 一條透明熱區，title 顯示該點所有數值
    for i in range(n):
        vals = "、".join(f"{nm} {tip_fmt.format(s[i])}" for nm, s in series.items()
                        if s[i] is not None and np.isfinite(s[i]))
        if vals:
            w = (W - PAD_L - PAD_R) / n
            out.append(f'<rect x="{x[i] - w / 2:.1f}" y="{PAD_T}" width="{w:.1f}" height="{H - PAD_T - PAD_B}" '
                       f'class="hit" fill="transparent"><title>{labels[i]}：{vals}</title></rect>')
    xl = [lab if i % max(1, n // 6) == 0 else "" for i, lab in enumerate(labels)]
    return _wrap(title, _axes(lo, hi, ticks, xl, fmt, zero) + "".join(out))


def _wrap(title, body):
    return (f'<figure><figcaption>{title}</figcaption>'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}">{body}</svg></figure>')


# ---------------------------------------------------------------- HTML

CSS = """
:root{--surface:#fcfcfb;--panel:#f4f3f0;--text:#0b0b0b;--text2:#52514e;--muted:#8a8984;--grid:#e4e3df;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#0ca30c;--bad:#d03b3b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--surface:#1a1a19;--panel:#242422;--text:#f0efec;
--text2:#c3c2b7;--muted:#8f8e86;--grid:#33332f;--s1:#3987e5;--s2:#d95926;--s3:#199e70}}
:root[data-theme="dark"]{--surface:#1a1a19;--panel:#242422;--text:#f0efec;--text2:#c3c2b7;--muted:#8f8e86;
--grid:#33332f;--s1:#3987e5;--s2:#d95926;--s3:#199e70}
body{background:var(--surface);color:var(--text);font:15px/1.6 -apple-system,"PingFang TC","Noto Sans TC",sans-serif;
margin:0;padding:24px 16px}
main{max-width:1100px;margin:0 auto}
h1{font-size:26px;margin:0}h2{font-size:18px;margin:32px 0 8px;border-bottom:1px solid var(--grid);padding-bottom:4px}
.sub{color:var(--text2)}.muted{color:var(--muted);font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--panel);border-radius:8px;padding:10px 12px}.tile b{display:block;font-size:20px}
.tile span{color:var(--text2);font-size:13px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
figure{margin:0;background:var(--panel);border-radius:8px;padding:8px 8px 4px}
figcaption{font-size:13px;color:var(--text2);margin:0 0 4px 4px}svg{width:100%;height:auto;display:block}
.grid{stroke:var(--grid);stroke-width:1}.zero{stroke:var(--muted);stroke-width:1}
.ref{stroke:var(--muted);stroke-dasharray:4 4}
.tick,.lab{fill:var(--text2);font-size:11px}.bar{fill:var(--s1)}.hit:hover .bar{opacity:.75}
.line{fill:none;stroke-width:2;stroke-linejoin:round}.s1{stroke:var(--s1)}.s2{stroke:var(--s2)}.s3{stroke:var(--s3)}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:5px 8px;text-align:right;
border-bottom:1px solid var(--grid)}th:first-child,td:first-child{text-align:left}th{color:var(--text2);font-weight:500}
.wrap{overflow-x:auto}
.bar-pct{display:inline-block;height:8px;border-radius:4px;background:var(--s1);vertical-align:middle;margin-right:6px}
ul{padding-left:20px}.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.box{background:var(--panel);border-radius:8px;padding:4px 16px}
.pos li::marker{content:"＋ ";color:var(--good)}.neg li::marker{content:"－ ";color:var(--bad)}
.disc{margin-top:40px;color:var(--muted);font-size:12px}
.ann-list{list-style:none;padding:0}.ann-list li{padding:6px 0;border-bottom:1px solid var(--grid)}
details summary{cursor:pointer;color:var(--text2);font-size:13px}
pre.ann{white-space:pre-wrap;font:13px/1.6 inherit;background:var(--panel);padding:8px 12px;border-radius:6px;margin:6px 0}
"""


def fmt_pct(x, nd=1, sign=True):
    return "—" if x is None else (f"{x * 100:+.{nd}f}%" if sign else f"{x * 100:.{nd}f}%")


def fmt_num(x, nd=2, suffix=""):
    return "—" if x is None else f"{x:,.{nd}f}{suffix}"


def pct_bar(p):
    if p is None:
        return "—"
    return f'<span class="bar-pct" style="width:{p * 60:.0f}px"></span>{p * 100:.0f}'


def render(facts: dict, series: dict, narrative: dict | None) -> str:
    e = html.escape
    p, rv, fn, sc = facts["price"], facts["revenue"], facts["financials"], facts["scores"]
    val = sc.get("valuation", {})
    tiles = [
        ("收盤", fmt_num(p["close"])), ("近 12 月含息報酬", fmt_pct(p["ret_12m"])),
        ("市值（億）", fmt_num(val.get("mcap_億"), 0)), ("本益比 TTM", fmt_num(val.get("pe_ttm"), 1)),
        ("股價淨值比", fmt_num(val.get("pb"), 2)),
        ("綜合分數百分位", "—" if sc.get("composite_pct") is None else f"{sc['composite_pct'] * 100:.0f}"),
    ]
    tiles_html = "".join(f'<div class="tile"><span>{k}</span><b>{v}</b></div>' for k, v in tiles)

    charts = []
    px = series["px"].set_index("date")["adj_close"].resample("W").last().dropna()
    charts.append(svg_lines([f"{d:%y/%m}" for d in px.index], {"還原收盤": list(px.values)},
                            "{:,.0f}", "還原收盤價（近 3 年，週）", "{:,.1f}"))
    r = series["rv"]
    if len(r):
        lab = [f"{d:%y/%m}" for d in r["period"]]
        charts.append(svg_bars(lab, list(r["以合併為主單月營收(千元)"] / 1e5), "{:,.0f}", "單月營收（億元）", "{:,.1f} 億"))
        charts.append(svg_lines(lab, {"年增率": list(r["以合併為主單月營收成長率％"])}, "{:,.0f}%",
                                "單月營收年增率（%）", "{:+.1f}%", zero=True))
    f = series["fn"]
    if len(f):
        lab = [f"{d.year % 100}Q{d.quarter}" for d in f["period"]]
        charts.append(svg_lines(lab, {"毛利率": list(f["營業毛利率"]), "營益率": list(f["營業利益率"]),
                                      "淨利率": list(f["稅後淨利率"])}, "{:,.0f}%", "單季利潤率（%）", "{:.1f}%",
                                zero=True))
    t = series["trend"]
    if len(t):
        charts.append(svg_lines([f"{d:%y/%m}" for d in t.index], {"百分位": list(t.values * 100)}, "{:.0f}",
                                "綜合分數百分位（虛線 = 前 20% 門檻）", "{:.0f}", lo=0, hi=100, ref=80))

    def row(*cells):
        return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"

    fin_rows = ""
    if len(f):
        for _, q in f.iloc[::-1].iterrows():
            fin_rows += row(f"{q['period'].year}Q{q['period'].quarter}", fmt_num(q["營業收入淨額"] / 1e5, 1),
                            fmt_num(q["營業毛利率"], 1), fmt_num(q["營業利益率"], 1), fmt_num(q["稅後淨利率"], 1),
                            fmt_num(q["每股盈餘"]), fmt_num(q["ROE(A)－稅後"], 2), fmt_num(q["負債比率"], 1))
    rev_rows = ""
    if len(r):
        for _, m in r.tail(12).iloc[::-1].iterrows():
            rev_rows += row(f"{m['period']:%Y-%m}", fmt_num(m["以合併為主單月營收(千元)"] / 1e5, 1),
                            fmt_num(m["以合併為主單月營收成長率％"], 1), fmt_num(m["單月營收與上月比％"], 1),
                            fmt_num(m["近3月累計營收成長率"], 1), str(m["avail_date"].date()))

    g = sc.get("groups_pct", {})
    grp_rows = "".join(row(k, pct_bar(v)) for k, v in g.items())
    fac_rows = "".join(row(e(v["desc"]), fmt_num(v["value"], 3), pct_bar(v["pct_good"]))
                       for v in sc.get("factors", {}).values())
    qx = sc.get("quintile_next_month_excess_pct_since_2019", {})
    peer_rows = "".join(row(f'{e(x["name"] or "")} {x["code"]}', fmt_num(x["mcap_億"], 0), fmt_num(x["rev_yoy_pct"], 1),
                            fmt_num(x["roe_quarter_pct"], 2), fmt_num(x["pe_ttm"], 1),
                            "—" if x["composite_pct"] is None else f'{x["composite_pct"] * 100:.0f}')
                        for x in sc.get("peers", []))
    ind = sc.get("industry", {})
    med = ind.get("median", {})

    if narrative:
        narr = (f'<p>{e(narrative["summary"])}</p>'
                + (f'<p><b>模型看法：</b>{e(narrative["outlook"])}</p>' if narrative.get("outlook") else "")
                + f'<div class="cols">'
                f'<div class="box pos"><h3>正面</h3><ul>{"".join(f"<li>{e(x)}</li>" for x in narrative["positives"])}</ul></div>'
                f'<div class="box neg"><h3>負面</h3><ul>{"".join(f"<li>{e(x)}</li>" for x in narrative["negatives"])}</ul></div>'
                f'<div class="box"><h3>追蹤項目</h3><ul>{"".join(f"<li>{e(x)}</li>" for x in narrative["watch_items"])}</ul></div>'
                f'</div><p class="muted">文字由 {MODEL} 依下方數據生成，只能引用事實表內容；請以表格數字為準。</p>')
    else:
        narr = '<p class="muted">未產生文字摘要（沒有 LLM 憑證或使用 --no-llm）。</p>'

    notes = "".join(f"<li>{n['period']}：{e(n['note'])}</li>" for n in rv.get("company_notes", []))
    events = "".join(row(x["date"], x["kind"], fmt_num(x["prev_close"]), fmt_num(x["ref"]))
                     for x in facts["corporate_actions"])
    chips_html = ""
    fl = facts.get("institutional_flows") or {}
    if fl:
        names_ = {"foreign": "外資", "trust": "投信", "dealer": "自營商", "total": "三大法人"}
        flow_rows = "".join(row(names_[k], *(f'{fmt_num(fl[w][k]["net_lots"], 0)}（{fmt_num(fl[w][k]["pct_of_volume"], 1)}%）'
                                              if w in fl else "—" for w in ("5d", "20d", "60d")))
                            for k in names_)
        c = series["chips"]
        chart = svg_lines([f"{d:%m/%d}" for d in c["date"]],
                          {"外資": list(c["foreign"].cumsum() / 1000), "投信": list(c["trust"].cumsum() / 1000)},
                          "{:,.0f}", "近 60 日累計買賣超（張）", "{:,.0f} 張", zero=True) if len(c) else ""
        st = fl.get("foreign_streak", {})
        chips_html = f"""<h2>三大法人</h2>
<p>資料至 {fl["last_date"]}；外資連續 {st.get("days", "—")} 日{st.get("direction", "")}。括號內為買賣超佔同期成交量比例。</p>
<div class="cols"><div class="wrap"><table><tr><th>法人</th><th>近 5 日</th><th>近 20 日</th><th>近 60 日</th></tr>{flow_rows}</table></div>
<div>{chart}</div></div>"""
    cr = facts.get("margin_foreign_holding") or {}
    if cr:
        m, q = cr.get("margin") or {}, cr.get("foreign_holding") or {}
        lines = []
        if q:
            lines.append(row("外資持股比率", f'{fmt_num(q["foreign_pct"], 2)}%',
                             *(f'{fmt_num(q[k], 2)} 點' if q.get(k) is not None else "—" for k in ("chg_20d_pt", "chg_60d_pt")),
                             f'{fmt_num(q["low_1y_pct"], 2)}% ~ {fmt_num(q["high_1y_pct"], 2)}%'))
        if m:
            lines.append(row("融資餘額（張）", fmt_num(m["margin_balance"], 0),
                             *(fmt_num(m[k], 0) if m.get(k) is not None else "—" for k in ("margin_chg_20d", "margin_chg_60d")),
                             f'使用率 {fmt_num(m["margin_usage_pct"], 1)}%' if m.get("margin_usage_pct") is not None else "—"))
            lines.append(row("融券餘額（張）", fmt_num(m["short_balance"], 0),
                             fmt_num(m["short_chg_20d"], 0) if m.get("short_chg_20d") is not None else "—", "—",
                             f'券資比 {fmt_num(m["short_to_margin_pct"], 1)}%' if m.get("short_to_margin_pct") is not None else "—"))
        date_ = q.get("last_date") or m.get("last_date")
        chips_html += f"""<h2>融資融券與外資持股</h2>
<p>資料至 {date_}。</p>
<div class="wrap"><table><tr><th>項目</th><th>最新</th><th>近 20 日變化</th><th>近 60 日變化</th><th>備註</th></tr>{"".join(lines)}</table></div>"""
    ann_html = ""
    an = series.get("ann")
    if an is not None and len(an):
        items = "".join(
            f'<li><span class="muted">{r.time:%Y-%m-%d %H:%M}</span> {e(r.subject or "")}'
            f'<details><summary>內文</summary><pre class="ann">{e(r.content or "")}</pre></details></li>'
            for r in an.itertuples())
        ann_html = f"""<h2>重大訊息（近 {facts["announcements"]["window_days"]} 天，{len(an)} 則）</h2>
<ul class="ann-list">{items}</ul>"""
    elif facts.get("announcements") is not None:
        ann_html = f'<h2>重大訊息</h2><p class="muted">近 90 天沒有重大訊息。</p>'
    ol = facts.get("outlook") or {}
    outlook_html = ""
    if ol:
        def _ex(v):
            return "—" if v is None else f"{v:+.2f}%"
        o_rows = "".join(row(
            k, f'<b>{v["label"] or "—"}</b>', "—" if v["pct"] is None else v["pct"], e(v["holding"]),
            " / ".join(_ex(v["history_by_label_oos"].get(l)) for l in ("看漲", "中性", "看跌")),
            "—" if (v["history_same_label_oos"] or {}).get("hit_pct") is None else f'{v["history_same_label_oos"]["hit_pct"]:.0f}%',
            f'{e(v["reliability"] or "—")}（t {v["t"]}）' if v["t"] is not None else e(v["reliability"] or "—"))
            for k, v in ol["horizons"].items())
        outlook_html = f"""<h2>短中長期看法（模型標籤）</h2>
<p class="muted">{ol["date"]} 的訊號。{e(ol["method"])}。{e(ol["hit_pct_note"])}。訊號組成見 outlook.py。</p>
<div class="wrap"><table><tr><th>期間</th><th>標籤</th><th>百分位</th><th>持有期</th>
<th>歷史平均超額 看漲 / 中性 / 看跌</th><th>同標籤勝率</th><th>可靠度</th></tr>{o_rows}</table></div>"""
    title = f'{p["name"]} {facts["code"]}'
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)} 個股報告</title>
<style>{CSS}</style></head><body><main>
<h1>{e(title)}</h1>
<div class="sub">{p["market"]} · {e(ind.get("name") or "—")} · 同業：{ind.get("peer_level") or "—"} {ind.get("tej_sub") if ind.get("peer_level") == "子產業" else ind.get("tej_main") or ""}（{e(ind.get("peer_group") or "")} 等 {ind.get("n_stocks", 0)} 檔） · 資料日 {facts["asof"]}
 · 營收至 {rv.get("latest_period", "—")} · 財報至 {fn.get("latest_quarter", "—")}</div>
<div class="tiles">{tiles_html}</div>

<h2>摘要</h2>{narr}
{outlook_html}

<h2>圖表</h2><div class="charts">{"".join(charts)}</div>

<h2>月營收</h2>
<p>最新 {rv.get("latest_period", "—")}（{rv.get("published", "—")} 公布）：{fmt_num(rv.get("revenue_億"), 1)} 億，
年增 {fmt_num(rv.get("yoy_pct"), 1)}%，近 3 月年增 {fmt_num(rv.get("yoy_3m_pct"), 1)}%，近 12 月年增 {fmt_num(rv.get("yoy_12m_pct"), 1)}%，
與歷史最高月比 {fmt_num(rv.get("vs_record_high_pct"), 1)}%。</p>
{f"<p class='muted'>公司說明：</p><ul>{notes}</ul>" if notes else ""}
<div class="wrap"><table><tr><th>月份</th><th>營收（億）</th><th>年增%</th><th>月增%</th><th>近3月年增%</th><th>公布日</th></tr>{rev_rows}</table></div>

<h2>季財報</h2>
<p>近四季 EPS {fmt_num(fn.get("eps_ttm"))} 元，近四季營業現金流 / 淨利 {fmt_num(fn.get("cfo_to_ni_ttm"))}，每股淨值 {fmt_num(fn.get("bvps"))} 元。</p>
<div class="wrap"><table><tr><th>季度</th><th>營收（億）</th><th>毛利率%</th><th>營益率%</th><th>淨利率%</th><th>EPS</th><th>ROE%（單季）</th><th>負債比%</th></tr>{fin_rows}</table></div>

<h2>估值與同業</h2>
<p>本益比 {fmt_num(val.get("pe_ttm"), 1)}（同業中位數 {fmt_num(med.get("pe_ttm"), 1)}），股價淨值比 {fmt_num(val.get("pb"))}（同業中位數 {fmt_num(med.get("pb"))}）。
和自己近 5 年比的便宜程度：本益比 {pct_bar(val.get("pe_cheapness_vs_own_5y_pct"))}、股價淨值比 {pct_bar(val.get("pb_cheapness_vs_own_5y_pct"))}
（越高越便宜；50 以下代表比過去多數時候貴）。</p>
<div class="wrap"><table><tr><th>同業（{ind.get("peer_level") or ""}市值前 8）</th><th>市值（億）</th><th>營收年增%</th><th>ROE%</th><th>本益比</th><th>綜合百分位</th></tr>{peer_rows}</table></div>

<h2>因子分數</h2>
<p>綜合分數百分位 {pct_bar(sc.get("composite_pct"))}（{sc["score_date"]}，{"在" if sc["in_universe"] else "不在"}回測股票池內）。
2019 年起，綜合分數 Q1~Q5 的下月平均超額：{"、".join(f"{k} {v:+.2f}%" for k, v in qx.items() if v is not None)}。</p>
<p class="muted">組別與綜合分數是「同產業、同市值等級」相比的百分位；右表是原始因子值在全股票池的百分位，
應計項目、資產成長、負債比、近 1 月報酬已反向（越高越有利）；市值與動能只是參考，不在綜合分數內。
所以營收年增很高的大型股，組別百分位仍可能偏低：同產業同規模的公司長得更快。</p>
<div class="cols"><div class="wrap"><table><tr><th>組別（中性化）</th><th>百分位</th></tr>{grp_rows}</table></div>
<div class="wrap"><table><tr><th>因子</th><th>數值</th><th>池內百分位</th></tr>{fac_rows}</table></div></div>

{chips_html}
{ann_html}

<h2>價格與風險</h2>
<div class="wrap"><table><tr><th>指標</th><th>數值</th></tr>
{row("近 1 / 3 / 6 / 12 月含息報酬", " / ".join(fmt_pct(p[k]) for k in ["ret_1m", "ret_3m", "ret_6m", "ret_12m"]))}
{row("距 52 週高點", fmt_pct(p["from_52w_high"]))}{row("近 60 日年化波動", fmt_pct(p["vol_60d_ann"], sign=False))}
{row("近 1 年最大回撤", fmt_pct(p["max_drawdown_1y"]))}{row("20 日均成交值（億）", fmt_num(p["avg_value_20d_億"]))}</table></div>
{f'<p class="muted">近期除權息 / 減資 / 變更面額：</p><div class="wrap"><table><tr><th>日期</th><th>類型</th><th>前收盤</th><th>參考價</th></tr>{events}</table></div>' if events else ""}

<p class="disc">本報告由程式依 TEJ 財報、證交所 / 櫃買中心行情、法人買賣超與公開資訊觀測站重大訊息自動產生，僅供學術研究。
看漲 / 看跌是模型依歷史統計的分組，回測與因子分數為歷史統計，不代表未來報酬。</p>
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("code")
    ap.add_argument("--asof", default=None, help="時點（預設最新資料日）")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    asof = pd.Timestamp(a.asof) if a.asof else pd.Timestamp.today().normalize()

    if not a.asof and (w := freshness_warning()):
        print(f"⚠ {w}")
    facts, series = build_facts(a.code, asof, extended=True)
    narrative = None if a.no_llm else llm_narrative(facts)
    if narrative:
        facts["narrative"] = narrative
    REPORTS.mkdir(exist_ok=True)
    stem = REPORTS / f"{a.code}_{facts['asof']}"
    stem.with_suffix(".json").write_text(json.dumps(facts, ensure_ascii=False, indent=1))
    stem.with_suffix(".html").write_text(render(facts, series, narrative))
    print(f"{stem}.html")


if __name__ == "__main__":
    main()
