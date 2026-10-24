# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15", "anthropic>=2.1"]
# ///
"""互動式台股問答分析師（命令列）：Claude 透過工具查本專案的資料回答問題。

用法:
  uv run analyst.py                         # 互動模式，輸入 exit 離開，/new 開新對話
  uv run analyst.py -q "台積電毛利率為什麼上升？"
  uv run analyst.py --asof 2025-06-30       # 以過去某天為資料時點
  uv run analyst.py --model haiku           # 模型：haiku / sonnet（預設）/ opus

工具（都只讀本機 parquet，不連網）:
  find_stock            名稱 / 代號查詢
  stock_facts           個股事實表（價格、月營收、季財報、估值、因子分數、法人、重大訊息）
  screen_stocks         依條件篩選全市場
  compare_peers         同子產業比較
  search_announcements  搜尋重大訊息
  get_announcement      讀單則重大訊息全文
  price_performance     區間含息報酬、最大回撤，與同期全市場等權比較
  factor_backtest       因子回測摘要（rank IC）
  outlook_picks         模型選股清單（短 / 中 / 長期都看漲）與歷史校準

需要 Anthropic 憑證（ANTHROPIC_API_KEY，可放專案 .env）。對話紀錄存 logs/analyst/。
引用 outlook.py 的模型看漲 / 看跌標籤（附歷史校準），不給目標價。學術研究用。
"""
import argparse
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import anthropic
import numpy as np
import pandas as pd
from anthropic import beta_tool

from build_panel import OUT
import outlook as OL
from report import build_facts, freshness_warning, industry_info, load_dotenv

MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}
MODEL = MODELS["sonnet"]   # main() / run_eval 依 --model 改寫；Opus 貴，預設用 Sonnet
LOGS = Path(__file__).parent / "logs" / "analyst"
ASOF = pd.Timestamp.today().normalize()   # main() 依 --asof 改寫

SYSTEM = """你是台股研究助理，替一位個人投資者回答關於台股個股與市場的問題。

資料與工具:
- 所有數字都要來自工具結果；不知道就說資料不足，不要用自己記得的數字、新聞或產業知識補。
- 先用 find_stock 確認代號，再查其他工具。使用者打的名稱查不到、你改用相近名稱才找到時（例如錯字），
  回答第一句要說「查不到『原輸入』，以下以 ○○（代號）回答，如果不是請告訴我」。需要比較或篩選時用 screen_stocks / compare_peers，不要自己憑印象列股票。
- 「為什麼」的問題：先查財報與營收數字，再用 search_announcements / get_announcement 找公司自己的說明；
  只能引用公告與營收說明裡寫的原因，資料裡沒有寫原因就明說「資料沒有說明原因」，不要自己推論這家公司的原因。
- 可以補充一般性的財務機制（例如低基期會放大年增率、景氣循環股獲利高峰時本益比反而低、近四季獲利含較差季度會讓本益比偏低），
  但要寫明是「一般而言」的機制，並說明資料無法確認是否適用於這家公司；不可寫成這家公司的原因（不要寫「本益比偏高主要是因為…」）。
  產業景氣、需求、價格、新聞等外部資訊不屬於一般機制，仍然不能用。
  寫法固定為「一般而言，X 會 Y；資料無法確認這家公司是否屬於這種情況」，後面不要接「所以」「因此」「有一部分反映」把機制套回這家公司。
- 不要把獲利、虧損歸到特定子公司或轉投資（例如「虧損主要來自某子公司」），除非公告或財報資料明寫。
- 產業分類只用工具給的產業名稱（例如「半導體業」）；不要用自己的知識把公司歸成記憶體、IC 設計、AI 伺服器等次產業。
- 公司公告的財測可以引用（註明來源與期間），但不要把財測和已公布數字相加推算全年或其他期間，也不要算任何未來期間的數字。
- 資料時點是 {asof}；營收、財報只用當時已公布的資料。季財報最新可能落後一季，必要時用重大訊息補充並註明來源。
- 2026Q3 起由公開資料補上的季度沒有現金流量表，現金流相關因子會缺值。

回答方式:
- 用繁體中文，先給結論再給數字；每個數字帶期別或日期。表格最多 10 列。
- 工具回傳的百分位已是 0~100 的整數，直接寫成「第 N 百分位」，不要自己換算；並說明是和誰比（同產業同市值中性化後，或全市場未中性化）。
- 「排第幾、最高、最低、前 N 名、第幾低」這類說法，只能用工具回傳的排序或 *_rank 欄位，不要自己目測比較數字；
  表格標題要和實際排序欄位一致（screen_stocks 的 sorted_by），沒有排序依據就不要寫名次。
- 寫範圍（「從 A 到 B」「都超過 X」）時，逐一核對工具結果裡的最大與最小值，端點的公司和數字都要正確。
- 本益比 pe = 市值 ÷ 近四季歸母淨利，不是股價 ÷ 近四季 EPS 加總，兩者因股數不同可能有差；不要說本益比是用 eps_ttm 算的。
- 不要推測組別百分位或綜合分數高低的原因（例如「同業成長也很快所以排名不高」「毛利率高所以排第一」）；
  工具沒有提供中性化的拆解，只能陳述各組百分位與原始數字，不要把兩者用因果連起來。
- 篩選時先說明用了哪些條件，結果數量很多時說明只列前幾名。
- 看漲 / 看跌只能引用模型標籤：stock_facts 的 outlook（個股短 / 中 / 長期）與 outlook_picks（選股清單）。
  引用時寫「模型標籤：看漲」並附持有期、樣本外同標籤平均超額和中性組比較、可靠度；不要自己另外判斷方向，
  不給目標價或具體價格、報酬數字預測。看漲組和中性組歷史平均超額差不到 0.5 個百分點時，要說這個看漲標籤區分力有限。
  被問「該不該買」時，提供模型標籤與歷史校準、相關數據，說明這是統計分組不是保證，由使用者自己判斷。
- 回測與因子分數是歷史統計，不代表未來報酬；引用時要說清楚。"""


def _j(x) -> str:
    return json.dumps(x, ensure_ascii=False, default=str)


def _f(x, nd=2):
    return None if x is None or pd.isna(x) else round(float(x), nd)


# ---------------------------------------------------------------- 共用資料（快取）

@lru_cache(maxsize=1)
def names() -> pd.DataFrame:
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "name", "market"])
    px = px[px["date"] <= ASOF]
    last = px.sort_values("date").groupby("code").last()
    last["active"] = last["date"] >= px["date"].max() - pd.Timedelta(days=10)
    return last


@lru_cache(maxsize=1)
def universe() -> pd.DataFrame:
    """最新換股日的全市場橫斷面：因子、綜合分數、產業、估值。"""
    fm = pd.read_parquet(OUT / "factors_monthly.parquet")
    fm = fm[fm["date"] <= ASOF]
    cur = fm[fm["date"] == fm["date"].max()].copy()
    sc = pd.read_parquet(OUT / "bt" / "composite_scores.parquet")
    sc = sc[sc["date"] == cur["date"].max()].set_index("code")
    rk = (sc[["comp_equal", "rev_mom", "quality", "value", "trend"]].rank(pct=True) * 100).round()
    cur = cur.join(rk.rename(columns={"comp_equal": "composite_pct", "rev_mom": "rev_mom_pct",
                                      "quality": "quality_pct", "value": "value_pct", "trend": "trend_pct"}), on="code")
    cur = cur.join(industry_info(ASOF), on="code").join(names()[["name", "market"]], on="code")
    cur["pe"] = np.where(cur["ep"] > 0, 1 / cur["ep"], np.nan)
    cur["pb"] = np.where(cur["bp"] > 0, 1 / cur["bp"], np.nan)
    cur["mcap_億"] = cur["mcap"] / 1e5
    cur["avg_value_億"] = cur["avg_value"] / 1e8
    cur["liquid"] = (cur["avg_value"] >= 5e6) & (cur["close"] >= 5) & (cur["n_obs"] >= 120)
    return cur.reset_index(drop=True)


SCREEN_FIELDS = {
    "rev_yoy": "單月營收年增率 %", "rev_yoy_3m": "近 3 月累計營收年增率 %", "rev_yoy_12m": "近 12 月累計營收年增率 %",
    "rev_accel": "近 3 月年增 − 近 12 月年增（百分點）", "rev_vs_high": "單月營收與歷史最高比 %（0 = 新高）",
    "roe": "單季 ROE %", "gross_margin": "單季毛利率 %", "gm_chg": "毛利率較去年同季變化（百分點）",
    "op_margin": "單季營益率 %", "debt_ratio": "負債比 %", "ni_yoy": "單季淨利年增率（0.5 = 50%）",
    "pe": "本益比 = 市值 ÷ 近四季歸母淨利（虧損為空）", "pb": "股價淨值比", "mcap_億": "市值（億元）", "avg_value_億": "20 日均成交值（億元）",
    "close": "收盤價", "mom_12_1": "過去 12 月扣最近 1 月報酬（0.3 = 30%）", "rev_1m": "近 1 月報酬",
    "composite_pct": "綜合分數百分位 0~100（整數，直接寫第 N 百分位）：營收動能、品質、價值、趨勢四組等權，各因子先做產業與市值中性；要說明排名原因只能引用 *_pct 四組百分位，不要用個別指標推論", "rev_mom_pct": "營收動能組百分位",
    "quality_pct": "品質組百分位", "value_pct": "價值組百分位",
    "trend_pct": "趨勢組百分位（距 52 週高點、收盤相對 200 日均線）",
}
SHOW = ["code", "name", "產業名稱", "mcap_億", "close", "rev_yoy", "rev_yoy_3m", "gross_margin", "roe", "pe", "pb",
        "composite_pct"]


def _rows(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    out = []
    for r in df[cols].to_dict("records"):
        out.append({k: (None if pd.isna(v) else int(v)) if k.endswith("_pct") and k in SCREEN_FIELDS or k.endswith("rank")
                    else (_f(v, 3) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    return out


PCTILE_KEYS = {"composite_pct", "pct_good", "pe_cheapness_vs_own_5y_pct", "pb_cheapness_vs_own_5y_pct"}


def _pctile(x, key=None):
    """report.build_facts 的百分位是 0~1；問答改成 0~100 整數，避免模型自己換算出錯。"""
    if isinstance(x, dict):
        return {k: (_pctile(v) if k != "groups_pct" else {g: _pctile(p, "composite_pct") for g, p in v.items()})
                if isinstance(v, (dict, list)) else (_to100(v) if k in PCTILE_KEYS else v) for k, v in x.items()}
    if isinstance(x, list):
        return [_pctile(v) for v in x]
    return _to100(x) if key in PCTILE_KEYS else x


def _to100(v):
    return None if v is None else int(round(v * 100))


# ---------------------------------------------------------------- 工具

@beta_tool(eager_input_streaming=True)
def find_stock(query: str) -> str:
    """用公司名稱或代號找股票代號。名稱可以只打部分（例如「台積」、「聯發科」、「2454」）。

    Args:
        query: 公司名稱的一部分或股票代號。
    """
    n = names().reset_index()
    q = query.strip()
    hit = n[(n["code"] == q) | n["name"].str.contains(q, regex=False, na=False)]
    hit = hit.sort_values(["active", "code"], ascending=[False, True]).head(10)
    rows = [{"code": r.code, "name": r.name, "market": str(r.market), "last_trade": str(r.date.date()),
             "still_trading": bool(r.active)} for r in hit.itertuples()]
    if not rows:
        return _j({"result": f"找不到「{q}」", "note": "可能是錯字或簡稱；可改用部分名稱再查。若改查到相近的公司，"
                   "回答開頭要說明原輸入查不到、改用哪一家，並請使用者確認"})
    return _j(rows)


@beta_tool(eager_input_streaming=True)
def stock_facts(code: str, sections: list[str] | None = None) -> str:
    """單一股票的事實表：價格與風險、月營收（近 24 月重點）、季財報、估值與同業、因子分數、三大法人、
    融資融券與外資持股比率、短中長期模型標籤、重大訊息摘要。

    Args:
        code: 股票代號，例如 "2330"。
        sections: 只要部分章節時填，可選 price、revenue、financials、scores、institutional_flows、
            margin_foreign_holding、outlook、announcements、corporate_actions；不填就全部回傳。
            outlook 是短 / 中 / 長期模型標籤（看漲 / 中性 / 看跌）與歷史校準。
    """
    try:
        facts, series = build_facts(code, ASOF)
    except SystemExit as e:
        return _j({"error": str(e)})
    # 事實表只有最新一期；問答需要看趨勢，補上近 8 季與近 13 個月明細
    fn, rv = series["fn"], series["rv"]
    if len(fn):
        facts["financials"]["quarters"] = [
            {"quarter": f"{q['period'].year}Q{q['period'].quarter}", "revenue_億": _f(q["營業收入淨額"] / 1e5, 1),
             "gross_margin_pct": _f(q["營業毛利率"]), "op_margin_pct": _f(q["營業利益率"]),
             "net_margin_pct": _f(q["稅後淨利率"]), "eps": _f(q["每股盈餘"]), "roe_pct": _f(q["ROE(A)－稅後"]),
             "net_income_total_億": _f(q["合併總損益"] / 1e5, 1), "net_income_parent_億": _f(q["歸屬母公司淨利（損）"] / 1e5, 1),
             "published": str(q["avail_date"].date())} for q in fn.to_dict("records")]
    if len(rv):
        facts["revenue"]["months"] = [
            {"month": f"{m['period']:%Y-%m}", "revenue_億": _f(m["以合併為主單月營收(千元)"] / 1e5, 1),
             "yoy_pct": _f(m["以合併為主單月營收成長率％"]), "mom_pct": _f(m["單月營收與上月比％"])}
            for m in rv.tail(13).to_dict("records")]
        yoy = pd.Series([m["yoy_pct"] for m in facts["revenue"]["months"]], dtype=float)
        for m, k in zip(facts["revenue"]["months"], yoy.rank(ascending=False, method="min")):
            m["yoy_rank_in_13m"] = None if pd.isna(k) else int(k)   # 1 = 這 13 個月裡年增率最高
    facts = _pctile(facts)
    facts["notes"] = [n.replace("百分位 0~1", "百分位 0~100 整數") for n in facts.get("notes", [])] + [
        "pe_ttm = 市值 ÷ 近四季歸母淨利，不等於 close ÷ eps_ttm（股數基礎不同）",
        "名次只能引用 *_rank 欄位或工具的排序；months[].yoy_rank_in_13m 是年增率在這 13 個月的名次（1 = 最高）"]
    if sections:
        facts = {k: v for k, v in facts.items() if k in sections or k in ("code", "asof", "notes")}
    return _j(facts)


@beta_tool(eager_input_streaming=True)
def screen_stocks(filters: list[dict[str, Any]], sort_by: str = "composite_pct", descending: bool = True,
                  industry: str | None = None, limit: int = 15, include_illiquid: bool = False) -> str:
    """依條件篩選全市場普通股（最新換股日的橫斷面）。

    可用欄位: rev_yoy、rev_yoy_3m、rev_yoy_12m、rev_accel、rev_vs_high、roe、gross_margin、gm_chg、op_margin、
    debt_ratio、ni_yoy、pe、pb、mcap_億、avg_value_億、close、mom_12_1、rev_1m、composite_pct、rev_mom_pct、
    quality_pct、value_pct、trend_pct（意義見回傳的 field_desc）。

    Args:
        filters: 條件陣列，每個是 {"field": 欄位, "op": ">"|">="|"<"|"<="|"==", "value": 數字}，例如
            [{"field": "rev_yoy_3m", "op": ">", "value": 30}, {"field": "pe", "op": "<", "value": 20}]。可以是空陣列。
        sort_by: 排序欄位（上面任一欄位）。
        descending: 是否由大到小。
        industry: 只看某產業，比對交易所產業名稱（例如「半導體」、「電子零組件」、「航運」），部分字串即可。
        limit: 回傳幾檔，最多 30。
        include_illiquid: 是否包含成交太少的股票（預設排除：股價 < 5、20 日均成交值 < 500 萬、上市未滿 120 天）。
    """
    u = universe()
    if not include_illiquid:
        u = u[u["liquid"]]
    if industry:
        u = u[u["產業名稱"].fillna("").str.contains(industry, regex=False)]
    ops = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal, "==": np.equal}
    for f in filters or []:
        fld, op, val = f.get("field"), f.get("op"), f.get("value")
        if fld not in SCREEN_FIELDS or op not in ops:
            return _j({"error": f"不支援的條件 {f}；欄位要在 {list(SCREEN_FIELDS)}，op 要在 {list(ops)}"})
        u = u[ops[op](u[fld], float(val))]
    if sort_by not in SCREEN_FIELDS:
        return _j({"error": f"sort_by 要在 {list(SCREEN_FIELDS)}"})
    u = u.sort_values(sort_by, ascending=not descending, na_position="last")
    u = u.assign(rank=np.arange(1, len(u) + 1))
    cols = list(dict.fromkeys(["rank"] + SHOW + [f.get("field") for f in filters or []] + [sort_by]))
    return _j({"as_of": str(u["date"].max().date()) if len(u) else None, "n_matched": int(len(u)),
               "sorted_by": f"{sort_by}（{SCREEN_FIELDS[sort_by]}）{'由大到小' if descending else '由小到大'}；rank 是這個排序的名次",
               "field_desc": {k: SCREEN_FIELDS[k] for k in cols if k in SCREEN_FIELDS},
               "rows": _rows(u.head(min(limit, 30)), cols)})


@beta_tool(eager_input_streaming=True)
def compare_peers(code: str, limit: int = 12) -> str:
    """列出同 TEJ 子產業（不足 3 檔時用主產業）的公司，含營收成長、獲利能力、估值與綜合分數，依市值排序。

    Args:
        code: 股票代號。
        limit: 最多列幾家（含本股）。
    """
    u = universe()
    me = u[u["code"] == code]
    if me.empty:
        return _j({"error": f"{code} 不在最新換股日的資料中"})
    sub, main = me["TEJ子產業代碼"].iloc[0], me["TEJ主產業代碼"].iloc[0]
    if pd.notna(sub) and (u["TEJ子產業代碼"] == sub).sum() >= 3:
        level, peers = "子產業", u[u["TEJ子產業代碼"] == sub]
    else:
        level, peers = "主產業", u[u["TEJ主產業代碼"] == main]
    peers = peers.assign(
        pe_rank=peers["pe"].rank(method="min"), pb_rank=peers["pb"].rank(method="min"),
        roe_rank=peers["roe"].rank(ascending=False, method="min"),
        rev_yoy_rank=peers["rev_yoy"].rank(ascending=False, method="min"),
    ).sort_values("mcap", ascending=False)
    top = peers.head(limit)
    if code not in set(top["code"]):
        top = pd.concat([top, me])
    cols = SHOW + ["rev_yoy_12m", "op_margin", "debt_ratio", "rev_mom_pct", "quality_pct", "value_pct", "trend_pct",
                   "pe_rank", "pb_rank", "roe_rank", "rev_yoy_rank"]
    med = {c: _f(peers[c].median()) for c in ["rev_yoy", "rev_yoy_3m", "gross_margin", "roe", "pe", "pb"]}
    return _j({"peer_level": level, "industry_code": sub if level == "子產業" else main,
               "industry_name": me["產業名稱"].iloc[0], "n_peers": int(len(peers)), "median": med,
               "rank_desc": "*_rank 是在全部 n_peers 家同業中的名次（含沒列出的）：pe_rank、pb_rank 1 = 最低（最便宜），"
                            "roe_rank、rev_yoy_rank 1 = 最高；虧損公司沒有本益比、不排名。百分位為 0~100 整數",
               "rows": _rows(top, cols)})


@beta_tool(eager_input_streaming=True)
def search_announcements(code: str | None = None, keyword: str | None = None, days: int = 60,
                         limit: int = 20) -> str:
    """搜尋公開資訊觀測站重大訊息（標題與內文節錄）。可以只指定公司、只指定關鍵字，或兩者都給。

    Args:
        code: 股票代號；不填就搜尋全市場。
        keyword: 標題或內文要包含的字（例如「澄清」、「處分」、「股利」、「法說會」）。
        days: 往回搜尋幾天（最多 180）。
        limit: 最多回傳幾則，最多 40。
    """
    a = pd.read_parquet(OUT / "announcements.parquet")
    end = ASOF + pd.Timedelta(days=1)
    a = a[(a["time"] < end) & (a["time"] >= end - pd.Timedelta(days=min(days, 180)))]
    if code:
        a = a[a["code"] == code]
    if keyword:
        a = a[a["subject"].str.contains(keyword, regex=False, na=False)
              | a["content"].str.contains(keyword, regex=False, na=False)]
    a = a.sort_values("time", ascending=False)
    return _j({"n_matched": int(len(a)),
               "items": [{"time": f"{r.time:%Y-%m-%d %H:%M}", "code": r.code, "name": r.name, "subject": r.subject,
                          "excerpt": (r.content or "")[:300]} for r in a.head(min(limit, 40)).itertuples()]})


@beta_tool(eager_input_streaming=True)
def get_announcement(code: str, time: str) -> str:
    """讀取單則重大訊息全文。

    Args:
        code: 股票代號。
        time: 發言時間，格式 "YYYY-MM-DD HH:MM"（從 search_announcements 的結果複製）。
    """
    a = pd.read_parquet(OUT / "announcements.parquet", filters=[("code", "==", code)])
    t = pd.Timestamp(time)
    hit = a[a["time"].dt.floor("min") == t.floor("min")]
    if hit.empty:
        return _j({"error": "找不到這則公告"})
    r = hit.iloc[0]
    return _j({"time": f"{r.time:%Y-%m-%d %H:%M}", "code": r.code, "name": r["name"], "subject": r.subject,
               "fact_date": str(r.fact_date.date()) if pd.notna(r.fact_date) else None, "content": r.content})


@beta_tool(eager_input_streaming=True)
def price_performance(code: str, start: str, end: str | None = None) -> str:
    """區間含息報酬（已處理除權息、減資）、最大回撤，以及同期全市場普通股等權平均報酬。

    Args:
        code: 股票代號。
        start: 起始日 "YYYY-MM-DD"（用該日或之前最後一個交易日的收盤當基準）。
        end: 結束日 "YYYY-MM-DD"，不填就到資料最新日。
    """
    end_t = min(pd.Timestamp(end), ASOF) if end else ASOF
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "close", "ret"],
                         filters=[("date", ">=", pd.Timestamp(start) - pd.Timedelta(days=10)), ("date", "<=", end_t)])
    px = px[px["code"].str.fullmatch(r"[1-9]\d{3}")]
    s = px[px["code"] == code].dropna(subset=["close"]).sort_values("date")
    s = s[s["date"] <= end_t]
    base = s[s["date"] <= pd.Timestamp(start)]
    if base.empty:
        return _j({"error": "起始日之前沒有行情"})
    t0 = base["date"].iloc[-1]
    w = s[s["date"] > t0]
    nav = (1 + w["ret"].fillna(0)).cumprod()
    allw = px[(px["date"] > t0)]
    mkt = (1 + allw.groupby("date")["ret"].mean()).prod() - 1
    return _j({"code": code, "from": str(t0.date()), "to": str(w["date"].max().date()) if len(w) else None,
               "total_return_incl_div": _f(nav.iloc[-1] - 1 if len(nav) else 0, 4),
               "max_drawdown": _f((nav / nav.cummax() - 1).min() if len(nav) else 0, 4),
               "close_start": _f(base["close"].iloc[-1]), "close_end": _f(s["close"].iloc[-1]),
               "market_equal_weight_return": _f(mkt, 4)})


@beta_tool(eager_input_streaming=True)
def factor_backtest(factor: str | None = None) -> str:
    """因子回測摘要：2011 年起每月 rank IC（產業與市值中性），分 2011–2018 與 2019 以後。
    不填 factor 就回傳全部因子的摘要；factor 可填 rev_yoy、rev_accel、rev_vs_high、roe、ep、bp 等。

    Args:
        factor: 因子名稱，不填就全部。
    """
    ic = pd.read_parquet(OUT / "bt" / "ic_ind_size.parquet")
    if factor:
        if factor not in ic:
            return _j({"error": f"沒有這個因子；可用 {list(ic.columns)}"})
        ic = ic[[factor]]
    out = {}
    for f in ic:
        s = ic[f].dropna()
        out[f] = {p: {"mean_ic": _f(x.mean(), 4), "t": _f(x.mean() / x.std() * np.sqrt(len(x)), 2),
                      "ic_positive_share": _f((x > 0).mean(), 2), "months": int(len(x))}
                  for p, x in [("2011-2018", s[s.index < "2019"]), ("2019-", s[s.index >= "2019"])]}
    return _j({"note": "rank IC = 因子分數與下月報酬排名的相關；|t| > 2 大致算顯著。歷史統計，不代表未來。",
               "factors": out})


@beta_tool(eager_input_streaming=True)
def outlook_picks() -> str:
    """模型選股清單：在「有統計支持」的期間（短 / 中 / 長期）都標成看漲、成交值夠的股票，
    附各期間的歷史校準與同條件清單過去的表現。個股的三期間標籤請用 stock_facts 的 outlook。
    """
    f = OUT / "outlook.parquet"
    if not f.exists():
        return _j({"error": "還沒跑 uv run outlook.py"})
    d, cal = OL.load_outlook(), OL.load_calib()
    d = d[d["date"] <= ASOF]
    pk = OL.picks(d, cal)
    nm = names()["name"].to_dict()
    return _j({
        "date": str(d["date"].max().date()), "conditions": f"{'、'.join(OL.HORIZONS[k]['name'] for k in cal['_picks']['horizons'])}"
        f"都看漲 + 20 日均成交值 >= {OL.MIN_VALUE / 1e6:.0f} 百萬，依這些期間的平均百分位排序",
        "picks": [{"rank": i + 1, "code": r.code, "name": nm.get(r.code),
                   **{OL.HORIZONS[k]["name"] + "_pct": int(getattr(r, f"{k}_pct")) for k in OL.HORIZONS}}
                  for i, r in enumerate(pk.itertuples())],
        "history_of_this_rule_by_holding": {OL.HORIZONS[k]["name"]: v for k, v in cal["_picks"].get("history", {}).items()},
        "horizon_reliability": {c["name"]: {"reliability": c["reliability"], "holding_days": c["days"],
                                            "oos_mean_ex_pct_by_label": {l: c["periods"][OL.OOS][l]["mean_ex_pct"] for l in OL.LABELS},
                                            "t": c["periods"][OL.OOS]["t"]} for k, c in cal.items() if k in OL.HORIZONS},
        "note": "超額 = 之後 N 個交易日相對同產業等權的報酬；歷史統計，不保證未來。勝率因報酬右偏普遍低於 50%，要互相比較"})


TOOLS = [find_stock, stock_facts, screen_stocks, compare_peers, search_announcements, get_announcement,
         price_performance, factor_backtest, outlook_picks]


# ---------------------------------------------------------------- 對話

def ask(client: anthropic.Anthropic, messages: list, verbose: bool = True, max_restarts: int = 2,
        stats: dict | None = None) -> str:
    """跑一輪（含工具呼叫），文字邊生成邊印出；整輪內容追加到 messages，回傳最後的文字回答。

    stats 不是 None 時累加每次 API 呼叫的 usage、實際服務的 model、最後的 stop_reason（評測用）。

    tool runner 會自己保留一份歷史，這裡另外鏡像到 messages：工具輸入 JSON 壞掉（ValueError）時
    用鏡像的歷史重開一個 runner 接著跑。
    """
    final = ""
    for _ in range(max_restarts + 1):
        runner = client.beta.messages.tool_runner(
            model=MODEL, max_tokens=16000, max_iterations=15, stream=True,
            system=SYSTEM.format(asof=ASOF.date()), tools=TOOLS, messages=messages,
            output_config={"effort": "medium"}, cache_control={"type": "ephemeral"},
            # Haiku 5.5 沒有伺服器端備援
            **({} if MODEL == MODELS["haiku"] else {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}),
        )
        try:
            for stream in runner:
                printed = False
                for event in stream:
                    if event.type == "content_block_start" and event.content_block.type == "tool_use" and verbose:
                        print(f"\033[2m  ↳ {event.content_block.name}\033[0m", flush=True)
                    elif event.type == "content_block_delta" and event.delta.type == "text_delta" and verbose:
                        if not printed:
                            print()
                            printed = True
                        print(event.delta.text, end="", flush=True)
                message = stream.get_final_message()
                messages.append({"role": "assistant", "content": message.content})
                if stats is not None:
                    u = message.usage
                    for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                        stats.setdefault("usage", {}).setdefault(k, 0)
                        stats["usage"][k] += getattr(u, k, 0) or 0
                    stats.setdefault("models", []).append(message.model)
                    stats["api_calls"] = stats.get("api_calls", 0) + 1
                    stats["stop_reason"] = message.stop_reason
                text = "".join(b.text for b in message.content if b.type == "text")
                if text:
                    final = text
                    if verbose:
                        print(flush=True)
                if message.stop_reason == "refusal":
                    return final or "（這個問題被模型拒絕回答。）"
                if message.stop_reason == "max_tokens" and any(b.type == "tool_use" for b in message.content):
                    return final or "（回答超過長度上限。）"
                tool_response = runner.generate_tool_call_response()
                if tool_response is not None:
                    messages.append(tool_response)
            return final
        except ValueError as e:  # 工具輸入 JSON 無法解析：丟掉未完成的那一輪，從鏡像歷史重來
            if stats is not None:
                stats["restarts"] = stats.get("restarts", 0) + 1
            print(f"\033[2m  （工具輸入解析失敗，重試：{str(e)[:60]}）\033[0m", flush=True)
            while messages and messages[-1]["role"] == "assistant":
                messages.pop()
    return final


def log_turn(path: Path, question: str, answer: str, messages: list):
    tools = [{"name": b.name, "input": b.input} for m in messages if m["role"] == "assistant"
             for b in m["content"] if getattr(b, "type", None) == "tool_use"]
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "asof": str(ASOF.date()),
                            "question": question, "answer": answer, "tool_calls": tools}, ensure_ascii=False) + "\n")


def main():
    global ASOF, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question")
    ap.add_argument("--asof", default=None)
    ap.add_argument("--model", choices=list(MODELS), default="sonnet")
    a = ap.parse_args()
    MODEL = MODELS[a.model]
    if a.asof:
        ASOF = pd.Timestamp(a.asof)
    load_dotenv()
    client = anthropic.Anthropic()
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    messages: list = []

    def turn(q: str):
        messages.append({"role": "user", "content": q})
        start = len(messages)
        try:
            ans = ask(client, messages)
        except anthropic.APIError as e:
            ans = f"（API 錯誤：{e}）"
            del messages[start - 1:]
            print(ans)
        print(flush=True)
        log_turn(log, q, ans, messages[start:])

    if not a.asof and (w := freshness_warning()):
        print(f"⚠ {w}")
    if a.question:
        turn(a.question)
        return
    print(f"台股問答分析師（資料時點 {ASOF.date()}，模型 {MODEL}）。輸入 exit 離開，/new 開新對話。")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("exit", "quit"):
            break
        if q == "/new":
            messages.clear()
            print("（已開新對話）")
            continue
        if q:
            turn(q)


if __name__ == "__main__":
    main()
