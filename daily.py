# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""每日速覽：觀察名單的變化 + 財報季新公布的公司 → reports/daily_YYYYMMDD.html。

用法:
  uv run daily.py                  # 和上次執行比較，結果記在 data/daily_state.json
  uv run daily.py --no-save        # 不更新狀態（測試用，下次仍和上上次比）
  uv run daily.py --codes 2330,2454   # 臨時換一份名單（不寫入 watchlist.txt）

觀察名單在 watchlist.txt：一行一檔代號，# 之後是註解。update.py 跑完會自動執行這支。

觀察名單（每檔）:
  - 新公布的月營收、季財報、重大訊息（上次執行之後才出現的；第一次執行看近 7 天）
  - 綜合分數百分位和上次比的變化
  - 近 5 個交易日：股價、外資持股比率、融資餘額、三大法人買賣超
  超過下方門檻的會標成提醒。
模型標籤（outlook.py）: 觀察名單各檔的短 / 中 / 長期看漲、看跌，與最新的模型選股清單（附歷史校準）。
模擬持股（paper.py）: 最新一期持股、進場以來各檔報酬、累積淨值對同池等權與 0050；
  paper_strategies.txt 裡各策略的即時紀錄（累積報酬、曝險、大盤濾網狀態、目前持股）。
財報季速覽（季底到申報期限後 30 天）:
  這一季已公布的公司：營收年增、毛利率與年增變化、EPS 與去年同季、毛利率在自己近 20 季的排名，
  另外依產業彙總。只描述資料，不判斷好壞或股價方向。
"""
import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from build_panel import OUT
from fetch_chips import load_chips, load_margin, load_qfii
import outlook as OL
from paper import ETF, TOP_N, load_strategy_summaries, summary as paper_summary
from report import CSS, industry_info
from update import season_status

ROOT = Path(__file__).parent
WATCHLIST = ROOT / "watchlist.txt"
STATE = OUT / "daily_state.json"
REPORTS = ROOT / "reports"
# 提醒門檻
ALERT_DAY_RET = 0.07          # 單日漲跌幅
ALERT_COMP_CHG = 15           # 綜合分數百分位變化（點）
ALERT_FOREIGN_5D = 1.0        # 外資持股比率 5 日變化（百分點）
ALERT_MARGIN_5D = 0.15        # 融資餘額 5 日變化（比例）
FIN_COLS = ["code", "name", "period", "avail_date", "單季(Q)/單半年(H)", "營業收入淨額", "營業毛利率",
            "營業利益率", "每股盈餘", "歸屬母公司淨利（損）"]


def read_watchlist(path: Path = WATCHLIST) -> list[str]:
    if not path.exists():
        return []
    codes = [ln.split("#")[0].strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return list(dict.fromkeys(c for c in codes if c))


def _f(x, nd=2):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def latest_prices(days: int = 15) -> pd.DataFrame:
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "name", "close", "ret"])
    return px[px["date"] >= px["date"].max() - pd.Timedelta(days=days)]


def composite_pct() -> tuple[str, pd.Series]:
    sc = pd.read_parquet(OUT / "bt" / "composite_scores.parquet", columns=["date", "code", "comp_equal"])
    d = sc["date"].max()
    now = sc[sc["date"] == d].dropna(subset=["comp_equal"])
    return str(d.date()), (now.set_index("code")["comp_equal"].rank(pct=True) * 100).round()


# ---------- 觀察名單 ----------

def watch_rows(codes: list[str], prev: dict, first_run: bool) -> tuple[list[dict], dict]:
    px = latest_prices()
    asof = px["date"].max()
    comp_date, comp = composite_pct()
    rev = pd.read_parquet(OUT / "tej_rev.parquet", columns=["code", "period", "avail_date", "單月營收(千元)",
                                                            "單月營收成長率％", "創新高/低(歷史)"],
                          filters=[("code", "in", codes)])
    fin = pd.read_parquet(OUT / "tej_fin.parquet", columns=FIN_COLS, filters=[("code", "in", codes)])
    ann = pd.read_parquet(OUT / "announcements.parquet", columns=["code", "time", "subject"],
                          filters=[("code", "in", codes)])
    recent = pd.Timestamp.today().normalize() - pd.Timedelta(days=7)    # 第一次執行時「新」的定義
    ol = OL.load_outlook() if (OUT / "outlook.parquet").exists() else pd.DataFrame(columns=["date", "code"])
    ol = ol[ol["date"] == ol["date"].max()].set_index("code")
    rows, state = [], {"asof": str(asof.date()), "comp_date": comp_date, "codes": {}}
    for c in codes:
        p = px[px["code"] == c].sort_values("date")
        ps = prev.get("codes", {}).get(c, {})
        r = {"code": c, "name": p["name"].iloc[-1] if len(p) else "", "alerts": [], "events": []}
        st = {}
        if p.empty:
            r["alerts"].append("近 15 天沒有行情（下市、停牌或代號錯誤）")
        else:
            r["close"] = _f(p["close"].iloc[-1])
            r["ret_1d"] = _f(p["ret"].iloc[-1], 4)
            r["ret_5d"] = _f((1 + p["ret"].tail(5)).prod() - 1, 4)
            if r["ret_1d"] is not None and abs(r["ret_1d"]) >= ALERT_DAY_RET:
                r["alerts"].append(f"單日 {r['ret_1d'] * 100:+.1f}%")
        # 綜合分數
        if c in comp.index:
            r["comp"] = int(comp[c])
            st["comp"] = r["comp"]
            if "comp" in ps:
                r["comp_chg"] = r["comp"] - ps["comp"]
                if abs(r["comp_chg"]) >= ALERT_COMP_CHG:
                    r["alerts"].append(f"綜合分數百分位 {ps['comp']} → {r['comp']}")
        # 月營收
        rv = rev[rev["code"] == c].sort_values("period")
        if len(rv):
            last = rv.iloc[-1]
            st["rev"] = str(last["period"].date())
            if (st["rev"] > ps["rev"]) if "rev" in ps else (first_run and last["avail_date"] >= recent):
                hi = f"，{last['創新高/低(歷史)']}" if pd.notna(last["創新高/低(歷史)"]) else ""
                r["events"].append(f"{last['period']:%Y-%m} 月營收 {last['單月營收(千元)'] / 1e5:,.2f} 億，"
                                   f"年增 {last['單月營收成長率％']:+.1f}%{hi}（{last['avail_date']:%m/%d} 公布）")
        # 季財報
        fn = fin[fin["code"] == c].sort_values("period")
        if len(fn):
            last = fn.iloc[-1]
            st["fin"] = str(last["period"].date())
            if (st["fin"] > ps["fin"]) if "fin" in ps else (first_run and last["avail_date"] >= recent):
                y = fn[fn["period"] == last["period"] - pd.DateOffset(years=1) + pd.offsets.MonthEnd(0)]
                eps_prev = f"（去年同季 {y['每股盈餘'].iloc[0]:.2f}）" if len(y) and pd.notna(y["每股盈餘"].iloc[0]) else ""
                r["events"].append(f"{last['period'].year}Q{last['period'].quarter} 財報：毛利率 {last['營業毛利率']:.1f}%，"
                                   f"EPS {last['每股盈餘']:.2f}{eps_prev}（{last['avail_date']:%m/%d} 公布）")
        # 重大訊息
        an = ann[ann["code"] == c].sort_values("time")
        if len(an):
            st["ann"] = str(an["time"].max())
            cut = pd.Timestamp(ps["ann"]) if "ann" in ps else (recent if first_run else an["time"].max())
            for a in an[an["time"] > cut].itertuples():
                r["events"].append(f"重大訊息 {a.time:%m/%d %H:%M}：{a.subject}")
        # 籌碼：近 5 個交易日
        qf = load_qfii(c).sort_values("date")
        if len(qf) > 5:
            r["foreign_pct"] = _f(qf["foreign_pct"].iloc[-1])
            r["foreign_5d"] = _f(qf["foreign_pct"].iloc[-1] - qf["foreign_pct"].iloc[-6])
            if abs(r["foreign_5d"]) >= ALERT_FOREIGN_5D:
                r["alerts"].append(f"外資持股 5 日 {r['foreign_5d']:+.2f} 點")
        mg = load_margin(c).sort_values("date")
        if len(mg) > 5 and mg["margin_bal"].iloc[-6] > 0:
            r["margin_bal"] = _f(mg["margin_bal"].iloc[-1], 0)
            r["margin_5d"] = _f(mg["margin_bal"].iloc[-1] / mg["margin_bal"].iloc[-6] - 1, 4)
            if abs(r["margin_5d"]) >= ALERT_MARGIN_5D and mg["margin_bal"].iloc[-1] >= 100:
                r["alerts"].append(f"融資 5 日 {r['margin_5d'] * 100:+.0f}%")
        if c in ol.index:
            for k, h in OL.HORIZONS.items():
                lab = ol.loc[c, f"{k}_label"]
                r[f"ol_{k}"] = None if pd.isna(lab) else lab
                if "ol" in ps and ps["ol"].get(k) not in (None, r[f"ol_{k}"]) and r[f"ol_{k}"]:
                    r["alerts"].append(f"{h['name']}標籤 {ps['ol'][k]} → {r[f'ol_{k}']}")
            st["ol"] = {k: r[f"ol_{k}"] for k in OL.HORIZONS}
        ch = load_chips(c).sort_values("date").tail(5)
        if len(ch):
            r["insti_5d"] = _f(ch["total"].sum() / 1000, 0)      # 股 → 張
        rows.append(r)
        state["codes"][c] = st
    return rows, state


# ---------- 財報季速覽 ----------

def season_rows(watch: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    ss = season_status()
    if not ss:
        return {}, pd.DataFrame(), pd.DataFrame()
    y, qn = int(ss["quarter"][:4]), int(ss["quarter"][-1])
    q = pd.Timestamp(year=y, month=qn * 3, day=1) + pd.offsets.MonthEnd(0)
    fin = pd.read_parquet(OUT / "tej_fin.parquet", columns=FIN_COLS)
    fin = fin[fin["單季(Q)/單半年(H)"] == "Q"]
    cur = fin[fin["period"] == q].copy()
    if cur.empty:
        return ss, cur, pd.DataFrame()
    hist = fin[fin["code"].isin(cur["code"]) & (fin["period"] > q - pd.DateOffset(years=5))]
    prev = fin[fin["period"] == q - pd.DateOffset(years=1) + pd.offsets.MonthEnd(0)].set_index("code")
    cur = cur.set_index("code")
    cur["rev_yoy"] = (cur["營業收入淨額"] / prev["營業收入淨額"] - 1) * 100
    cur["gm_chg"] = cur["營業毛利率"] - prev["營業毛利率"]
    cur["eps_prev"] = prev["每股盈餘"]
    # 毛利率、營收在自己近 20 季（含本季）的名次，1 = 最高
    for col, out in (("營業毛利率", "gm_rank"), ("營業收入淨額", "rev_rank")):
        rk = hist.assign(r=hist.groupby("code")[col].rank(ascending=False, method="min"))
        cur[out] = rk[rk["period"] == q].set_index("code")["r"]
    cur["n_q"] = hist.groupby("code").size()
    fm = pd.read_parquet(OUT / "factors_monthly.parquet", columns=["date", "code", "mcap"])   # 行情的上市股沒有股數
    cur["mcap_億"] = fm[fm["date"] == fm["date"].max()].set_index("code")["mcap"] / 1e5
    cur["產業"] = industry_info(pd.Timestamp.today())["產業名稱"]
    cur["watch"] = cur.index.isin(watch)
    cur = cur.reset_index().sort_values(["avail_date", "mcap_億"], ascending=[False, False])
    ind = cur.groupby(cur["產業"].fillna("（未分類）")).agg(
        n=("code", "size"), rev_yoy_med=("rev_yoy", "median"), gm_chg_med=("gm_chg", "median"),
        rev_up=("rev_yoy", lambda s: (s > 0).mean() * 100)).sort_values("n", ascending=False)
    return ss, cur, ind


# ---------- HTML ----------

def _pct(x, nd=1, mult=1):
    return "—" if x is None or pd.isna(x) else f"{x * mult:+.{nd}f}%"


def _num(x, nd=2, sign=False):
    return "—" if x is None or pd.isna(x) else (f"{x:+,.{nd}f}" if sign else f"{x:,.{nd}f}")


def strategy_section(ps: list[dict], watch: set) -> str:
    e = html.escape
    rows, details = [], []
    for x in ps:
        rn = x.get("regime_now")
        reg = "—" if not rn else (f'{"均線上" if rn["above"] else "均線下"}（{rn["gap"] * 100:+.1f}%，'
                                   f'{"每日" if rn["freq"] == "daily" else "每月"}判斷）')
        has = "ret" in x
        rows.append([e(x["name"]), x["start"], x["n_periods"], f'{x["target_exposure"] * 100:.0f}%', reg,
                     _pct(x.get("ret"), 2, 100) if has else "等進場", _pct(x.get("etf"), 2, 100), _pct(x.get("pool"), 2, 100),
                     _pct(x.get("mdd"), 2, 100), f'{x["cash_pct"] * 100:.0f}%' if has else "—",
                     x.get("exits", "—") if has else "—"])
        if has and x["holdings"]:
            body = "".join(f'<tr><td>{"★ " if h["code"] in watch else ""}{e(h["code"])}</td><td>{e(h["name"] or "")}</td>'
                           f'<td>{h["value"]:,.0f}</td><td>{h["weight"] * 100:.1f}%</td></tr>' for h in x["holdings"])
            head = "<tr><th>代號</th><th>名稱</th><th>市值（元）</th><th>佔資產</th></tr>"
            label = f'目前持股 {len(x["holdings"])} 檔（{x["nav_date"]}，資產 {x["nav"]:,.0f} 元）'
        else:
            body = "".join(f'<tr><td>{"★ " if t["code"] in watch else ""}{e(t["code"])}</td><td>{e(t["name"] or "")}</td>'
                           f'<td>{t["weight"] * 100:.1f}%</td></tr>' for t in x["targets"])
            head = "<tr><th>代號</th><th>名稱</th><th>目標權重</th></tr>"
            label = f'本期目標 {len(x["targets"])} 檔（訊號日 {x["signal_date"]}）'
        details.append(f'<details><summary>{e(x["name"])}：{label}</summary><p class="muted">{e(x["description"])}</p>'
                       f'<div class="wrap"><table>{head}{body}</table></div></details>')
    head = ["策略", "起點", "期數", "目標曝險", "大盤濾網", "累積報酬", ETF, "同池等權", "最大回落", "現金", "出場次數"]
    table = (f'<div class="wrap"><table><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr>'
             + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + "</table></div>")
    cap = ps[0]["capital"] if ps else 0
    return (f'<h2>多策略模擬（{len(ps)} 個）</h2><p class="muted">策略清單在 paper_strategies.txt，設定在 strategies/。'
            f'每個策略各用 {cap:,} 元，扣手續費與證交稅；停損、大盤濾網、回落控制照策略設定逐日執行。'
            f'即時紀錄，起點之前沒有資料；{ETF} 與同池等權不扣成本。</p>{table}{"".join(details)}')


def render(rows, comp_date, ss, cur, ind, asof, prev_asof, pp=None, pk=None, ps=None) -> str:
    e = html.escape
    parts = []
    # 觀察名單
    alerts = [(r, a) for r in rows for a in r["alerts"]]
    events = [(r, ev) for r in rows for ev in r["events"]]
    since = f"和上次執行（資料日 {prev_asof}）比較" if prev_asof else "第一次執行，「新公布」指近 7 天"
    parts.append(f'<h2>觀察名單（{len(rows)} 檔）</h2><p class="muted">{since}。名單在 watchlist.txt。</p>')
    if not rows:
        parts.append('<p class="muted">watchlist.txt 是空的。</p>')
    else:
        parts.append('<div class="cols"><div class="box"><h3>提醒</h3>' +
                     ("<ul>" + "".join(f"<li><b>{e(r['code'])} {e(r['name'])}</b>：{e(a)}</li>" for r, a in alerts) + "</ul>"
                      if alerts else '<p class="muted">沒有超過門檻的變化。</p>') + "</div>")
        parts.append('<div class="box"><h3>新公布</h3>' +
                     ('<ul class="ann-list">' + "".join(f"<li><b>{e(r['code'])} {e(r['name'])}</b>　{e(ev)}</li>"
                                                         for r, ev in events) + "</ul>"
                      if events else '<p class="muted">沒有新的營收、財報或重大訊息。</p>') + "</div></div>")
        head = ["代號", "名稱", "短 / 中 / 長期", "收盤", "1 日", "5 日", f"綜合分數百分位（{comp_date}）", "與上次比",
                "外資持股 %", "5 日變化（點）", "融資（張）", "融資 5 日", "法人 5 日（張）"]
        body = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in [
                e(r["code"]), e(r["name"]), " / ".join(r.get(f"ol_{k}") or "—" for k in OL.HORIZONS), _num(r.get("close")), _pct(r.get("ret_1d"), mult=100),
                _pct(r.get("ret_5d"), mult=100), r.get("comp", "—"), _num(r.get("comp_chg"), 0, True),
                _num(r.get("foreign_pct")), _num(r.get("foreign_5d"), 2, True), _num(r.get("margin_bal"), 0),
                _pct(r.get("margin_5d"), 0, 100), _num(r.get("insti_5d"), 0, True)]) + "</tr>" for r in rows)
        parts.append(f'<div class="wrap"><table><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr>{body}</table></div>')
    # 模型選股清單
    if pk is not None:
        cal, tab = pk
        use = cal["_picks"]["horizons"]
        rel = "；".join(f'{c["name"]} {c["reliability"]}（樣本外 看漲 {c["periods"][OL.OOS]["看漲"]["mean_ex_pct"]:+.2f}% / '
                       f'中性 {c["periods"][OL.OOS]["中性"]["mean_ex_pct"]:+.2f}% / 看跌 {c["periods"][OL.OOS]["看跌"]["mean_ex_pct"]:+.2f}%，'
                       f'持有 {c["days"]} 日）' for k, c in cal.items() if k in OL.HORIZONS)
        hist = cal["_picks"].get("history", {})
        hist_txt = "；".join(f'持有{OL.HORIZONS[k]["name"]}（{OL.HORIZONS[k]["days"]} 日）平均超額 '
                             f'{v[OL.OOS]["mean_ex_pct"]:+.2f}%、勝率 {v[OL.OOS]["hit_pct"]:.0f}%' for k, v in hist.items())
        watch = set(r["code"] for r in rows)
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in [
            i + 1, ("★ " if t.code in watch else "") + e(t.code), e(t.name or ""),
            *[int(getattr(t, f"{k}_pct")) for k in OL.HORIZONS]]) + "</tr>" for i, t in enumerate(tab.itertuples()))
        parts.append(f'<h2>模型選股清單（{len(tab)} 檔）</h2><p class="muted">條件：{"、".join(OL.HORIZONS[k]["name"] for k in use)}'
                     f'都看漲（百分位 &gt;= {OL.BULL}），20 日均成交值 &gt;= {OL.MIN_VALUE / 1e6:.0f} 百萬，依平均百分位排序。'
                     f'各期間：{rel}。同條件在樣本外（2019 起）的歷史：{hist_txt}。超額相對同產業等權；歷史統計，不保證未來。</p>'
                     f'<div class="wrap"><table><tr><th>#</th><th>代號</th><th>名稱</th>'
                     + "".join(f"<th>{h['name']}百分位</th>" for h in OL.HORIZONS.values()) + f"</tr>{body}</table></div>")
    # 模擬持股
    if pp:
        head = (f'自 {pp["start"]} 起共 {pp["n_periods"]} 期；最新一期訊號日 {pp["signal_date"]}，'
                + (f'{pp["entry_date"]} 收盤進場' if pp["entry_date"] else "下一個交易日收盤進場"))
        perf = ""
        if "port" in pp:
            perf = ('<div class="tiles">' + "".join(
                f'<div class="tile"><b>{v * 100:+.2f}%</b><span>{k}</span></div>' for k, v in
                [("模擬持股（扣成本）", pp["port"]), ("同池等權", pp["pool"]), (ETF, pp["etf"]), ("最大回落", pp["mdd"])])
                + f'</div><p class="muted">累積到 {pp["nav_date"]}。</p>')
        watch = set(r["code"] for r in rows)
        body = "".join(f'<tr><td>{h["rank"]}</td><td>{"★ " if h["code"] in watch else ""}{e(h["code"])}</td>'
                       f'<td>{e(h["name"] or "")}</td><td>{_pct(h["ret_since_entry"], mult=100)}</td></tr>'
                       for h in pp["holdings"])
        parts.append(f'<h2>模擬持股（綜合分數前 {TOP_N} 名）</h2><p class="muted">{head}。每月換股一次，'
                     f'規則見 paper.py；這是即時紀錄，起點之前沒有資料。</p>{perf}'
                     f'<details><summary>本期 {len(pp["holdings"])} 檔持股</summary><div class="wrap"><table>'
                     f'<tr><th>名次</th><th>代號</th><th>名稱</th><th>進場以來</th></tr>{body}</table></div></details>')
    if ps:
        parts.append(strategy_section(ps, set(r["code"] for r in rows)))
    # 財報季
    if ss:
        parts.append(f'<h2>財報季速覽 {ss["quarter"]}</h2><p class="muted">申報期限 {ss["deadline"]}（還有 '
                     f'{ss["days_to_deadline"]} 天）。上一季有財報且仍在交易的 {ss["expected"]:,} 家中，已公布 '
                     f'{ss["reported"]:,} 家（{ss["reported"] / max(ss["expected"], 1) * 100:.0f}%）。'
                     f'名次 = 在自己近 20 季中的排名，1 = 最高。★ = 觀察名單。</p>')
        if len(cur):
            n_show = 300
            show = cur.head(n_show)
            head = ["公布日", "代號", "名稱", "產業", "市值（億）", "營收（億）", "營收年增", "營收名次", "毛利率",
                    "毛利率年增（點）", "毛利率名次", "營益率", "EPS", "去年同季 EPS"]
            body = "".join(
                "<tr>" + "".join(f"<td>{c}</td>" for c in [
                    f"{r.avail_date:%m/%d}", ("★ " if r.watch else "") + e(r.code), e(r.name), e(str(r.產業 or "—")),
                    _num(r.mcap_億, 0), _num(r.營業收入淨額 / 1e5), _pct(r.rev_yoy),
                    "—" if pd.isna(r.rev_rank) else f"{r.rev_rank:.0f}/{r.n_q}", _num(r.營業毛利率, 1),
                    _num(r.gm_chg, 1, True), "—" if pd.isna(r.gm_rank) else f"{r.gm_rank:.0f}/{r.n_q}",
                    _num(r.營業利益率, 1), _num(r.每股盈餘), _num(r.eps_prev)]) + "</tr>"
                for r in show.itertuples())
            more = f'<p class="muted">只列最近公布的 {n_show} 家，共 {len(cur):,} 家。</p>' if len(cur) > n_show else ""
            parts.append(f'<div class="wrap"><table><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr>{body}</table></div>{more}')
            body = "".join(f"<tr><td>{e(k)}</td><td>{int(v.n)}</td><td>{_pct(v.rev_yoy_med)}</td><td>{_num(v.gm_chg_med, 1, True)}</td>"
                           f"<td>{v.rev_up:.0f}%</td></tr>" for k, v in ind.iterrows())
            parts.append('<h3>依產業</h3><div class="wrap"><table><tr><th>產業</th><th>已公布</th><th>營收年增中位數</th>'
                         f'<th>毛利率年增中位數（點）</th><th>營收年增 &gt; 0 的比例</th></tr>{body}</table></div>')
        else:
            parts.append('<p class="muted">這一季還沒有公司公布。</p>')
    return (f'<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>每日速覽 {asof}</title>'
            f'<style>{CSS}h3{{font-size:15px;margin:12px 0 4px}}</style></head><body><main>'
            f'<h1>每日速覽</h1><p class="sub">資料日 {asof}，產生於 {datetime.now():%Y-%m-%d %H:%M}</p>'
            + "".join(parts) +
            '<p class="disc">學術研究用。看漲 / 看跌是模型依歷史統計的分組，不保證未來報酬。</p></main></body></html>')


def model_picks():
    if not (OUT / "outlook.parquet").exists():
        return None
    d, cal = OL.load_outlook(), OL.load_calib()
    tab = OL.picks(d, cal)
    names = latest_prices().sort_values("date").groupby("code")["name"].last()
    return cal, tab.assign(name=tab["code"].map(names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", help="逗號分隔，臨時取代 watchlist.txt")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    codes = a.codes.split(",") if a.codes else read_watchlist()
    prev = json.loads(STATE.read_text()) if STATE.exists() else {}
    rows, state = watch_rows(codes, prev, first_run=not prev)
    ss, cur, ind = season_rows(codes)
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"daily_{pd.Timestamp(state['asof']):%Y%m%d}.html"
    out.write_text(render(rows, state["comp_date"], ss, cur, ind, state["asof"], prev.get("asof"),
                          paper_summary(), model_picks(), load_strategy_summaries()), encoding="utf-8")
    if not a.no_save and not a.codes:
        # 名單外的舊狀態保留，之後加回名單時仍能比較
        state["codes"] = {**prev.get("codes", {}), **state["codes"]}
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    n_alert = sum(len(r["alerts"]) for r in rows)
    n_event = sum(len(r["events"]) for r in rows)
    print(f"觀察名單 {len(rows)} 檔：提醒 {n_alert} 則、新公布 {n_event} 則"
          + (f"；財報季 {ss['quarter']} 已公布 {len(cur)} 家" if ss else "") + f" → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
