# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""訊號矩陣：技術、籌碼、基本面、綜合四個維度各給 多 / 空 / 中性，算一致數與矛盾清單，並回測（學術研究用）。

用法:
  uv run matrix.py               # 回測一致數，寫 data/matrix_calib.json，嘗試記進 data/strategy/trials.jsonl

維度規則（事先定好、沒有依回測結果調整）:
  技術    outlook 短期標籤：看漲 → 多、看跌 → 空、中性 → 中性
  籌碼    外資 + 投信近 20 個交易日合計買賣超佔同期成交量 >= +2% → 多、<= −2% → 空（資料 2024-10 起）
  基本面  近 3 月累計營收年增率與毛利率年變化（同季比）都 > 0 → 多、都 < 0 → 空，一正一負 → 中性
  綜合    outlook 中期 + 長期標籤（看漲 +1、看跌 −1）加總 > 0 → 多、< 0 → 空
一致數 = 偏多維度數 − 偏空維度數；資料不足的維度不算。
矛盾清單（只列出，不改變一致數）：方向相反的兩個維度、技術方向和外資連續 5 日以上反向買賣超、
法人賣超但融資 20 日增加 10% 以上、技術方向和成立中且「有統計支持」的反向形態（patterns.py）。

回測（每月換股日 = outlook.parquet 的日期，之後 H 個交易日的產業中性超額，NW t，lag = H/21）:
  A  三維（技術、基本面、綜合）一致數各組；全部一致偏多 − 全部一致偏空、>= +2 − <= −2
  C  綜合偏多的股票中，技術與基本面也都偏多 − 其他（看一致數有沒有比單看綜合多出東西）
  D  四維（含籌碼）一致數 >= +2 − <= −2，只有 2024-11 起
  對照：綜合單看（中期標籤 看漲 − 看跌），同一批樣本
  A、C、D × H（20、60）各算一次嘗試，記進 trials.jsonl（沒有月 Sharpe，只增加嘗試次數）。
"""
import argparse
import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd

import technicals as tc
from build_panel import OUT

DIMS = ["技術", "籌碼", "基本面", "綜合"]
DIRS = {1: "多", -1: "空", 0: "中性"}
CHIP_PCT = 2.0
STREAK_DAYS = 5
MARGIN_JUMP = 0.10
HORIZONS = [20, 60]
PERIODS = {"樣本內 2011–2018": ("2011-01-01", "2018-12-31"), "樣本外 2019–": ("2019-01-01", "2100-01-01"),
           "近期 2024–": ("2024-01-01", "2100-01-01")}
OOS = "樣本外 2019–"
LAB = {"看漲": 1, "看跌": -1, "中性": 0}
TRIALS = OUT / "strategy" / "trials.jsonl"
RULES = {"tech": "短期標籤", "chips": f"外資+投信 20 日 / 成交量 ±{CHIP_PCT}%",
         "fund": "營收近 3 月年增與毛利率年變化同號", "comp": "中期 + 長期標籤加總"}


# ---------------------------------------------------------------- 維度判定（回測與個股共用）

def dimensions(d: pd.DataFrame) -> pd.DataFrame:
    """輸入欄位 short_label、mid_label、long_label、inst20_pct、rev_yoy_3m、gm_chg；回傳四維 +1 / −1 / 0（缺值 NaN）。"""
    out = pd.DataFrame(index=d.index)
    out["技術"] = d["short_label"].map(LAB)
    x = d["inst20_pct"]
    out["籌碼"] = np.where(x >= CHIP_PCT, 1.0, np.where(x <= -CHIP_PCT, -1.0, 0.0))
    out["籌碼"] = out["籌碼"].where(x.notna())
    r, g = d["rev_yoy_3m"], d["gm_chg"]
    out["基本面"] = np.where((r > 0) & (g > 0), 1.0, np.where((r < 0) & (g < 0), -1.0, 0.0))
    out["基本面"] = out["基本面"].where(r.notna() & g.notna())
    m, lg = d["mid_label"].map(LAB), d["long_label"].map(LAB)
    out["綜合"] = np.sign(m.fillna(0) + lg.fillna(0)).where(m.notna() | lg.notna())
    return out


def net(dm: pd.DataFrame, cols: list[str]) -> pd.Series:
    """一致數 = 偏多數 − 偏空數；全部維度都缺值時為 NaN。"""
    x = dm[cols]
    return x.sum(axis=1, min_count=1)


# ---------------------------------------------------------------- 個股

def _basis(dim: str, row: dict) -> str:
    f = lambda v: f"{v:+.1f}"
    if dim == "技術":
        return f"短期模型標籤：{row['short_label']}" if row["short_label"] else "沒有短期標籤"
    if dim == "籌碼":
        x = row["inst20_pct"]
        return "沒有法人買賣超資料" if x is None else \
            f"外資 + 投信近 20 日{'買' if x >= 0 else '賣'}超，佔成交量 {abs(x):.1f}%"
    if dim == "基本面":
        r, g = row["rev_yoy_3m"], row["gm_chg"]
        if r is None or g is None:
            return "營收或毛利率資料不足"
        return f"近 3 月營收年增 {f(r)}%、毛利率較去年同季 {f(g)} 個百分點"
    return f"中期模型標籤：{row['mid_label'] or '—'}、長期：{row['long_label'] or '—'}"


def stock_matrix(facts: dict, patterns: dict | None = None) -> dict:
    """從 report.build_facts 的事實表建個股的訊號矩陣（數字和報告其他章節同源）。"""
    hz = (facts.get("outlook") or {}).get("horizons", {})
    lab = lambda k: (hz.get(k) or {}).get("label")
    fl20 = (facts.get("institutional_flows") or {}).get("20d") or {}
    pcts = [(fl20.get(k) or {}).get("pct_of_volume") for k in ("foreign", "trust")]
    yq = (facts.get("financials") or {}).get("yoy_same_quarter") or {}
    row = {"short_label": lab("短期"), "mid_label": lab("中期"), "long_label": lab("長期"),
           "inst20_pct": None if any(p is None for p in pcts) else round(sum(pcts), 2),
           "rev_yoy_3m": (facts.get("revenue") or {}).get("yoy_3m_pct"), "gm_chg": yq.get("gross_margin_chg_pt")}
    dm = dimensions(pd.DataFrame([row]).astype({k: float for k in ("inst20_pct", "rev_yoy_3m", "gm_chg")})).iloc[0]
    dims = {k: {"direction": None if pd.isna(dm[k]) else DIRS[int(dm[k])], "basis": _basis(k, row)} for k in DIMS}
    ok = dm.dropna()
    agree = {"多": int((ok == 1).sum()), "空": int((ok == -1).sum()), "中性": int((ok == 0).sum()),
             "資料不足": int(dm.isna().sum()), "net": int(ok.sum())}

    con = []
    for i, a in enumerate(DIMS):
        for b in DIMS[i + 1:]:
            if dm[a] * dm[b] == -1:
                hi, lo = (a, b) if dm[a] == 1 else (b, a)
                con.append(f"{hi}偏多（{dims[hi]['basis']}），但{lo}偏空（{dims[lo]['basis']}）")
    st = (facts.get("institutional_flows") or {}).get("foreign_streak") or {}
    t = dm["技術"]
    if t in (1, -1) and st.get("days", 0) >= STREAK_DAYS and st.get("direction") == ("賣超" if t == 1 else "買超"):
        con.append(f"技術面偏{DIRS[int(t)]}（{dims['技術']['basis']}），但外資連 {st['days']} 日{st['direction']}")
    mg = (facts.get("margin_foreign_holding") or {}).get("margin") or {}
    bal, chg = mg.get("margin_balance"), mg.get("margin_chg_20d")
    if row["inst20_pct"] is not None and row["inst20_pct"] <= -CHIP_PCT and bal and chg is not None \
            and bal - chg > 0 and chg / (bal - chg) >= MARGIN_JUMP:
        con.append(f"法人近 20 日賣超（佔成交量 {abs(row['inst20_pct']):.1f}%），但融資 20 日增加 "
                   f"{chg / (bal - chg) * 100:.0f}%（{chg:,.0f} 張）")
    if t in (1, -1) and patterns and patterns.get("date") == facts.get("asof"):
        opp = "看跌形態" if t == 1 else "看漲形態"
        for p in patterns.get("patterns", []):
            if p["side"] == opp and p["reliability"] == "有統計支持":
                con.append(f"技術面偏{DIRS[int(t)]}（{dims['技術']['basis']}），但成立中{p['name']}"
                           f"（{opp}，{p['formed']} 成立，歷史有統計支持）")

    cal = json.loads(f.read_text()) if (f := OUT / "matrix_calib.json").exists() else {}
    n3 = None if dm[["技術", "基本面", "綜合"]].isna().all() else int(dm[["技術", "基本面", "綜合"]].sum())
    hist = {f"{h} 日": (cal.get("A", {}).get(str(h), {}).get(OOS, {}).get("groups", {}).get(str(n3))
                       if n3 is not None else None) for h in HORIZONS}
    return {"method": "四個維度各依固定規則判斷多 / 空 / 中性（規則見 matrix.py），一致數 = 偏多數 − 偏空數；"
                      "矛盾清單只列出，不改變一致數",
            "rules": RULES, "dimensions": dims, "agreement": agree, "contradictions": con,
            "history_oos_same_net_3dims": {"net_3dims": n3, **hist},
            "note": "歷史 = 樣本外（2019 起）三維（技術、基本面、綜合）一致數相同的股票之後 H 日產業中性超額 % 與勝率；"
                    "籌碼只有 2024-10 起的資料，四維的回測樣本很短"}


# ---------------------------------------------------------------- 回測

def chips20(dates: pd.DatetimeIndex) -> pd.DataFrame:
    ch = pd.read_parquet(OUT / "insti_daily.parquet", columns=["date", "code", "foreign", "trust"])
    ch["ft"] = ch["foreign"].fillna(0) + ch["trust"].fillna(0)
    ft = ch.pivot(index="date", columns="code", values="ft").fillna(0)
    vol = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "volume"],
                          filters=[("date", ">=", ft.index.min())]) \
        .pivot(index="date", columns="code", values="volume").reindex(index=ft.index, columns=ft.columns)
    pct = ft.rolling(20, min_periods=20).sum() / vol.rolling(20, min_periods=20).sum().replace(0, np.nan) * 100
    pct = pct.loc[pct.index.intersection(dates)]
    return pct.stack().rename("inst20_pct").reset_index()


def panel() -> pd.DataFrame:
    o = pd.read_parquet(OUT / "outlook.parquet", columns=["date", "code", "short_label", "mid_label", "long_label"])
    f = pd.read_parquet(OUT / "factors_monthly.parquet", columns=["date", "code", "industry", "rev_yoy_3m", "gm_chg"])
    d = o.merge(f, on=["date", "code"], how="left")
    dates = pd.DatetimeIndex(sorted(d["date"].unique()))
    d = d.merge(chips20(dates), on=["date", "code"], how="left")
    w = tc.load()
    dates = dates[dates.isin(w["adj_close"].index)]
    for h in HORIZONS:
        fwd = tc.forward(w, h).loc[dates].stack().rename(f"fwd{h}").reset_index()
        d = d.merge(fwd, on=["date", "code"], how="left")
        d[f"ex{h}"] = d[f"fwd{h}"] - d.groupby(["date", "industry"])[f"fwd{h}"].transform("mean")
    return pd.concat([d, dimensions(d)], axis=1)


def _group(d: pd.DataFrame, mask: pd.Series, ex: str) -> tuple[dict, pd.Series]:
    g = d[mask & d[ex].notna()]
    m = g.groupby("date")[ex].mean()
    return {"mean_ex_pct": round(float(m.mean() * 100), 2) if len(m) else None,
            "hit_pct": round(float((g[ex] > 0).mean() * 100), 1) if len(g) else None,
            "avg_n": round(float(g.groupby("date").size().mean()), 1) if len(m) else 0}, m


def _spread(d, a, b, ex, lag) -> dict:
    ga, ma = _group(d, a, ex)
    gb, mb = _group(d, b, ex)
    sp = (ma - mb).dropna()
    return {"long": ga, "short": gb, "spread_pct": round(float(sp.mean() * 100), 2) if len(sp) else None,
            "t": round(float(t), 2) if len(sp) >= 20 and not np.isnan(t := tc.nw_t(sp, lag)) else None,
            "n_dates": int(len(sp))}


def backtest(d: pd.DataFrame) -> dict:
    d = d.assign(net3=net(d, ["技術", "基本面", "綜合"]), net4=net(d, DIMS))
    out = {"A": {}, "C": {}, "D": {}, "baseline": {}}
    for h in HORIZONS:
        ex, lag = f"ex{h}", max(1, round(h / 21))
        for k in out:
            out[k][h] = {}
        for p, (a, b) in PERIODS.items():
            x = d[(d["date"] >= a) & (d["date"] <= b)]
            out["A"][h][p] = {
                "groups": {str(int(n)): _group(x, x["net3"] == n, ex)[0] for n in range(-3, 4)},
                "all_agree": _spread(x, x["net3"] == 3, x["net3"] == -3, ex, lag),
                "ge2_vs_le_minus2": _spread(x, x["net3"] >= 2, x["net3"] <= -2, ex, lag)}
            comp_bull = x["綜合"] == 1
            out["C"][h][p] = _spread(x, comp_bull & (x["技術"] == 1) & (x["基本面"] == 1),
                                     comp_bull & ~((x["技術"] == 1) & (x["基本面"] == 1)), ex, lag)
            out["baseline"][h][p] = _spread(x, x["mid_label"] == "看漲", x["mid_label"] == "看跌", ex, lag)
            y = x[x["籌碼"].notna()]
            out["D"][h][p] = {"groups": {str(int(n)): _group(y, y["net4"] == n, ex)[0] for n in range(-4, 5)},
                              "ge2_vs_le_minus2": _spread(y, y["net4"] >= 2, y["net4"] <= -2, ex, lag)}
    return out


def log_trials() -> int:
    """A、C、D × H 各記一次（同設定不重複記）。沒有月 Sharpe，只讓 Deflated Sharpe 的嘗試次數反映真實次數。"""
    TRIALS.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in TRIALS.read_text().splitlines() if l.strip()] if TRIALS.exists() else []
    seen, new = {r["hash"] for r in rows}, 0
    names = {"A": "三維一致數", "C": "綜合偏多再看技術與基本面", "D": "四維一致數（含籌碼）"}
    with open(TRIALS, "a", encoding="utf-8") as f:
        for k, name in names.items():
            for h in HORIZONS:
                cfg = {"test": k, "horizon": h, "rules": RULES}
                hsh = hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
                if hsh in seen:
                    continue
                f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "id": f"matrix_{k}_h{h}",
                                    "name": f"訊號矩陣：{name}（H{h}）", "hash": hsh, "sharpe_m": None,
                                    "sharpe_ex": None, "cagr": None, "config": cfg}, ensure_ascii=False) + "\n")
                new += 1
    return new


def main():
    argparse.ArgumentParser().parse_args()
    d = panel()
    cal = backtest(d)
    (OUT / "matrix_calib.json").write_text(json.dumps(cal, ensure_ascii=False, indent=1))
    print(f"新記嘗試 {log_trials()} 筆（{TRIALS}）")
    fmt = lambda s: "—" if s["spread_pct"] is None else f"{s['spread_pct']:+.2f}（t {s['t']}，{s['n_dates']} 期）"
    for h in HORIZONS:
        print(f"\n== H{h}：之後 {h} 個交易日的產業中性超額 %")
        rows = []
        for p in PERIODS:
            a = cal["A"][h][p]
            rows.append({"期間": p, "綜合單看 漲−跌": fmt(cal["baseline"][h][p]), "A 三維全一致": fmt(a["all_agree"]),
                         "A ≥+2 − ≤−2": fmt(a["ge2_vs_le_minus2"]), "C 綜合多內 一致−其他": fmt(cal["C"][h][p]),
                         "D 四維 ≥+2 − ≤−2": fmt(cal["D"][h][p]["ge2_vs_le_minus2"])})
        print(pd.DataFrame(rows).set_index("期間").to_string())
        g = cal["A"][h][OOS]["groups"]
        print("A 樣本外各一致數（平均超額 %, 勝率 %, 平均檔數）:",
              {n: (v["mean_ex_pct"], v["hit_pct"], v["avg_n"]) for n, v in g.items()})


if __name__ == "__main__":
    main()
