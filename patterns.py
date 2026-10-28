# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""規則式價格形態偵測，附和 technicals.py 同一套的歷史校準（學術研究用）。

用法:
  uv run patterns.py               # 偵測全部歷史、校準，存最新一天成立中的形態
  uv run patterns.py --latest      # 只偵測最近一段、存最新形態（沿用既有校準，給 update.py 用）
  uv run patterns.py --code 2330   # 另外印出個股成立中的形態

形態（全部用還原價，關鍵價位輸出時換回最新一天的價格基準）:
  帶量突破 / 帶量跌破  收盤創前 60 日收盤新高（新低），且成交值 > 前 60 日均值 × 2
  箱型整理            近 20 日最高 − 最低 < 收盤 × 10%，且當天沒有突破 / 跌破（中性，只標箱頂、箱底）
  箱型突破 / 跌破      前一天還在箱型內，收盤站上箱頂（跌破箱底）
  W 底 / M 頭         兩個相近（差 < 3%）的波段低點（高點），間隔 10–60 日，中間頸線離低點至少 5%；
                      第二個低點確認後（之後 5 日都沒更低）30 日內收盤突破頸線才成立，期間跌破低點就作廢
  回測均線 / 反彈均線  21 > 60 > 200 日線多頭排列時，最低價碰到 21 或 60 日線 1% 內、收盤守住均線
                      （空頭排列時最高價碰到均線、收盤沒站上）
  向上 / 向下跳空      當天最低 > 前一天最高（當天最高 < 前一天最低），缺口當天沒回補
事件型形態（突破、W 底、跳空…）在成立後 5 個交易日內都算「成立中」，和 technicals.py 的事件訊號一樣；
狀態型（箱型、回測均線）只看當天。

校準（technicals.py 的方法）:
  每 5 個交易日取樣、股票池同 technicals.py；之後 H 日的產業中性超額報酬（隔天收盤進場）
  每個形態：成立中股票的平均超額（每期至少 5 檔才算）、跑贏同產業的比例、Newey-West t
  可靠度看樣本外（2019 起）H=20：方向和形態相同、t >= 2 且樣本內同方向 → 有統計支持；
  方向相同但不夠 → 弱；方向相反 → 歷史上不可靠。中性的箱型整理不給可靠度。
  technicals.py 的反轉類訊號樣本內就不顯著，傳統形態大多可能沒有預測力；不可靠的形態照樣輸出，
  但不能當買賣理由。形態 × 期間共做了 len(PATTERNS) × len(HORIZONS) 次檢定，單一個 t >= 2 要打折看。

輸出 data/patterns_calib.json（校準表）、data/patterns_latest.parquet（最新一天成立中的形態與關鍵價位）。
"""
import argparse
import json

import numpy as np
import pandas as pd

import technicals as tc
from build_panel import OUT

HORIZONS = [10, 20, 60]
PRIMARY = 20                 # 可靠度看的持有期（同 outlook.py 短期）
EVENT_DAYS = 5
MIN_N = 5
PERIODS = {"樣本內 2011–2018": ("2011-01-01", "2018-12-31"), "樣本外 2019–": ("2019-01-01", "2100-01-01"),
           "近期 2024–": ("2024-01-01", "2100-01-01")}
OOS, IS = "樣本外 2019–", "樣本內 2011–2018"
LATEST_ROWS = 320            # --latest 只用最近這麼多個交易日（200 日線 + 60 日視窗 + W 底回看）

BRK_DAYS, VOL_MULT = 60, 2.0
BOX_DAYS, BOX_PCT = 20, 0.10
PIVOT_K, DBL_GAP, DBL_TOL, DBL_DEPTH, DBL_CONFIRM = 5, (10, 60), 0.03, 0.05, 30
MA_TOUCH = 0.01

# 名稱 -> (方向 +1 看漲形態 / −1 看跌形態 / 0 中性, 類別, 中文名, 關鍵價位名稱)
PATTERNS = {
    "vol_breakout": (1, "event", "帶量突破", ["突破價位"]),
    "vol_breakdown": (-1, "event", "帶量跌破", ["跌破價位"]),
    "box": (0, "state", "箱型整理", ["箱頂", "箱底"]),
    "box_up": (1, "event", "箱型向上突破", ["箱頂", "箱底"]),
    "box_dn": (-1, "event", "箱型向下跌破", ["箱頂", "箱底"]),
    "w_bottom": (1, "event", "W 底", ["頸線", "底部低點"]),
    "m_top": (-1, "event", "M 頭", ["頸線", "頭部高點"]),
    "pullback_ma": (1, "state", "多頭排列回測均線", ["均線"]),
    "rally_ma": (-1, "state", "空頭排列反彈均線", ["均線"]),
    "gap_up": (1, "event", "向上跳空", ["缺口上緣", "缺口下緣"]),
    "gap_dn": (-1, "event", "向下跳空", ["缺口上緣", "缺口下緣"]),
}
SIDE = {1: "看漲形態", -1: "看跌形態", 0: "中性"}


# ---------------------------------------------------------------- 偵測

def _double(lo: np.ndarray, hi: np.ndarray, cl: np.ndarray, piv: np.ndarray):
    """W 底（M 頭傳入負號的序列）：回傳 (成立日 bool, 頸線, 底部低點)，未成立處為 NaN。"""
    n = len(cl)
    ev, neck_lv, low_lv = np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan)
    p = np.flatnonzero(piv)
    for a, b in zip(p[:-1], p[1:]):
        if not DBL_GAP[0] <= b - a <= DBL_GAP[1]:
            continue
        bottom = min(lo[a], lo[b])
        if abs(lo[b] - lo[a]) > DBL_TOL * abs(lo[a]) or np.nanmin(lo[a:b + 1]) < bottom:
            continue
        neck = np.nanmax(hi[a:b + 1])
        if neck - max(lo[a], lo[b]) < DBL_DEPTH * abs(max(lo[a], lo[b])):
            continue
        for t in range(b + PIVOT_K, min(n, b + DBL_CONFIRM + 1)):     # 第二個低點要等 K 日後才確認
            if lo[t] < bottom:
                break
            if cl[t] > neck:
                ev[t], neck_lv[t], low_lv[t] = True, neck, bottom
                break
    return ev, neck_lv, low_lv


def doubles(lo: pd.DataFrame, hi: pd.DataFrame, cl: pd.DataFrame, sign: int):
    """sign=+1 W 底、−1 M 頭；逐檔掃波段點。"""
    lo, hi, cl = (lo, hi, cl) if sign == 1 else (-hi, -lo, -cl)
    w = 2 * PIVOT_K + 1
    piv = (lo == lo.rolling(w, center=True, min_periods=w).min()).values
    L, H, C = lo.values, hi.values, cl.values
    ev, neck, bot = np.zeros(L.shape, bool), np.full(L.shape, np.nan), np.full(L.shape, np.nan)
    for j in range(L.shape[1]):
        ev[:, j], neck[:, j], bot[:, j] = _double(L[:, j], H[:, j], C[:, j], piv[:, j])
    f = lambda x: pd.DataFrame(sign * x, index=cl.index, columns=cl.columns)
    return pd.DataFrame(ev, index=cl.index, columns=cl.columns), f(neck), f(bot)


def detect(w: dict, keep_levels: int = EVENT_DAYS) -> tuple[dict, dict]:
    """回傳 (hits, levels)：hits[名稱] 是形態成立當天的 bool 表；levels[名稱][價位名] 只留最後 keep_levels 天。"""
    c = w["adj_close"].ffill(limit=5)
    h = w["adj_high"].ffill(limit=5)
    lo = w["adj_low"].ffill(limit=5)
    v = w["value"].fillna(0)
    hits, levels = {}, {}

    def put(name, hit, *lv):
        hits[name] = hit.fillna(False).astype(bool) & c.notna()
        levels[name] = {k: x.iloc[-keep_levels:].where(hits[name].iloc[-keep_levels:])
                        for k, x in zip(PATTERNS[name][3], lv)}

    vavg = v.shift(1).rolling(BRK_DAYS, min_periods=BRK_DAYS).mean()
    big = v > VOL_MULT * vavg.replace(0, np.nan)
    ph = c.shift(1).rolling(BRK_DAYS, min_periods=BRK_DAYS).max()
    pl = c.shift(1).rolling(BRK_DAYS, min_periods=BRK_DAYS).min()
    put("vol_breakout", (c > ph) & big, ph)
    put("vol_breakdown", (c < pl) & big, pl)
    del vavg, big, ph, pl

    top = h.rolling(BOX_DAYS, min_periods=BOX_DAYS).max()
    bot = lo.rolling(BOX_DAYS, min_periods=BOX_DAYS).min()
    inbox = (top - bot) < BOX_PCT * c
    was = inbox.shift(1, fill_value=False)
    up, dn = was & (c > top.shift(1)), was & (c < bot.shift(1))
    put("box", inbox & ~up & ~dn, top, bot)             # 當天已突破 / 跌破就不再算整理中
    put("box_up", up, top.shift(1), bot.shift(1))
    put("box_dn", dn, top.shift(1), bot.shift(1))
    del top, bot, inbox, was, up, dn

    for name, sign in (("w_bottom", 1), ("m_top", -1)):
        ev, neck, ext = doubles(lo, h, c, sign)
        put(name, ev, neck, ext)
    del ev, neck, ext

    m21, m60, m200 = tc.ma(c, 21), tc.ma(c, 60), tc.ma(c, 200)
    up, dn = (m21 > m60) & (m60 > m200), (m21 < m60) & (m60 < m200)
    t21 = (lo <= m21 * (1 + MA_TOUCH)) & (c >= m21)
    t60 = (lo <= m60 * (1 + MA_TOUCH)) & (c >= m60)
    put("pullback_ma", up & (t21 | t60), m21.where(t21, m60))
    r21 = (h >= m21 * (1 - MA_TOUCH)) & (c <= m21)
    r60 = (h >= m60 * (1 - MA_TOUCH)) & (c <= m60)
    put("rally_ma", dn & (r21 | r60), m21.where(r21, m60))
    del m21, m60, m200, up, dn, t21, t60, r21, r60

    put("gap_up", lo > h.shift(1), lo, h.shift(1))
    put("gap_dn", h < lo.shift(1), lo.shift(1), h)
    return hits, levels


def active(name: str, hit: pd.DataFrame) -> pd.DataFrame:
    """成立中：事件型 = 最近 EVENT_DAYS 天內成立過；狀態型 = 當天成立。"""
    if PATTERNS[name][1] == "state":
        return hit
    return hit.astype("int8").rolling(EVENT_DAYS, min_periods=1).max().astype(bool)


# ---------------------------------------------------------------- 校準

def calibrate(w: dict, uni: pd.DataFrame, hits: dict) -> dict:
    idx = w["adj_close"].index
    sample = idx[idx >= "2011-01-01"][::tc.STEP]
    ind = tc.industries(w["adj_close"].columns)
    u = uni.loc[sample]
    act = {k: active(k, x).loc[sample] & u for k, x in hits.items()}
    out = {k: {"name": PATTERNS[k][2], "side": PATTERNS[k][0], "horizons": {}} for k in PATTERNS}
    for hz in HORIZONS:
        ex = tc.industry_excess(tc.forward(w, hz).loc[sample].where(u), ind)
        lag = max(1, hz // tc.STEP)
        for k in PATTERNS:
            g = ex.where(act[k])
            n = g.notna().sum(axis=1)
            m = g.mean(axis=1).where(n >= MIN_N)
            res = {}
            for p, (a, b) in PERIODS.items():
                mp, gp, np_ = m.loc[a:b], g.loc[a:b], n.loc[a:b]
                cnt = int(gp.notna().sum().sum())
                res[p] = {"mean_ex_pct": round(float(mp.mean() * 100), 2) if mp.notna().any() else None,
                          "hit_pct": round(float((gp > 0).sum().sum() / cnt * 100), 1) if cnt else None,
                          "t": None if np.isnan(t := tc.nw_t(mp, lag)) else round(float(t), 2),
                          "avg_n": round(float(np_[np_ > 0].mean()), 1) if (np_ > 0).any() else 0,
                          "n_dates": int(mp.notna().sum())}
            out[k]["horizons"][hz] = res
    for k, r in out.items():
        r["reliability"] = reliability(r["side"], r["horizons"][PRIMARY])
    out["_meta"] = {"primary_horizon": PRIMARY, "n_tests": len(PATTERNS) * len(HORIZONS),
                    "note": "平均超額 = 成立中股票之後 H 日的產業中性超額（%）；勝率 = 跑贏同產業的比例。"
                            "個股報酬右偏，各組勝率常低於 50%，要和其他形態比較。"}
    return out


def reliability(side: int, res: dict) -> str | None:
    if side == 0:
        return None
    o, i = res[OOS], res[IS]
    if o["mean_ex_pct"] is None or o["mean_ex_pct"] * side <= 0:
        return "歷史上不可靠"
    if (o["t"] or 0) * side >= 2 and (i["mean_ex_pct"] or 0) * side > 0:
        return "有統計支持"
    return "弱"


# ---------------------------------------------------------------- 最新形態

def latest(w: dict, uni: pd.DataFrame, hits: dict, levels: dict) -> pd.DataFrame:
    idx = w["adj_close"].index
    last, tail = idx[-1], idx[-EVENT_DAYS:]
    traded = w["close"].loc[last].notna()
    base = (w["adj_close"] / w["close"]).loc[last]            # 還原價 → 最新一天的價格基準
    rows = []
    for k, (side, kind, name, _) in PATTERNS.items():
        hk = hits[k].loc[tail if kind == "event" else [last]]
        hk = hk.loc[:, hk.any() & traded.reindex(hk.columns, fill_value=False)]
        for code in hk.columns:
            d = hk.index[hk[code].values][-1]
            lv = {nm: round(float(f.at[d, code] / base[code]), 2) for nm, f in levels[k].items()}
            rows.append({"date": last, "code": code, "pattern": k, "name": name, "side": side,
                         "formed": d, "levels": json.dumps(lv, ensure_ascii=False),
                         "in_universe": bool(uni.at[last, code])})
    return pd.DataFrame(rows, columns=["date", "code", "pattern", "name", "side", "formed", "levels", "in_universe"])


def stock_patterns(code: str) -> dict:
    """給事實表：個股最新一天成立中的形態、關鍵價位與歷史校準（樣本外、PRIMARY 日）。"""
    f = OUT / "patterns_latest.parquet"
    if not f.exists():
        return {}
    d = pd.read_parquet(f)
    cf = OUT / "patterns_calib.json"
    cal = json.loads(cf.read_text()) if cf.exists() else {}
    sec = {"date": str(d["date"].max().date()) if len(d) else None,
           "method": "規則式偵測（見 patterns.py）；歷史 = 樣本外（2019 起）形態成立後 "
                     f"{PRIMARY} 個交易日的產業中性超額報酬",
           "note": "形態是技術面常用的價位描述，不是預測；可靠度「歷史上不可靠」或「弱」的不能當買賣理由",
           "patterns": []}
    for r in d[d["code"] == code].itertuples():
        c = cal.get(r.pattern, {})
        sec["patterns"].append({"name": r.name, "side": SIDE[r.side], "formed": str(r.formed.date()),
                                "levels": json.loads(r.levels), "reliability": c.get("reliability"),
                                "history_oos": c.get("horizons", {}).get(str(PRIMARY), {}).get(OOS)})
    return sec


# ---------------------------------------------------------------- 主程式

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", action="store_true")
    ap.add_argument("--code")
    a = ap.parse_args()
    w = tc.load()
    if a.latest:
        w = {k: v.iloc[-LATEST_ROWS:] for k, v in w.items()}
    uni = tc.universe(w)
    hits, levels = detect(w)
    lt = latest(w, uni, hits, levels)
    lt.to_parquet(OUT / "patterns_latest.parquet", index=False)
    if not a.latest:
        cal = calibrate(w, uni, hits)
        (OUT / "patterns_calib.json").write_text(json.dumps(cal, ensure_ascii=False, indent=1))
        print(f"形態校準：成立中股票之後 H 日的產業中性超額 %（勝率 %，NW t，平均每期檔數）；可靠度看樣本外 H{PRIMARY}")
        for p in PERIODS:
            rows = []
            for k, r in ((k, cal[k]) for k in PATTERNS):
                row = {"形態": r["name"], "方向": SIDE[r["side"]]}
                for hz in HORIZONS:
                    x = r["horizons"][hz][p]
                    row[f"H{hz}"] = "—" if x["mean_ex_pct"] is None else \
                        f"{x['mean_ex_pct']:+.2f}（{x['hit_pct']:.0f}%，{x['t'] if x['t'] is not None else '—'}，{x['avg_n']:.0f}）"
                row["可靠度"] = r["reliability"] or "—"
                rows.append(row)
            print(f"\n== {p}")
            print(pd.DataFrame(rows).set_index("形態").to_string())
    print(f"\npatterns_latest.parquet：{w['adj_close'].index[-1]:%Y-%m-%d}，{len(lt)} 筆（{lt['code'].nunique()} 檔）")
    print(lt.groupby("name").size().to_string())
    if a.code:
        print(json.dumps(stock_patterns(a.code), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
