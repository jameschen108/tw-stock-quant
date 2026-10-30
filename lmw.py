# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""Lo, Mamaysky & Wang (2000) 的核迴歸形態辨識，和 patterns.py 的規則式形態用同一套校準比較（學術研究用）。

Lo, A. W., Mamaysky, H., & Wang, J. (2000). Foundations of Technical Analysis: Computational Algorithms,
Statistical Inference, and Empirical Implementation. Journal of Finance, 55(4), 1705–1765.

用法:
  uv run lmw.py               # 偵測全部歷史、校準、和 patterns.py 比較，存最新一天成立中的形態
  uv run lmw.py --latest      # 只偵測最近一段、存最新形態（沿用既有校準）

方法（照原文，標「改編」的是本專案的調整）:
  - 每個交易日 T 取 L = l + d = 35 + 3 天的還原收盤價視窗，對時間做 Nadaraya-Watson 核迴歸（高斯核）
  - 頻寬：每個視窗用留一交叉驗證選 h*，實際用 0.3 × h*（原文認為 CV 頻寬太平滑，和技術分析看圖的尺度不合）
  - 平滑曲線斜率由正轉負為局部高點、由負轉正為局部低點；極值價用原始收盤價在該日 ±1 日內的最高（最低）
  - 形態用最後 5 個交替極值 E1…E5 定義，E5 必須落在視窗第 l 天（T − d），也就是只在形態剛完成、
    多看 d 天確認後記一次，同一個形態不會在相鄰視窗重複記
  - 10 種形態（下表），門檻照原文：頭肩 1.5%、矩形 0.75%、雙重頂底 1.5% 且兩個高（低）點相隔 > 22 日
  - 雙重頂底（改編）：原文的 Ea 取視窗內 E1 之後最高（最低）的極值；這裡限定在同一組 E1…E5 內
  - 方向（改編）：原文只檢定形態後報酬分布是否不同，不指定漲跌；這裡照技術分析慣例，頂部形態記看跌、
    底部形態記看漲，再用 patterns.py 的產業中性超額與 Newey-West t 校準
  - 分布檢定：形態後隔日報酬（用前 250 日平均與標準差標準化，改編：原文用整段期間）對全部股票日的
    二樣本 Kolmogorov-Smirnov 檢定；樣本有時間與橫斷面相依，p 值偏樂觀

輸出 data/patterns_lmw_calib.json、data/patterns_lmw_latest.parquet。和 patterns.py 一樣，
沒通過校準的形態照樣輸出、標「歷史上不可靠」，不能當買賣理由；形態 × 期間的多重檢定要打折看。
"""
import argparse
import json
import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

import patterns as P
import technicals as tc
from build_panel import OUT

L_DAYS, D_LAG = 35, 3
WIN = L_DAYS + D_LAG
H_GRID = np.geomspace(0.5, 20, 24)       # 交叉驗證的頻寬候選（交易日）
H_SCALE = 0.3
HS_TOL, RECT_TOL, DBL_TOL, DBL_GAP = 0.015, 0.0075, 0.015, 22
NORM_DAYS = 250
KS_SAMPLE = 300_000                        # 無條件分布抽樣數（固定亂數種子）

LEVELS = ["上緣", "下緣"]                  # E1…E5 中高點的平均、低點的平均
# 名稱 -> (方向, 類別, 中文名, 關鍵價位名稱)；格式同 patterns.PATTERNS
PATTERNS = {
    "hs": (-1, "event", "頭肩頂", LEVELS),
    "ihs": (1, "event", "頭肩底", LEVELS),
    "btop": (-1, "event", "擴張頂", LEVELS),
    "bbot": (1, "event", "擴張底", LEVELS),
    "ttop": (-1, "event", "三角頂", LEVELS),
    "tbot": (1, "event", "三角底", LEVELS),
    "rtop": (-1, "event", "矩形頂", LEVELS),
    "rbot": (1, "event", "矩形底", LEVELS),
    "dtop": (-1, "event", "雙重頂", LEVELS),
    "dbot": (1, "event", "雙重底", LEVELS),
}


# ---------------------------------------------------------------- 核迴歸

def _smoothers() -> tuple[np.ndarray, np.ndarray]:
    """每個候選頻寬的 (留一平滑矩陣, 0.3 倍頻寬的平滑矩陣)，形狀 (len(H_GRID), WIN, WIN)。

    視窗的 x 是 0…WIN−1，矩陣只和頻寬有關，所以所有股票、所有視窗共用。
    """
    x = np.arange(WIN)
    dx = x[:, None] - x[None, :]
    loo, fit = [], []
    for h in H_GRID:
        k = np.exp(-0.5 * (dx / h) ** 2)
        k0 = k.copy()
        np.fill_diagonal(k0, 0)
        loo.append(k0 / k0.sum(1, keepdims=True))
        ks = np.exp(-0.5 * (dx / (H_SCALE * h)) ** 2)
        fit.append(ks / ks.sum(1, keepdims=True))
    return np.stack(loo), np.stack(fit)


LOO, FIT = _smoothers()


def cv_bandwidth(win: np.ndarray) -> np.ndarray:
    """視窗 (N, WIN) → 每列 CV 選到的 H_GRID 位置。"""
    return np.stack([((win - win @ s.T) ** 2).mean(1) for s in LOO], axis=1).argmin(1)


def smooth(win: np.ndarray) -> np.ndarray:
    """視窗 (N, WIN) → 平滑曲線；每列用自己的 CV 頻寬 × 0.3。"""
    best = cv_bandwidth(win)
    out = np.empty_like(win)
    for k in np.unique(best):
        r = best == k
        out[r] = win[r] @ FIT[k].T
    return out


def extrema(m: np.ndarray) -> np.ndarray:
    """平滑曲線 (N, WIN) → 每天是 +1 局部高點、−1 局部低點、0 都不是。"""
    d = np.diff(m, axis=1)
    t = np.zeros(m.shape, np.int8)
    t[:, 1:-1][(d[:, :-1] > 0) & (d[:, 1:] <= 0)] = 1
    t[:, 1:-1][(d[:, :-1] < 0) & (d[:, 1:] >= 0)] = -1
    return t


# ---------------------------------------------------------------- 形態

def _within(vals: list[np.ndarray], tol: float) -> np.ndarray:
    avg = sum(vals) / len(vals)
    return np.all([np.abs(v - avg) <= tol * np.abs(avg) for v in vals], axis=0)


def classify(e: np.ndarray, t: np.ndarray, first: np.ndarray) -> dict[str, np.ndarray]:
    """E1…E5 的價 e、日 t（形狀 (n, 5)）、E1 是高點（+1）或低點（−1）→ 各形態是否成立。"""
    e1, e2, e3, e4, e5 = e.T
    mx, mn = first == 1, first == -1
    out = {
        "hs": mx & (e3 > e1) & (e3 > e5) & _within([e1, e5], HS_TOL) & _within([e2, e4], HS_TOL),
        "ihs": mn & (e3 < e1) & (e3 < e5) & _within([e1, e5], HS_TOL) & _within([e2, e4], HS_TOL),
        "btop": mx & (e1 < e3) & (e3 < e5) & (e2 > e4),
        "bbot": mn & (e1 > e3) & (e3 > e5) & (e2 < e4),
        "ttop": mx & (e1 > e3) & (e3 > e5) & (e2 < e4),
        "tbot": mn & (e1 < e3) & (e3 < e5) & (e2 > e4),
    }
    odd, even = [e1, e3, e5], [e2, e4]
    rect = _within(odd, RECT_TOL) & _within(even, RECT_TOL)
    out["rtop"] = mx & rect & (np.min(odd, 0) > np.max(even, 0))
    out["rbot"] = mn & rect & (np.max(odd, 0) < np.min(even, 0))
    # 雙重頂（底）：E1 之後同類極值（E3、E5）中最高（最低）的一個，和 E1 相差 1.5% 內、相隔 > 22 日
    a = np.where(e5 * first > e3 * first, 4, 2)
    ea, ta = e[np.arange(len(e)), a], t[np.arange(len(t)), a]
    dbl = _within([e1, ea], DBL_TOL) & (ta - t[:, 0] > DBL_GAP)
    out["dtop"], out["dbot"] = mx & dbl, mn & dbl
    return out


def detect_one(c: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """單一股票的還原收盤 (T,) → (各形態成立日 bool (T,), 上緣 (T,), 下緣 (T,))。成立日 = 視窗最後一天。"""
    n = len(c)
    hits = {k: np.zeros(n, bool) for k in PATTERNS}
    top, bot = np.full(n, np.nan), np.full(n, np.nan)
    if n < WIN:
        return hits, top, bot
    win = sliding_window_view(c, WIN)
    ok = ~np.isnan(win).any(1)
    rows = np.flatnonzero(ok)
    if not len(rows):
        return hits, top, bot
    w = win[rows]
    typ = extrema(smooth(w))
    last = L_DAYS - 1
    cand = typ[:, last] != 0
    w, typ, rows = w[cand], typ[cand, :L_DAYS], rows[cand]
    is_ext = typ != 0
    cnt = np.cumsum(is_ext[:, ::-1], axis=1)[:, ::-1]           # 從這天到 E5 共有幾個極值
    enough = cnt[:, 0] >= 5
    w, typ, rows, is_ext, cnt = w[enough], typ[enough], rows[enough], is_ext[enough], cnt[enough]
    if not len(rows):
        return hits, top, bot
    pos = np.stack([np.argmax(is_ext & (cnt == k), axis=1) for k in (5, 4, 3, 2, 1)], axis=1)
    r = np.arange(len(rows))[:, None]
    kinds = typ[r, pos]
    alt = np.all(kinds[:, 1:] == -kinds[:, :-1], axis=1)
    hi3 = np.maximum(np.maximum(w[:, :-2], w[:, 1:-1]), w[:, 2:])   # 第 τ 日 ±1 日內的最高（τ = 1…）
    lo3 = np.minimum(np.minimum(w[:, :-2], w[:, 1:-1]), w[:, 2:])
    e = np.where(kinds == 1, hi3[r, pos - 1], lo3[r, pos - 1])
    res = classify(e, pos, kinds[:, 0])
    day = rows + WIN - 1
    with np.errstate(invalid="ignore", divide="ignore"):         # 不交替的列會沒有高點或低點，下面會濾掉
        hi_avg = np.where(kinds == 1, e, 0).sum(1) / (kinds == 1).sum(1)
        lo_avg = np.where(kinds == -1, e, 0).sum(1) / (kinds == -1).sum(1)
    for k, m in res.items():
        m = m & alt
        hits[k][day[m]] = True
        top[day[m]], bot[day[m]] = hi_avg[m], lo_avg[m]
    return hits, top, bot


def detect(w: dict, keep_levels: int = P.EVENT_DAYS) -> tuple[dict, dict]:
    """格式同 patterns.detect：hits[名稱] bool 寬表；levels[名稱][價位名] 只留最後 keep_levels 天。"""
    c = w["adj_close"].ffill(limit=5)
    shape = c.shape
    H = {k: np.zeros(shape, bool) for k in PATTERNS}
    top, bot = np.full(shape, np.nan), np.full(shape, np.nan)
    for j in range(shape[1]):
        h, tp, bt = detect_one(c.iloc[:, j].to_numpy(float))
        for k in PATTERNS:
            H[k][:, j] = h[k]
        top[:, j], bot[:, j] = tp, bt
    df = lambda x: pd.DataFrame(x, index=c.index, columns=c.columns)
    hits = {k: df(v) & c.notna() for k, v in H.items()}
    top, bot = df(top), df(bot)
    levels = {k: {"上緣": top.iloc[-keep_levels:].where(hits[k].iloc[-keep_levels:]),
                  "下緣": bot.iloc[-keep_levels:].where(hits[k].iloc[-keep_levels:])} for k in PATTERNS}
    return hits, levels


# ---------------------------------------------------------------- 分布檢定

def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """二樣本 Kolmogorov-Smirnov：(D, 漸近 p 值)。"""
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    d = float(np.max(np.abs(np.searchsorted(a, x, "right") / len(a) - np.searchsorted(b, x, "right") / len(b))))
    ne = len(a) * len(b) / (len(a) + len(b))
    lam = (math.sqrt(ne) + 0.12 + 0.11 / math.sqrt(ne)) * d
    if lam < 1.18:      # 小 λ 時交錯級數收斂差，改用 Kolmogorov 分布的另一個展開式
        cdf = 0.0 if lam < 1e-3 else math.sqrt(2 * math.pi) / lam * sum(
            math.exp(-((2 * k - 1) ** 2) * math.pi ** 2 / (8 * lam * lam)) for k in range(1, 6))
        p = 1 - cdf
    else:
        p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, min(max(p, 0.0), 1.0)


def ks_tests(w: dict, uni: pd.DataFrame, hits: dict) -> dict:
    """形態成立隔日的標準化報酬 vs 全部股票日；分樣本內、樣本外。"""
    c = w["adj_close"].ffill(limit=5)
    r = c.pct_change(fill_method=None).where(~w["ret_suspect"])
    mu = r.rolling(NORM_DAYS, min_periods=120).mean().shift(1)
    sd = r.rolling(NORM_DAYS, min_periods=120).std().shift(1)
    z = ((r - mu) / sd).shift(-1).where(uni)                  # T 日成立 → T+1 日的報酬
    rng = np.random.default_rng(0)
    out = {}
    for p in (P.IS, P.OOS):
        a, b = P.PERIODS[p]
        zp = z.loc[a:b]
        allz = zp.to_numpy().ravel()
        allz = allz[~np.isnan(allz)]
        base = rng.choice(allz, min(KS_SAMPLE, len(allz)), replace=False) if len(allz) else allz
        out[p] = {}
        for k, h in hits.items():
            v = zp.where(h.loc[a:b]).to_numpy().ravel()
            v = v[~np.isnan(v)]
            if len(v) < 20 or not len(base):
                out[p][k] = {"n": int(len(v)), "D": None, "p": None}
                continue
            d, pv = ks_2samp(v, base)
            out[p][k] = {"n": int(len(v)), "D": round(d, 4), "p": round(pv, 4), "mean_z": round(float(v.mean()), 3)}
    return out


# ---------------------------------------------------------------- 主程式

def diagnose(w: dict, n_codes: int = 60) -> dict:
    """抽樣股票：CV 頻寬落在候選下限的比例、每 l 日視窗的極值個數（看平滑程度）。"""
    c = w["adj_close"].ffill(limit=5)
    cols = np.random.default_rng(0).choice(c.columns, min(n_codes, c.shape[1]), replace=False)
    best, n_ext = [], []
    for col in cols:
        win = sliding_window_view(c[col].to_numpy(float), WIN) if len(c) >= WIN else np.empty((0, WIN))
        win = win[~np.isnan(win).any(1)]
        if len(win):
            best.append(cv_bandwidth(win))
            n_ext.append((extrema(smooth(win))[:, :L_DAYS] != 0).sum(1))
    if not best:
        return {}
    b, e = np.concatenate(best), np.concatenate(n_ext)
    return {"at_grid_min_pct": round(float((b == 0).mean() * 100), 1),
            "median_h_cv": round(float(np.median(H_GRID[b])), 2),
            "median_extrema_per_window": float(np.median(e)), "n_codes": len(cols)}


def compare(cal: dict) -> pd.DataFrame:
    """樣本外 H = PRIMARY：LMW 形態 vs patterns.py 規則式形態（讀 data/patterns_calib.json）。"""
    rows = []
    old_f = OUT / "patterns_calib.json"
    old = json.loads(old_f.read_text()) if old_f.exists() else {}
    for src, c, spec in (("LMW", cal, PATTERNS), ("規則式", old, P.PATTERNS)):
        for k in spec:
            if k not in c or spec[k][0] == 0:
                continue
            x = c[k]["horizons"][str(P.PRIMARY)][P.OOS] if str(P.PRIMARY) in c[k]["horizons"] \
                else c[k]["horizons"][P.PRIMARY][P.OOS]
            rows.append({"來源": src, "形態": c[k]["name"], "方向": P.SIDE[spec[k][0]], "平均超額%": x["mean_ex_pct"],
                         "勝率%": x["hit_pct"], "NW t": x["t"], "每期檔數": x["avg_n"], "期數": x["n_dates"],
                         "可靠度": c[k]["reliability"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", action="store_true")
    a = ap.parse_args()
    w = tc.load()
    if a.latest:
        w = {k: v.iloc[-P.LATEST_ROWS:] for k, v in w.items()}
    uni = tc.universe(w)
    hits, levels = detect(w)
    lt = P.latest(w, uni, hits, levels, PATTERNS)
    lt.to_parquet(OUT / "patterns_lmw_latest.parquet", index=False)
    if not a.latest:
        cal = P.calibrate(w, uni, hits, PATTERNS)
        cal["_meta"]["method"] = "Lo, Mamaysky & Wang (2000) 核迴歸形態；方向照技術分析慣例（改編）"
        cal["_meta"]["ks"] = ks_tests(w, uni, hits)
        cal["_meta"]["bandwidth"] = diagnose(w)
        (OUT / "patterns_lmw_calib.json").write_text(json.dumps(cal, ensure_ascii=False, indent=1))
        n = {k: int(v.to_numpy().sum()) for k, v in hits.items()}
        print("頻寬診斷：", json.dumps(cal["_meta"]["bandwidth"], ensure_ascii=False))
        print("LMW 形態成立次數（全部股票、全部歷史）：", json.dumps({PATTERNS[k][2]: v for k, v in n.items()},
                                                          ensure_ascii=False))
        print(f"\n樣本外（2019 起）H{P.PRIMARY} 產業中性超額：LMW vs 規則式")
        print(compare(cal).to_string(index=False))
        print("\nKS 檢定：形態成立隔日標準化報酬 vs 全部股票日（p 值未校正多重比較，樣本相依會偏樂觀）")
        rows = [{"形態": PATTERNS[k][2], **{f"{p} {m}": cal["_meta"]["ks"][p][k].get(m) for p in (P.IS, P.OOS)
                                          for m in ("n", "D", "p")}} for k in PATTERNS]
        print(pd.DataFrame(rows).to_string(index=False))
    print(f"\npatterns_lmw_latest.parquet：{w['adj_close'].index[-1]:%Y-%m-%d}，{len(lt)} 筆（{lt['code'].nunique()} 檔）")


if __name__ == "__main__":
    main()
