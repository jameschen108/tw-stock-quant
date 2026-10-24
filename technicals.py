# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""短 / 中 / 長期技術面訊號與大盤多空判斷，用日價格回測對之後報酬的預測力。

用法:
  uv run technicals.py                 # 個股訊號 + 大盤多空
  uv run technicals.py --part stocks   # 只跑個股訊號
  uv run technicals.py --part market   # 只跑大盤多空

個股訊號（仿 Trading Central 的事件，分三類）:
  state  每天都有值的多空狀態（+1 多 / −1 空），例：收盤在 50 日均線之上
  event  最近 5 個交易日內發生的交叉事件（+1 多方事件 / −1 空方事件 / 0 沒有）
  level  連續數值，例：CCI、RSI、過去報酬
評估（每 5 個交易日取樣一次）:
  - 報酬 = 訊號日隔天收盤進場，持有 H 個交易日的還原報酬，減同日同產業等權平均（產業中性超額）
  - state / event：多方組平均超額 − 空方組平均超額（每組至少 MIN_GROUP 檔）
  - level：與超額報酬的 rank IC
  - t 值用 Newey-West（lag = H / 5），因為持有期重疊
  - H = 10、30（短期 2–6 週）與 60、190（中期 6 週–9 個月）；分 2011–2018 / 2019–2026 兩段

大盤多空:
  - 指數 = 股票池市值加權（權重用上月底市值），另附等權
  - 狀態: 指數在 200 日均線上下、50/200 日均線、股票池站上自身 200 日均線的比例（廣度）、
    距 252 日高點回檔 20%
  - 評估: 各狀態之後 H 日指數報酬；擇時（多頭持有、空頭空手，隔天收盤執行，空手報酬 0）與一直持有比較

股票池: 普通股、當天有成交、上市滿 120 個交易日、收盤 >= 5、20 日均成交值 >= 500 萬。
產業用 factors_monthly 的 TEJ 主產業（每家公司取最後一個值，近似）。
"""
import argparse

import numpy as np
import pandas as pd

from build_panel import OUT

HORIZONS = [10, 30, 60, 190]
STEP = 5
MIN_GROUP = 20
MIN_EVENT = 5          # 事件訊號每組最少檔數（黃金 / 死亡交叉很少）
PERIODS = [("2011–2018", "2011-01-01", "2018-12-31"), ("2019–2026", "2019-01-01", "2026-12-31")]

# 名稱 -> (類別, 期別, 說明)
SIGNALS = {
    "px_ma21":     ("state", "短", "收盤在 21 日均線之上 / 之下"),
    "triple_ma":   ("state", "短", "4 > 9 > 18 日均線多頭排列 / 空頭排列"),
    "kst":         ("state", "短", "短期 KST 在訊號線之上 / 之下"),
    "x_ma21":      ("event", "短", "5 日內收盤向上 / 向下穿越 21 日均線"),
    "x_cci":       ("event", "短", "5 日內 CCI 向上穿越 −100 / 向下穿越 +100"),
    "x_triple":    ("event", "短", "5 日內轉為多頭排列 / 空頭排列（4/9/18 日）"),
    "cci20":       ("level", "短", "CCI(20)"),
    "rsi14":       ("level", "短", "RSI(14)"),
    "ret_20d":     ("level", "短", "過去 20 日報酬"),
    "bb_pct":      ("level", "短", "布林通道 %b(20, 2)"),
    "vol_surge":   ("level", "短", "5 日均成交值 / 60 日均成交值"),
    "macd":        ("state", "中", "MACD 柱狀體 > 0 / < 0"),
    "px_ma50":     ("state", "中", "收盤在 50 日均線之上 / 之下"),
    "x_macd":      ("event", "中", "5 日內 MACD 向上 / 向下穿越訊號線"),
    "x_ma50":      ("event", "中", "5 日內收盤向上 / 向下穿越 50 日均線"),
    "ret_6_1":     ("level", "中", "過去 6 個月扣最近 1 個月報酬"),
    "from_high":   ("level", "中", "距 252 日最高收盤 %"),
    "px_ma200":    ("state", "長", "收盤在 200 日均線之上 / 之下"),
    "golden":      ("state", "長", "50 日均線在 200 日之上 / 之下"),
    "x_ma200":     ("event", "長", "5 日內收盤向上 / 向下穿越 200 日均線"),
    "x_golden":    ("event", "長", "5 日內黃金交叉 / 死亡交叉（50/200）"),
    "ret_12_1":    ("level", "長", "過去 12 個月扣最近 1 個月報酬"),
}


# ---------------------------------------------------------------- 資料

def load() -> dict[str, pd.DataFrame]:
    px = pd.read_parquet(OUT / "prices_daily.parquet",
                         columns=["date", "code", "high", "low", "close", "value", "adj_close", "ret", "ret_suspect"])
    px = px[px["code"].str.fullmatch(r"[1-9]\d{3}")]
    px["adj"] = px["adj_close"] / px["close"]
    px["adj_high"] = px["high"] * px["adj"]
    px["adj_low"] = px["low"] * px["adj"]
    w = {c: px.pivot(index="date", columns="code", values=c)
         for c in ["close", "value", "adj_close", "adj_high", "adj_low", "ret", "ret_suspect"]}
    w["ret_suspect"] = w["ret_suspect"].fillna(False).astype(bool)
    return w


def universe(w: dict) -> pd.DataFrame:
    traded = w["value"] > 0
    n_obs = w["close"].notna().cumsum()
    avg_val = w["value"].fillna(0).rolling(20, min_periods=1).mean()
    return traded & (n_obs >= 120) & (w["close"] >= 5) & (avg_val >= 5e6)


def industries(codes) -> pd.Series:
    f = pd.read_parquet(OUT / "factors_monthly.parquet", columns=["date", "code", "industry"])
    return f.dropna(subset=["industry"]).sort_values("date").groupby("code")["industry"].last() \
        .reindex(codes).fillna("NA")


def forward(w: dict, h: int) -> pd.DataFrame:
    """t+1 收盤到 t+1+h 收盤的還原報酬；區間內有可疑報酬就設缺值。"""
    c = w["adj_close"].ffill(limit=5)
    fwd = c.shift(-(1 + h)) / c.shift(-1) - 1
    bad = w["ret_suspect"].astype(int).cumsum()
    bad_in = bad.shift(-(1 + h)) - bad.shift(-1)
    return fwd.where(bad_in == 0)


# ---------------------------------------------------------------- 指標

def ma(x, n):
    return x.rolling(n, min_periods=n).mean()


def crosses(state: pd.DataFrame, days: int = 5) -> pd.DataFrame:
    """state 為 +1 / −1；回傳最近 days 日內的翻多(+1) / 翻空(−1) 事件，沒有為 0。"""
    prev = state.shift(1)
    up = ((state == 1) & (prev == -1)).astype(float).where(state.notna() & prev.notna())
    dn = ((state == -1) & (prev == 1)).astype(float).where(state.notna() & prev.notna())
    up5 = up.rolling(days, min_periods=1).max()
    dn5 = dn.rolling(days, min_periods=1).max()
    ev = up5 - dn5                        # 5 日內兩種都有時抵銷成 0
    return ev.where(state.notna())


def sign(x: pd.DataFrame) -> pd.DataFrame:
    return np.sign(x).replace(0, np.nan)


def signals(w: dict, sample: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """回傳只留取樣日的訊號（float32）；日資料中間值用完即丟，避免記憶體爆掉。"""
    c = w["adj_close"].ffill(limit=5)
    h = w["adj_high"].ffill(limit=5)
    lo = w["adj_low"].ffill(limit=5)
    s = {}

    def put(name, x):
        s[name] = x.loc[sample].astype("float32")

    m = {n: ma(c, n) for n in [4, 9, 18, 21, 50, 200]}
    for name, x in (("px_ma21", sign(c - m[21])), ("px_ma50", sign(c - m[50])),
                    ("px_ma200", sign(c - m[200])), ("golden", sign(m[50] - m[200]))):
        put(name, x)
        put("x_" + name.removeprefix("px_"), crosses(x))
    bull = (m[4] > m[9]) & (m[9] > m[18])
    bear = (m[4] < m[9]) & (m[9] < m[18])
    tri = pd.DataFrame(np.where(bull, 1.0, np.where(bear, -1.0, np.nan)),
                       index=c.index, columns=c.columns).where(m[18].notna())
    put("triple_ma", tri)
    put("x_triple", crosses(tri.ffill().where(m[18].notna())))   # 中間經過不排列也算轉換
    sd = c.rolling(20, min_periods=20).std()
    put("bb_pct", (c - (ma(c, 20) - 2 * sd)) / (4 * sd))
    del m, bull, bear, tri, sd

    # KST（Pring 短期日線版）
    roc = lambda n: c / c.shift(n) - 1
    kst = ma(roc(10), 10) + 2 * ma(roc(15), 10) + 3 * ma(roc(20), 10) + 4 * ma(roc(30), 15)
    put("kst", sign(kst - ma(kst, 9)))
    del kst

    # MACD(12, 26, 9)
    macd = c.ewm(span=12, adjust=False, min_periods=26).mean() - c.ewm(span=26, adjust=False, min_periods=26).mean()
    st = sign(macd - macd.ewm(span=9, adjust=False, min_periods=9).mean())
    put("macd", st)
    put("x_macd", crosses(st))
    del macd, st

    # CCI(20)：平均絕對離差逐期展開
    tp = (h + lo + c) / 3
    sma = ma(tp, 20)
    mad = sum((tp.shift(k) - sma).abs() for k in range(20)) / 20
    cci = (tp - sma) / (0.015 * mad.replace(0, np.nan))
    put("cci20", cci)
    cci_up = ((cci > -100) & (cci.shift(1) <= -100)).astype(float)
    cci_dn = ((cci < 100) & (cci.shift(1) >= 100)).astype(float)
    put("x_cci", (cci_up.rolling(5, min_periods=1).max() - cci_dn.rolling(5, min_periods=1).max()).where(cci.notna()))
    del tp, sma, mad, cci, cci_up, cci_dn

    # RSI(14)，Wilder 平滑
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    put("rsi14", 100 - 100 / (1 + gain / loss.replace(0, np.nan)))
    del d, gain, loss

    v = w["value"].fillna(0)
    put("vol_surge", ma(v, 5) / ma(v, 60).replace(0, np.nan))
    put("ret_20d", c / c.shift(20) - 1)
    put("ret_6_1", c.shift(21) / c.shift(126) - 1)
    put("ret_12_1", c.shift(21) / c.shift(252) - 1)
    put("from_high", c / c.rolling(252, min_periods=200).max() - 1)
    return s


# ---------------------------------------------------------------- 評估

def nw_t(x: pd.Series, lag: int) -> float:
    x = x.dropna()
    n = len(x)
    if n < 20:
        return np.nan
    e = x - x.mean()
    v = (e @ e) / n
    for k in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - k / (lag + 1)) * (e[k:].values @ e[:-k].values) / n
    return x.mean() / np.sqrt(v / n)


def row_rank_ic(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    ok = a.notna() & b.notna()
    ra = a.where(ok).rank(axis=1)
    rb = b.where(ok).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    ic = (ra * rb).sum(axis=1) / np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    return ic.where(ok.sum(axis=1) >= 100)


def spread(sig: pd.DataFrame, ex: pd.DataFrame, min_n: int = MIN_GROUP) -> pd.Series:
    """多方組 − 空方組的平均超額。"""
    up, dn = ex.where(sig == 1), ex.where(sig == -1)
    ok = (up.notna().sum(axis=1) >= min_n) & (dn.notna().sum(axis=1) >= min_n)
    return (up.mean(axis=1) - dn.mean(axis=1)).where(ok)


def industry_excess(fwd: pd.DataFrame, ind: pd.Series) -> pd.DataFrame:
    t = fwd.T
    return (t - t.groupby(ind.reindex(t.index).values).transform("mean")).T


def run_stocks(w: dict, uni: pd.DataFrame):
    idx = w["adj_close"].index
    sample = idx[(idx >= "2011-01-01")][::STEP]
    s = signals(w, sample)
    ind = industries(w["adj_close"].columns)
    u = uni.loc[sample]

    res = {}
    for h in HORIZONS:
        ex = industry_excess(forward(w, h).loc[sample].where(u), ind)
        for name, (kind, _, _) in SIGNALS.items():
            x = s[name].where(u)
            res[(name, h)] = row_rank_ic(x, ex) if kind == "level" \
                else spread(x, ex, MIN_EVENT if kind == "event" else MIN_GROUP)

    cov = {name: s[name].where(u).replace(0, np.nan).notna().sum(axis=1).mean()
           for name, (kind, _, _) in SIGNALS.items() if kind == "event"}
    print("個股技術訊號：產業中性超額報酬（state/event = 多方組 − 空方組，%；level = rank IC），括號為 Newey-West t\n"
          f"每 {STEP} 個交易日取樣；H 為持有交易日數（10≈2 週、30≈6 週、60≈3 個月、190≈9 個月）\n")
    for p, a, b in PERIODS:
        rows = []
        for name, (kind, term, desc) in SIGNALS.items():
            r = {"訊號": name, "期": term, "類": kind}
            for h in HORIZONS:
                x = res[(name, h)].loc[a:b]
                v = x.mean() * (1 if kind == "level" else 100)
                r[f"H{h}"] = f"{v:+.3f}({nw_t(x, h // STEP):+.1f})" if kind == "level" \
                    else f"{v:+.2f}({nw_t(x, h // STEP):+.1f})"
            rows.append(r)
        print(f"== {p}")
        print(pd.DataFrame(rows).set_index("訊號").to_string(), "\n")
    print("事件訊號平均每期有事件的檔數:", {k: round(v) for k, v in cov.items()})
    print("\n說明:", "; ".join(f"{k} {v[2]}" for k, v in SIGNALS.items()))
    return s


# ---------------------------------------------------------------- 大盤

def market_index(w: dict, uni: pd.DataFrame) -> pd.DataFrame:
    r = w["ret"].where(~w["ret_suspect"]).where(uni.shift(1, fill_value=False))
    f = pd.read_parquet(OUT / "factors_monthly.parquet", columns=["date", "code", "mcap"])
    mc = f.pivot(index="date", columns="code", values="mcap").reindex(columns=r.columns)
    wgt = mc.reindex(r.index, method="ffill").shift(1)     # 上月底（前一交易日以前）已知的市值
    cap = (r * wgt).sum(axis=1) / wgt.where(r.notna()).sum(axis=1)
    eq = r.mean(axis=1)
    out = pd.DataFrame({"cap": cap, "eq": eq}).loc["2010-03-01":]
    return out.fillna(0)


def run_market(w: dict, uni: pd.DataFrame, s: dict | None):
    rets = market_index(w, uni)
    c = w["adj_close"].ffill(limit=5)
    m200 = ma(c, 200)
    above200 = (c > m200).astype(float).where(m200.notna() & uni)
    breadth = above200.mean(axis=1)
    del m200

    for col, label in (("cap", "市值加權"), ("eq", "等權")):
        lvl = (1 + rets[col]).cumprod()
        m50, m200 = ma(lvl, 50), ma(lvl, 200)
        states = {
            "指數 > 200 日線": lvl > m200,
            "50 日線 > 200 日線": m50 > m200,
            "廣度 > 50%": breadth.reindex(lvl.index) > 0.5,
            "距高點回檔 < 20%": lvl / lvl.rolling(252, min_periods=200).max() - 1 > -0.2,
            "200 日線且廣度": (lvl > m200) & (breadth.reindex(lvl.index) > 0.5),
        }
        valid = m200.notna() & (lvl.index >= "2011-01-01")
        print(f"\n==== 大盤（股票池{label}指數）")
        rows = []
        for name, st in states.items():
            st = st[valid]
            for p, a, b in PERIODS:
                r = {"狀態": name, "期間": p, "多頭天數比": st.loc[a:b].mean()}
                for h in HORIZONS:
                    fwd = (lvl.shift(-(1 + h)) / lvl.shift(-1) - 1)[valid].loc[a:b]
                    stp = st.loc[a:b]
                    diff = fwd[stp].mean() - fwd[~stp].mean()
                    # 差異的 t：以狀態虛擬變數回歸，NW lag = h
                    z = pd.Series(np.where(stp, 1 / stp.mean(), -1 / (1 - stp.mean())), index=stp.index) \
                        * (fwd - fwd.mean())
                    r[f"H{h} 多/空"] = f"{fwd[stp].mean() * 100:+.1f}/{fwd[~stp].mean() * 100:+.1f}" \
                                       f"({nw_t(z.dropna(), h):+.1f})"
                rows.append(r)
        tab = pd.DataFrame(rows).set_index(["狀態", "期間"])
        tab["多頭天數比"] = tab["多頭天數比"].map("{:.0%}".format)
        print("之後 H 日指數報酬 %（多頭狀態 / 空頭狀態），括號為差異的 NW t")
        print(tab.to_string())

        rows = []
        for p, a, b in PERIODS:
            base = rets[col].loc[a:b]
            for name, st in [("一直持有", None)] + list(states.items()):
                x = base if st is None else base * st.shift(1).reindex(base.index).fillna(False)
                nav = (1 + x).cumprod()
                yrs = len(x) / 250
                rows.append({"期間": p, "策略": name, "年化": nav.iloc[-1] ** (1 / yrs) - 1,
                             "波動": x.std() * np.sqrt(250), "Sharpe": x.mean() / x.std() * np.sqrt(250),
                             "最大回撤": (nav / nav.cummax() - 1).min(),
                             "持有比例": 1.0 if st is None else st.shift(1).reindex(base.index).fillna(False).mean(),
                             "換手次數/年": 0 if st is None else st.astype(int).diff().abs().loc[a:b].sum() / yrs})
        tab = pd.DataFrame(rows).set_index(["期間", "策略"])
        for k, f in {"年化": "{:+.1%}", "波動": "{:.1%}", "Sharpe": "{:.2f}", "最大回撤": "{:.0%}",
                     "持有比例": "{:.0%}", "換手次數/年": "{:.1f}"}.items():
            tab[k] = tab[k].map(f.format)
        print("\n擇時：多頭持有指數、空頭空手（隔天收盤執行，空手報酬 0、未扣成本）")
        print(tab.to_string())

    # 多空狀態下，個股訊號的效果是否不同（H30，市值加權指數 > 200 日線）
    if s is None:
        return
    lvl = (1 + rets["cap"]).cumprod()
    bull = (lvl > ma(lvl, 200))
    idx = w["adj_close"].index
    sample = idx[idx >= "2011-01-01"][::STEP]
    ind = industries(w["adj_close"].columns)
    u = uni.loc[sample]
    ex = industry_excess(forward(w, 30).loc[sample].where(u), ind)
    b = bull.reindex(sample).fillna(False)
    rows = []
    for name, (kind, term, _) in SIGNALS.items():
        x = s[name].where(u)
        v = row_rank_ic(x, ex) if kind == "level" \
            else spread(x, ex, MIN_EVENT if kind == "event" else MIN_GROUP) * 100
        rows.append({"訊號": name, "期": term, "類": kind,
                     "多頭時": f"{v[b].mean():+.3f}({nw_t(v[b], 6):+.1f})",
                     "空頭時": f"{v[~b].mean():+.3f}({nw_t(v[~b], 6):+.1f})"})
    print(f"\n個股訊號在大盤多 / 空頭時的效果（H30，2011–2026；多頭期 {b.mean():.0%}）")
    print(pd.DataFrame(rows).set_index("訊號").to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["all", "stocks", "market"], default="all")
    a = ap.parse_args()
    w = load()
    uni = universe(w)
    s = run_stocks(w, uni) if a.part in ("all", "stocks") else None
    if a.part in ("all", "market"):
        run_market(w, uni, s)


if __name__ == "__main__":
    main()
