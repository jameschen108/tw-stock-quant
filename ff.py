# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""因子模型拆解：台股自建 Fama-French／Carhart 代理因子，拆解策略報酬與持股的因子曝險。

用法:
  uv run ff.py build                                   # 建因子 → data/ff_factors_daily.parquet、ff_factors_monthly.parquet
  uv run ff.py                                         # 迴歸＋持股曝險，輸出 reports/factor_model.html
  uv run ff.py strategies/top30.toml strategies/mid_bull_invvol.toml   # 指定要重跑持股曝險的策略（預設見 HOLDINGS）

因子（自建代理因子，不是 Kenneth French 網站的官方因子）:
  - MKT = 全市場市值加權 − rf；SMB、HML 為 size × B/M 的 2×3 排序（Fama & French 1993）
  - RMW、CMA 為 size × 營業獲利率、size × 資產成長的 2×3 排序；FF5 的 SMB 取三組排序平均（Fama & French 2015）
  - MOM 為 size × 過去 12 個月扣掉最近 1 個月報酬的 2×3 排序（Carhart 1997）
  - 與原文的差異：每月重組（原文每年 6 月）、分位用全體台股普通股（原文 NYSE 斷點）、
    權益與獲利用換股日前已公布的最新季報（TTM），不是上一個會計年度
  - 時點同 factors.py：換股日 t = 每月最後交易日，只用 t 之前公布的財報；t 後第 LAG 個交易日收盤組成，
    買進持有到下一個換股日後第 LAG 個交易日（期間不再平衡，權重隨價漂移）
  - rf：refdata/tw_rf_1y.csv 五大銀行一年期定存固定利率，當月年利率 ÷ 12（日頻 ÷ 252）

分析:
  1. 策略報酬：data/strategy/results/*.parquet 的扣成本日報酬，複利成月報酬，對 CAPM、FF3、Carhart、FF5、
     FF5+MOM 迴歸，Newey-West 標準誤（lag 3）；另畫滾動 36 個月的 Carhart beta
  2. 持股特徵（參考 Daniel, Grinblatt, Titman & Wermers 1997 的特徵基準）：每檔持股在換股日的橫斷面百分位
     （0.5 = 市場中位數），組合等權平均；小型股比例 = 市值低於中位數（FF 的 S 組）的持股比例
     策略重跑 strategy.simulate 取每期持股；LLM 回放比 buy 與其他候選；前向三組讀 mine_ledger
  3. 前向三組的報酬迴歸要有 RET_MIN_DAYS 個交易日以上才算，不足時只列持股特徵
"""
import argparse
import html
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import factors as F
import strategy as S
from build_panel import OUT
from report import CSS, svg_lines
from robust import _p, _t, table

ROOT = Path(__file__).parent
RF_CSV = ROOT / "refdata" / "tw_rf_1y.csv"
DAILY = OUT / "ff_factors_daily.parquet"
MONTHLY = OUT / "ff_factors_monthly.parquet"
LAG = F.LAG
NW_LAG = 3
ROLL = 36            # 滾動迴歸的月數
RET_MIN_DAYS = 120   # 前向組合日報酬少於這麼多天不做迴歸
N_PERM = 5000
HOLDINGS = ["strategies/top30.toml", "strategies/mid_bull_invvol.toml", "strategies/top30_mt.toml",
            "strategies/top30_regime_daily.toml"]
FORWARD = ["LLM篩選-haiku", "量化全買", "隨機對照-haiku"]

MODELS = {
    "CAPM": ["MKT"],
    "FF3": ["MKT", "SMB", "HML"],
    "Carhart": ["MKT", "SMB", "HML", "MOM"],
    "FF5": ["MKT", "SMB5", "HML", "RMW", "CMA"],
    "FF5+MOM": ["MKT", "SMB5", "HML", "RMW", "CMA", "MOM"],
}
# 排序變數 -> (說明, 高分位的腿是多方嗎)；CMA 是保守（資產成長低）減積極
SORTS = {"bm": ("B/M", True), "op": ("營業獲利率", True), "inv": ("資產成長", False), "mom": ("動能 12-1", True)}
CHARS = {"size": "市值", "bm": "B/M", "op": "營業獲利率", "inv": "資產成長", "mom": "動能 12-1"}


# ---------------------------------------------------------------- 資料

def load_rf() -> pd.Series:
    """月 → 一年期定存年利率（%），索引是月份 Period。"""
    d = pd.read_csv(RF_CSV, comment="#")
    return d.set_index(pd.PeriodIndex(d["month"], freq="M"))["rate_1y"]


def rf_on(index: pd.DatetimeIndex, per_year: int, rf: pd.Series | None = None) -> pd.Series:
    """每個日期（或月底）的無風險報酬：當月年利率 ÷ per_year；資料還沒公布的月份沿用最後一個月。"""
    rf = load_rf() if rf is None else rf
    months = index.to_period("M")
    full = rf.reindex(pd.period_range(rf.index.min(), max(months.max(), rf.index.max()), freq="M")).ffill()
    return pd.Series(full.reindex(months).values / 100 / per_year, index=index)


def op_table() -> pd.DataFrame:
    """營業獲利率（Fama & French 2015）：TTM(營業收入 − 營業成本 − 營業費用 − 財務成本) ÷ 母公司股東權益。"""
    f = pd.read_parquet(OUT / "tej_fin.parquet", columns=["code", "period", "avail_date", "單季(Q)/單半年(H)",
                                                          "營業收入淨額", "營業成本", "營業費用", "財務成本",
                                                          "母公司股東權益合計"])
    f = f.sort_values(["code", "period"]).reset_index(drop=True)
    g = f.groupby("code", sort=False)
    months = f["period"].dt.year * 12 + f["period"].dt.month
    q = f["單季(Q)/單半年(H)"].eq("Q")
    ttm_ok = q & (months - g["period"].shift(3).pipe(lambda s: s.dt.year * 12 + s.dt.month) == 9) \
        & g["單季(Q)/單半年(H)"].transform(lambda s: s.eq("Q").rolling(4).sum() == 4).fillna(False)
    prof = f["營業收入淨額"] - f["營業成本"] - f["營業費用"] - f["財務成本"].fillna(0)
    ttm = prof.groupby(f["code"]).transform(lambda s: s.rolling(4).sum()).where(ttm_ok)
    eq = f["母公司股東權益合計"]
    return f[["code", "period", "avail_date"]].assign(op=(ttm / eq).where(eq > 0))


def characteristics() -> pd.DataFrame:
    """換股日 × 股票：mcap、bm、op、inv、mom、industry。權益 ≤ 0 的 B/M、OP、資產成長設缺值（同 FF 只排正權益）。"""
    fm = pd.read_parquet(OUT / "factors_monthly.parquet",
                         columns=["date", "code", "mcap", "bp", "asset_growth", "mom_12_1", "industry"])
    fm = fm[fm["mcap"] > 0]
    ch = F.attach(fm, op_table(), "fin", ["op"]).drop(columns="fin_period")
    pos = ch["bp"] > 0
    ch = ch.rename(columns={"bp": "bm", "asset_growth": "inv", "mom_12_1": "mom"})
    for c in ("bm", "op", "inv"):
        ch[c] = ch[c].where(pos)
    return ch.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def load_rets() -> pd.DataFrame:
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "ret", "ret_suspect"])
    return S.clean_returns(px)


# ---------------------------------------------------------------- 因子

def sort_2x3(day: pd.DataFrame, col: str) -> pd.Series:
    """單一換股日：size 中位數切 S/B，col 的 30/70 分位切 L/M/H；回傳每檔的組別（例 'S_H'），排不進去的不列。"""
    d = day.dropna(subset=["mcap", col])
    if len(d) < 10:
        return pd.Series(dtype=str)
    size = np.where(d["mcap"] <= d["mcap"].median(), "S", "B")
    lo, hi = d[col].quantile([0.3, 0.7])
    lvl = np.where(d[col] <= lo, "L", np.where(d[col] > hi, "H", "M"))
    return pd.Series([f"{a}_{b}" for a, b in zip(size, lvl)], index=d["code"].values)


def hold_returns(rets: pd.DataFrame, start: int, end: int, groups: dict[str, pd.Series]) -> pd.DataFrame:
    """rets 第 start 列之後到第 end 列（含）的日報酬；每組依期初權重買進持有、權重隨價漂移。"""
    win = rets.iloc[start + 1:end + 1]
    out = {}
    for name, w in groups.items():
        w = w[w.index.isin(win.columns) & (w > 0)]
        if not len(w):
            out[name] = pd.Series(np.nan, index=win.index)
            continue
        grow = (1 + win[w.index].fillna(0)).cumprod()
        v = grow @ (w / w.sum()).values
        out[name] = v / v.shift(1, fill_value=1.0) - 1
    return pd.DataFrame(out, index=win.index)


def legs(day: pd.DataFrame) -> dict[str, pd.Series]:
    """單一換股日的所有市值加權組合：MKT 與四個 2×3 排序的 6 組。"""
    w = day.set_index("code")["mcap"]
    out = {"MKT": w}
    for col in SORTS:
        lab = sort_2x3(day, col)
        for g in lab.unique():
            out[f"{col}:{g}"] = w.reindex(lab.index[lab == g])
    return out


def combine(p: pd.DataFrame) -> pd.DataFrame:
    """6 組組合報酬 → 多空因子。缺任何一組的期間給缺值。"""
    def avg(col, side, lv):
        return p[[f"{col}:{side}_{x}" for x in lv]].mean(axis=1, skipna=False)

    def spread(col, long_hi):
        hi = (p[f"{col}:S_H"] + p[f"{col}:B_H"]) / 2
        lo = (p[f"{col}:S_L"] + p[f"{col}:B_L"]) / 2
        return hi - lo if long_hi else lo - hi

    for c in [f"{col}:{s}_{x}" for col in SORTS for s in "SB" for x in "LMH"]:
        if c not in p:
            p[c] = np.nan
    smb = {col: avg(col, "S", "LMH") - avg(col, "B", "LMH") for col in SORTS}
    return pd.DataFrame({
        "MKT_RAW": p["MKT"],
        "SMB": smb["bm"],
        "SMB5": (smb["bm"] + smb["op"] + smb["inv"]) / 3,
        "HML": spread("bm", True), "RMW": spread("op", True), "CMA": spread("inv", False), "MOM": spread("mom", True),
    })


def build_factors(ch: pd.DataFrame, rets: pd.DataFrame, lag: int = LAG) -> pd.DataFrame:
    """日頻因子報酬（未扣 rf 的 MKT_RAW 與多空因子）。換股日 t 組成，持有 (t+lag, 下一個換股日+lag]。"""
    days = rets.index
    reb = pd.DatetimeIndex(sorted(ch["date"].unique()))
    reb = reb[reb.isin(days)]
    pos = days.get_indexer(reb)
    parts = []
    for k, t in enumerate(reb):
        start = pos[k] + lag
        end = pos[k + 1] + lag if k + 1 < len(reb) else len(days) - 1
        if start >= len(days) - 1 or end <= start:
            continue
        p = hold_returns(rets, start, min(end, len(days) - 1), legs(ch[ch["date"] == t]))
        parts.append(combine(p))
    return pd.concat(parts).sort_index()


def finish(raw: pd.DataFrame, rf: pd.Series | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """日頻 → (日表, 月表)，都加上 RF 並把 MKT 改成扣 rf 後的超額。月報酬是日報酬在月內複利。"""
    d = raw.dropna(subset=["MKT_RAW"]).copy()
    d["RF"] = rf_on(d.index, 252, rf)
    m = (1 + d.drop(columns="RF")).resample("ME").prod() - 1
    m[d.drop(columns="RF").isna().resample("ME").sum() > 0] = np.nan      # 月內有缺值的因子當月不算
    m["RF"] = rf_on(m.index, 12, rf)
    for x in (d, m):
        x["MKT"] = x["MKT_RAW"] - x["RF"]
    return d, m


# ---------------------------------------------------------------- 迴歸

def nw_ols(y: pd.Series, X: pd.DataFrame, lag: int = NW_LAG) -> dict:
    """OLS y = a + X b，Newey-West 標準誤（同 robust.nw_se_alpha 的多變數版）。lag = 0 即 White 標準誤。"""
    d = pd.concat([y.rename("_y"), X], axis=1).dropna()
    yv, Xv = d["_y"].values, np.column_stack([np.ones(len(d)), d[X.columns].values])
    n, k = Xv.shape
    if n <= k + 2:
        return {"n": n}
    beta, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    e = yv - Xv @ beta
    u = Xv * e[:, None]
    Sm = u.T @ u / n
    for j in range(1, min(lag, n - 1) + 1):
        g = u[j:].T @ u[:-j] / n
        Sm += (1 - j / (lag + 1)) * (g + g.T)
    inv = np.linalg.pinv(Xv.T @ Xv / n)
    se = np.sqrt(np.diag(inv @ Sm @ inv / n))
    names = ["alpha", *X.columns]
    return {"n": n, "coef": pd.Series(beta, index=names), "t": pd.Series(beta / se, index=names),
            "r2": 1 - (e @ e) / ((yv - yv.mean()) @ (yv - yv.mean())), "start": d.index.min(), "end": d.index.max()}


def to_monthly(daily: pd.Series) -> pd.Series:
    m = (1 + daily.fillna(0)).resample("ME").prod() - 1
    return m[daily.resample("ME").count() > 0]


def regress(excess_m: pd.Series, fm: pd.DataFrame) -> dict[str, dict]:
    return {name: nw_ols(excess_m, fm[cols]) for name, cols in MODELS.items()}


def rolling_betas(excess_m: pd.Series, fm: pd.DataFrame, cols: list[str], window: int = ROLL) -> pd.DataFrame:
    d = pd.concat([excess_m.rename("_y"), fm[cols]], axis=1).dropna()
    rows = {}
    for i in range(window, len(d) + 1):
        w = d.iloc[i - window:i]
        Xv = np.column_stack([np.ones(window), w[cols].values])
        b, *_ = np.linalg.lstsq(Xv, w["_y"].values, rcond=None)
        rows[d.index[i - 1]] = pd.Series(b, index=["alpha", *cols])
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- 持股特徵

def pct_ranks(ch: pd.DataFrame) -> pd.DataFrame:
    """每個換股日各特徵的橫斷面百分位（0–1）；size 用市值。"""
    src = ch.rename(columns={"mcap": "size"})
    r = src.groupby("date")[list(CHARS)].rank(pct=True)
    return pd.concat([src[["date", "code", "industry"]], r], axis=1)


def snapshot(pr: pd.DataFrame, codes, when: pd.Timestamp) -> pd.DataFrame:
    """when 當天或之前最近一個換股日的特徵百分位（只列 codes）。"""
    dates = pr["date"].unique()
    d = dates[dates <= when]
    if not len(d):
        return pr.iloc[0:0]
    day = pr[pr["date"] == d.max()].set_index("code")
    return day.reindex([c for c in codes if c in day.index])


def exposure(pr: pd.DataFrame, codes, when: pd.Timestamp) -> dict:
    s = snapshot(pr, codes, when)
    out = {c: s[c].mean() for c in CHARS}
    out["small"] = (s["size"] <= 0.5).mean() if len(s) else np.nan
    out["n"] = len(s)
    top = s["industry"].value_counts(normalize=True)
    out["top_ind"], out["top_ind_share"] = (top.index[0], top.iloc[0]) if len(top) else ("", np.nan)
    return out


def exposure_path(pr: pd.DataFrame, picks: dict[pd.Timestamp, list[str]]) -> pd.DataFrame:
    return pd.DataFrame({t: exposure(pr, c, t) for t, c in sorted(picks.items()) if len(c)}).T


def perm_diff(df: pd.DataFrame, col: str, flag: str = "buy", by: str = "batch", n: int = N_PERM,
              seed: str = "ff") -> tuple[float, float]:
    """flag 組減其他組的 col 平均，與「批次內打亂 flag」的雙尾 p 值。"""
    d = df.dropna(subset=[col])
    def diff(f):
        return d.loc[f, col].mean() - d.loc[~f, col].mean()
    obs = diff(d[flag].values)
    rng = random.Random(seed)
    groups = [np.flatnonzero((d[by] == b).values) for b in d[by].unique()]
    base = d[flag].values
    hits = 0
    for _ in range(n):
        f = base.copy()
        for g in groups:
            vals = list(f[g])
            rng.shuffle(vals)
            f[g] = vals
        hits += abs(diff(f)) >= abs(obs) - 1e-12
    return obs, (hits + 1) / (n + 1)


# ---------------------------------------------------------------- 各分析

def strategy_returns() -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(S.RESULTS.glob("*.parquet")):
        if p.stem.endswith("_fills"):
            continue
        out[p.stem] = pd.read_parquet(p).set_index("date")
    return out


def strategy_picks(path: str, panel, rets) -> dict[pd.Timestamp, list[str]]:
    res = S.simulate(S.load_config(Path(path)), panel, rets)
    return {pd.Timestamp(x["signal"]): list(x["picks"]) for x in (res or {}).get("picks", [])}


def replay_exposure(pr: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """LLM 回放每檔候選的特徵百分位；batch 是資料時點，用它當天或之前最近的換股日。"""
    ev = pd.read_csv(OUT / "paper" / "llm_replay_returns.csv", dtype={"code": str}, parse_dates=["batch"])
    ev = ev[ev["code"].str.fullmatch(r"[1-9]\d{3}")]
    rows = []
    for b, g in ev.groupby("batch"):
        s = snapshot(pr, g["code"], b)
        rows.append(g.merge(s[list(CHARS) + ["industry"]], left_on="code", right_index=True, how="left"))
    ev = pd.concat(rows, ignore_index=True)
    ev["buy"] = ev["decision"].eq("buy")
    ev["small"] = (ev["size"] <= 0.5).astype(float).where(ev["size"].notna())
    tests = {c: perm_diff(ev, c) for c in [*CHARS, "small"]}
    return ev, tests


def adjusted_buy(ev: pd.DataFrame, y: str = "r20") -> dict:
    """y = a + b·buy（＋批次固定效果）；再加 size、B/M、動能百分位。同批同窗口，個股報酬互相相關，t 只作參考。"""
    d = ev.dropna(subset=[y, "size", "bm", "mom"]).copy()
    fe = pd.get_dummies(d["batch"].dt.strftime("%m%d"), prefix="b", drop_first=True, dtype=float)
    X0 = pd.concat([d["buy"].astype(float), fe], axis=1)
    X1 = pd.concat([X0, d[["size", "bm", "mom"]]], axis=1)
    return {"raw": nw_ols(d[y], X0, lag=0), "adj": nw_ols(d[y], X1, lag=0), "n": len(d)}


def forward_groups(pr: pd.DataFrame, fd: pd.DataFrame) -> list[dict]:
    import mine as M
    led = M.load_ledger()
    out = []
    for name in FORWARD:
        sigs = M._latest_signals(led, name)
        if not sigs:
            continue
        path = exposure_path(pr, {t: list(w[w > 0].index) for t, w in sigs.items()})
        res = M.replay(name)
        reg, days = None, 0
        if res is not None:
            r = res["daily"]["strategy"]
            days = len(r)
            if days >= RET_MIN_DAYS:
                ex = r - fd["RF"].reindex(r.index)
                reg = {k: nw_ols(ex, fd[c].reindex(r.index), lag=5) for k, c in MODELS.items()}
        out.append({"name": name, "path": path, "days": days, "reg": reg})
    return out


# ---------------------------------------------------------------- 輸出

def fmt_reg(o: dict, cols: list[str]) -> list[str]:
    if "coef" not in o:
        return ["—"] * (3 + len(cols))
    c, t = o["coef"], o["t"]
    return ([f"{_p(c['alpha'] * 12)}<br><span class='muted'>t {_t(t['alpha'])}</span>"]
            + [f"{c[x]:+.2f}<br><span class='muted'>t {_t(t[x])}</span>" for x in cols]
            + [f"{o['r2']:.2f}", str(o["n"])])


def factor_summary(fm: pd.DataFrame, etf_m: pd.Series) -> str:
    cols = ["MKT", "SMB", "SMB5", "HML", "RMW", "CMA", "MOM"]
    rows = []
    for c in cols:
        x = fm[c].dropna()
        from technicals import nw_t
        rows.append([c, _p(x.mean() * 12), _p(x.std() * np.sqrt(12)), _t(nw_t(x, NW_LAG)),
                     f"{x.index.min():%Y-%m}–{x.index.max():%Y-%m}"])
    corr = fm[cols].corr().round(2)
    ctab = table(["", *cols], [[a, *[f"{corr.at[a, b]:+.2f}" for b in cols]] for a in cols])
    e = pd.concat([fm["MKT"] + fm["RF"], etf_m], axis=1).dropna()
    note = f"<p>自建 MKT（含 rf）與 0050 月報酬相關 {e.iloc[:, 0].corr(e.iloc[:, 1]):.2f}（{len(e)} 個月）。</p>"
    return (table(["因子", "年化平均", "年化波動", "NW t", "期間"], rows) + note
            + f"<h3>因子相關係數</h3><div class='wrap'>{ctab}</div>")


def exposure_chart(path: pd.DataFrame, title: str) -> str:
    lab = [f"{d:%y/%m}" for d in path.index]
    series = {CHARS[c]: [float(v) * 100 for v in path[c]] for c in CHARS}
    series["小型股比例"] = [float(v) * 100 for v in path["small"]]
    return svg_lines(lab, series, "{:.0f}", title, "{:.0f}", lo=0, hi=100, ref=50)


def exposure_table(rows: dict[str, dict]) -> str:
    out = []
    for name, e in rows.items():
        top = "—" if not e["top_ind"] else f"{html.escape(str(e['top_ind']))} {e['top_ind_share'] * 100:.0f}%"
        out.append([html.escape(name), f"{e['n']:.0f}", *[f"{e[c] * 100:.0f}" for c in CHARS],
                    f"{e['small'] * 100:.0f}%", top])
    return table(["組合", "檔數", *[f"{v} 百分位" for v in CHARS.values()], "小型股比例", "最大產業"], out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run")
    ap.add_argument("tomls", nargs="*")
    a = ap.parse_args()
    if a.cmd == "build":
        print("characteristics ..."); ch = characteristics()
        print("returns ..."); rets = load_rets()
        print("factors ..."); fd, fm = finish(build_factors(ch, rets))
        fd.to_parquet(DAILY); fm.to_parquet(MONTHLY)
        print(f"{len(fd):,} 日、{fm['MKT'].notna().sum()} 個月 → {DAILY.name}、{MONTHLY.name}")
        print((fm.drop(columns=["MKT_RAW", "RF"]).mean() * 12).round(4).to_string())
        return
    tomls = ([a.cmd] if a.cmd.endswith(".toml") else []) + a.tomls or HOLDINGS
    if not DAILY.exists():
        sys.exit("先跑 uv run ff.py build")
    fd, fm = pd.read_parquet(DAILY), pd.read_parquet(MONTHLY)
    ch = characteristics()
    pr = pct_ranks(ch)
    sections, console = [], []

    # 1. 策略報酬拆解
    strats = strategy_returns()
    etf_m = to_monthly(next(iter(strats.values()))["etf"])
    sections.append(f"<h2>自建因子</h2>{factor_summary(fm, etf_m)}")
    regs, ex_by = {}, {}
    for sid, d in strats.items():
        ex_by[sid] = to_monthly(d["strategy"]) - fm["RF"]
        regs[sid] = regress(ex_by[sid], fm)
    first = next(iter(strats.values()))
    for sid, col in (("同池等權（不扣成本）", "pool"), ("0050", "etf")):
        ex_by[sid] = to_monthly(first[col]) - fm["RF"]
        regs[sid] = regress(ex_by[sid], fm)
    blocks = []
    for model, cols in MODELS.items():
        rows = [[html.escape(sid), *fmt_reg(r[model], cols)] for sid, r in regs.items()]
        blocks.append(f"<h3>{model}</h3><div class='wrap'>{table(['策略', 'α / 年', *cols, 'R²', '月數'], rows)}</div>")
    for sid, r in regs.items():
        o, c = r["Carhart"], r["CAPM"]
        if "coef" in o:
            console.append({"策略": sid, "CAPM α/年": _p(c["coef"]["alpha"] * 12), "t": _t(c["t"]["alpha"]),
                            "Carhart α/年": _p(o["coef"]["alpha"] * 12), "t ": _t(o["t"]["alpha"]),
                            "MKT": round(o["coef"]["MKT"], 2), "SMB": round(o["coef"]["SMB"], 2),
                            "HML": round(o["coef"]["HML"], 2), "MOM": round(o["coef"]["MOM"], 2),
                            "FF5+MOM α/年": _p(r["FF5+MOM"]["coef"]["alpha"] * 12),
                            "t  ": _t(r["FF5+MOM"]["t"]["alpha"])})
    sections.append("<h2>策略報酬的因子迴歸</h2><p class='muted'>被解釋變數 = 策略扣成本月報酬 − rf；"
                    "α 年化 = 月 α × 12；t 用 Newey-West（lag 3）。係數下方小字是 t。</p>"
                    "<p>SMB 係數是相對市值加權的 MKT 而言：台股市值集中在少數大型股，等權持有中型股就會有正的 SMB，"
                    "所以 SMB 高不等於持股是小型股，要對照下面的持股市值百分位一起看。</p>" + "".join(blocks))

    # 2. 滾動曝險與持股特徵
    stem = {Path(t).stem: t for t in tomls}
    charts = []
    for sid in stem:
        if sid in ex_by:
            rb = rolling_betas(ex_by[sid], fm, MODELS["Carhart"])
            lab = [f"{d:%y/%m}" for d in rb.index]
            charts.append(svg_lines(lab, {c: list(rb[c]) for c in MODELS["Carhart"]}, "{:.1f}",
                                    f"{sid}：滾動 {ROLL} 個月 Carhart beta", "{:+.2f}", zero=True))
    print("重跑策略取持股 ...")
    panel, rets = S.load_data()
    hold_rows, hold_charts = {}, []
    for sid, path in stem.items():
        picks = strategy_picks(path, panel, rets)
        if not picks:
            continue
        ep = exposure_path(pr, picks)
        hold_rows[f"{sid}（全期平均）"] = {**ep[list(CHARS) + ["small", "n"]].astype(float).mean().to_dict(),
                                       "top_ind": "", "top_ind_share": np.nan}
        sm = ep[list(CHARS) + ["small"]].astype(float).rolling(12, min_periods=1).mean()
        hold_charts.append(exposure_chart(sm, f"{sid}：持股特徵百分位（12 期移動平均，50 = 中位數）"))
    sections.append(f"<h2>因子曝險隨時間變化</h2><h3>報酬迴歸：滾動 beta</h3><div class='charts'>{''.join(charts)}</div>"
                    f"<h3>持股特徵</h3><p class='muted'>每期持股在換股日的橫斷面百分位等權平均；"
                    f"小型股比例 = 市值低於全市場中位數的持股比例。</p>{exposure_table(hold_rows)}"
                    f"<div class='charts'>{''.join(hold_charts)}</div>")

    # 3. LLM 回放
    ev, tests = replay_exposure(pr)
    rows = []
    for b, g in [*ev.groupby(ev["batch"].dt.strftime("%Y-%m-%d")), ("合計", ev)]:
        for lab, f in (("buy", g["buy"]), ("其他候選", ~g["buy"]), ("全部候選", g["buy"] | True)):
            x = g[f]
            rows.append([b, lab, str(len(x)), *[f"{x[c].mean() * 100:.0f}" for c in CHARS], f"{x['small'].mean() * 100:.0f}%"])
    trow = [["buy − 其他", "", "", *[f"{tests[c][0] * 100:+.1f}<br><span class='muted'>p {tests[c][1]:.2f}</span>"
                                     for c in CHARS], f"{tests['small'][0] * 100:+.1f} 點<br><span class='muted'>p {tests['small'][1]:.2f}</span>"]]
    head = ["批次", "組", "檔數", *[f"{v} 百分位" for v in CHARS.values()], "小型股比例"]
    adj = adjusted_buy(ev)
    def coef(o):
        return "—" if "coef" not in o else f"{_p(o['coef']['buy'])}（t {_t(o['t']['buy'])}）"
    ind = pd.crosstab(ev["industry"], ev["buy"], normalize="columns").rename(columns={True: "buy", False: "其他"})
    ind = ind.assign(差=ind.get("buy", 0) - ind.get("其他", 0)).sort_values("差")
    ind_rows = [[html.escape(str(i)), f"{r.get('buy', 0) * 100:.0f}%", f"{r.get('其他', 0) * 100:.0f}%",
                 f"{r['差'] * 100:+.0f} 點"] for i, r in pd.concat([ind.head(3), ind.tail(3)]).iterrows()]
    sections.append(
        "<h2>LLM 回放：buy 是不是偏小型股</h2><p class='muted'>data/paper/llm_replay_returns.csv（7–9 月三批）。"
        "特徵用各批資料時點當天或之前最近的月底。p 值是批次內打亂 buy 標記 "
        f"{N_PERM} 次的雙尾 permutation 檢定。</p>"
        f"<div class='wrap'>{table(head, rows + trow)}</div>"
        f"<h3>扣掉特徵後的 buy 效果（20 日報酬）</h3><p>只控制批次：buy − 其他 {coef(adj['raw'])}；"
        f"再控制市值、B/M、動能百分位：{coef(adj['adj'])}，{adj['n']} 檔。"
        "同批股票同一段期間，報酬互相相關，White 標準誤會高估 t，只作參考。</p>"
        f"<h3>產業分布差最多的產業</h3>{table(['產業', 'buy', '其他', '差'], ind_rows)}")
    print("LLM 回放 buy − 其他（百分位點）：" + "、".join(f"{CHARS.get(c, '小型股比例')} {tests[c][0] * 100:+.1f}"
                                                 f"（p {tests[c][1]:.2f}）" for c in tests))
    print(f"20 日報酬 buy 係數：只控制批次 {coef(adj['raw'])}；加特徵 {coef(adj['adj'])}")

    # 4. 前向三組
    fw = forward_groups(pr, fd)
    snap = {}
    for g in fw:
        last = g["path"].iloc[-1]
        snap[f"{g['name']}（{g['path'].index[-1]:%Y-%m-%d}）"] = last.to_dict()
    msg = []
    for g in fw:
        if g["reg"] is None:
            msg.append(f"{html.escape(g['name'])}：{g['days']} 個交易日，未滿 {RET_MIN_DAYS} 日，不做報酬迴歸")
        else:
            o = g["reg"]["Carhart"]
            msg.append(f"{html.escape(g['name'])}：Carhart α/年 {_p(o['coef']['alpha'] * 252)}（t {_t(o['t']['alpha'])}）")
    sections.append("<h2>前向三組模擬持股</h2>" + (exposure_table(snap) if snap else "<p>還沒有持股</p>")
                    + "<ul>" + "".join(f"<li>{m}</li>" for m in msg) + "</ul>")
    for k, e in snap.items():
        print(f"{k}：市值百分位 {e['size'] * 100:.0f}、小型股比例 {e['small'] * 100:.0f}%、{int(e['n'])} 檔")

    sections.append("""<h2>限制</h2><ul>
<li>因子是台股自建代理因子：每月重組、全體普通股分位、最新季報 TTM，與 Fama-French 原始規格不同；未扣交易成本。</li>
<li>rf 用五大銀行一年期定存，非短期國庫券；近幾個月只有一家銀行報價。</li>
<li>策略與因子都在 2011 年後同一段資料開發，α 不是樣本外證據。</li>
<li>LLM 回放只有三批、前向三組只有一批，持股特徵是描述，不是因果。</li></ul>
<p class="muted">Fama &amp; French (1993, JFE; 2015, JFE)、Carhart (1997, JF)、Daniel, Grinblatt, Titman &amp; Wermers
(1997, JF)、Newey &amp; West (1987, Econometrica)。</p>""")

    extra = "\n.s4{stroke:#9b59b6}.s5{stroke:#c9a227}.s6{stroke:var(--muted)}"
    page = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>因子模型拆解</title>
<style>{CSS}{extra}</style></head><body><main>
<h1>因子模型拆解</h1>
<p class="muted">台股自建 Fama-French／Carhart 代理因子；資料到 {fd.index.max():%Y-%m-%d}。學術研究用。</p>
{''.join(sections)}
</main></body></html>"""
    S.REPORTS.mkdir(exist_ok=True)
    out = S.REPORTS / "factor_model.html"
    out.write_text(page, encoding="utf-8")
    pd.set_option("display.width", 220)
    print(pd.DataFrame(console).to_string(index=False))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
