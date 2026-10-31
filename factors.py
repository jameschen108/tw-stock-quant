# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""建立月頻因子表：每個換股日（每月最後一個交易日）× 股票，一列含因子值、市值、產業、下期報酬。

用法: uv run factors.py        # 輸出 data/factors_monthly.parquet

時點規則（避免偷看未來）:
  - 換股日 t 只用 avail_date < t 的營收 / 財報（公布當天盤後的資料要到隔天才能用）
  - fwd_ret 是 t+lag 收盤到下一個換股日 +lag 收盤的含息報酬（lag 預設 1：訊號日隔天收盤才進場）
  - 視窗內有 ret_suspect 的股票，fwd_ret 設為缺值
"""
import numpy as np
import pandas as pd

from build_panel import MAX_STALE_DAYS, OUT, latest_known

LAG = 1

# 因子名稱 -> 說明；方向一律「值越大預期越好」以外的，回測時看 IC 正負號即可
FACTORS = {
    # 月營收
    "rev_yoy":       "單月營收年增率（以合併為主）",
    "rev_yoy_3m":    "近 3 月累計營收年增率",
    "rev_yoy_12m":   "近 12 月累計營收年增率",
    "rev_accel":     "近 3 月年增率 − 近 12 月年增率",
    "rev_vs_high":   "單月營收與歷史最高比 %",
    # 季財報
    "roe":           "ROE（單季稅後）",
    "gross_margin":  "營業毛利率",
    "gm_chg":        "毛利率較去年同季變化",
    "op_margin":     "營業利益率",
    "ep":            "近四季歸母淨利 / 市值",
    "bp":            "母公司股東權益 / 市值",
    "sp":            "近四季營收 / 市值",
    "cfo_ta":        "近四季營業現金流 / 總資產",
    "accruals":      "（近四季淨利 − 營業現金流）/ 總資產",
    "asset_growth":  "總資產年增率",
    "debt_ratio":    "負債比率",
    "ni_yoy":        "單季稅後淨利年增率",
    # 價格（對照組）
    "size":          "log 市值",
    "mom_12_1":      "過去 12 個月扣掉最近 1 個月的報酬",
    "rev_1m":        "最近 1 個月報酬（短期反轉）",
}

# TEJ 月營收表的「流通在外股數」會用之後的股數回填過去（例：2603 在 2022-09 減資，
# 2022-01 起就已是減資後股數），不能拿來算市值；股數改用 point_in_time_shares()
REV_COLS = ["以合併為主單月營收成長率％", "近3月累計營收成長率", "近12月累計營收成長率",
            "與歷史最高單月營收比%"]


def load_prices():
    px = pd.read_parquet(OUT / "prices_daily.parquet",
                         columns=["date", "code", "close", "value", "ret", "ret_suspect", "shares"])
    px = px[px["code"].str.fullmatch(r"[1-9]\d{3}")]               # 只留普通股（00 開頭是 ETF，例如 0050）
    wide = {c: px.pivot(index="date", columns="code", values=c)
            for c in ["close", "value", "ret", "ret_suspect", "shares"]}
    return wide


def point_in_time_shares(df: pd.DataFrame, close: pd.DataFrame) -> pd.Series:
    """換股日當時的普通股股數（千股）。

    上櫃：櫃買每日行情的發行股數。上市沒有每日股數：用最新已公布財報的季底普通股市值 ÷ 季底收盤
    得出季底股數（面額不是 10 元也對），再乘上季底之後到換股日之間減資、變更面額的股數變化
    （前收盤 / 參考價；減資退還現金時是近似值）。季底之後的盈餘轉增資要等下一季財報才會反映。
    """
    pe = df[["code", "fin_period"]].dropna().drop_duplicates()
    cl = close.ffill().stack().rename("close_pe").reset_index()  # date, code, close_pe
    pe = pd.merge_asof(pe.sort_values("fin_period"), cl.sort_values("date"), left_on="fin_period",
                       right_on="date", by="code", direction="backward").drop(columns="date")
    out = df.merge(pe, on=["code", "fin_period"], how="left")
    stmt = (out["季底普通股市值"] / out["close_pe"]).where(out["close_pe"] > 0).fillna(out["普通股股本"] / 10)

    ev = pd.read_parquet(OUT / "price_events.parquet", columns=["date", "code", "kind", "factor"])
    ev = ev[ev["kind"].isin(["reduction", "par_change"])].sort_values(["code", "date"])
    ev["cum"] = (-np.log(ev["factor"])).groupby(ev["code"]).cumsum()
    ev = ev.groupby(["code", "date"])["cum"].last().reset_index().sort_values("date")

    def cum_at(col):
        x = out[["code", col]].reset_index().dropna(subset=[col]).sort_values(col)
        x = pd.merge_asof(x, ev.rename(columns={"date": "ev_date"}), left_on=col, right_on="ev_date",
                          by="code", direction="backward")
        return x.set_index("index")["cum"].reindex(out.index).fillna(0)

    stmt = stmt * np.exp(cum_at("date") - cum_at("fin_period"))
    return (out["shares_ex"] / 1000).where(out["shares_ex"] > 0).fillna(stmt).set_axis(df.index)


def rebalance_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(dates, index=dates)
    return pd.DatetimeIndex(s.groupby(dates.to_period("M")).max().values)


def price_features(w: dict, lag: int = LAG, reb: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """reb = None 用每月最後一個交易日；也可以傳任意交易日（研究用的快照，例如 LLM 批次的資料時點）。"""
    dates = w["close"].index
    reb = rebalance_dates(dates) if reb is None else pd.DatetimeIndex(reb)
    pos = dates.get_indexer(reb)
    logtr = np.log1p(w["ret"].fillna(0)).cumsum()            # 停牌、下市後持平
    bad = w["ret_suspect"].fillna(False).astype(int).cumsum()
    avg_val = w["value"].fillna(0).rolling(20, min_periods=1).mean()
    n_obs = w["close"].notna().cumsum()

    def at(df, p):
        p = np.clip(p, 0, len(dates) - 1)
        return df.iloc[p].set_axis(reb[: len(p)])

    out = {
        "close": at(w["close"], pos),
        "avg_value": at(avg_val, pos),
        "n_obs": at(n_obs, pos),
        "shares_ex": at(w["shares"], pos),                     # 只有上櫃有

        # 動能：t−1 月底對 t−12 月底；反轉：t−1 月底對 t
        "mom_12_1": np.exp(at(logtr, np.r_[[0], pos[:-1]]).values - at(logtr, np.r_[[0] * 12, pos][:len(pos)]).values) - 1,
        "rev_1m": np.exp(at(logtr, pos).values - at(logtr, np.r_[[0], pos[:-1]]).values) - 1,
    }
    out["mom_12_1"] = pd.DataFrame(out["mom_12_1"], index=reb, columns=w["close"].columns)
    out["rev_1m"] = pd.DataFrame(out["rev_1m"], index=reb, columns=w["close"].columns)
    out["mom_12_1"].iloc[:12] = np.nan
    out["rev_1m"].iloc[:1] = np.nan

    # 下期報酬：(pos_k + lag, pos_{k+1} + lag]
    start, end = pos + lag, np.r_[pos[1:], [len(dates) - 1 - lag]] + lag
    valid = end > start
    fwd = np.exp(at(logtr, end).values - at(logtr, start).values) - 1
    sus = (at(bad, end).values - at(bad, start).values) > 0
    fwd[sus | ~valid[:, None]] = np.nan
    out["fwd_ret"] = pd.DataFrame(fwd, index=reb, columns=w["close"].columns)
    out["fwd_ret"].iloc[-1] = np.nan                           # 最後一期還沒走完

    df = pd.concat({k: v.stack(future_stack=True) for k, v in out.items()}, axis=1)
    df.index.names = ["date", "code"]
    return df.reset_index()


def fin_features() -> pd.DataFrame:
    f = pd.read_parquet(OUT / "tej_fin.parquet")
    f = f.sort_values(["code", "period"]).reset_index(drop=True)
    g = f.groupby("code", sort=False)
    months = f["period"].dt.year * 12 + f["period"].dt.month
    q = f["單季(Q)/單半年(H)"].eq("Q")
    ttm_ok = q & (months - g["period"].shift(3).pipe(lambda s: s.dt.year * 12 + s.dt.month) == 9) \
        & g["單季(Q)/單半年(H)"].transform(lambda s: s.eq("Q").rolling(4).sum() == 4).fillna(False)
    yoy_ok = (months - g["period"].shift(4).pipe(lambda s: s.dt.year * 12 + s.dt.month) == 12)

    def ttm(col):
        return g[col].transform(lambda s: s.rolling(4).sum()).where(ttm_ok)

    out = f[["code", "period", "avail_date", "TEJ主產業代碼", "普通股股本", "季底普通股市值", "資產總額"]].copy()
    out["ni_ttm"] = ttm("歸屬母公司淨利（損）")
    out["sales_ttm"] = ttm("營業收入淨額")
    out["cfo_ttm"] = ttm("來自營運之現金流量")
    out["equity"] = f["母公司股東權益合計"]
    out["roe"] = f["ROE(A)－稅後"]
    out["gross_margin"] = f["營業毛利率"]
    out["gm_chg"] = (f["營業毛利率"] - g["營業毛利率"].shift(4)).where(yoy_ok)
    out["op_margin"] = f["營業利益率"]
    out["cfo_ta"] = out["cfo_ttm"] / f["資產總額"]
    out["accruals"] = (out["ni_ttm"] - out["cfo_ttm"]) / f["資產總額"]
    out["asset_growth"] = (f["資產總額"] / g["資產總額"].shift(4) - 1).where(yoy_ok)
    out["debt_ratio"] = f["負債比率"]
    ni, ni_ly = f["歸屬母公司淨利（損）"], g["歸屬母公司淨利（損）"].shift(4)
    out["ni_yoy"] = ((ni - ni_ly) / ni_ly.abs()).where(yoy_ok & (ni_ly.abs() > 0))
    return out


def attach(base: pd.DataFrame, tbl: pd.DataFrame, kind: str, cols: list[str]) -> pd.DataFrame:
    """把 avail_date < 換股日 的最新一期接到 base。"""
    reb = pd.DatetimeIndex(sorted(base["date"].unique()))
    key = f"{kind}_period"
    lk = latest_known(tbl, reb - pd.Timedelta(days=1), key)
    lk["date"] = lk["date"] + pd.Timedelta(days=1)
    lk.loc[lk["date"] - lk[key] > pd.Timedelta(days=MAX_STALE_DAYS[kind]), key] = pd.NaT
    lk = lk.merge(tbl[["code", "period", *cols]].rename(columns={"period": key}), on=["code", key], how="left")
    return base.merge(lk, on=["code", "date"], how="left")


def build(reb: pd.DatetimeIndex | None = None, same_day: bool = False) -> pd.DataFrame:
    """reb 見 price_features。same_day=True 時連換股日當天公布的營收 / 財報也算已知（給盤後才做決策的快照用，
    例如 LLM 批次在資料時點收盤後決策）；預設 False，同 factors_monthly.parquet。"""
    print("prices ..."); w = load_prices(); df = price_features(w, reb=reb)
    df = df[df["close"].notna()]                               # 換股日當天有成交
    shift = pd.Timedelta(days=1 if same_day else 0)
    df["date"] += shift
    print("revenue ...")
    rev = pd.read_parquet(OUT / "tej_rev.parquet", columns=["code", "period", "avail_date", *REV_COLS])
    df = attach(df, rev, "rev", REV_COLS)
    print("financials ...")
    fin = fin_features()
    fin_cols = [c for c in fin.columns if c not in ("code", "period", "avail_date")]
    df = attach(df, fin, "fin", fin_cols)
    df["date"] -= shift

    df["mcap"] = df["close"] * point_in_time_shares(df, w["close"])  # 千元
    df["size"] = np.log(df["mcap"])
    df["rev_yoy"] = df["以合併為主單月營收成長率％"]
    df["rev_yoy_3m"] = df["近3月累計營收成長率"]
    df["rev_yoy_12m"] = df["近12月累計營收成長率"]
    df["rev_accel"] = df["rev_yoy_3m"] - df["rev_yoy_12m"]
    df["rev_vs_high"] = df["與歷史最高單月營收比%"]
    df["ep"] = df["ni_ttm"] / df["mcap"]
    df["bp"] = df["equity"] / df["mcap"]
    df["sp"] = df["sales_ttm"] / df["mcap"]
    # 產業用該股最後已知的分類（分類幾乎不變，視為靜態）
    df["industry"] = df.groupby("code")["TEJ主產業代碼"].transform(lambda s: s.ffill().bfill())
    keep = ["date", "code", "close", "avg_value", "n_obs", "mcap", "industry", "rev_period", "fin_period",
            *FACTORS, "fwd_ret"]
    df = df[keep].replace([np.inf, -np.inf], np.nan)
    return df.sort_values(["date", "code"]).reset_index(drop=True)


def main():
    df = build()
    df.to_parquet(OUT / "factors_monthly.parquet", index=False)
    print(f"{len(df):,} rows, {df['date'].nunique()} dates, {df['code'].nunique()} codes")
    print("coverage:", df[list(FACTORS)].notna().mean().round(2).to_dict())


if __name__ == "__main__":
    main()
