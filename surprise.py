# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15", "numpy>=1.26"]
# ///
"""財報公布後追價：月營收驚喜、盈餘驚喜的事件名單（給自選組合當名單來源，也可單獨看）。

用法:
  uv run surprise.py                 # 目前持有期內的月營收驚喜股票
  uv run surprise.py --kind eps      # 盈餘驚喜
  uv run surprise.py --bad           # 壞消息（最低 10%，避開用）

規則（2018–2025 回測、2026 年 1～9 月樣本外都贏同條件的一般股票，但沒有贏 0050；規則事先寫定，不在這裡調參數）:
  - 驚喜分數：月營收 SUR＝(d_t − 前 24 個月 d 的平均) ÷ 前 24 個月 d 的標準差，d_t＝單月營收 − 去年同月（至少 12 個月）
              盈餘 SUE＝同樣算法，用單季歸屬母公司淨利、去年同季、前 8 季（至少 4 季）；分數截在 ±10
  - 只看上市普通股、公布日當天近 60 日成交金額中位數不在上市股最低 20% 的公布事件
  - 門檻：公布月份之前 12 個月的事件分 10 組（事件數不夠的月份不出名單）；最高一組＝驚喜，最低一組＝壞消息
  - 公布日 = tej_rev / tej_fin 的 avail_date（推估的公布日不算）；公布日當天或之前最後一個交易日當訊號日，
    隔一個交易日收盤進場，持有 月營收 20、盈餘 60 個交易日
  - 和原始研究的差別：原本是公布後第一個交易日「開盤」進場；這裡沿用回測引擎的收盤成交（晚半天，偏保守）
"""
import argparse
from functools import lru_cache

import numpy as np
import pandas as pd

from build_panel import OUT

KINDS = {
    "rev": {"name": "月營收驚喜", "bad": "月營收壞消息", "table": "tej_rev.parquet", "col": "以合併為主單月營收(千元)",
            "freq": "M", "lag": 12, "window": 24, "min_obs": 12, "hold": 20, "min_n": 2000},
    "eps": {"name": "盈餘驚喜", "bad": "盈餘壞消息", "table": "tej_fin.parquet", "col": "歸屬母公司淨利（損）",
            "freq": "Q", "lag": 4, "window": 8, "min_obs": 4, "hold": 60, "min_n": 1500},
}
LIQ_DAYS, LIQ_CUT, THRESH_MONTHS = 60, 0.2, 12


def scores(kind: str) -> pd.DataFrame:
    """每個公布事件的驚喜分數：code、period、pub（公布日）、x。"""
    k = KINDS[kind]
    t = pd.read_parquet(OUT / k["table"], columns=["code", "period", "avail_date", "avail_imputed", k["col"]])
    t = t[t["code"].str.match(r"^[1-9]\d{3}$") & ~t["avail_imputed"].astype(bool)]
    t = t.assign(per=pd.PeriodIndex(t["period"], freq=k["freq"])).drop_duplicates(["code", "per"], keep="last")
    out = []
    for code, g in t.groupby("code"):
        s = g.set_index("per")[k["col"]]
        s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq=k["freq"]))
        d = s - s.shift(k["lag"])
        prev = d.shift(1).rolling(k["window"], min_periods=k["min_obs"])
        x = ((d - prev.mean()) / prev.std().where(lambda v: v > 0)).clip(-10, 10)
        out.append(pd.DataFrame({"code": code, "period": g["period"].values, "pub": g["avail_date"].values,
                                 "value": g[k["col"]].values, "x": x.reindex(g["per"]).values}))
    if not out:
        return pd.DataFrame(columns=["code", "period", "pub", "value", "x"])
    return pd.concat(out, ignore_index=True).dropna(subset=["x", "pub"])


@lru_cache(maxsize=1)
def market() -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """交易日與「可買」表（上市普通股、近 60 日成交金額中位數不在當天上市股最低 20%）。"""
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "market", "value"],
                         filters=[("market", "==", "TWSE")])
    px = px[px["code"].str.match(r"^[1-9]\d{3}$")]
    v = px.pivot(index="date", columns="code", values="value").sort_index()
    present = v.notna()
    liq = v.where(v > 0).rolling(LIQ_DAYS, min_periods=20).median().where(present)
    return v.index, (liq.rank(axis=1, pct=True) > LIQ_CUT) & present


@lru_cache(maxsize=2)
def events(kind: str) -> pd.DataFrame:
    """可買的公布事件加上分組：dec（1～10）、sig_i（訊號日在交易日序列的位置）、entry（進場日，還沒到的是 NaT）。"""
    k = KINDS[kind]
    days, ok = market()
    ev = scores(kind)
    ev = ev[ev["pub"] >= days[0] + pd.Timedelta(days=30)].copy()
    ev["sig_i"] = days.searchsorted(ev["pub"].values, side="right") - 1      # 公布日當天或之前最後一個交易日
    col = ok.columns.get_indexer(ev["code"])
    ev = ev[(col >= 0) & (ev["sig_i"] >= 0)]
    col = ok.columns.get_indexer(ev["code"])
    ev = ev[ok.values[ev["sig_i"].values, col]].copy()
    ev["m"] = ev["pub"].dt.to_period("M")
    bps = {}
    for m in ev["m"].unique():
        past = ev["x"][(ev["m"] < m) & (ev["m"] >= m - THRESH_MONTHS)]
        bps[m] = np.quantile(past, np.linspace(0.1, 0.9, 9)) if len(past) >= k["min_n"] else None
    ev = ev[ev["m"].map(lambda m: bps[m] is not None)].copy()
    ev["dec"] = [int(np.searchsorted(bps[m], x, side="right")) + 1 for m, x in zip(ev["m"], ev["x"])]
    e = ev["sig_i"] + 1
    ev["entry"] = pd.Series(days[np.minimum(e, len(days) - 1)], index=ev.index).where(e < len(days))
    return ev.drop(columns="m").sort_values(["pub", "x"], ascending=[True, False]).reset_index(drop=True)


def active(kind: str, sig: pd.Timestamp, bad: bool = False) -> pd.DataFrame:
    """訊號日 sig 時還在持有期內的事件（驚喜：最高一組；bad：最低一組），分數高到低。"""
    days, _ = market()
    j = days.searchsorted(pd.Timestamp(sig), side="right") - 1
    ev = events(kind)
    ev = ev[(ev["dec"] == (1 if bad else 10)) & (ev["sig_i"] <= j) & (ev["sig_i"] > j - KINDS[kind]["hold"])]
    ev = ev.sort_values("x", ascending=bad).drop_duplicates("code")
    return ev.assign(day=j - ev["sig_i"] + 1)     # day：持有期的第幾天（訊號日算第 1 天）


def active_codes(kind: str, sig: pd.Timestamp) -> list[str]:
    return active(kind, sig)["code"].tolist()


def schedule(kind: str, start: pd.Timestamp, end: pd.Timestamp | None = None) -> dict[pd.Timestamp, list[str]]:
    """start～end 每個交易日的持有名單，只留名單有變動的日子（回測用）。"""
    days, _ = market()
    ev = events(kind)
    ev = ev[ev["dec"] == 10]
    h = KINDS[kind]["hold"]
    out, prev = {}, None
    for j in np.flatnonzero((days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end or days[-1]))):
        cur = ev[(ev["sig_i"] <= j) & (ev["sig_i"] > j - h)].sort_values("x", ascending=False)["code"].drop_duplicates().tolist()
        if cur != prev and (cur or prev):
            out[days[j]] = cur
        prev = cur
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=list(KINDS), default="rev")
    ap.add_argument("--bad", action="store_true")
    a = ap.parse_args()
    k = KINDS[a.kind]
    days, _ = market()
    act = active(a.kind, days[-1], a.bad)
    names = pd.read_parquet(OUT / "prices_daily.parquet", columns=["code", "name"], filters=[("date", "==", days[-1])])
    act = act.merge(names, on="code", how="left")
    print(f"{k['bad'] if a.bad else k['name']}（資料日 {days[-1]:%Y-%m-%d}，持有 {k['hold']} 個交易日）：{len(act)} 檔")
    for r in act.itertuples():
        print(f"  {r.code} {r.name or '':<6} 期別 {pd.Timestamp(r.period):%Y-%m}  公布 {r.pub:%Y-%m-%d}  "
              f"分數 {r.x:+.2f}  第 {r.day}/{k['hold']} 天")


if __name__ == "__main__":
    main()
