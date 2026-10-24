# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""驗證 prices_daily.parquet：原始價對 FinMind、事件覆蓋率、可疑報酬、對 TEJ 季底市值。

用法: uv run check_prices.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
FINMIND = ROOT / "data" / "finmind" / "price"   # 選用：FinMind 原始收盤價，沒有就跳過第 1 項

px = pd.read_parquet(ROOT / "data" / "prices_daily.parquet")
ev = pd.read_parquet(ROOT / "data" / "price_events.parquet")
print(f"rows {len(px):,}  codes {px.code.nunique()}  {px.date.min().date()} ~ {px.date.max().date()}")
print("trading days per year:", px.groupby(px.date.dt.year).date.nunique().to_dict())

# 1. 原始收盤對 FinMind（未還原）
if FINMIND.exists():
    fm = []
    for p in sorted(FINMIND.glob("*.json")):
        d = json.loads(p.read_text())
        d = d.get("data", d) if isinstance(d, dict) else d
        if d:
            fm.append(pd.DataFrame(d)[["date", "stock_id", "close"]])
    fm = pd.concat(fm).rename(columns={"stock_id": "code", "close": "fm_close"})
    fm["date"] = pd.to_datetime(fm["date"])
    m = fm[fm.fm_close > 0].merge(px[["date", "code", "close"]], on=["date", "code"], how="left")
    print(f"\n[1] vs FinMind: {len(m):,} rows, missing in ours {m.close.isna().mean():.3%}, "
          f"close mismatch {(m.close.notna() & (m.close - m.fm_close).abs().gt(1e-6)).mean():.3%}")
else:
    print("\n[1] 沒有 data/finmind/price，略過")

# 2. 事件覆蓋：證交所「X 不比價」日 / 櫃買「次日參考價 != 收盤」
evk = set(zip(ev.code, ev.date))
tw = px[(px.market == "TWSE") & px.no_compare.fillna(False) & px.close.notna()]
hit = np.array([(c, d) in evk for c, d in zip(tw.code, tw.date)])
print(f"\n[2] TWSE 不比價日 {len(tw):,}，有對應事件 {hit.mean():.1%}")
tp = px[px.market == "TPEX"].sort_values(["code", "date"]).copy()
tp["next_date"] = tp.groupby("code").date.shift(-1)
sig = tp[tp.close.notna() & tp.ref_next.notna() & (tp.ref_next - tp.close).abs().gt(1e-6) & tp.next_date.notna()]
hit = np.array([(c, d) in evk for c, d in zip(sig.code, sig.next_date)])
print(f"    TPEX 次日參考價≠收盤 {len(sig):,}，隔日有對應事件 {hit.mean():.1%}")
print("    未對應範例:", sig.loc[~hit, ["code", "name", "date", "close", "ref_next"]].head(5).to_dict("records"))

# 3. 可疑報酬
s = px[px.ret_suspect]
print(f"\n[3] ret_suspect {len(s):,} rows / {s.code.nunique()} codes; by year:",
      s.groupby(s.date.dt.year).size().to_dict())
print(s.reindex(s.ret.abs().sort_values(ascending=False).index)[["date", "code", "name", "close", "ret"]]
      .head(10).to_string(index=False))

# 4. 對 TEJ 季底普通股市值：收盤 × 普通股股本/10
fin = pd.read_parquet(ROOT / "data" / "tej_fin.parquet", columns=["code", "period", "季底普通股市值", "普通股股本"])
q = fin.dropna().query("普通股股本 > 0")
last = px.dropna(subset=["close"]).sort_values("date")
q = pd.merge_asof(q.sort_values("period"), last[["date", "code", "close"]].rename(columns={"date": "period"}),
                  on="period", by="code", direction="backward", tolerance=pd.Timedelta(days=7)).dropna()
ratio = q["close"] * q["普通股股本"] / 10 / q["季底普通股市值"]
print(f"\n[4] vs TEJ 季底市值: {len(q):,} 筆，比值中位數 {ratio.median():.4f}，"
      f"|比值-1|<1% 的比例 {(ratio.sub(1).abs() < 0.01).mean():.1%}")

# 5. 抽查：2330 年度含息報酬、2603 減資恢復日
t = px[px.code == "2330"].set_index("date")
yr = (1 + t.ret).groupby(t.index.year).prod() - 1
print("\n[5] 2330 yearly total return:", yr.round(3).to_dict())
print(px[(px.code == "2603") & px.date.between("2022-09-08", "2022-09-21")]
      [["date", "close", "ev_factor", "ret", "ret_suspect"]].to_string(index=False))
