# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""自選股模擬：自己挑股票和起訖日，用和回測相同的成本、股數取整、漲跌停規則重播，對比同組等權與 0050。

用法:
  uv run custom.py 2330,2317,2454 --start 2023-01-03 --end 2024-12-31
  uv run custom.py 2330:50,2317:30,2454:20 --start 2023-01-03           # 冒號後是權重（自動正規化），end 預設到最新
  uv run custom.py 2330,2317 --start 2023-01-03 --rebalance monthly      # 每月最後一個交易日調回目標權重
  uv run custom.py 2330,2317 --start 2023-01-03 --capital 1000000

規則:
  - 進場：start 當天（不是交易日就順延）當訊號日，隔一個交易日收盤成交（同回測 lag 1）
  - --rebalance none（預設）：買進持有，期間不再平衡；monthly：每月最後一個交易日調回目標權重，交易隔一個交易日成交
  - 成本、零股取整、漲停買不到 / 跌停賣不掉 / 沒行情不交易：同 strategy.simulate，費率取 strategy.DEFAULTS["cost"]
  - 基準：同一組股票依目標權重買進持有、不扣成本（「自選」）與 0050
  - 沒有行情的代號會列出來並略過，剩下的權重重新正規化
"""
import argparse
from copy import deepcopy

import pandas as pd

import strategy as S
from build_panel import OUT
from strategy import clean_returns


def parse_codes(spec: str) -> pd.Series:
    """'2330,2317:30' → 權重 Series（正規化到總和 1）；沒寫權重的當 1。"""
    w = {}
    for tok in spec.replace(" ", "").split(","):
        if not tok:
            continue
        code, _, wt = tok.partition(":")
        w[code] = float(wt) if wt else 1.0
    s = pd.Series(w, dtype=float)
    if s.empty or (s <= 0).any():
        raise ValueError("股票代號不能是空的，權重要大於 0")
    return s / s.sum()


def load_rets(start: pd.Timestamp) -> pd.DataFrame:
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "ret", "ret_suspect"],
                         filters=[("date", ">=", start - pd.Timedelta(days=10))])
    return clean_returns(px)


def names_of(codes: list[str]) -> dict[str, str]:
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "name"],
                         filters=[("code", "in", codes)])
    return px.sort_values("date").groupby("code")["name"].last().to_dict()


def run(weights: pd.Series, start, end=None, rebalance: str = "none", capital: float = 100_000) -> dict:
    """回傳 simulate 的結果加上 missing（略過的代號）、weights（實際用的權重）、perf。無法模擬時 raise ValueError。"""
    start = pd.Timestamp(start)
    rets = load_rets(start)
    if end is not None:
        rets = rets[rets.index <= pd.Timestamp(end)]
    days = rets.index[rets.index >= start]
    if len(days) < 3:
        raise ValueError("起訖日之間的交易日太少")
    ok = [c for c in weights.index if c in rets.columns and rets.loc[days, c].notna().any()]
    missing = [c for c in weights.index if c not in ok]
    if not ok:
        raise ValueError("這些代號在期間內都沒有行情：" + "、".join(missing))
    w = weights[ok] / weights[ok].sum()

    sigs = [days[0]]
    if rebalance == "monthly":
        me = pd.Series(days, index=days).groupby(days.to_period("M")).max()
        sigs += [d for d in me if d > days[0]]
    fixed = {s: {"tw": w, "tw_full": w, "rinfo": {}, "pool": ok, "picks": ok} for s in sigs}

    cfg = deepcopy(S.DEFAULTS)
    cfg.update(id="custom", name="自選股", description="", start=str(days[0].date()), capital=capital)
    res = S.simulate(cfg, pd.DataFrame(), rets, fixed=fixed)
    if res is None:
        raise ValueError("期間內沒有成交日")
    # 基準「自選」要照目標權重，不是 simulate 內建的等權；0050 沿用
    r = rets.loc[res["nav"].index].reindex(columns=ok).fillna(0)
    bh = (1 + r).cumprod().mul(w, axis=1).sum(axis=1) if rebalance == "none" else (1 + r.mul(w, axis=1).sum(axis=1)).cumprod()
    res["daily"]["pool"] = bh.pct_change().fillna(0)
    res["perf"] = {k: S.perf(res["daily"][k]) for k in ("strategy", "pool", "etf")}
    res["missing"], res["weights"], res["rebalance"] = missing, w, rebalance
    return res


def _fmt(p: dict) -> str:
    if not p:
        return "期間太短（不到 20 個交易日），不算年化指標"
    return (f"總報酬 {p['total']:+.1%}  年化 {p['cagr']:+.1%}  波動 {p['vol']:.1%}  "
            f"Sharpe {p['sharpe']:.2f}  最大回落 {p['mdd']:.1%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("codes", help="代號，逗號分隔，可加 :權重，例 2330:50,2317:30")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end")
    ap.add_argument("--rebalance", choices=["none", "monthly"], default="none")
    ap.add_argument("--capital", type=float, default=100_000)
    a = ap.parse_args()
    try:
        res = run(parse_codes(a.codes), a.start, a.end, a.rebalance, a.capital)
    except ValueError as e:
        raise SystemExit(f"無法模擬：{e}")
    nv, w = res["nav"], res["weights"]
    nm = names_of(list(w.index))
    print(f"期間 {nv.index[0]:%Y-%m-%d} → {nv.index[-1]:%Y-%m-%d}（{len(nv)} 個交易日）  "
          f"{'買進持有' if a.rebalance == 'none' else '每月調回目標權重'}  本金 {a.capital:,.0f}")
    if res["missing"]:
        print("略過（期間內沒有行情）：" + "、".join(res["missing"]))
    print("持股：" + "  ".join(f"{c}{nm.get(c, '')} {x:.1%}" for c, x in w.items()))
    print(f"\n自選（扣成本）  {_fmt(res['perf']['strategy'])}")
    print(f"自選（不扣成本）{_fmt(res['perf']['pool'])}")
    print(f"0050            {_fmt(res['perf']['etf'])}")
    t = res["trades"]
    print(f"\n期末資產 {nv.iloc[-1]:,.0f}  手續費 {t['fees'].sum():,.0f}  證交稅 {t['tax'].sum():,.0f}  現金 {res['cash']:,.0f}")
    for _, row in t.iterrows():
        if row["blocked"]:
            print(f"  {row['exe']:%Y-%m-%d} 未成交：{'、'.join(row['blocked'])}")


if __name__ == "__main__":
    main()
