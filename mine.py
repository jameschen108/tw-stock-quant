# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""自選股即時模擬：自己挑股票，從最新資料日起往後記錄並追蹤報酬（只記當下，不補過去；要看過去用 custom.py）。

用法:
  uv run mine.py new 科技 2330:50,2317:30,2454:20 [--monthly] [--capital 1000000]   # 建立組合，今天起算
  uv run mine.py edit 科技 2330:40,2317:30,3711:30                                 # 改持股 / 權重，從最新資料日起生效
  uv run mine.py archive 科技                                                      # 封存（紀錄保留，不再更新）
  uv run mine.py                                                                   # 月調組合記新月份，重算並印出各組合摘要（update.py 會跑）

規則（同 paper.py、strategy.simulate）:
  - 訊號日 = 建立 / 修改當下的最新資料日；隔一個交易日收盤成交，成本、零股取整、漲跌停規則同回測
  - 預設只在你改動時才調整，期間不再平衡；--monthly 的組合在資料日進入新月份後，第一次執行時調回目標權重
  - 最新資料日距今超過 7 天不接受新紀錄（先更新資料）
  - 紀錄 data/paper/mine_ledger.parquet 只追加：portfolio、signal_date、code、weight、recorded_at；
    同一組合同一訊號日有多次記錄時取最後一次。組合設定（本金、調整方式、封存）在 mine_meta.json
  - 淨值每次由已記錄的權重重播，不存檔；基準是同組股票等權買進持有（不扣成本）與 0050
"""
import argparse
import json
from datetime import datetime

import pandas as pd

import strategy as S
from build_panel import OUT
from custom import load_rets, parse_codes

PAPER = OUT / "paper"
LEDGER = PAPER / "mine_ledger.parquet"
META = PAPER / "mine_meta.json"
MAX_STALE_DAYS = 7
COLS = ["portfolio", "signal_date", "code", "weight", "recorded_at"]


def load_ledger() -> pd.DataFrame:
    return pd.read_parquet(LEDGER) if LEDGER.exists() else pd.DataFrame(columns=COLS)


def load_meta() -> dict:
    return json.loads(META.read_text()) if META.exists() else {}


def _latest_signals(led: pd.DataFrame, name: str) -> dict[pd.Timestamp, pd.Series]:
    """{訊號日: 權重 Series}，同一訊號日取最後一次記錄。"""
    g = led[led["portfolio"] == name]
    out = {}
    for s, d in g.groupby("signal_date"):
        d = d[d["recorded_at"] == d["recorded_at"].max()]
        out[pd.Timestamp(s)] = d.set_index("code")["weight"]
    return dict(sorted(out.items()))


def data_date() -> pd.Timestamp:
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date"], filters=[("date", ">=", pd.Timestamp.today() - pd.Timedelta(days=30))])
    return px["date"].max()


def _append(name: str, w: pd.Series, sig: pd.Timestamp) -> None:
    PAPER.mkdir(exist_ok=True)
    new = pd.DataFrame({"portfolio": name, "signal_date": sig, "code": w.index, "weight": w.values,
                        "recorded_at": pd.Timestamp(datetime.now())})
    led = load_ledger()
    pd.concat([led, new], ignore_index=True).to_parquet(LEDGER, index=False)


def _fresh_signal_date() -> pd.Timestamp:
    last = data_date()
    if (pd.Timestamp.today().normalize() - last).days > MAX_STALE_DAYS:
        raise ValueError(f"最新資料日 {last:%Y-%m-%d} 距今超過 {MAX_STALE_DAYS} 天，先更新資料")
    return last


def _check_codes(w: pd.Series, sig: pd.Timestamp) -> None:
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code"], filters=[("date", "==", sig)])
    bad = [c for c in w.index if c not in set(px["code"])]
    if bad:
        raise ValueError("最新資料日沒有行情：" + "、".join(bad))


def create(name: str, w: pd.Series, monthly: bool = False, capital: float = 1_000_000,
           signal_date: pd.Timestamp | None = None) -> pd.Timestamp:
    """signal_date：另外指定訊號日（例如決策當天，晚於最新資料日），成交日 = 它之後的第一個交易日。"""
    meta = load_meta()
    if not name.strip() or name in meta:
        raise ValueError("組合名稱不能空白，也不能和既有的重複（含已封存的）")
    sig = _fresh_signal_date()
    _check_codes(w, sig)
    sig = signal_date if signal_date is not None else sig
    _append(name, w, sig)
    meta[name] = {"capital": float(capital), "rebalance": "monthly" if monthly else "none",
                  "created": str(sig.date()), "archived": False}
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return sig


def edit(name: str, w: pd.Series, signal_date: pd.Timestamp | None = None) -> pd.Timestamp:
    m = load_meta().get(name)
    if not m or m["archived"]:
        raise ValueError("找不到這個組合，或已封存")
    sig = _fresh_signal_date()
    _check_codes(w, sig)
    sig = signal_date if signal_date is not None else sig
    _append(name, w, sig)
    return sig


def archive(name: str) -> None:
    meta = load_meta()
    if name not in meta:
        raise ValueError("找不到這個組合")
    meta[name]["archived"] = True
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=1))


def record_monthly() -> list[str]:
    """月調組合：資料日進入新月份就把最近一次的目標權重再記一期。"""
    led, msgs = load_ledger(), []
    for name, m in load_meta().items():
        if m["archived"] or m["rebalance"] != "monthly":
            continue
        sigs = _latest_signals(led, name)
        last = max(sigs)
        try:
            sig = _fresh_signal_date()
        except ValueError as e:
            msgs.append(f"[{name}] 不記：{e}")
            continue
        if sig.to_period("M") <= last.to_period("M"):
            continue
        _append(name, sigs[last], sig)
        msgs.append(f"[{name}] 記新一期（訊號日 {sig:%Y-%m-%d}）")
    return msgs


def replay(name: str, rets: pd.DataFrame | None = None) -> dict | None:
    """由已記錄的權重重播。還沒有成交日（等下一個交易日收盤進場）回傳 None。"""
    meta = load_meta()[name]
    sigs = _latest_signals(load_ledger(), name)
    if rets is None:
        rets = load_rets(min(sigs))
    fixed = {s: {"tw": w, "tw_full": w, "rinfo": {}, "pool": w.index.tolist(), "picks": w.index.tolist()}
             for s, w in sigs.items()}
    cfg = {**json.loads(json.dumps(S.DEFAULTS)), "id": name, "name": name, "description": "",
           "start": str(min(sigs).date()), "capital": meta["capital"]}
    res = S.simulate(cfg, pd.DataFrame(), rets, fixed=fixed)
    if res is None:
        return None
    res["perf"] = {k: S.perf(res["daily"][k]) for k in ("strategy", "pool", "etf")}
    res["signals"] = sigs
    res["meta"] = meta
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    n = sub.add_parser("new")
    n.add_argument("name")
    n.add_argument("codes")
    n.add_argument("--monthly", action="store_true")
    n.add_argument("--capital", type=float, default=1_000_000)
    e = sub.add_parser("edit")
    e.add_argument("name")
    e.add_argument("codes")
    sub.add_parser("archive").add_argument("name")
    a = ap.parse_args()
    try:
        if a.cmd == "new":
            print(f"已建立「{a.name}」，訊號日 {create(a.name, parse_codes(a.codes), a.monthly, a.capital):%Y-%m-%d}，隔一個交易日收盤進場")
        elif a.cmd == "edit":
            print(f"已記錄「{a.name}」的新持股，訊號日 {edit(a.name, parse_codes(a.codes)):%Y-%m-%d}")
        elif a.cmd == "archive":
            archive(a.name)
            print(f"已封存「{a.name}」")
    except ValueError as err:
        raise SystemExit(f"失敗：{err}")
    if a.cmd:
        return
    for m in record_monthly():
        print(m)
    meta = {k: v for k, v in load_meta().items() if not v["archived"]}
    if not meta:
        print("還沒有自選組合（uv run mine.py new 名稱 代號,代號）")
        return
    rets = load_rets(pd.Timestamp(min(v["created"] for v in meta.values())))
    for name, m in meta.items():
        res = replay(name, rets)
        if res is None:
            print(f"[{name}] 自 {m['created']} 起，等下一個交易日收盤進場")
            continue
        p, b = res["perf"]["strategy"], res["daily"]
        tot = lambda k: (1 + b[k]).prod() - 1
        print(f"[{name}] 自 {m['created']} 起 {len(res['signals'])} 次訊號・{'每月調回' if m['rebalance'] == 'monthly' else '只在改動時調'}；"
              f"累積 {tot('strategy'):+.2%}（同組等權 {tot('pool'):+.2%}、{S.ETF} {tot('etf'):+.2%}）"
              + (f"、最大回落 {p['mdd']:.2%}" if p else "") + f"、資產 {res['nav'].iloc[-1]:,.0f}")


if __name__ == "__main__":
    main()
