# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""自選股即時模擬：自己挑股票，從最新資料日起往後記錄並追蹤報酬（只記當下，不補過去；要看過去用 custom.py）。

用法:
  uv run mine.py new 科技 2330:50,2317:30,2454:20 [--monthly] [--capital 1000000]   # 建立組合，今天起算
  uv run mine.py new 低波 2330,2317,2454 --weighting inv_vol --max-weight 0.4 --rule stop_loss=0.1 --rule regime=ma
  uv run mine.py new 前30 - --source strategies/top30.toml --monthly                # 名單從策略帶入，每月照規則重選
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
  - 選用的規則（建立時決定，之後不改；要換規則就建新組合，免得回頭改到過去的淨值）:
      名單來源 source：觀察名單（watchlist）、模型選股（model_picks，短中長期都看漲）、strategies/*.toml 的選股條件
        （只用它的選股，倉位與風險規則用這個組合自己的）。搭配 --monthly 時每月照同一規則重選
      加權 weighting：manual（照輸入的權重）、equal、inv_vol（過去 60 日波動倒數）、score（綜合分數高的多買）；
        max_weight 單檔上限，分不完的留現金。manual 以外在每次訊號日用當時的資料重算
      風險規則 risk：strategy.DEFAULTS["risk"] 的欄位（停損、停利、持有天數、大盤濾網、資產回落、目標波動），
        重播時帶進 strategy.simulate；大盤濾網（每月判斷）與目標波動在每個訊號日用當時以前的資料決定曝險
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import strategy as S
from build_panel import OUT
from custom import load_rets, parse_codes

PAPER = OUT / "paper"
LEDGER = PAPER / "mine_ledger.parquet"
META = PAPER / "mine_meta.json"
MAX_STALE_DAYS = 7
COLS = ["portfolio", "signal_date", "code", "weight", "recorded_at"]
ROOT = Path(__file__).parent
WEIGHTINGS = {"manual": "照輸入的權重", "equal": "等權", "inv_vol": "波動倒數（波動小的多買）", "score": "綜合分數高的多買"}
SOURCES = {"watchlist": "觀察名單", "model_picks": "模型選股（短中長期都看漲）"}   # 另外可用 strategies/*.toml
RISK_KEYS = tuple(S.DEFAULTS["risk"])
HIST_DAYS = 420     # 大盤濾網（均線）與目標波動要往前看的日曆天數，重播時多載入這段報酬


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


def _latest_on_or_before(path: Path, cols: list[str] | None, sig: pd.Timestamp) -> pd.DataFrame:
    """月表（panel 類）在 sig 當天或之前最近一期的資料。"""
    d = pd.read_parquet(path, columns=cols, filters=[("date", "<=", sig), ("date", ">=", sig - pd.Timedelta(days=45))])
    return d[d["date"] == d["date"].max()] if len(d) else d


def panel_on(sig: pd.Timestamp) -> pd.DataFrame:
    """訊號日的選股欄位，和 strategy.load_data 同一套合併（只取一天，比較快）。"""
    ol = _latest_on_or_before(OUT / "outlook.parquet", None, sig).drop(columns=["date", "avg_value"])
    sc = _latest_on_or_before(OUT / "bt" / "composite_scores.parquet", None, sig).drop(columns=["fwd_ret", "avg_value"])
    fm = _latest_on_or_before(OUT / "factors_monthly.parquet", None, sig).drop(columns=["date", "fwd_ret"])
    day = sc.merge(ol, on="code", how="left").merge(fm, on="code", how="left")
    r = load_rets(sig - pd.Timedelta(days=120)).loc[:sig]
    return day.assign(vol_60d=day["code"].map(r.rolling(60, min_periods=40).std().iloc[-1]))


def source_label(source: str | None) -> str:
    if not source:
        return "自己輸入"
    if source in SOURCES:
        return SOURCES[source]
    try:
        return f"策略「{S.load_config(ROOT / source)['name']}」的選股"
    except (OSError, ValueError):
        return source


def source_codes(source: str, sig: pd.Timestamp | None = None) -> list[str]:
    """名單來源在訊號日 sig 的股票（依來源自己的排序）。"""
    sig = sig if sig is not None else data_date()
    if source == "watchlist":
        from daily import read_watchlist
        return read_watchlist()
    if source == "model_picks":
        import outlook as OL
        asof = _latest_on_or_before(OUT / "outlook.parquet", ["date"], sig)["date"].max()
        return OL.picks(OL.load_outlook(), OL.load_calib(), asof)["code"].tolist()
    path = ROOT / source
    if path.suffix != ".toml" or not path.exists():
        raise ValueError(f"不認得的名單來源：{source}")
    _, pick = S.select(panel_on(sig), S.load_config(path))
    return pick["code"].tolist()


def make_weights(w: pd.Series, weighting: str = "manual", max_weight: float = 1.0,
                 sig: pd.Timestamp | None = None) -> pd.Series:
    """目標權重。manual 照輸入的權重；其他只看代號，用訊號日 sig 當時的資料算（同 strategy.target_weights）。
    超過單檔上限的部分分給其他股票，都到上限就留現金。"""
    if weighting not in WEIGHTINGS:
        raise ValueError(f"加權方式要是 {'、'.join(WEIGHTINGS)} 其中之一")
    if not 0 < max_weight <= 1:
        raise ValueError("單檔上限要在 0 到 1 之間")
    if weighting == "manual":
        out = w / w.sum()
        return S._cap(out.copy(), None, max_weight).clip(upper=max_weight) if max_weight < 1 else out
    sig = sig if sig is not None else data_date()
    pick = pd.DataFrame({"code": list(w.index)})
    if weighting == "inv_vol":
        r = load_rets(sig - pd.Timedelta(days=120)).loc[:sig]
        pick["vol_60d"] = pick["code"].map(r.reindex(columns=pick["code"]).rolling(60, min_periods=40).std().iloc[-1])
    elif weighting == "score":
        sc = _latest_on_or_before(OUT / "bt" / "composite_scores.parquet", ["date", "code", "comp_equal"], sig)
        v = pick["code"].map(sc.set_index("code")["comp_equal"])
        pick["comp_equal"] = v.fillna(v.median() if v.notna().any() else 0.0)   # 沒有分數的（例如 ETF）給中位數
    pos = {**S.DEFAULTS["position"], "weighting": weighting, "max_weight": max_weight}
    return S.target_weights(pick, pos, "comp_equal", False)     # 分數高的排前面、權重大


def _check_rules(weighting: str, max_weight: float, risk: dict | None, source: str | None) -> dict:
    """檢查選用的規則，回傳和預設不同的風險規則（只存這些）。"""
    if weighting not in WEIGHTINGS:
        raise ValueError(f"加權方式要是 {'、'.join(WEIGHTINGS)} 其中之一")
    if not 0 < max_weight <= 1:
        raise ValueError("單檔上限要在 0 到 1 之間")
    if source and source not in SOURCES and not (source.endswith(".toml") and (ROOT / source).exists()):
        raise ValueError(f"不認得的名單來源：{source}")
    risk = dict(risk or {})
    bad = [k for k in risk if k not in RISK_KEYS]
    if bad:
        raise ValueError("不認得的風險規則：" + "、".join(bad))
    return {k: v for k, v in risk.items() if v != S.DEFAULTS["risk"][k]}


def create(name: str, w: pd.Series | None, monthly: bool = False, capital: float = 1_000_000,
           signal_date: pd.Timestamp | None = None, weighting: str = "manual", max_weight: float = 1.0,
           risk: dict | None = None, source: str | None = None) -> pd.Timestamp:
    """signal_date：另外指定訊號日（例如決策當天，晚於最新資料日），成交日 = 它之後的第一個交易日。
    w 是 None 時名單從 source 帶入；weighting / max_weight / risk / source 見檔頭說明。"""
    meta = load_meta()
    if not name.strip() or name in meta:
        raise ValueError("組合名稱不能空白，也不能和既有的重複（含已封存的）")
    risk = _check_rules(weighting, max_weight, risk, source)
    sig = _fresh_signal_date()
    if w is None:
        if not source:
            raise ValueError("沒有股票：輸入代號，或選一個名單來源")
        w = pd.Series(1.0, index=source_codes(source, sig))
        if w.empty:
            raise ValueError(f"{source_label(source)}目前沒有股票")
    _check_codes(w, sig)
    w = make_weights(w, weighting, max_weight, sig)
    sig = signal_date if signal_date is not None else sig
    _append(name, w, sig)
    meta[name] = {"capital": float(capital), "rebalance": "monthly" if monthly else "none",
                  "created": str(sig.date()), "archived": False}
    # 選用的規則只在有用到時寫入，沒用到的組合設定和以前一樣
    extra = {"weighting": weighting if weighting != "manual" else None,
             "max_weight": max_weight if max_weight < 1 else None, "risk": risk or None, "source": source or None}
    meta[name].update({k: v for k, v in extra.items() if v is not None})
    META.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return sig


def edit(name: str, w: pd.Series, signal_date: pd.Timestamp | None = None) -> pd.Timestamp:
    m = load_meta().get(name)
    if not m or m["archived"]:
        raise ValueError("找不到這個組合，或已封存")
    sig = _fresh_signal_date()
    _check_codes(w, sig)
    w = make_weights(w, m.get("weighting", "manual"), m.get("max_weight", 1.0), sig)
    sig = signal_date if signal_date is not None else sig
    _append(name, w, sig)
    return sig


def describe(m: dict) -> list[str]:
    """組合用了哪些規則（給畫面與 CLI 摘要）。"""
    risk = {**S.DEFAULTS["risk"], **m.get("risk", {})}
    out = [f"名單：{source_label(m.get('source'))}"
           + ("，每月照規則重選" if m.get("source") and m["rebalance"] == "monthly" else ""),
           f"加權：{WEIGHTINGS[m.get('weighting', 'manual')]}"
           + (f"，單檔上限 {m['max_weight']:.0%}" if m.get("max_weight") else "")]
    if risk["stop_loss"]:
        out.append(f"停損 {risk['stop_loss']:.0%}（{'從進場價' if risk['stop_type'] == 'fixed' else '從進場後高點'}）")
    if risk["take_profit"]:
        out.append(f"停利 {risk['take_profit']:.0%}")
    if risk["max_hold_days"]:
        out.append(f"持有滿 {risk['max_hold_days']} 個交易日出場")
    if risk["regime"] == "ma":
        out.append(f"大盤濾網：{'0050' if risk['regime_index'] == '0050' else '全市場等權'}跌破 {risk['regime_ma']} 日均線"
                   f"留 {risk['regime_off']:.0%}（{'每天' if risk['regime_freq'] == 'daily' else '每月'}判斷）")
    if risk["dd_limit"]:
        out.append(f"資產從高點回落 {risk['dd_limit']:.0%} 就降到 {risk['dd_exposure']:.0%}")
    if risk["target_vol"]:
        out.append(f"目標年化波動 {risk['target_vol']:.0%}")
    return out


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
        w = sigs[last]
        try:
            if m.get("source"):                          # 照名單來源重選，再照加權方式算權重
                w = pd.Series(1.0, index=source_codes(m["source"], sig))
                if w.empty:
                    msgs.append(f"[{name}] 不記：{source_label(m['source'])}這期沒有股票，維持上期持股")
                    continue
            if m.get("source") or m.get("weighting", "manual") != "manual":
                w = make_weights(w, m.get("weighting", "manual"), m.get("max_weight", 1.0), sig)
        except ValueError as e:
            msgs.append(f"[{name}] 不記：{e}")
            continue
        _append(name, w, sig)
        msgs.append(f"[{name}] 記新一期（訊號日 {sig:%Y-%m-%d}，{len(w)} 檔）")
    return msgs


def replay(name: str, rets: pd.DataFrame | None = None) -> dict | None:
    """由已記錄的權重重播。還沒有成交日（等下一個交易日收盤進場）回傳 None。"""
    meta = load_meta()[name]
    sigs = _latest_signals(load_ledger(), name)
    risk = {**S.DEFAULTS["risk"], **meta.get("risk", {})}
    if rets is None:
        rets = load_rets(min(sigs) - pd.Timedelta(days=HIST_DAYS))
    idx = S.regime_index(rets, risk) if risk["regime"] == "ma" else None
    fixed = {}
    for s, w in sigs.items():
        mult, info = S.exposure(s, w, rets, risk, idx)       # 沒開大盤濾網、目標波動時是 1
        fixed[s] = {"tw": w * mult, "tw_full": w * info.get("m_vol", 1.0), "rinfo": info,
                    "pool": w.index.tolist(), "picks": w.index.tolist()}
    cfg = {**json.loads(json.dumps(S.DEFAULTS)), "id": name, "name": name, "description": "",
           "start": str(min(sigs).date()), "capital": meta["capital"], "risk": risk}
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
    n.add_argument("--weighting", default="manual", choices=list(WEIGHTINGS))
    n.add_argument("--max-weight", type=float, default=1.0)
    n.add_argument("--source", help="watchlist、model_picks 或 strategies/xxx.toml；codes 填 - 表示名單全部從來源帶入")
    n.add_argument("--rule", action="append", default=[], help="風險規則，例 stop_loss=0.1、regime=ma，可重複")
    e = sub.add_parser("edit")
    e.add_argument("name")
    e.add_argument("codes")
    sub.add_parser("archive").add_argument("name")
    a = ap.parse_args()
    try:
        if a.cmd == "new":
            risk = {}
            for kv in a.rule:
                k, _, v = kv.partition("=")
                d = S.DEFAULTS["risk"].get(k)
                risk[k] = type(d)(float(v)) if isinstance(d, (int, float)) else v
            w = None if a.codes == "-" else parse_codes(a.codes)
            sig = create(a.name, w, a.monthly, a.capital, weighting=a.weighting, max_weight=a.max_weight,
                         risk=risk, source=a.source)
            print(f"已建立「{a.name}」，訊號日 {sig:%Y-%m-%d}，隔一個交易日收盤進場。" + "；".join(describe(load_meta()[a.name])))
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
    rets = load_rets(pd.Timestamp(min(v["created"] for v in meta.values())) - pd.Timedelta(days=HIST_DAYS))
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
