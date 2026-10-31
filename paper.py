# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""模擬持股：每月記下綜合分數前 N 名，往後追蹤實際報酬，累積真正的樣本外紀錄。

用法:
  uv run paper.py              # 有新月份的訊號就記一期，重算淨值並印出摘要
  uv run paper.py --dry-run    # 只算淨值，不記新的一期

規則（和 composite.py 的 Top-N 回測一致）:
  - 訊號：composite_scores.parquet 最新一天，20 日均成交值 >= 2,000 萬元的股票中 comp_equal 前 30 名，等權
  - 換股：資料日進入新的月份後，第一次執行時記一期（所以換股日是「每月第一個有跑的交易日」，
    不是回測用的月底；手動執行的時點會影響結果）
  - 只記「當下」的訊號：最新資料日距今超過 7 天就不記，避免把過去的訊號補記成即時紀錄
  - 處置股、注意股（訊號日當天公布）、全額交割股不新買進，上一期已在前 N 名的照常續抱；基準池不排除（2026-11 起）
  - 進場：訊號日隔一個交易日收盤（同回測的 lag 1），持有到下一期進場日，期間不再平衡
  - 成本：換手部分扣 買 0.1425% + 賣 0.1425% + 證交稅 0.3%（第一期換手 100%）
  - 停牌 / 下市：沒有行情的日子報酬算 0（價值凍結），不另外處理下市損失；可疑報酬同 strategy.clean_returns 設為 0
  - 基準：同一池（成交值門檻內全部股票）等權買進持有、0050

紀錄 data/paper/ledger.parquet（只追加不改寫）：signal_date、code、name、score、rank、in_top、recorded_at；
in_top = False 的列是同一期的基準池。淨值 data/paper/nav.parquet 每次重算。

多策略（paper_strategies.txt 列出的 strategies/*.toml）:
  - 每個策略每月記一期當下的目標權重（含總曝險、大盤濾網狀態），規則同上：新月份、資料不超過 7 天
  - 記到 data/paper/strat_ledger.parquet（只追加）；設定內容存在 strat_configs.json（以設定雜湊為鍵）。
    策略檔改過內容 → 雜湊不同 → 從改的那期重新起算一條新紀錄，舊紀錄保留
  - 淨值用 strategy.simulate 重播已記錄的權重（不重新選股），停損、每日大盤濾網、回落控制照策略設定逐日執行，
    資金、成本、漲跌停規則同回測
  - 摘要寫到 data/paper/strat_summary.json，給 daily.py 讀；淨值序列寫到 data/paper/strat_nav.parquet
    （strategy、date、nav、pool、etf，起點 = 1），給前端畫圖
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest import COST_ROUND_TRIP
from build_panel import OUT
import restricted
import strategy as S
from strategy import clean_returns

PAPER = OUT / "paper"
LEDGER = PAPER / "ledger.parquet"
NAV = PAPER / "nav.parquet"
TOP_N = 30
MIN_VALUE = 2e7
MAX_STALE_DAYS = 7
ETF = "0050"
STRATS_FILE = Path(__file__).parent / "paper_strategies.txt"
STRAT_LEDGER = PAPER / "strat_ledger.parquet"
STRAT_CONFIGS = PAPER / "strat_configs.json"
STRAT_SUMMARY = PAPER / "strat_summary.json"
STRAT_NAV = PAPER / "strat_nav.parquet"


def load_ledger() -> pd.DataFrame:
    return pd.read_parquet(LEDGER) if LEDGER.exists() else pd.DataFrame(
        columns=["signal_date", "code", "name", "score", "rank", "in_top", "recorded_at"])


def maybe_record(ledger: pd.DataFrame, today: pd.Timestamp) -> tuple[pd.DataFrame, str]:
    sc = pd.read_parquet(OUT / "bt" / "composite_scores.parquet", columns=["date", "code", "avg_value", "comp_equal"])
    last = sc["date"].max()
    if len(ledger) and last.to_period("M") <= ledger["signal_date"].max().to_period("M"):
        return ledger, f"本月已有一期（訊號日 {ledger['signal_date'].max():%Y-%m-%d}）"
    if (today - last).days > MAX_STALE_DAYS:
        return ledger, f"最新資料日 {last:%Y-%m-%d} 距今超過 {MAX_STALE_DAYS} 天，不記（先更新資料）"
    pool = sc[(sc["date"] == last) & (sc["avg_value"] >= MIN_VALUE)].copy()
    # 處置／注意／全額交割股不新買進（上一期已持有的照常續抱），不參與排名；基準池照舊全部
    days = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date"], filters=[("code", "==", ETF)])["date"]
    held = set(ledger.loc[ledger["signal_date"] == ledger["signal_date"].max()].query("in_top")["code"]) if len(ledger) else set()
    ok = ~restricted.flags(pool, pd.DatetimeIndex(sorted(days))) | pool["code"].isin(held)
    pool["rank"] = pool["comp_equal"].where(ok).rank(ascending=False, method="first")
    pool["in_top"] = pool["rank"] <= TOP_N
    names = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "name"])
    names = names[names["date"] == last].set_index("code")["name"]
    new = pd.DataFrame({"signal_date": last, "code": pool["code"], "name": pool["code"].map(names),
                        "score": pool["comp_equal"], "rank": pool["rank"], "in_top": pool["in_top"],
                        "recorded_at": pd.Timestamp(datetime.now())})
    PAPER.mkdir(exist_ok=True)
    ledger = pd.concat([ledger, new], ignore_index=True) if len(ledger) else new
    ledger.to_parquet(LEDGER, index=False)
    return ledger, f"記下新的一期：訊號日 {last:%Y-%m-%d}，前 {TOP_N} 名（池內 {len(pool)} 檔）"


def _hold(rets: pd.DataFrame, codes: list[str]) -> pd.Series:
    """等權買進持有：每檔價值 = 累積(1+報酬)，組合價值 = 平均；回傳期間內的日報酬。"""
    v = (1 + rets.reindex(columns=codes).fillna(0)).cumprod().mean(axis=1)
    return v.pct_change().fillna(v.iloc[0] - 1)


def compute_nav(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "ret", "ret_suspect"],
                         filters=[("date", ">=", ledger["signal_date"].min())])
    rets = clean_returns(px)
    days = rets.index
    sigs = sorted(ledger["signal_date"].unique())
    entries = [days[days > s][0] if (days > s).any() else None for s in sigs]   # 隔一個交易日收盤進場
    out, prev_top = [], None
    for k, (s, e) in enumerate(zip(sigs, entries)):
        if e is None:
            break
        end = entries[k + 1] if k + 1 < len(entries) and entries[k + 1] is not None else days[-1]
        win = rets[(rets.index > e) & (rets.index <= end)]
        led = ledger[ledger["signal_date"] == s]
        top = led.loc[led["in_top"], "code"].tolist()
        turn = 1.0 if prev_top is None else 1 - len(set(top) & set(prev_top)) / len(top)
        cost = pd.Series(0.0, index=[e])
        cost.iloc[0] = -turn * COST_ROUND_TRIP          # 換股成本記在進場日
        prev_top = top
        if win.empty:
            seg = pd.DataFrame({"port": cost, "pool": 0.0, "etf": 0.0})
        else:
            seg = pd.DataFrame({"port": _hold(win, top), "pool": _hold(win, led["code"].tolist()),
                                "etf": win[ETF].fillna(0) if ETF in win else 0.0})
            seg = pd.concat([pd.DataFrame({"port": cost, "pool": 0.0, "etf": 0.0}), seg])
        seg["signal_date"] = s
        out.append(seg)
    if not out:
        return pd.DataFrame()
    r = pd.concat(out)
    r = r.groupby(level=0).agg({"port": "sum", "pool": "sum", "etf": "sum", "signal_date": "last"})
    nav = (1 + r[["port", "pool", "etf"]]).cumprod()
    nav["signal_date"] = r["signal_date"]
    nav.index.name = "date"
    return nav.reset_index()


def summary(ledger: pd.DataFrame | None = None, nav: pd.DataFrame | None = None) -> dict:
    """給 daily.py 用：最新一期持股（含進場以來報酬）與累積績效。"""
    ledger = load_ledger() if ledger is None else ledger
    if ledger.empty:
        return {}
    nav = (pd.read_parquet(NAV) if NAV.exists() else pd.DataFrame()) if nav is None else nav
    s = ledger["signal_date"].max()
    cur = ledger[(ledger["signal_date"] == s) & ledger["in_top"]].sort_values("rank")
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "ret"],
                         filters=[("date", ">", s), ("code", "in", cur["code"].tolist())])
    days = sorted(px["date"].unique())
    held = px[px["date"] > days[0]] if days else px                 # 進場日之後才開始算
    since = (1 + held.pivot(index="date", columns="code", values="ret").fillna(0)).prod() - 1 if len(held) else None
    rows = [{"rank": int(r.rank), "code": r.code, "name": r.name,
             "ret_since_entry": None if since is None or r.code not in since else float(since[r.code])}
            for r in cur.itertuples()]
    res = {"start": str(ledger["signal_date"].min().date()), "signal_date": str(s.date()),
           "entry_date": str(pd.Timestamp(days[0]).date()) if days else None,
           "n_periods": int(ledger["signal_date"].nunique()), "holdings": rows}
    if len(nav):
        last = nav.iloc[-1]
        res.update({"nav_date": str(pd.Timestamp(last["date"]).date()), "port": float(last["port"] - 1),
                    "pool": float(last["pool"] - 1), "etf": float(last["etf"] - 1),
                    "mdd": float((nav["port"] / nav["port"].cummax() - 1).min())})
    return res


# ---------------------------------------------------------------- 多策略

def read_strategies() -> list[Path]:
    if not STRATS_FILE.exists():
        return []
    root = STRATS_FILE.parent
    return [root / l.split("#")[0].strip() for l in STRATS_FILE.read_text(encoding="utf-8").splitlines()
            if l.split("#")[0].strip()]


def signal_rows(cfg: dict, panel: pd.DataFrame, rets: pd.DataFrame, s: pd.Timestamp, held=()) -> pd.DataFrame:
    """訊號日 s 的目標權重（一列一檔，含基準池），和 strategy.simulate 的選股、倉位、曝險同一套。
    held：上一期的選股（處置等股票續抱判斷用）。"""
    pool, pick = S.select(panel[panel["date"] == s], cfg, held=held)
    base = S.target_weights(pick, cfg["position"], cfg["select"]["sort"], cfg["select"]["ascending"])
    idx = S.regime_index(rets, cfg["risk"]) if cfg["risk"]["regime"] == "ma" else None
    mult, info = S.exposure(s, base, rets, cfg["risk"], idx)
    return pd.DataFrame({"signal_date": s, "code": pool["code"].values,
                         "weight": pool["code"].map(base * mult).fillna(0.0).values,
                         "weight_full": pool["code"].map(base * info.get("m_vol", 1.0)).fillna(0.0).values,
                         "in_pick": pool["code"].isin(pick["code"]).values,
                         "pick_order": pool["code"].map({c: i for i, c in enumerate(pick["code"])}).values,
                         "regime_on": info.get("regime_on", True), "m_vol": info.get("m_vol", 1.0),
                         "est_vol": info.get("est_vol", float("nan"))})


def fixed_from(led: pd.DataFrame) -> dict:
    out = {}
    for s, g in led.groupby("signal_date"):
        p = g[g["in_pick"]].sort_values("pick_order")          # 同權重時買進順序和回測一樣
        rinfo = {"regime_on": bool(g["regime_on"].iloc[0]), "m_vol": float(g["m_vol"].iloc[0])}
        out[pd.Timestamp(s)] = {"tw": p.set_index("code")["weight"], "tw_full": p.set_index("code")["weight_full"],
                                "rinfo": rinfo, "pool": g["code"].tolist(), "picks": p["code"].tolist()}
    return out


def record_strategies(panel, rets, today: pd.Timestamp) -> list[str]:
    led = pd.read_parquet(STRAT_LEDGER) if STRAT_LEDGER.exists() else pd.DataFrame()
    confs = json.loads(STRAT_CONFIGS.read_text()) if STRAT_CONFIGS.exists() else {}
    last = panel["date"].max()
    msgs, new = [], []
    for path in read_strategies():
        cfg = S.load_config(path)
        h = S.config_hash(cfg)
        mine = led[(led["strategy"] == cfg["id"]) & (led["hash"] == h)] if len(led) else led
        if len(mine) and last.to_period("M") <= mine["signal_date"].max().to_period("M"):
            continue
        if (today - last).days > MAX_STALE_DAYS:
            msgs.append(f"{cfg['id']}：最新資料日 {last:%Y-%m-%d} 距今超過 {MAX_STALE_DAYS} 天，不記")
            continue
        prev = mine[mine["signal_date"] == mine["signal_date"].max()] if len(mine) else mine
        held = prev.loc[prev["in_pick"], "code"].tolist() if len(prev) else []
        rows = signal_rows(cfg, panel, rets, last, held).assign(strategy=cfg["id"], hash=h,
                                                          recorded_at=pd.Timestamp(datetime.now()))
        new.append(rows)
        confs[h] = cfg
        restart = len(led) and (led["strategy"] == cfg["id"]).any() and not len(mine)
        msgs.append(f"{cfg['id']}：記下訊號日 {last:%Y-%m-%d}，{int(rows['in_pick'].sum())} 檔，目標曝險 "
                    f"{rows['weight'].sum():.0%}" + ("（設定改過，重新起算）" if restart else ""))
    if new:
        PAPER.mkdir(exist_ok=True)
        led = pd.concat([led, *new], ignore_index=True) if len(led) else pd.concat(new, ignore_index=True)
        led.to_parquet(STRAT_LEDGER, index=False)
        STRAT_CONFIGS.write_text(json.dumps(confs, ensure_ascii=False, indent=1, default=str))
    return msgs


def strategy_summaries(panel, rets) -> tuple[list[dict], pd.DataFrame]:
    """目前在 paper_strategies.txt 裡的策略，各自最新設定那條紀錄的績效與持股，加上淨值序列
    （strategy、date、nav、pool、etf，三者起點都是 1）。"""
    navs = []
    if not STRAT_LEDGER.exists():
        return [], pd.DataFrame(columns=["strategy", "date", "nav", "pool", "etf"])
    led = pd.read_parquet(STRAT_LEDGER)
    confs = json.loads(STRAT_CONFIGS.read_text())
    names = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "name"],
                            filters=[("date", ">=", led["signal_date"].max())]).groupby("code")["name"].last()
    out = []
    for path in read_strategies():
        cur = S.load_config(path)
        h = S.config_hash(cur)
        mine = led[(led["strategy"] == cur["id"]) & (led["hash"] == h)]
        if mine.empty:
            continue
        cfg = S.with_defaults(confs[h])                  # 舊紀錄沒有後來才加的設定（例如停利），補預設值
        res = S.simulate(cfg, panel, rets, fixed_from(mine))
        s_last = mine["signal_date"].max()
        lastp = mine[(mine["signal_date"] == s_last) & mine["in_pick"]].sort_values("pick_order")
        item = {"id": cfg["id"], "name": cfg["name"], "description": cfg["description"], "capital": cfg["capital"],
                "start": str(mine["signal_date"].min().date()), "signal_date": str(s_last.date()),
                "n_periods": int(mine["signal_date"].nunique()), "target_exposure": float(lastp["weight"].sum()),
                "regime_on": bool(lastp["regime_on"].iloc[0]) if len(lastp) else None,
                "targets": [{"code": c, "name": names.get(c), "weight": float(w)} for c, w in zip(lastp["code"], lastp["weight"])]}
        r = cfg["risk"]
        if r["regime"] == "ma":
            idx = S.regime_index(rets, r)
            ma = idx.rolling(r["regime_ma"]).mean()
            item["regime_now"] = {"date": str(idx.index[-1].date()), "above": bool(idx.iloc[-1] >= ma.iloc[-1]),
                                  "gap": float(idx.iloc[-1] / ma.iloc[-1] - 1), "freq": r["regime_freq"]}
        if res is not None:
            nv, d = res["nav"], res["daily"]
            total = res["cash"] + sum(res["hold"].values())
            item.update({"nav_date": str(nv.index[-1].date()), "nav": float(nv.iloc[-1]),
                         "ret": float(nv.iloc[-1] / cfg["capital"] - 1), "pool": float((1 + d["pool"]).prod() - 1),
                         "etf": float((1 + d["etf"]).prod() - 1), "mdd": float((nv / nv.cummax() - 1).min()),
                         "cash_pct": float(res["cash"] / total),
                         "exits": int(res["trades"]["exits"].sum()) if "exits" in res["trades"] else 0,
                         "fees": float(res["trades"]["fees"].sum() + res["trades"]["tax"].sum()),
                         "holdings": [{"code": c, "name": names.get(c), "value": float(v), "weight": float(v / total)}
                                      for c, v in sorted(res["hold"].items(), key=lambda x: -x[1])]})
            cum = (1 + d[["pool", "etf"]]).cumprod()
            navs.append(pd.DataFrame({"strategy": cfg["id"], "date": nv.index, "nav": (nv / cfg["capital"]).values,
                                      "pool": cum["pool"].reindex(nv.index).values,
                                      "etf": cum["etf"].reindex(nv.index).values}))
        out.append(item)
    return out, (pd.concat(navs, ignore_index=True) if navs else
                 pd.DataFrame(columns=["strategy", "date", "nav", "pool", "etf"]))


def load_strategy_summaries() -> list[dict]:
    return json.loads(STRAT_SUMMARY.read_text()) if STRAT_SUMMARY.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ledger = load_ledger()
    if not a.dry_run:
        ledger, msg = maybe_record(ledger, pd.Timestamp.today().normalize())
        print(msg)
    nav = compute_nav(ledger)
    if len(nav):
        PAPER.mkdir(exist_ok=True)
        nav.to_parquet(NAV, index=False)
    s = summary(ledger, nav)
    if not s:
        print("還沒有任何一期")
    else:
        print(f"模擬持股自 {s['start']} 起，共 {s['n_periods']} 期；最新一期訊號日 {s['signal_date']}，"
              + (f"進場日 {s['entry_date']}" if s["entry_date"] else "等下一個交易日收盤進場"))
    if "port" in s:
        print(f"累積到 {s['nav_date']}：模擬持股 {s['port']:+.2%}（扣成本）、同池等權 {s['pool']:+.2%}、"
              f"{ETF} {s['etf']:+.2%}；最大回落 {s['mdd']:.2%}")
    if not read_strategies():
        return
    panel, rets = S.load_data()
    if not a.dry_run:
        for m in record_strategies(panel, rets, pd.Timestamp.today().normalize()):
            print(m)
    sums, navs = strategy_summaries(panel, rets)
    PAPER.mkdir(exist_ok=True)
    STRAT_SUMMARY.write_text(json.dumps(sums, ensure_ascii=False, indent=1))
    navs.to_parquet(STRAT_NAV, index=False)
    for x in sums:
        perf = (f"累積 {x['ret']:+.2%}（{ETF} {x['etf']:+.2%}）、最大回落 {x['mdd']:.2%}、現金 {x['cash_pct']:.0%}"
                if "ret" in x else "等下一個交易日收盤進場")
        print(f"[{x['id']}] 自 {x['start']} 起 {x['n_periods']} 期，目標曝險 {x['target_exposure']:.0%}；{perf}")


if __name__ == "__main__":
    main()
