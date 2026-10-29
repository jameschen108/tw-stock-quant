# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""策略回測：讀 strategies/*.toml 的策略設定（選股 → 倉位），用真實資金模擬月頻換股。

用法:
  uv run strategy.py strategies/top30.toml                 # 回測一個策略，輸出 reports/strategy_top30.html
  uv run strategy.py strategies/*.toml                     # 多個策略，另外印出比較表
  uv run strategy.py strategies/top30.toml --no-log        # 不寫入嘗試紀錄（只是重看結果）

設定檔（TOML）:
  name / description / start / capital（初始資金，元）
  [universe] filters   股票池條件（pandas query 字串），同時是「同池等權」基準
  [select]   filters / sort / ascending / n   選股條件、排序欄位、持股檔數
  [position] weighting = "equal" | "inv_vol" | "score"；max_weight（單檔上限）；max_industry（單一產業上限）；
             exposure（股票部位佔資產比例，其餘現金）；min_trade（元）：續抱股票的加碼 / 減碼金額小於這個數就不調，
             避免每筆最低手續費吃掉小額調整（新買進、全部賣出不受影響）
  lag        訊號日後第幾個交易日收盤成交（預設 1；robust.py 用來測換股日偏移）
  [risk]     總曝險控制（每次換股時決定股票部位比例，乘在 position.exposure 上，其餘放現金）:
             regime = "none" | "ma"：大盤指標在 regime_ma 日均線之上才滿倉，之下只留 regime_off（0–1）
             regime_index = "0050"（0050 含息指數）| "ew"（全市場等權指數）
             target_vol：目標年化波動（例 0.15），用目標持股過去 vol_window 日的組合波動估計，
             曝險 = min(1, target_vol / 估計波動)；0 = 不用
             以下是每日檢查的出場規則（收盤判斷、隔一個交易日收盤成交，跌停或沒行情就再等一天）:
             stop_loss（例 0.15）：個股從進場價（stop_type = "fixed"）或進場後高點（"trailing"）跌超過這個幅度就賣，
               賣掉的股票這期不再買回，下次換股若還在名單內才重新買
             regime_freq = "daily"：大盤濾網每天判斷；跌破均線就把股票降到 regime_off，站回均線就買回這期名單
             dd_limit（例 0.15）：資產從高點回落超過這個幅度，股票降到 dd_exposure（預設 0），到下次換股才恢復，
               恢復時高點重設為當時資產
             take_profit（例 0.30）：個股從進場價漲超過這個幅度就賣（停利），賣掉的這期不再買回，同 stop_loss
             max_hold_days（例 60）：個股持有滿這麼多個交易日就賣（從第一次買進算，續抱不重算），這期不再買回
  [cost]     fee_rate、fee_discount（券商折扣，0.28 = 2.8 折）、min_fee（每筆最低手續費，元）、tax（賣出證交稅）
  可用欄位：outlook.parquet（short/mid/long 的 _pct、_label）、composite_scores（comp_equal、rev_mom、quality、
  value、trend）、factors_monthly（close、avg_value、mcap、industry、rev_yoy、roe、ep、bp…）、vol_60d（日報酬 60 日標準差）

模擬規則:
  - 訊號日 = 每月最後一個交易日（加上最新資料日）；隔一個交易日收盤成交（同回測的 lag 1）
  - 股數取整（可買零股，1 股為單位），成交價 = 當天收盤；零股實際成交價可能較差，這裡沒有模擬
  - 手續費 = max(min_fee, 成交金額 × fee_rate × fee_discount)，每檔每次一筆；賣出另扣 tax
  - 成交日收盤漲停（2015-06 前 ±7%，之後 ±10%，以含息報酬近似）買不到、跌停賣不掉、沒有行情不交易，
    買不到的部位留現金到下次換股
  - 持有期間不再平衡，部位價值隨含息日報酬變動（股利視為再投入該股）；現金不計利息
  - 基準：同池等權（每期買進持有、不扣成本）、0050

每次回測（設定內容不同才算一次）都記到 data/strategy/trials.jsonl。成績單附 Deflated Sharpe：
把「總共試過 N 個設定」的運氣成分扣掉後，月 Sharpe 仍大於 0 的機率（Bailey & López de Prado 2014）；
另附檢定「月超額報酬（策略 − 0050）的 Sharpe 大於 0」的版本（dsr_excess）。
每日報酬與成績單另存 data/strategy/results/<id>.parquet / .json，給前端讀（不用重跑）。
"""
import argparse
import hashlib
import html
import json
import math
import tomllib
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from build_panel import OUT
from report import CSS, svg_lines

ROOT = Path(__file__).parent
LATE = {"position": {"min_trade"}, "risk": {"stop_loss", "stop_type", "regime_freq", "dd_limit", "dd_exposure",
                                             "take_profit", "max_hold_days"}}
TRIALS = OUT / "strategy" / "trials.jsonl"
RESULTS = OUT / "strategy" / "results"
REPORTS = ROOT / "reports"
ETF = "0050"
DELIST_RET = 0.0    # 下市當下的報酬假設；沒有下市原因（合併收購多半 ≈ 0、財務危機可能大虧），預設 0，敏感度見 README
DELIST_GAP = 30     # 最後一筆行情早於資料最後一天幾個日曆天以上才算下市
PERIODS = {"全期": ("2000-01-01", "2100-01-01"), "樣本內 2011–2018": ("2011-01-01", "2018-12-31"),
           "樣本外 2019–": ("2019-01-01", "2100-01-01"), "近期 2024–": ("2024-01-01", "2100-01-01")}
DEFAULTS = {"start": "2011-01-01", "capital": 100_000,
            "universe": {"filters": ["avg_value >= 2e7"]},
            "select": {"filters": [], "sort": "comp_equal", "ascending": False, "n": 30},
            "position": {"weighting": "equal", "max_weight": 1.0, "max_industry": 1.0, "exposure": 1.0, "min_trade": 0},
            "risk": {"regime": "none", "regime_index": "0050", "regime_ma": 200, "regime_off": 0.0,
                     "target_vol": 0.0, "vol_window": 60,
                     "stop_loss": 0.0, "stop_type": "trailing", "regime_freq": "monthly", "dd_limit": 0.0, "dd_exposure": 0.0,
                     "take_profit": 0.0, "max_hold_days": 0},
            "cost": {"fee_rate": 0.001425, "fee_discount": 1.0, "min_fee": 20, "tax": 0.003}}


# ---------------------------------------------------------------- 設定與資料

def load_config(path: Path) -> dict:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg = {k: ({**v, **raw.get(k, {})} if isinstance(v, dict) else raw.get(k, v)) for k, v in DEFAULTS.items()}
    cfg["name"] = raw.get("name", path.stem)
    cfg["description"] = raw.get("description", "")
    cfg["id"] = path.stem
    if "lag" in raw:                                     # 不在 DEFAULTS 裡（沒寫就是 1，舊設定的雜湊不變），要另外帶進來
        cfg["lag"] = int(raw["lag"])
    return cfg


def config_hash(cfg: dict) -> str:
    body = {k: v for k, v in cfg.items() if k not in ("name", "description", "id")}
    # 後來才加的設定維持預設值時不進雜湊，舊設定的雜湊和加之前一樣（不會被重複算成新的嘗試）
    for sec, keys in LATE.items():
        body[sec] = {k: v for k, v in body[sec].items() if not (k in keys and v == DEFAULTS[sec][k])}
    if body["risk"] == {k: v for k, v in DEFAULTS["risk"].items() if k not in LATE["risk"]}:
        del body["risk"]
    return hashlib.sha1(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """panel：訊號日 × 股票的所有可用欄位；rets：日含息報酬（列 = 日期、欄 = 代號）。"""
    ol = pd.read_parquet(OUT / "outlook.parquet").drop(columns=["avg_value"])
    sc = pd.read_parquet(OUT / "bt" / "composite_scores.parquet").drop(columns=["fwd_ret", "avg_value"])
    fm = pd.read_parquet(OUT / "factors_monthly.parquet").drop(columns=["fwd_ret"])
    panel = sc.merge(ol, on=["date", "code"], how="left").merge(fm, on=["date", "code"], how="left")
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "ret", "ret_suspect"],
                         filters=[("date", ">=", pd.Timestamp("2010-06-01"))])
    rets = clean_returns(px)
    vol = rets.rolling(60, min_periods=40).std()
    v = vol.reindex(panel["date"].unique()).stack().rename("vol_60d").reset_index()
    panel = panel.merge(v.rename(columns={"level_0": "date"}), on=["date", "code"], how="left")
    return panel, rets


def clean_returns(px: pd.DataFrame) -> pd.DataFrame:
    """日報酬寬表：可疑報酬（超過漲跌幅又沒有事件，多半是缺事件資料）設為缺值（持有時視為 0）。
    下市（最後一筆行情早於資料最後一天 DELIST_GAP 天以上）：最後一筆的隔一個交易日記一筆 DELIST_RET。"""
    w = px.assign(ret=px["ret"].where(~px["ret_suspect"].fillna(False).astype(bool))) \
        .pivot(index="date", columns="code", values="ret").sort_index()
    if DELIST_RET:
        for code, d in delisted(px).items():
            nxt = w.index[w.index > d]
            if len(nxt):
                w.at[nxt[0], code] = DELIST_RET
    return w


def delisted(px: pd.DataFrame) -> pd.Series:
    """{代號: 最後一筆行情日}，只含已經不再交易的（停牌中、還會恢復的看不出來，最後一筆離資料尾端夠遠才算）。"""
    last = px.groupby("code")["date"].max()
    return last[last < px["date"].max() - pd.Timedelta(days=DELIST_GAP)]


# ---------------------------------------------------------------- 倉位

def _cap(w: pd.Series, groups: pd.Series | None, cap: float) -> pd.Series:
    """把超過上限的部分按比例分給還沒到上限的；都到上限時多出的變現金。"""
    for _ in range(20):
        tot = w.groupby(groups).transform("sum") if groups is not None else w
        over = tot > cap + 1e-9
        if not over.any():
            break
        scale = (cap / tot).where(over, 1.0)
        excess = (w * (1 - scale)).sum()
        w = w * scale
        free = ~over
        if not free.any() or w[free].sum() == 0:
            break
        w[free] += excess * w[free] / w[free].sum()
    return w


def target_weights(pick: pd.DataFrame, pos: dict, sort: str, asc: bool) -> pd.Series:
    if pick.empty:
        return pd.Series(dtype=float)
    if pos["weighting"] == "inv_vol":
        raw = 1 / pick["vol_60d"].where(pick["vol_60d"] > 0)
        raw = raw.fillna(raw.median() if raw.notna().any() else 1.0)
    elif pos["weighting"] == "score":
        raw = pick[sort].rank(ascending=not asc)          # 排在前面（選股時先取）的名次數字大、權重大
    else:
        raw = pd.Series(1.0, index=pick.index)
    w = raw / raw.sum()
    ind = pick["industry"].fillna("NA") if pos["max_industry"] < 1 else None
    # 產業上限把多出的權重分給其他股票時，可能又讓個股超過單檔上限，所以兩個上限輪流套到都滿足為止
    for _ in range(20):
        w = _cap(w, None, pos["max_weight"])
        if ind is None:
            break
        w = _cap(w, ind, pos["max_industry"])
        if w.max() <= pos["max_weight"] + 1e-9:
            break
    w = w.clip(upper=pos["max_weight"])                  # 還是分不完（例如選到的產業太少）就變現金
    if ind is not None:
        w = w * (pos["max_industry"] / w.groupby(ind).transform("sum")).clip(upper=1.0)
    return (w * pos["exposure"]).set_axis(pick["code"].values)


def select(day: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = day
    for q in cfg["universe"]["filters"]:
        pool = pool.query(q)
    pick = pool
    for q in cfg["select"]["filters"]:
        pick = pick.query(q)
    s = cfg["select"]
    pick = pick.dropna(subset=[s["sort"]]).sort_values(s["sort"], ascending=s["ascending"]).head(s["n"])
    return pool, pick


# ---------------------------------------------------------------- 總曝險

def regime_index(rets: pd.DataFrame, risk: dict) -> pd.Series:
    r = rets[ETF] if risk["regime_index"] == "0050" else rets.mean(axis=1)
    return (1 + r.fillna(0)).cumprod()


def exposure(s: pd.Timestamp, tw: pd.Series, rets: pd.DataFrame, risk: dict, idx: pd.Series | None) -> tuple[float, dict]:
    """訊號日 s 的曝險倍數（0–1）與當時的判斷依據。只用 s 當天以前的資料。"""
    m, info = 1.0, {}
    if risk["regime"] == "ma" and idx is not None:
        h = idx.loc[:s]
        ma = h.iloc[-risk["regime_ma"]:].mean() if len(h) >= risk["regime_ma"] else np.nan
        on = not (h.iloc[-1] < ma)                         # 資料不夠時視為多頭
        info["regime_on"] = on
        info["m_regime"] = 1.0 if on else risk["regime_off"]
        m *= info["m_regime"]
    if risk["target_vol"] > 0 and len(tw) and tw.sum() > 0:
        hist = rets.loc[:s].iloc[-risk["vol_window"]:].reindex(columns=tw.index).fillna(0)
        est = float((hist @ (tw / tw.sum())).std() * np.sqrt(252))
        info["est_vol"] = est
        info["m_vol"] = min(1.0, risk["target_vol"] / est) if est > 0 else 1.0
        m *= info["m_vol"]
    return m, info


# ---------------------------------------------------------------- 每日出場規則

def _limits(t: pd.Timestamp, r: pd.Series, p: pd.Series):
    lim = 0.07 if t < pd.Timestamp("2015-06-01") else 0.10
    ok = lambda x: pd.notna(p.get(x, np.nan)) and p.get(x) > 0
    up = lambda x: r.get(x, np.nan) >= lim - 0.005
    down = lambda x: r.get(x, np.nan) <= -lim + 0.005
    return ok, up, down


def _fill(t, code, side, px, v, why) -> dict:
    """一筆成交紀錄。股數 = 金額 / 成交價（買進時金額本來就是整股；全部賣出時是當天市值，四捨五入到整數）。"""
    return {"date": t, "code": code, "side": side, "price": float(px) if pd.notna(px) else np.nan,
            "shares": int(round(v / px)) if pd.notna(px) and px > 0 else 0, "amount": float(v), "reason": why}


def _hold_daily(win, prices, hold, cash, tw_full, st, risk, pos, c, fee, use_regime):
    """逐日持有：收盤更新市值 → 執行前一天的待辦（賣出 / 降曝險 / 買回）→ 依收盤判斷出場規則，排到隔天。
    hold、st 原地更新；回傳（現金, 每日資產, 這段的手續費 / 稅 / 出場次數）。"""
    ex = {"fees": 0.0, "tax": 0.0, "exits": 0, "fills": []}
    sell_q, scale_q, reenter = {}, None, False          # 待辦：要全賣的股票（→ 原因）、要把部位降到幾成、要不要買回
    scale_why = ""
    stopped = set()
    navs = []
    def sell(code, v, why):
        nonlocal cash
        ex["fills"].append(_fill(t, code, "賣", prices.at[t, code], v, why))
        f, tx = fee(v), v * c["tax"]
        cash += v - f - tx
        hold[code] -= v
        ex["fees"] += f
        ex["tax"] += tx
        if hold[code] < 1e-6:
            del hold[code]
            st["factor"].pop(code, None)
            st["peak"].pop(code, None)
            st["age"].pop(code, None)
    for t, r in win.iterrows():
        for code in hold:
            g = 1 + (r.get(code) if pd.notna(r.get(code)) else 0.0)
            hold[code] *= g
            st["factor"][code] = st["factor"].get(code, 1.0) * g
            st["peak"][code] = max(st["peak"].get(code, 1.0), st["factor"][code])
            st["age"][code] = st["age"].get(code, 0) + 1
        if sell_q or scale_q is not None or reenter:
            ok, up, down = _limits(t, r, prices.loc[t] if t in prices.index else pd.Series(dtype=float))
            if scale_q is not None:
                for code in list(hold):
                    if not ok(code) or down(code):
                        continue                         # 賣不掉的留著，明天再試（scale_q 不清掉）
                    v = hold[code] * (1 - scale_q) if scale_q > 0 else hold[code]
                    if scale_q > 0:
                        v = math.floor(v / prices.at[t, code]) * prices.at[t, code]
                    if v > 0:
                        sell(code, v, scale_why)
                if scale_q == 0 and not hold or scale_q > 0:
                    scale_q = None
            for code in list(sell_q):
                if code not in hold:
                    sell_q.pop(code)
                elif ok(code) and not down(code):
                    sell(code, hold[code], sell_q.pop(code))
                    stopped.add(code)
                    ex["exits"] += 1
            if reenter:
                total = cash + sum(hold.values())
                for code, wt in tw_full.sort_values(ascending=False).items():
                    if code in stopped:
                        continue
                    need = wt * total - hold.get(code, 0.0)
                    if need <= max(pos["min_trade"], 0) or not ok(code) or up(code):
                        continue
                    px = prices.at[t, code]
                    v = math.floor(min(need, cash - fee(need)) / px) * px
                    if v <= 0:
                        continue
                    f = fee(v)
                    cash -= v + f
                    ex["fees"] += f
                    ex["fills"].append(_fill(t, code, "買", px, v, "大盤回升買回" if code not in hold else "大盤回升加碼"))
                    if code not in hold:
                        st["factor"][code] = st["peak"][code] = 1.0
                        st["age"][code] = 0
                    hold[code] = hold.get(code, 0.0) + v
                reenter = False
        nav_t = cash + sum(hold.values())
        navs.append(nav_t)
        # 收盤判斷，排到隔天
        if risk["stop_loss"] > 0:
            for code in hold:
                base = 1.0 if risk["stop_type"] == "fixed" else st["peak"][code]
                if st["factor"][code] / base <= 1 - risk["stop_loss"]:
                    sell_q.setdefault(code, "停損")
        for code in hold:
            if risk["take_profit"] > 0 and st["factor"][code] >= 1 + risk["take_profit"]:
                sell_q.setdefault(code, "停利")
            if risk["max_hold_days"] > 0 and st["age"].get(code, 0) >= risk["max_hold_days"]:
                sell_q.setdefault(code, "持有期滿")
        if use_regime and risk["regime_freq"] == "daily":
            on = bool(st["above"].get(t, True))
            if st["regime_on"] and not on:
                scale_q, reenter, scale_why = risk["regime_off"], False, "大盤跌破均線降曝險"
            elif not st["regime_on"] and on and not st["dd_out"]:
                reenter, scale_q = True, None
            st["regime_on"] = on
        if risk["dd_limit"] > 0:
            st["nav_peak"] = max(st["nav_peak"], nav_t)
            if not st["dd_out"] and nav_t / st["nav_peak"] - 1 <= -risk["dd_limit"]:
                st["dd_out"], scale_q, reenter, scale_why = True, risk["dd_exposure"], False, "回落控制降曝險"
    return cash, pd.Series(navs, index=win.index, dtype=float), ex


# ---------------------------------------------------------------- 模擬

FILL_COLS = ["date", "code", "side", "price", "shares", "amount", "reason"]

def simulate(cfg: dict, panel: pd.DataFrame, rets: pd.DataFrame, fixed: dict | None = None) -> dict | None:
    """fixed：{訊號日: {"tw", "tw_full", "rinfo", "pool", "picks"}}，給模擬持股用已記錄的訊號重播（不重新選股）。
    還沒有任何成交日時回傳 None。"""
    c, pos = cfg["cost"], cfg["position"]
    fee = lambda v: max(c["min_fee"], v * c["fee_rate"] * c["fee_discount"]) if v > 0 else 0.0
    days = rets.index
    sigs = sorted(fixed) if fixed else sorted(d for d in panel["date"].unique() if d >= pd.Timestamp(cfg["start"]))
    prices = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "close"],
                             filters=[("date", ">=", sigs[0])]).pivot(index="date", columns="code", values="close")
    risk = cfg["risk"]
    idx = regime_index(rets, risk) if risk["regime"] == "ma" else None
    daily_mode = (risk["stop_loss"] > 0 or risk["dd_limit"] > 0 or risk["take_profit"] > 0 or risk["max_hold_days"] > 0
                  or (risk["regime"] == "ma" and risk["regime_freq"] == "daily"))
    st = {"factor": {}, "peak": {}, "age": {}, "nav_peak": float(cfg["capital"]), "dd_out": False}
    if idx is not None:
        ma = idx.rolling(risk["regime_ma"], min_periods=risk["regime_ma"]).mean()
        st["above"] = ~(idx < ma)                       # 均線還算不出來時視為多頭
    cash, hold = float(cfg["capital"]), {}              # hold: code -> 部位市值
    gone = {}
    if len(prices):                                     # 訊號日之後還沒有行情（模擬持股剛建立）時是空表
        last_px = pd.to_datetime(prices.apply(pd.Series.last_valid_index), errors="coerce")
        gone = last_px[last_px < days[-1] - pd.Timedelta(days=DELIST_GAP)].to_dict()
    delist_out = []
    nav, bench, trades, logs, fills = [], [], [], [], []
    for k, s in enumerate(sigs):
        after = days[days > s]
        if not len(after):
            break
        lag = cfg.get("lag", 1)                          # 訊號日後第幾個交易日成交（沒寫就是 1，不進設定雜湊）
        exe = after[min(lag, len(after)) - 1]
        nxt = days[days > sigs[k + 1]] if k + 1 < len(sigs) else []
        end = nxt[min(lag, len(nxt)) - 1] if len(nxt) else days[-1]
        if fixed:
            f = fixed[s]
            tw, tw_full, rinfo, pool_codes, pick_codes = f["tw"], f["tw_full"], f["rinfo"], f["pool"], f["picks"]
        else:
            pool, pick = select(panel[panel["date"] == s], cfg)
            base_tw = target_weights(pick, pos, cfg["select"]["sort"], cfg["select"]["ascending"])
            mult, rinfo = exposure(s, base_tw, rets, risk, idx)
            tw = base_tw * mult
            tw_full = base_tw * rinfo.get("m_vol", 1.0)  # 不含大盤濾網的目標，每日濾網站回均線時買回用
            pool_codes, pick_codes = pool["code"].tolist(), pick["code"].tolist()
        # 已下市（成交日之前就沒有行情了）的部位按最後價值轉成現金，否則永遠賣不掉、也不會再投入
        for code in [x for x in hold if x in gone and gone[x] < exe]:
            v = hold.pop(code)
            cash += v
            delist_out.append(code)
            fills.append(_fill(exe, code, "賣", np.nan, v, "下市轉現金"))
        total = cash + sum(hold.values())
        if st["dd_out"]:                                 # 回落控制觸發後，換股時恢復並重設高點
            st["dd_out"], st["nav_peak"] = False, total
        r_exe, p_exe = rets.loc[exe], prices.loc[exe] if exe in prices.index else pd.Series(dtype=float)
        lim = 0.07 if exe < pd.Timestamp("2015-06-01") else 0.10
        up = lambda x: r_exe.get(x, np.nan) >= lim - 0.005
        down = lambda x: r_exe.get(x, np.nan) <= -lim + 0.005
        ok = lambda x: pd.notna(p_exe.get(x, np.nan)) and p_exe.get(x) > 0
        fees = tax = traded = 0.0
        blocked = []
        # 先賣
        for code in list(hold):
            tgt = tw.get(code, 0.0) * total
            if tgt >= hold[code]:
                continue
            if not ok(code) or down(code):
                blocked.append(f"{code} 賣不掉")
                continue
            if tgt > 0 and hold[code] - tgt < pos["min_trade"]:
                continue
            px = p_exe[code]
            v = hold[code] if tgt == 0 else math.floor((hold[code] - tgt) / px) * px
            if v <= 0:
                continue
            fills.append(_fill(exe, code, "賣", px, v, "換股出場" if tgt == 0 else "減碼"))
            f, t = fee(v), v * c["tax"]
            cash += v - f - t
            hold[code] -= v
            fees, tax, traded = fees + f, tax + t, traded + v
            if hold[code] < 1e-6:
                del hold[code]
        # 再買（目標權重大的先買）
        for code, wt in tw.sort_values(ascending=False).items():
            need = wt * total - hold.get(code, 0.0)
            if need <= 0 or (code in hold and need < pos["min_trade"]):
                continue
            if not ok(code) or up(code):
                blocked.append(f"{code} 買不到")
                continue
            px = p_exe[code]
            v = math.floor(min(need, cash - fee(need)) / px) * px
            if v <= 0:
                continue
            f = fee(v)
            cash -= v + f
            fills.append(_fill(exe, code, "買", px, v, "新買進" if code not in hold else "加碼"))
            if code not in hold:
                st["factor"][code] = st["peak"][code] = 1.0
                st["age"][code] = 0
            hold[code] = hold.get(code, 0.0) + v
            fees, traded = fees + f, traded + v
        trades.append({"signal": s, "exe": exe, "total": total, "n_hold": len(hold), "n_target": len(tw),
                       "turnover": traded / total / 2,
                       "fees": fees, "tax": tax, "blocked": blocked, "cash_pct": cash / total,
                       "exposure": float(tw.sum()), "delisted": delist_out, **rinfo})
        delist_out = []
        # 持有到下一個成交日
        win = rets[(rets.index > exe) & (rets.index <= end)]
        if daily_mode:
            st["regime_on"] = rinfo.get("regime_on", True)
            cash, nav_win, ex = _hold_daily(win, prices, hold, cash, tw_full, st, risk, pos, c, fee, idx is not None)
            trades[-1]["fees"] += ex["fees"]
            trades[-1]["tax"] += ex["tax"]
            trades[-1]["exits"] = ex["exits"]
            fills += ex["fills"]
        elif len(hold):
            g = (1 + win.reindex(columns=list(hold)).fillna(0)).cumprod()
            vals = g.mul(pd.Series(hold), axis=1)
            nav_win = vals.sum(axis=1) + cash
            if len(vals):
                hold = vals.iloc[-1].to_dict()
        else:
            nav_win = pd.Series(cash, index=win.index)
        nav.append(pd.concat([pd.Series(total - fees - tax, index=[exe]), nav_win]))
        # 同池等權（不扣成本）與 0050
        pw = (1 + win.reindex(columns=pool_codes).fillna(0)).cumprod().mean(axis=1)
        bench.append(pd.DataFrame({"pool": pw.pct_change().fillna(pw.iloc[0] - 1) if len(pw) else pw,
                                   "etf": win[ETF].fillna(0) if ETF in win else 0.0}))
        logs.append({"signal": str(pd.Timestamp(s).date()), "picks": pick_codes})
    if not nav:
        return None
    nv = pd.concat(nav)
    nv = nv[~nv.index.duplicated(keep="last")]
    b = pd.concat(bench)
    b = b[~b.index.duplicated(keep="first")].reindex(nv.index).fillna(0)
    daily = pd.DataFrame({"strategy": nv.pct_change().fillna(nv.iloc[0] / cfg["capital"] - 1),
                          "pool": b["pool"], "etf": b["etf"]})
    return {"nav": nv, "daily": daily, "trades": pd.DataFrame(trades), "picks": logs,
            "fills": pd.DataFrame(fills, columns=FILL_COLS).sort_values("date", kind="stable", ignore_index=True), "hold": hold, "cash": cash}


# ---------------------------------------------------------------- 成績

def perf(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {}
    nav = (1 + r).cumprod()
    yrs = len(r) / 252
    dd = nav / nav.cummax() - 1
    under = (dd < 0).astype(int)
    runs = under.groupby((under == 0).cumsum()).cumsum()
    m = (1 + r).resample("ME").prod() - 1
    return {"cagr": nav.iloc[-1] ** (1 / yrs) - 1, "vol": r.std() * np.sqrt(252),
            "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
            "mdd": dd.min(), "mdd_days": int(runs.max()), "total": nav.iloc[-1] - 1,
            "sharpe_m": m.mean() / m.std() if m.std() > 0 else np.nan, "months": len(m)}


def _monthly(r: pd.Series) -> pd.Series:
    return (1 + r.dropna()).resample("ME").prod() - 1


def excess_monthly(d: pd.DataFrame) -> pd.Series:
    """月超額報酬：策略 − 0050。"""
    return _monthly(d["strategy"]) - _monthly(d["etf"])


def _dsr(m: pd.Series, n_trials: int, sr_var: float) -> float | None:
    t = len(m)
    if t < 24 or m.std() == 0:
        return None
    sr = m.mean() / m.std()
    g3, g4 = m.skew(), m.kurt() + 3
    nd, gamma = NormalDist(), 0.5772156649
    sr0 = 0.0 if n_trials <= 1 else math.sqrt(max(sr_var, 1e-12)) * (
        (1 - gamma) * nd.inv_cdf(1 - 1 / n_trials) + gamma * nd.inv_cdf(1 - 1 / (n_trials * math.e)))
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2, 1e-12))
    return nd.cdf((sr - sr0) * math.sqrt(t - 1) / denom)


def deflated_sharpe(r: pd.Series, n_trials: int, sr_var: float) -> float | None:
    """Bailey & López de Prado（2014）的 Deflated Sharpe Ratio，用月報酬。"""
    return _dsr(_monthly(r), n_trials, sr_var)


def deflated_sharpe_excess(d: pd.DataFrame, n_trials: int, sr_var: float | None) -> float | None:
    """同上，但檢定的是相對 0050 的月超額報酬（資訊比率）大於 0；sr_var 是各設定超額 Sharpe 的變異數。"""
    return None if sr_var is None else _dsr(excess_monthly(d), n_trials, sr_var)


def _sharpe_ex(d: pd.DataFrame) -> float | None:
    m = excess_monthly(d)
    return float(m.mean() / m.std()) if len(m) > 1 and m.std() > 0 else None


def trial_stats(cfg_id: str) -> tuple[int, int, float, float | None]:
    """讀嘗試紀錄（不寫入），回傳（這個策略名稱試過幾個設定、全部設定數、月 Sharpe 變異數、超額 Sharpe 變異數）。"""
    rows = [json.loads(l) for l in TRIALS.read_text().splitlines() if l.strip()] if TRIALS.exists() else []
    srs = [r["sharpe_m"] for r in rows if r.get("sharpe_m") is not None]
    exs = [r["sharpe_ex"] for r in rows if r.get("sharpe_ex") is not None]
    return (sum(r.get("id") == cfg_id for r in rows), max(len(rows), 1), float(np.var(srs)) if len(srs) > 1 else 0.0,
            float(np.var(exs)) if len(exs) > 1 else None)


def log_trial(cfg: dict, res: dict) -> tuple[int, int, float]:
    """寫入嘗試紀錄（同設定不重複算），回傳（這個策略名稱試過幾個設定、全部設定數、各設定月 Sharpe 變異數）。"""
    TRIALS.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in TRIALS.read_text().splitlines()] if TRIALS.exists() else []
    h = config_hash(cfg)
    if h not in {r["hash"] for r in rows}:
        p = perf(res["daily"]["strategy"])
        rows.append({"time": datetime.now().isoformat(timespec="seconds"), "id": cfg["id"], "name": cfg["name"],
                     "hash": h, "sharpe_m": p.get("sharpe_m"), "sharpe_ex": _sharpe_ex(res["daily"]),
                     "cagr": p.get("cagr"), "config": cfg})
        with open(TRIALS, "a", encoding="utf-8") as f:
            f.write(json.dumps(rows[-1], ensure_ascii=False, default=str) + "\n")
    srs = [r["sharpe_m"] for r in rows if r.get("sharpe_m") is not None]
    return sum(r["id"] == cfg["id"] for r in rows), len(rows), float(np.var(srs)) if len(srs) > 1 else 0.0


def scorecard(cfg: dict, res: dict, n_same: int, n_all: int, sr_var: float, sr_var_ex: float | None = None) -> dict:
    """sr_var_ex 沒給就從嘗試紀錄讀。"""
    d, tr = res["daily"], res["trades"]
    out = {"periods": {}}
    for p, (a, b) in PERIODS.items():
        x = d.loc[a:b]
        out["periods"][p] = {k: perf(x[k]) for k in ("strategy", "pool", "etf")}
    yrs = len(d) / 252
    out["turnover_month"] = float(tr["turnover"].mean())
    # 每次換股的成本佔當時資產的比例，加總後換算成每年
    out["cost_per_year_pct"] = float(((tr["fees"] + tr["tax"]) / tr["total"]).sum() / yrs * 100)
    out["fees_total"], out["tax_total"] = float(tr["fees"].sum()), float(tr["tax"].sum())
    out["blocked"] = int(tr["blocked"].map(len).sum())
    out["avg_hold"], out["avg_cash_pct"] = float(tr["n_hold"].mean()), float(tr["cash_pct"].mean())
    out["avg_exposure"] = float(tr["exposure"].mean())
    out["regime_off_pct"] = float((~tr["regime_on"].astype(bool)).mean()) if "regime_on" in tr else None
    out["final"] = float(res["nav"].iloc[-1])
    out["n_same"], out["n_all"] = n_same, n_all
    out["dsr"] = deflated_sharpe(d["strategy"], n_all, sr_var)
    if sr_var_ex is None:
        sr_var_ex = trial_stats(cfg["id"])[3]
    out["dsr_excess"] = deflated_sharpe_excess(d, n_all, sr_var_ex)
    out["hash"], out["asof"] = config_hash(cfg), str(d.index[-1].date())
    return out


def _clean(x):
    """JSON 用：numpy 數字轉成 Python，NaN / inf 轉成 None。"""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def with_names(fills: pd.DataFrame) -> pd.DataFrame:
    """成交紀錄加上股票名稱（取價格表裡最後一次的名稱）。"""
    out = fills.copy()
    if out.empty:
        out.insert(2, "name", pd.Series(dtype=str))
        return out
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "name"],
                         filters=[("code", "in", out["code"].unique().tolist())])
    names = px.sort_values("date").groupby("code")["name"].last()
    out.insert(2, "name", out["code"].map(names).fillna(""))
    return out


def save_result(cfg: dict, res: dict, sc: dict) -> Path:
    """每日報酬（strategy、pool、etf）存 results/<id>.parquet，成績單存 results/<id>.json，成交明細存
    results/<id>_fills.parquet，給前端讀。"""
    RESULTS.mkdir(parents=True, exist_ok=True)
    with_names(res["fills"]).to_parquet(RESULTS / f"{cfg['id']}_fills.parquet", index=False)
    d = res["daily"][["strategy", "pool", "etf"]].rename_axis("date").reset_index()
    d.to_parquet(RESULTS / f"{cfg['id']}.parquet", index=False)
    out = RESULTS / f"{cfg['id']}.json"
    out.write_text(json.dumps(_clean(sc), ensure_ascii=False, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------- 輸出

def _p(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.{nd}f}%"


def _fills_html(fills: pd.DataFrame, n_days: int = 3) -> str:
    """最近 n_days 個有成交的日子，一列一筆。"""
    if fills.empty:
        return "<p>（沒有任何成交）</p>"
    e = html.escape
    recent = fills[fills["date"].isin(sorted(fills["date"].unique())[-n_days:])].sort_values(
        ["date", "side", "amount"], ascending=[False, True, False])
    rows = "".join(
        f"<tr><td>{r.date:%Y-%m-%d}</td><td>{r.side}</td><td>{e(r.code)} {e(r.name)}</td>"
        f"<td>{'—' if pd.isna(r.price) else f'{r.price:,.2f}'}</td><td>{r.shares:,}</td><td>{r.amount:,.0f}</td>"
        f"<td>{e(r.reason)}</td></tr>" for r in recent.itertuples())
    return ('<div class="wrap"><table><tr><th>成交日</th><th>買賣</th><th>股票</th><th>成交價</th><th>股數</th>'
            f'<th>金額</th><th>原因</th></tr>{rows}</table></div>')


def render(cfg: dict, res: dict, sc: dict) -> str:
    e = html.escape
    d = res["daily"]
    w = (1 + d).cumprod().resample("W").last()
    chart = svg_lines([f"{t:%y/%m}" for t in w.index],
                      {"策略": list(w["strategy"]), "同池等權": list(w["pool"]), ETF: list(w["etf"])},
                      "{:,.1f}", "累積淨值（起點 = 1，週）", "{:,.2f}")
    dd = w["strategy"] / w["strategy"].cummax() - 1
    chart_dd = svg_lines([f"{t:%y/%m}" for t in dd.index], {"回落": list(dd * 100)}, "{:,.0f}%",
                         "策略從高點回落（%）", "{:.1f}%", zero=True)
    rows = ""
    for p, v in sc["periods"].items():
        s, pl, et = v["strategy"], v["pool"], v["etf"]
        if not s:
            continue
        rows += (f"<tr><td>{p}</td><td>{_p(s['cagr'])}</td><td>{_p(pl.get('cagr'))}</td><td>{_p(et.get('cagr'))}</td>"
                 f"<td>{_p(s['vol'])}</td><td>{s['sharpe']:.2f}</td><td>{_p(s['mdd'])}</td><td>{s['mdd_days']}</td></tr>")
    yr = (1 + d).groupby(d.index.year).prod() - 1
    yrows = "".join(f"<tr><td>{y}</td><td>{_p(r.strategy)}</td><td>{_p(r.pool)}</td><td>{_p(r.etf)}</td>"
                    f"<td>{_p(r.strategy - r.pool)}</td></tr>" for y, r in yr.iloc[::-1].iterrows())
    last = res["picks"][-1] if res["picks"] else {"signal": "—", "picks": []}
    dsr = "—" if sc["dsr"] is None else f"{sc['dsr'] * 100:.0f}%"
    dsr_ex = "—" if sc.get("dsr_excess") is None else f"{sc['dsr_excess'] * 100:.0f}%"
    cfg_txt = e(json.dumps({k: cfg[k] for k in ("start", "capital", "universe", "select", "position", "risk", "cost")},
                           ensure_ascii=False, indent=1))
    tiles = [("期末資產", f"{sc['final']:,.0f} 元"), ("全期年化", _p(sc["periods"]["全期"]["strategy"]["cagr"])),
             ("樣本外年化", _p(sc["periods"]["樣本外 2019–"]["strategy"].get("cagr"))),
             ("最大回落", _p(sc["periods"]["全期"]["strategy"]["mdd"])),
             ("交易成本 / 年（佔資產）", f"{sc['cost_per_year_pct']:.1f}%"), ("Deflated Sharpe", dsr),
             (f"Deflated Sharpe（對 {ETF}）", dsr_ex)]
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(cfg['name'])} 策略回測</title>
<style>{CSS}</style></head><body><main>
<h1>{e(cfg['name'])}</h1><p class="sub">{e(cfg['description'])}　·　初始資金 {cfg['capital']:,} 元　·
{res['daily'].index[0]:%Y-%m-%d} ~ {res['daily'].index[-1]:%Y-%m-%d}</p>
<div class="tiles">{"".join(f'<div class="tile"><span>{k}</span><b>{v}</b></div>' for k, v in tiles)}</div>
<p class="muted">這是 strategy.py 試過的第 {sc['n_all']} 個設定（同一個策略檔 {sc['n_same']} 個）。
Deflated Sharpe = 扣掉「試了 {sc['n_all']} 個設定挑最好的」運氣成分後，月 Sharpe 仍大於 0 的機率；
低於 95% 代表成績可能主要來自嘗試次數。「對 {ETF}」版檢定的是月超額報酬（策略 − {ETF}）的 Sharpe 大於 0，
比較能回答「有沒有贏過直接買 {ETF}」。</p>
<h2>淨值</h2><div class="charts">{chart}{chart_dd}</div>
<h2>各期間</h2><div class="wrap"><table><tr><th>期間</th><th>策略年化</th><th>同池等權</th><th>{ETF}</th><th>波動</th>
<th>Sharpe</th><th>最大回落</th><th>最長回落天數</th></tr>{rows}</table></div>
<p class="muted">同池等權與 {ETF} 不扣成本；策略已扣手續費（每筆最低 {cfg['cost']['min_fee']} 元）與證交稅。
平均每月換手 {sc['turnover_month'] * 100:.0f}%，平均持有 {sc['avg_hold']:.1f} 檔、現金 {sc['avg_cash_pct'] * 100:.1f}%，
平均目標曝險 {sc['avg_exposure'] * 100:.0f}%{f"（{sc['regime_off_pct'] * 100:.0f}% 的月份大盤在均線下）" if sc['regime_off_pct'] is not None else ""}，
累計手續費 {sc['fees_total']:,.0f} 元、證交稅 {sc['tax_total']:,.0f} 元，漲跌停或無行情擋下 {sc['blocked']} 筆。</p>
<h2>每年</h2><div class="wrap"><table><tr><th>年</th><th>策略</th><th>同池等權</th><th>{ETF}</th><th>對同池超額</th></tr>{yrows}</table></div>
<h2>最新一期選股（訊號日 {last['signal']}）</h2><p>{"、".join(last['picks']) or "（沒有符合條件的股票）"}</p>
<h2>最近三個成交日的買賣明細</h2>{_fills_html(with_names(res["fills"]))}
<p class="muted">完整明細（全期每一筆）在前端「策略」頁，或 data/strategy/results/{cfg['id']}_fills.parquet。</p>
<h2>設定</h2><pre class="ann">{cfg_txt}</pre>
<p class="disc">學術研究用。回測假設以收盤價成交、零股可成交，實際成交價與流動性可能更差；歷史結果不代表未來。</p>
</main></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", type=Path)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    panel, rets = load_data()
    REPORTS.mkdir(exist_ok=True)
    summary = []
    for path in a.configs:
        cfg = load_config(path)
        res = simulate(cfg, panel, rets)
        if not a.no_log:
            log_trial(cfg, res)
        n_same, n_all, var, var_ex = trial_stats(cfg["id"])
        sc = scorecard(cfg, res, n_same, n_all, var, var_ex)
        save_result(cfg, res, sc)
        out = REPORTS / f"strategy_{cfg['id']}.html"
        out.write_text(render(cfg, res, sc), encoding="utf-8")
        f, o = sc["periods"]["全期"], sc["periods"]["樣本外 2019–"]
        summary.append({"策略": cfg["name"], "期末資產": round(sc["final"]), "全期年化": _p(f["strategy"]["cagr"]),
                        "樣本外年化": _p(o["strategy"].get("cagr")), "同池等權樣本外": _p(o["pool"].get("cagr")),
                        f"{ETF}樣本外": _p(o["etf"].get("cagr")), "Sharpe": round(f["strategy"]["sharpe"], 2),
                        "最大回落": _p(f["strategy"]["mdd"]), "成本/年": f"{sc['cost_per_year_pct']:.1f}%",
                        "DSR": "—" if sc["dsr"] is None else f"{sc['dsr'] * 100:.0f}%",
                        f"DSR對{ETF}": "—" if sc["dsr_excess"] is None else f"{sc['dsr_excess'] * 100:.0f}%"})
        print(f"{cfg['name']} → {out.relative_to(ROOT)}")
    print(pd.DataFrame(summary).set_index("策略").to_string())


if __name__ == "__main__":
    main()
