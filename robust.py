# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""策略穩健性檢查：參數微調、分段績效、相對 0050 的超額報酬與 t 值。

用法:
  uv run robust.py strategies/top30.toml strategies/mid_bull_invvol.toml   # 輸出 reports/robust.html
  uv run robust.py strategies/*.toml --no-log                              # 不寫入嘗試紀錄
  uv run robust.py strategies/top30_*.toml --out robust_exposure           # 輸出 reports/robust_exposure.html

每個策略會跑這些變體（其餘設定不變）:
  - 持股檔數 × 0.5 / × 1.5（至少 3 檔）
  - 換股日偏移：訊號日後第 3、5、10 個交易日成交（原本第 1 個）
  - 成本 × 2（手續費與證交稅都加倍）、券商 2.8 折
  - 初始資金 100 萬（最低手續費 20 元的影響變小）
  - 有大盤濾網：均線 100 / 250 日、均線下留五成；有目標波動：目標 × 0.8 / × 1.2
  - 有停損：幅度 × 0.75 / × 1.25、固定 / 移動停損互換；有回落控制：門檻 × 2/3 / × 4/3
所有變體都記進 data/strategy/trials.jsonl，讓 Deflated Sharpe 的「試過幾次」反映真實次數。

相對 0050：用月報酬算
  - 超額 = 策略 − 0050，年化平均與 Newey-West t（lag 3）
  - CAPM：策略 = α + β × 0050，年化 α 與它的 t（殘差 NW 標準誤）
  分段：2011–2015、2016–2018、2019–2021、2022–2023、2024–
"""
import argparse
import copy
import html
from pathlib import Path

import numpy as np
import pandas as pd

from report import CSS
from strategy import ETF, REPORTS, deflated_sharpe, load_config, load_data, log_trial, perf, simulate
from technicals import nw_t

SEGMENTS = {"2011–2015": ("2011-01-01", "2015-12-31"), "2016–2018": ("2016-01-01", "2018-12-31"),
            "2019–2021": ("2019-01-01", "2021-12-31"), "2022–2023": ("2022-01-01", "2023-12-31"),
            "2024–": ("2024-01-01", "2100-01-01")}
OOS = ("2019-01-01", "2100-01-01")


def variants(cfg: dict) -> list[tuple[str, dict]]:
    out = [("原設定", cfg)]
    def v(tag, f):
        c = copy.deepcopy(cfg)
        f(c)
        c["id"] = f"{cfg['id']}__{tag}"
        out.append((tag, c))
    n = cfg["select"]["n"]
    for k in sorted({max(3, round(n * 0.5)), round(n * 1.5)} - {n}):
        v(f"n{k}", lambda c, k=k: c["select"].update(n=k))
    for lag in (3, 5, 10):
        v(f"lag{lag}", lambda c, lag=lag: c.update(lag=lag))
    v("cost2x", lambda c: c["cost"].update(fee_discount=c["cost"]["fee_discount"] * 2, tax=c["cost"]["tax"] * 2))
    v("disc28", lambda c: c["cost"].update(fee_discount=0.28))
    v("cap1m", lambda c: c.update(capital=1_000_000))
    r = cfg["risk"]
    if r["regime"] == "ma":
        for ma in (100, 250):
            v(f"ma{ma}", lambda c, ma=ma: c["risk"].update(regime_ma=ma))
        v("off50", lambda c: c["risk"].update(regime_off=0.5))
    if r["stop_loss"] > 0:
        for k in (0.75, 1.25):
            v(f"stop{round(r['stop_loss'] * k * 100)}", lambda c, k=k: c["risk"].update(stop_loss=r["stop_loss"] * k))
        other = "fixed" if r["stop_type"] == "trailing" else "trailing"
        v(other, lambda c: c["risk"].update(stop_type=other))
    if r["dd_limit"] > 0:
        for k in (2 / 3, 4 / 3):
            v(f"dd{round(r['dd_limit'] * k * 100)}", lambda c, k=k: c["risk"].update(dd_limit=r["dd_limit"] * k))
    if r["target_vol"] > 0:
        for k in (0.8, 1.2):
            v(f"tvol{round(r['target_vol'] * k * 100)}", lambda c, k=k: c["risk"].update(target_vol=r["target_vol"] * k))
    return out


def nw_se_alpha(y: pd.Series, x: pd.Series, lag: int = 3) -> tuple[float, float, float]:
    """OLS y = a + b x，回傳 (a, b, a 的 t)；標準誤用 Newey-West。"""
    X = np.column_stack([np.ones(len(x)), x.values])
    beta, *_ = np.linalg.lstsq(X, y.values, rcond=None)
    e = y.values - X @ beta
    n = len(y)
    u = X * e[:, None]
    S = u.T @ u / n
    for k in range(1, min(lag, n - 1) + 1):
        g = u[k:].T @ u[:-k] / n
        S += (1 - k / (lag + 1)) * (g + g.T)
    inv = np.linalg.inv(X.T @ X / n)
    cov = inv @ S @ inv / n
    return beta[0], beta[1], beta[0] / np.sqrt(cov[0, 0])


def vs_etf(d: pd.DataFrame, a: str, b: str) -> dict:
    m = (1 + d.loc[a:b, ["strategy", "etf"]]).resample("ME").prod() - 1
    m = m[m.index >= pd.Timestamp(a)]
    if len(m) < 12:
        return {}
    ex = m["strategy"] - m["etf"]
    al, be, t_al = nw_se_alpha(m["strategy"], m["etf"])
    p = perf(d.loc[a:b, "strategy"])
    return {"cagr": p["cagr"], "etf_cagr": perf(d.loc[a:b, "etf"])["cagr"], "mdd": p["mdd"],
            "excess": ex.mean() * 12, "t_excess": nw_t(ex, 3), "alpha": al * 12, "beta": be, "t_alpha": t_al,
            "win": (ex > 0).mean(), "months": len(m)}


def _p(x, nd=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:+.{nd}f}%"


def _t(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.2f}"


def table(head: list[str], rows: list[list[str]]) -> str:
    h = "".join(f"<th>{html.escape(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--out", default="robust", help="輸出 reports/<out>.html")
    a = ap.parse_args()
    panel, rets = load_data()
    sections, console, originals = [], [], {}
    trial_info = None
    for path in a.configs:
        base = load_config(Path(path))
        res_by = {}
        for tag, cfg in variants(base):
            res = simulate(cfg, panel, rets)
            res_by[tag] = res
            if not a.no_log:
                trial_info = log_trial(cfg, res)
            print(f"  {base['id']} {tag} 完成")
        # 參數微調：樣本外
        rows = []
        for tag, res in res_by.items():
            o = vs_etf(res["daily"], *OOS)
            tr = res["trades"]
            yrs = len(res["daily"]) / 252
            cost = ((tr["fees"] + tr["tax"]) / tr["total"]).sum() / yrs
            rows.append([tag, _p(o["cagr"]), _p(o["etf_cagr"]), _p(o["excess"]), _t(o["t_excess"]),
                         _p(o["alpha"]), f"{o['beta']:.2f}", _t(o["t_alpha"]), _p(o["mdd"]), f"{cost * 100:.1f}%"])
            console.append({"策略": base["id"], "變體": tag, "樣本外年化": _p(o["cagr"]), "超額/年": _p(o["excess"]),
                            "t": _t(o["t_excess"]), "α/年": _p(o["alpha"]), "β": round(o["beta"], 2),
                            "α t": _t(o["t_alpha"]), "回落": _p(o["mdd"])})
        grid = table(["變體", "樣本外年化", "0050", "超額/年", "超額 t", "α/年", "β", "α t", "最大回落", "成本/年"], rows)
        # 分段：原設定
        d = res_by["原設定"]["daily"]
        originals[base["name"]] = d["strategy"]
        seg_rows = []
        for name, (s, e) in SEGMENTS.items():
            o = vs_etf(d, s, e)
            if not o:
                continue
            seg_rows.append([name, _p(o["cagr"]), _p(o["etf_cagr"]), _p(o["excess"]), _t(o["t_excess"]),
                             _p(o["alpha"]), f"{o['beta']:.2f}", _t(o["t_alpha"]), f"{o['win'] * 100:.0f}%",
                             _p(o["mdd"])])
        segs = table(["期間", "年化", "0050", "超額/年", "超額 t", "α/年", "β", "α t", "贏 0050 月份", "最大回落"],
                     seg_rows)
        sections.append(f"<h2>{html.escape(base['name'])}</h2><p class='muted'>{html.escape(base['description'])}</p>"
                        f"<h3>參數微調（樣本外 2019–）</h3>{grid}<h3>分段（原設定）</h3>{segs}")
    dsr_note = ""
    if trial_info:
        n_same, n_all, var = trial_info
        dsr = {k: deflated_sharpe(r, n_all, var) for k, r in originals.items()}
        dsr_note = (f"<p>嘗試紀錄目前共 {n_all} 個設定（各設定月 Sharpe 變異數 {var:.4f}）。扣掉試了這麼多次的運氣後，"
                    "原設定的 Deflated Sharpe：" + "、".join(f"{html.escape(k)} {v * 100:.0f}%" for k, v in dsr.items()
                                                          if v is not None) + "</p>")
        print(f"嘗試 {n_all} 次；DSR：" + "、".join(f"{k} {v * 100:.0f}%" for k, v in dsr.items() if v is not None))
    page = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>策略穩健性檢查</title>
<style>{CSS}</style></head><body><main>
<h1>策略穩健性檢查</h1>
<p class="muted">超額 = 策略月報酬 − {ETF}，年化平均；t 用 Newey-West（lag 3）。α、β 是策略對 {ETF} 的迴歸，
α t 大於 2 左右才算統計上站得住。學術研究用。</p>
{dsr_note}{''.join(sections)}
</main></body></html>"""
    REPORTS.mkdir(exist_ok=True)
    out = REPORTS / f"{a.out}.html"
    out.write_text(page, encoding="utf-8")
    pd.set_option("display.width", 200)
    print(pd.DataFrame(console).to_string(index=False))
    print(f"→ {out}")


if __name__ == "__main__":
    main()
