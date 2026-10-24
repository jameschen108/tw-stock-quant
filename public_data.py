# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""把 OpenAPI 快照（fetch_public.py）轉成 TEJ 欄位格式，補上 TEJ 匯出檔之後的新月份 / 新季度。

用法: uv run public_data.py      # 跟 TEJ 重疊的期別逐欄比對（驗證對應關係）

build_panel.py 會呼叫 merge_rev / merge_fin：TEJ 有的 (code, period) 一律用 TEJ，只補 TEJ 沒有的。
可用日 avail_date = 該期第一次出現在快照的日期（快照是收盤後抓的，當天即可用）。

對應方式:
  月營收   當月營收、年增、月增直接取；近 3 月 / 近 12 月年增、與歷史最高比用 TEJ + 公開的月營收序列重算
  季財報   OpenAPI 是年初累計數，減去同年前幾季（先找快照，沒有就用 TEJ 單季加總）得單季
           毛利率、營益率、淨利率、負債比自己算；ROE = 單季繼續營業單位損益 / 期初期末總權益平均（與 TEJ ROE(A) 相同）
           季底普通股市值 = 基本資料的已發行股數 × 季底收盤
           現金流量表不在 OpenAPI：營業現金流由 fetch_cashflow.py 逐家查觀測站補上，還沒查到的季度留空
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
SNAP = ROOT / "data" / "raw" / "openapi"
OUT = ROOT / "data"

# 英文 / 中文鍵名統一
KEYS = {"SecuritiesCompanyCode": "公司代號", "CompanyName": "公司名稱", "Year": "年度", "Season": "季別",
        "Date": "出表日期", "IssueShares": "已發行普通股數或TDR原股發行股數",
        "ParValueOfCommonStock": "普通股每股面額"}
# 各產業版本的同義欄位，依序取第一個有值的
IS_FIELDS = {
    "rev": ["營業收入", "淨收益"],
    "gp": ["營業毛利（毛損）淨額", "營業毛利（毛損）"],
    "op": ["營業利益（損失）"],
    "ni": ["本期淨利（淨損）", "本期稅後淨利（淨損）"],
    "ni_parent": ["淨利（淨損）歸屬於母公司業主", "淨利（損）歸屬於母公司業主"],
    "ni_cont": ["繼續營業單位本期淨利（淨損）"],
    "eps": ["基本每股盈餘（元）"],
}
BS_FIELDS = {
    "assets": ["資產總計", "資產總額"],
    "liab": ["負債總計", "負債總額"],
    "capital": ["股本"],
    "equity_parent": ["歸屬於母公司業主之權益合計"],
    "equity_total": ["權益總計", "權益總額"],
    "bvps": ["每股參考淨值"],
}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def _roc_ym(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.zfill(5)
    return pd.to_datetime((s.str[:3].astype(int) + 1911).astype(str) + s.str[3:] + "01") + pd.offsets.MonthEnd(0)


def snapshots(prefix: str) -> pd.DataFrame:
    """讀 data/raw/openapi/{prefix}*/ 下所有快照，加上 snap_date。"""
    frames = []
    for d in sorted(SNAP.glob(f"{prefix}*")):
        for p in sorted(d.glob("*.json")):
            df = pd.DataFrame(json.loads(p.read_text())).rename(columns=KEYS)
            if df.empty:
                continue
            df["snap_date"] = pd.Timestamp(p.stem)
            df["template"] = d.name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["code"] = df["公司代號"].astype(str).str.strip()
    return df


def first_seen(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """同一 (code, period) 取最新快照的數值、最早快照的日期當可用日。"""
    df = df.sort_values("snap_date")
    avail = df.groupby(keys)["snap_date"].min().rename("avail_date")
    return df.drop_duplicates(keys, keep="last").merge(avail, on=keys)


def _pick(df: pd.DataFrame, names: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for n in names:
        if n in df:
            out = out.fillna(_num(df[n]))
    return out


# ---------------------------------------------------------------- 月營收

def public_rev() -> pd.DataFrame:
    s = snapshots("rev_")
    if s.empty:
        return s
    s["period"] = _roc_ym(s["資料年月"])
    s = first_seen(s, ["code", "period"])
    note = s["備註"].astype(str).str.strip()
    return pd.DataFrame({
        "code": s["code"], "name": s["公司名稱"], "period": s["period"], "avail_date": s["avail_date"],
        "以合併為主單月營收(千元)": _num(s["營業收入-當月營收"]),
        "以合併為主單月營收成長率％": _num(s["營業收入-去年同月增減(%)"]),
        "單月營收與上月比％": _num(s["營業收入-上月比較增減(%)"]),
        "以合併為主累計營收(千元)": _num(s["累計營業收入-當月累計營收"]),
        "營收變化說明": note.where(~note.isin(["-", "", "nan"])),
    })


def merge_rev(tej: pd.DataFrame) -> pd.DataFrame:
    pub = public_rev()
    if pub.empty:
        return tej.assign(source=tej.get("source", "tej"))
    have = set(zip(tej["code"], tej["period"]))
    new = pub[[k not in have for k in zip(pub["code"], pub["period"])]].copy()
    if new.empty:
        return tej.assign(source=tej.get("source", "tej"))
    # 用 TEJ + 新月份的營收序列重算累計類欄位
    col = "以合併為主單月營收(千元)"
    ser = pd.concat([tej[["code", "period", col]], new[["code", "period", col]]]).sort_values(["code", "period"])
    ser = ser.set_index(["code", "period"])[col]

    def lookup(code, period, months):
        return ser.get((code, period - pd.offsets.MonthEnd(months)), np.nan) if months else ser.get((code, period))

    rows = []
    for r in new.itertuples():
        last3 = [lookup(r.code, r.period, k) for k in range(3)]
        prev3 = [lookup(r.code, r.period, k + 12) for k in range(3)]
        last12 = [lookup(r.code, r.period, k) for k in range(12)]
        prev12 = [lookup(r.code, r.period, k + 12) for k in range(12)]
        hist = ser.loc[r.code] if r.code in ser.index.get_level_values(0) else pd.Series(dtype=float)
        hist = hist[hist.index <= r.period]
        rows.append({
            "近3月累計營收成長率": (np.sum(last3) / np.sum(prev3) - 1) * 100 if np.all(np.isfinite(last3 + prev3)) else np.nan,
            "近12月累計營收成長率": (np.sum(last12) / np.sum(prev12) - 1) * 100 if np.all(np.isfinite(last12 + prev12)) else np.nan,
            "與歷史最高單月營收比%": (getattr(r, "_5") / hist.max() - 1) * 100 if len(hist) else np.nan,
        })
    new = pd.concat([new.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    new["avail_imputed"] = False
    new["source"] = "public"
    out = pd.concat([tej.assign(source=tej.get("source", "tej")), new], ignore_index=True)
    return out.sort_values(["code", "period"]).reset_index(drop=True)


# ---------------------------------------------------------------- 季財報

def public_fin_cum() -> pd.DataFrame:
    """快照裡的年初累計數（每 code × 期別一列）。"""
    is_ = snapshots("is_")
    bs = snapshots("bs_")
    if is_.empty or bs.empty:
        return pd.DataFrame()
    is_ = is_[_num(is_["年度"]).notna() & _num(is_["季別"]).notna()].copy()   # 櫃買部分版本只有一列空白
    bs = bs[_num(bs["年度"]).notna() & _num(bs["季別"]).notna()].copy()
    for df in (is_, bs):
        df["period"] = pd.to_datetime((_num(df["年度"]).astype(int) + 1911).astype(str) + "-"
                                      + (_num(df["季別"]).astype(int) * 3).astype(str) + "-01") + pd.offsets.MonthEnd(0)
    is_ = first_seen(is_, ["code", "period"])
    bs = first_seen(bs, ["code", "period"])
    a = pd.DataFrame({"code": is_["code"], "period": is_["period"], "avail_is": is_["avail_date"],
                      "template": is_["template"],
                      **{k: _pick(is_, v) for k, v in IS_FIELDS.items()}})
    b = pd.DataFrame({"code": bs["code"], "period": bs["period"], "avail_bs": bs["avail_date"],
                      **{k: _pick(bs, v) for k, v in BS_FIELDS.items()}})
    df = a.merge(b, on=["code", "period"], how="inner")
    df["avail_date"] = df[["avail_is", "avail_bs"]].max(axis=1)
    return df.drop(columns=["avail_is", "avail_bs"])


TEJ_FLOW = {"rev": "營業收入淨額", "gp": "營業毛利", "op": "營業利益", "ni": "合併總損益",
            "ni_parent": "歸屬母公司淨利（損）", "ni_cont": "繼續營業單位損益", "eps": "每股盈餘"}


def issued_shares() -> pd.DataFrame:
    s = snapshots("basic_")
    if s.empty:
        return s
    s["shares_k"] = _num(s["已發行普通股數或TDR原股發行股數"]) / 1000
    return s[["code", "snap_date", "shares_k"]].dropna().sort_values("snap_date")


def quarter_end_close(codes, periods) -> pd.DataFrame:
    px = pd.read_parquet(OUT / "prices_daily.parquet", columns=["date", "code", "close"],
                         filters=[("code", "in", sorted(set(codes)))]).dropna()
    q = pd.DataFrame({"code": list(codes), "period": list(periods)}).drop_duplicates()
    return pd.merge_asof(q.sort_values("period"), px.sort_values("date"), left_on="period", right_on="date",
                         by="code", direction="backward", tolerance=pd.Timedelta(days=10))[["code", "period", "close"]]


def public_cfo(codes: pd.Series, periods: pd.Series, tej: pd.DataFrame) -> np.ndarray:
    """單季營業現金流（千元）= 年初累計（fetch_cashflow.py）− 同年前幾季累計（先找同來源，沒有就用 TEJ 單季加總）。"""
    f = OUT / "cashflow_public.parquet"
    if not f.exists():
        return np.full(len(codes), np.nan)
    cf = pd.read_parquet(f).set_index(["code", "period"])["cfo_cum"]
    tq = tej[tej["單季(Q)/單半年(H)"].eq("Q")].set_index(["code", "period"])["來自營運之現金流量"]
    out = []
    for c, p in zip(codes, periods):
        cum = cf.get((c, p), np.nan)
        if np.isnan(cum) or p.quarter == 1:
            out.append(cum)
            continue
        prev = cf.get((c, p - pd.offsets.QuarterEnd(1)), np.nan)
        if np.isnan(prev):
            qs = [tq.get((c, pd.Timestamp(year=p.year, month=3 * k, day=1) + pd.offsets.MonthEnd(0)), np.nan)
                  for k in range(1, p.quarter)]
            prev = sum(qs) if not any(np.isnan(qs)) else np.nan
        out.append(cum - prev)
    return np.array(out, dtype=float)


def merge_fin(tej: pd.DataFrame) -> pd.DataFrame:
    cum = public_fin_cum()
    if cum.empty:
        return tej.assign(source=tej.get("source", "tej"))
    have = set(zip(tej["code"], tej["period"]))
    new = cum[[k not in have for k in zip(cum["code"], cum["period"])]].copy()
    if new.empty:
        return tej.assign(source=tej.get("source", "tej"))

    # 同年前幾季的累計：先找公開快照，沒有就用 TEJ 單季加總（金融業的營收定義跟 TEJ 不同，不能混用）
    tq = tej[tej["單季(Q)/單半年(H)"].eq("Q")]
    flows = list(IS_FIELDS)
    new = new.reset_index(drop=True)
    prior, src = [], []
    for r in new.itertuples():
        q = r.period.quarter
        if q == 1:
            prior.append({k: 0.0 for k in flows}); src.append("none")
            continue
        prev_p = r.period - pd.offsets.QuarterEnd(1)
        c = cum[(cum["code"] == r.code) & (cum["period"] == prev_p)]
        if len(c):
            prior.append({k: c[k].iloc[0] for k in flows}); src.append("public")
            continue
        t = tq[(tq["code"] == r.code) & (tq["period"].dt.year == r.period.year) & (tq["period"] < r.period)]
        ok = len(t) == q - 1
        prior.append({k: t[TEJ_FLOW[k]].sum() if ok else np.nan for k in flows}); src.append("tej" if ok else "missing")
    prior = pd.DataFrame(prior)
    for k in flows:
        new[k] = new[k] - prior[k]
    new["prior_src"] = src

    # 上一季總權益（TEJ 的 ROE(A) = 繼續營業單位損益 / 期初期末總權益平均）
    prev_eq = []
    for r in new.itertuples():
        prev_p = r.period - pd.offsets.QuarterEnd(1)
        c = cum[(cum["code"] == r.code) & (cum["period"] == prev_p)]
        t = tej[(tej["code"] == r.code) & (tej["period"] == prev_p)]
        prev_eq.append(c["equity_total"].iloc[0] if len(c) else
                       (t["股東權益總額"].iloc[0] if len(t) else np.nan))
    new["avg_eq"] = (new["equity_total"] + pd.Series(prev_eq)) / 2

    sh = issued_shares()
    new = pd.merge_asof(new.sort_values("avail_date"), sh.rename(columns={"snap_date": "avail_date"}),
                        on="avail_date", by="code", direction="backward")
    qc = quarter_end_close(new["code"], new["period"])
    new = new.merge(qc, on=["code", "period"], how="left")
    last_tej = tej.sort_values("period").groupby("code").last()

    out = pd.DataFrame({
        "code": new["code"], "period": new["period"], "avail_date": new["avail_date"], "avail_imputed": False,
        "單季(Q)/單半年(H)": "Q", "合併(Y/N)": "Y",
        "營業收入淨額": new["rev"], "營業毛利": new["gp"], "營業利益": new["op"], "合併總損益": new["ni"],
        "歸屬母公司淨利（損）": new["ni_parent"], "每股盈餘": new["eps"],
        "營業毛利率": new["gp"] / new["rev"] * 100, "營業利益率": new["op"] / new["rev"] * 100,
        "稅後淨利率": new["ni"] / new["rev"] * 100,
        "ROE(A)－稅後": new["ni_cont"] / new["avg_eq"] * 100, "繼續營業單位損益": new["ni_cont"],
        "股東權益總額": new["equity_total"],
        "資產總額": new["assets"], "負債總額": new["liab"], "負債比率": new["liab"] / new["assets"] * 100,
        "普通股股本": new["capital"], "股本": new["capital"], "母公司股東權益合計": new["equity_parent"],
        "每股淨值(B)": new["bvps"], "季底普通股市值": new["shares_k"] * new["close"],
        "來自營運之現金流量": public_cfo(new["code"], new["period"], tej),
        "TEJ主產業代碼": new["code"].map(last_tej["TEJ主產業代碼"]) if "TEJ主產業代碼" in last_tej else np.nan,
        "name": new["code"].map(last_tej["name"]) if "name" in last_tej else np.nan,
        "source": "public",
    })
    # 前幾季對不起來（TEJ 缺季、半年報）的，流量欄位不可信
    flow_cols = ["營業收入淨額", "營業毛利", "營業利益", "合併總損益", "歸屬母公司淨利（損）", "每股盈餘",
                 "營業毛利率", "營業利益率", "稅後淨利率", "ROE(A)－稅後"]
    out.loc[(new["prior_src"] == "missing").values, flow_cols] = np.nan
    # 金融業：營收 / 毛利定義和 TEJ 不同，前幾季來自 TEJ 時只保留淨利、EPS
    fin_tpl = ~new["template"].str.startswith("is_ci").values
    out.loc[fin_tpl & (new["prior_src"] == "tej").values,
            ["營業收入淨額", "營業毛利", "營業利益", "營業毛利率", "營業利益率", "稅後淨利率"]] = np.nan
    res = pd.concat([tej.assign(source=tej.get("source", "tej")), out], ignore_index=True)
    return res.sort_values(["code", "period"]).reset_index(drop=True)


# ---------------------------------------------------------------- 驗證

def compare(pub: pd.DataFrame, tej: pd.DataFrame, cols: list[str], label: str):
    m = pub.merge(tej, on=["code", "period"], suffixes=("_pub", "_tej"))
    print(f"\n[{label}] 重疊 {len(m)} 筆（期別 {sorted(m['period'].dt.strftime('%Y-%m').unique())}）")
    for c in cols:
        a, b = m[f"{c}_pub"], m[f"{c}_tej"]
        ok = a.notna() & b.notna()
        tol = np.maximum(0.005 * b.abs(), 0.051)
        hit = ((a - b).abs() <= tol)[ok]
        print(f"  {c}: 兩邊都有 {ok.sum()}，一致率 {hit.mean():.1%}，中位數差 {(a - b)[ok].abs().median():.3f}")
        bad = m[ok & ~((a - b).abs() <= tol)]
        if len(bad):
            print("    例:", bad[["code", f"{c}_pub", f"{c}_tej"]].head(3).to_dict("records"))


def main():
    tej_rev = pd.read_parquet(OUT / "tej_rev.parquet")
    pub = public_rev()
    compare(pub, tej_rev, ["以合併為主單月營收(千元)", "以合併為主單月營收成長率％", "單月營收與上月比％"], "月營收")

    tej_fin = pd.read_parquet(OUT / "tej_fin.parquet")
    # 假裝 TEJ 沒有快照那一季，用 merge_fin 算出來，再跟 TEJ 實際值比
    cum = public_fin_cum()
    target = set(zip(cum["code"], cum["period"]))
    held_out = tej_fin[[k not in target for k in zip(tej_fin["code"], tej_fin["period"])]]
    merged = merge_fin(held_out)
    pub_fin = merged[merged["source"] == "public"]
    compare(pub_fin, tej_fin, ["營業收入淨額", "歸屬母公司淨利（損）", "每股盈餘", "營業毛利率", "營業利益率",
                               "稅後淨利率", "ROE(A)－稅後", "負債比率", "母公司股東權益合計", "季底普通股市值", "來自營運之現金流量"], "季財報")


if __name__ == "__main__":
    main()
