# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15"]
# ///
"""把 TEJ 匯出的月營收 / 季財報 CSV 整理成 parquet，並建立按發布日對齊的月底面板。

用法: uv run build_panel.py

輸出 (data/):
  tej_rev.parquet    月營收，一列 = 公司 × 營收月份
  tej_fin.parquet    季財報，一列 = 公司 × 財報季
  tej_*_agg.parquet  指數/產業彙總列（不進面板）
  panel_monthly.parquet
                     一列 = 公司 × 月底，rev_period / fin_period 指向該月底「已公布」的最新一期，
                     用 load_panel() 依需要的欄位 join 回來
"""
import io
from pathlib import Path

import pandas as pd

RAW = Path("/Users/jameschen/Project/data")
OUT = Path(__file__).parent / "data"

REV_CSV = RAW / "20261022103241.csv"
FIN_CSV = RAW / "20261022104526.csv"

# 保留為字串的欄位（其餘能轉數值的都轉）
TEXT_COLS = {
    "證券代碼", "營收變化說明", "創新高/低(歷史)", "創新高/低(近一年)", "創新高/低(同月)",
    "合併創新高/低(歷史)", "合併營收(Y/N)", "合併盈餘(Y/N)",
    "合併(Y/N)", "單季(Q)/單半年(H)", "幣別", "財報附註TEJ是否完成Y/N",
    "市場別", "交易所主產業代碼", "交易所子產業代碼", "TEJ主產業代碼", "TEJ子產業代碼",
}
# 年月型態欄位 (YYYYMM)
YM_COLS = {"歷史最高單月營收-年月", "歷史最低單月營收-年月"}
DATE_COLS = {"營收發布日", "預估營收發佈日", "預估盈餘發佈日", "盈餘發布日", "財報發布日"}
MAX_STALE_DAYS = {"rev": 186, "fin": 460}  # 面板最多沿用約 6 個月 / 15 個月前的期別
DROP_COLS = {"年月", "季別", "月份", "財報年月起日", "財報年月迄日"}  # 已轉成 period


def repair_lines(text: str, free_col: str = "營收變化說明") -> str:
    """自由文字欄內含 tab 會多出欄位，把多出的部分併回該欄。"""
    lines = text.splitlines()
    header = [c.strip() for c in lines[0].split("\t")]
    if free_col not in header:
        return text
    n, k = len(header), header.index(free_col)
    fixed = 0
    for i in range(1, len(lines)):
        f = lines[i].split("\t")
        if len(f) > n:
            extra = len(f) - n
            f[k:k + extra + 1] = [" ".join(s.strip() for s in f[k:k + extra + 1])]
            lines[i] = "\t".join(f)
            fixed += 1
    print(f"  repaired {fixed} lines with tab in {free_col}")
    return "\n".join(lines)


def read_tej(path: Path) -> pd.DataFrame:
    text = repair_lines(path.read_text(encoding="utf-16"))
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str, na_values=["."],
                     skipinitialspace=True)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    df = df.apply(lambda s: s.str.strip()).copy()
    code_name = df["證券代碼"].str.split(n=1, expand=True)
    df.insert(0, "code", code_name[0])
    df.insert(1, "name", code_name[1])
    df.insert(2, "period", pd.to_datetime(df["年月"], format="%Y%m") + pd.offsets.MonthEnd(0))
    return df


def to_types(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c in ("code", "name", "period") or c in TEXT_COLS or c in DROP_COLS:
            continue
        if c in DATE_COLS:
            df[c] = pd.to_datetime(df[c].str.replace("/", ""), format="%Y%m%d", errors="coerce")
        elif c in YM_COLS:
            df[c] = pd.to_datetime(df[c], format="%Y%m", errors="coerce") + pd.offsets.MonthEnd(0)
        else:
            num = pd.to_numeric(df[c].str.replace(",", ""), errors="coerce")
            lost = df[c].notna() & num.isna()
            if lost.any():
                print(f"  [keep str] {c}: {lost.sum()} non-numeric, e.g. {df.loc[lost, c].iloc[0]!r}")
                continue
            df[c] = num
    df[[c for c in TEXT_COLS if c in df]] = df[[c for c in TEXT_COLS if c in df]].astype("string")
    return df.drop(columns=[c for c in DROP_COLS if c in df] + ["證券代碼"]).copy()


def fin_deadline(period: pd.Series) -> pd.Series:
    """法定申報期限：Q1 5/15、Q2 8/14、Q3 11/14、Q4(年報) 隔年 3/31。"""
    y, m = period.dt.year, period.dt.month
    s = pd.Series(pd.NaT, index=period.index, dtype="datetime64[ns]")
    for mm, (dy, md) in {3: (0, "05-15"), 6: (0, "08-14"), 9: (0, "11-14"), 12: (1, "03-31")}.items():
        hit = m == mm
        s[hit] = pd.to_datetime((y[hit] + dy).astype(str) + "-" + md)
    return s


def build_rev() -> pd.DataFrame:
    df = to_types(read_tej(REV_CSV))
    # 2010–2012 沒有發布日：用法定期限「次月 10 日」
    fallback = df["period"] + pd.Timedelta(days=10)
    df.insert(3, "avail_date", df["營收發布日"].fillna(fallback))
    df.insert(4, "avail_imputed", df["營收發布日"].isna())
    return df.sort_values(["code", "period"]).reset_index(drop=True)


def build_fin() -> pd.DataFrame:
    df = to_types(read_tej(FIN_CSV))
    # 發布日早於季末的（3087、DR 股，會計年度跟 TEJ 期別對不上）也改用法定期限
    bad = df["財報發布日"].isna() | (df["財報發布日"] < df["period"])
    df.insert(3, "avail_date", df["財報發布日"].mask(bad, fin_deadline(df["period"])))
    df.insert(4, "avail_imputed", bad)
    return df.sort_values(["code", "period"]).reset_index(drop=True)


def split_agg(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """英文字母開頭的代碼是指數/產業彙總 (M1100 水泥類指數、OTC88、Y9999 加權指數)，不是公司。"""
    agg = df["code"].str.match(r"[A-Z]")
    return df[~agg].reset_index(drop=True), df[agg].reset_index(drop=True)


def latest_known(tbl: pd.DataFrame, dates: pd.DatetimeIndex, col: str) -> pd.DataFrame:
    """每個 (code, date) 找 avail_date <= date 的最新一期。

    依 avail_date 排序後取 period 的累計最大值，避免晚公布的舊期（補件/重編）蓋掉較新的期別。
    """
    ev = tbl[["code", "avail_date", "period"]].sort_values(["code", "avail_date", "period"])
    ev[col] = ev.groupby("code")["period"].cummax()
    ev = ev.drop_duplicates(["code", "avail_date"], keep="last")[["code", "avail_date", col]]
    grid = pd.MultiIndex.from_product([sorted(tbl["code"].unique()), dates],
                                      names=["code", "date"]).to_frame(index=False)
    return pd.merge_asof(grid.sort_values("date"), ev.sort_values("avail_date"),
                         left_on="date", right_on="avail_date", by="code",
                         direction="backward").drop(columns="avail_date")


def build_panel(rev: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    end = min(max(rev["avail_date"].max(), fin["avail_date"].max()), pd.Timestamp.today())
    dates = pd.date_range("2010-01-31", end, freq="ME")
    p = latest_known(rev, dates, "rev_period").merge(
        latest_known(fin, dates, "fin_period"), on=["code", "date"], how="outer")
    # 下市/停止申報後不再無限沿用舊資料
    p.loc[p["date"] - p["rev_period"] > pd.Timedelta(days=MAX_STALE_DAYS["rev"]), "rev_period"] = pd.NaT
    p.loc[p["date"] - p["fin_period"] > pd.Timedelta(days=MAX_STALE_DAYS["fin"]), "fin_period"] = pd.NaT
    p = p.dropna(subset=["rev_period", "fin_period"], how="all")
    return p.sort_values(["code", "date"]).reset_index(drop=True)


def load_panel(rev_cols=(), fin_cols=()) -> pd.DataFrame:
    """讀月底面板並 join 指定欄位，例: load_panel(["單月營收成長率％"], ["ROE(A)－稅後"])。"""
    p = pd.read_parquet(OUT / "panel_monthly.parquet")
    if rev_cols:
        r = pd.read_parquet(OUT / "tej_rev.parquet", columns=["code", "period", *rev_cols])
        p = p.merge(r.rename(columns={"period": "rev_period"}), on=["code", "rev_period"], how="left")
    if fin_cols:
        f = pd.read_parquet(OUT / "tej_fin.parquet", columns=["code", "period", *fin_cols])
        p = p.merge(f.rename(columns={"period": "fin_period"}), on=["code", "fin_period"], how="left")
    return p


def main():
    OUT.mkdir(exist_ok=True)
    from public_data import merge_fin, merge_rev
    print("rev ..."); rev, rev_agg = split_agg(build_rev())
    print("fin ..."); fin, fin_agg = split_agg(build_fin())
    # TEJ 匯出檔之後的新月份 / 新季度，用證交所、櫃買 OpenAPI 快照補上
    n0, m0 = len(rev), len(fin)
    rev, fin = merge_rev(rev), merge_fin(fin)
    print(f"  公開資料補入: 月營收 {len(rev) - n0} 筆，季財報 {len(fin) - m0} 筆")
    rev.to_parquet(OUT / "tej_rev.parquet", index=False)
    fin.to_parquet(OUT / "tej_fin.parquet", index=False)
    rev_agg.to_parquet(OUT / "tej_rev_agg.parquet", index=False)
    fin_agg.to_parquet(OUT / "tej_fin_agg.parquet", index=False)
    print("panel ..."); panel = build_panel(rev, fin)
    panel.to_parquet(OUT / "panel_monthly.parquet", index=False)
    for n, d in [("rev", rev), ("fin", fin), ("rev_agg", rev_agg), ("fin_agg", fin_agg), ("panel", panel)]:
        print(f"{n}: {d.shape}")


if __name__ == "__main__":
    main()
