"""測試用的合成行情：真實資料（data/）不在 repo 裡，CI 上只能用假資料測核心邏輯。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import strategy as S  # noqa: E402

DAYS = pd.bdate_range("2025-01-02", periods=60)
CODES = ["1101", "1102", "1103", "1104", S.ETF]


def make_prices(daily_ret: dict[str, float] | None = None, start_px: float = 10.0) -> pd.DataFrame:
    """每檔每天固定報酬的長表（date、code、close、ret、ret_suspect）；daily_ret 沒給的代號每天 +0.1%。"""
    rows = []
    for code in CODES:
        r = (daily_ret or {}).get(code, 0.001)
        close = start_px * np.cumprod(np.full(len(DAYS), 1 + r))
        rets = np.r_[np.nan, close[1:] / close[:-1] - 1]
        rows.append(pd.DataFrame({"date": DAYS, "code": code, "close": close, "ret": rets, "ret_suspect": False}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def market(tmp_path, monkeypatch):
    """把 strategy.OUT 指到暫存資料夾，寫入合成的 prices_daily.parquet；回傳寫入行情並取得 rets 的函式。"""
    monkeypatch.setattr(S, "OUT", tmp_path)

    def build(px: pd.DataFrame | None = None) -> pd.DataFrame:
        px = make_prices() if px is None else px
        px.to_parquet(tmp_path / "prices_daily.parquet", index=False)
        return S.clean_returns(px)

    return build


def zero_cost_cfg(**position) -> dict:
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in S.DEFAULTS.items()}
    cfg.update(id="test", name="test", description="", capital=1_000_000)
    cfg["cost"] = {"fee_rate": 0.0, "fee_discount": 1.0, "min_fee": 0, "tax": 0.0}
    cfg["position"] = {**cfg["position"], **position}
    return cfg


def fixed_signals(weights: dict[pd.Timestamp, pd.Series]) -> dict:
    return {s: {"tw": w, "tw_full": w, "rinfo": {}, "pool": list(w.index), "picks": list(w.index)}
            for s, w in weights.items()}
