"""evals/analyst/numcheck.py：評測數字比對要看正負號，但看不出方向時不判錯。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals" / "analyst"))
import numcheck as N  # noqa: E402


def ok(text: str, expected: float, tol: float = 0.1) -> bool:
    return N.hit(N.numbers_in(text), expected, tol)


def test_explicit_sign():
    assert ok("外資 20 日買賣超 -70,272 張", -70272, 1)
    assert ok("報酬 −5.0%", -5)
    assert not ok("報酬 +5.0%", -5)
    assert not ok("報酬 -5.0%", 5)


def test_direction_words():
    assert ok("近 20 日下跌 5.0%", -5)
    assert ok("外資賣超 70,272 張", -70272, 1)
    assert not ok("近 20 日上漲 5.0%", -5)
    assert not ok("外資買超 70,272 張", -70272, 1)
    assert ok("營收年減 12.3%", -12.3)
    assert not ok("營收年增 12.3%", -12.3)
    assert ok("呈現負成長 3.1%", -3.1)


def test_unknown_direction_passes():
    assert ok("外資 20 日買賣超 70,272 張", -70272, 1)    # 「買賣超」兩個方向都有
    assert ok("漲跌幅 5.0%", -5)
    assert ok("報酬為 5.0%", -5)


def test_levels_and_dates_not_signed():
    assert ok("股價跌破 100 元", 100, 0.5)
    assert ok("資料日期 2026-10-27，收盤 27", 27, 0.5)   # 日期的「-」不是負號
    assert ok("營收年減 3%，達 50.2 億", 50.2)             # 逗號後是新的一句


def test_gold_label_sets_direction():
    def ok_l(text, expected, label, tol=1):
        return N.hit(N.numbers_in(text), expected, tol, label)
    lab = "融資 20 日減少（張）"                              # 數值存正的減少量
    assert ok_l("融資近 20 日減少 47,980 張", 47980, lab)
    assert ok_l("margin_chg_20d 為 -47,980 張", 47980, lab)
    assert not ok_l("融資近 20 日增加 47,980 張", 47980, lab)
    assert ok_l("外資賣超 70,272 張", -70272, "外資 20 日買賣超（張）")
    assert ok_l("營收年增率 54.65%", 54.65, "年增率%", 0.1)
    assert ok_l("營收年減 5.2%", -5.2, "年增率%", 0.1)


def test_level_phrases_not_signed():
    assert ok("毛利率下降至 30.5%", 30.5)
    assert ok("融資餘額減少至 12,000 張", 12000, 1)
    assert ok("股價下滑到 88.5 元", 88.5)
