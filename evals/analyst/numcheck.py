"""評測的數字比對：從回答抽出數字並判斷正負號，再和標準答案比對。

中文回答常寫「下跌 5%」「賣超 7 萬張」而不寫負號，所以正負號看兩處：數字前的負號，以及同一句裡緊接在數字前的方向詞。
都看不出方向時不判錯（只比絕對值）；明確寫反方向（例如答案 −5，回答「上漲 5%」或「+5%」）才算沒命中。
"""
import re

NUM_RE = re.compile(r"(?<![\d.])([+\-−–]?)(\d[\d,]*\.?\d*)")
NEG_WORDS = ("跌", "減", "衰退", "下滑", "下降", "虧", "賣超", "淨賣", "流出", "負")
POS_WORDS = ("漲", "增", "成長", "上升", "買超", "淨買", "流入")
NEUTRAL = ("漲跌", "增減", "買賣超", "正負", "升降",       # 兩個方向都有
           "跌破", "跌至", "跌到", "漲至", "漲到", "站上")   # 後面接的是價位，不是變動
WINDOW = 8                                                # 往前看幾個字
STOP = "，。；：,;:、\n()（）|"


def _direction(before: str) -> int:
    """數字前同一句的文字 → 1 / −1 / 0（看不出來）。"""
    for ch in STOP:
        before = before.rsplit(ch, 1)[-1]
    before = before.replace("負成長", "衰退")
    for w in NEUTRAL:
        before = before.replace(w, "")
    neg = any(w in before for w in NEG_WORDS)
    pos = any(w in before for w in POS_WORDS)
    return -1 if neg and not pos else 1 if pos and not neg else 0


def numbers_in(text: str) -> list[tuple[float, int]]:
    """回答裡的數字 → [(絕對值, 方向)]，方向 1 / −1 / 0。日期裡的「-」（2026-10-27）不當負號。"""
    out, prev_end = [], 0
    for m in NUM_RE.finditer(text):
        sign, digits = m.groups()
        try:
            v = float(digits.replace(",", ""))
        except ValueError:
            continue
        if sign and sign in "-−–":
            d = -1
        elif sign == "+":
            d = 1
        else:
            d = _direction(text[max(prev_end, m.start() - WINDOW):m.start()])
        out.append((v, d))
        prev_end = m.end()
    return out


def hit(nums: list[tuple[float, int]], expected: float, tol: float) -> bool:
    """有沒有一個數字在容差內，且方向沒有和標準答案相反。"""
    want = (expected > 0) - (expected < 0)
    return any(abs(v - abs(expected)) <= tol + 1e-9 and (d == 0 or want == 0 or d == want) for v, d in nums)
