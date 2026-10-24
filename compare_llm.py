# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=15", "anthropic>=0.70", "openai>=1.50"]
# ///
"""比較不同模型寫的個股摘要：同一份事實表各跑一次，再核對輸出裡的數字與期別是否都在事實表內。

用法:
  uv run compare_llm.py                       # reports/*.json × 預設模型，已有輸出的組合不重跑
  uv run compare_llm.py --models grok-4.3     # 只跑指定模型
  uv run compare_llm.py --force               # 已有輸出也重跑
  uv run compare_llm.py --check-only          # 不呼叫 API，只重新核對 logs/llm_compare/ 裡已有的輸出

Claude 需要 ANTHROPIC_API_KEY 或 `ant auth login`；Grok 需要 XAI_API_KEY。
輸出 logs/llm_compare/{code}_{asof}.{model}.json 與 logs/llm_compare/summary.md。
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

from report import NARRATIVE_SCHEMA, REPORTS, SYSTEM

OUT_DIR = Path(__file__).parent / "logs" / "llm_compare"
# $/百萬 token（輸入, 輸出）
PRICES = {"claude-opus-5-5": (4.00, 20.00), "claude-sonnet-5-5": (2.00, 10.00), "claude-haiku-5-5": (0.10, 0.50),
          "grok-4.3": (1.25, 2.50)}
DEFAULT_MODELS = ["claude-sonnet-5-5", "claude-haiku-5-5"]

NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
PERIOD = re.compile(r"\d{4}Q[1-4]|\d{4}-\d{2}(?:-\d{2})?|\d{4}\s*年\s*\d{1,2}\s*月|\d{4}\s*年\s*第?\s*[一二三四1-4]\s*季")


def call_claude(model: str, facts: dict) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    kw = dict(model=model, max_tokens=16000, system=SYSTEM,
              output_config={"effort": "medium", "format": {"type": "json_schema", "schema": NARRATIVE_SCHEMA}},
              messages=[{"role": "user", "content": json.dumps(facts, ensure_ascii=False)}])
    if model == "claude-haiku-5-5":  # Haiku 5.5 沒有伺服器端備援
        resp = client.messages.create(**kw)
    else:
        resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
    return {"model": resp.model, "stop_reason": resp.stop_reason,
            "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
            "text": next((b.text for b in resp.content if b.type == "text"), "")}


def call_grok(model: str, facts: dict) -> dict:
    from openai import OpenAI
    client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ["XAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model, max_tokens=16000, reasoning_effort="medium",
        response_format={"type": "json_schema",
                         "json_schema": {"name": "narrative", "schema": NARRATIVE_SCHEMA, "strict": True}},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}])
    u = resp.usage
    details = getattr(u, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) or 0
    # xAI 文件沒寫 completion_tokens 是否含推理 token，用 total_tokens 判斷；推理 token 以輸出價計費
    out = u.completion_tokens + (reasoning if u.total_tokens == u.prompt_tokens + u.completion_tokens + reasoning else 0)
    choice = resp.choices[0]
    return {"model": resp.model, "stop_reason": "refusal" if getattr(choice.message, "refusal", None) else choice.finish_reason,
            "input_tokens": u.prompt_tokens, "output_tokens": out, "reasoning_tokens": reasoning,
            "text": choice.message.content or ""}


def call(model: str, facts: dict) -> dict:
    return call_grok(model, facts) if model.startswith("grok") else call_claude(model, facts)


def walk(obj, key=""):
    """事實表裡所有 (key, 值)。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, k)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v, key)
    else:
        yield key, obj


def allowed_values(facts: dict) -> tuple[list[float], set[str]]:
    nums, periods = [], set()
    for key, v in walk(facts):
        nums += [float(x) for x in re.findall(r"\d+", key)]  # ret_12m、high_52w 這類視窗長度
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (int, float)):
            nums += [v, abs(v)]
            if abs(v) <= 2:  # 比率或百分位，常被寫成百分比
                nums += [v * 100, abs(v) * 100]
        elif isinstance(v, str):
            periods |= {_norm_period(p) for p in PERIOD.findall(v)}
            nums += [float(x) for x in NUM.findall(v.replace(",", "")) if not PERIOD.fullmatch(x)]
    return nums, periods


def _norm_period(p: str) -> str:
    p = p.replace(" ", "")
    if m := re.fullmatch(r"(\d{4})年(\d{1,2})月", p):
        return f"{m[1]}-{int(m[2]):02d}"
    if m := re.fullmatch(r"(\d{4})年第?([一二三四1-4])季", p):
        return f"{m[1]}Q{'一二三四'.find(m[2]) + 1 if m[2] in '一二三四' else m[2]}"
    return p


def check(narr: dict, facts: dict) -> list[dict]:
    """回傳輸出中找不到出處的數字與期別。"""
    nums, periods = allowed_values(facts)
    month_periods = {p[:7] for p in periods}
    issues = []
    for field, val in narr.items():
        for i, s in enumerate([val] if isinstance(val, str) else val):
            where = field if isinstance(val, str) else f"{field}[{i}]"
            for p in PERIOD.findall(s):
                q = _norm_period(p)
                if q not in periods and q not in month_periods:
                    issues.append({"where": where, "kind": "期別", "token": p, "text": s})
            for tok in NUM.findall(PERIOD.sub(" ", s)):
                x = float(tok.replace(",", ""))
                dec = len(tok.split(".")[1]) if "." in tok else 0
                tol = 0.5 * 10 ** -dec + 1e-9  # 容許四捨五入到最後一位
                if not any(abs(x - v) <= tol for v in nums):
                    issues.append({"where": where, "kind": "數字", "token": tok, "text": s})
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(PRICES))
    ap.add_argument("--force", action="store_true", help="已有輸出也重跑")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, details = [], []
    for fp in sorted(REPORTS.glob("*.json")):
        facts = json.loads(fp.read_text())
        for model in args.models:
            p_in, p_out = PRICES[model]
            out = OUT_DIR / f"{fp.stem}.{model}.json"
            if not args.check_only and (args.force or not out.exists()):
                t0 = time.time()
                rec = call(model, facts)
                rec = {**rec, "seconds": round(time.time() - t0, 1)}
                out.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
                print(f"{fp.stem} {model}: {rec['seconds']}s")
            elif out.exists():
                rec = json.loads(out.read_text())
            else:
                continue
            cost = (rec["input_tokens"] * p_in + rec["output_tokens"] * p_out) / 1e6
            if rec["stop_reason"] == "refusal":
                rows.append(f"| {fp.stem} | {model} | 拒絕 | | | {rec['seconds']} | ${cost:.4f} |")
                continue
            try:
                narr = json.loads(rec["text"])
            except json.JSONDecodeError:
                rows.append(f"| {fp.stem} | {model} | 非 JSON（{rec['stop_reason']}） | | | {rec['seconds']} | ${cost:.4f} |")
                continue
            issues = check(narr, facts)
            n_claims = 1 + sum(len(narr[k]) for k in ("positives", "negatives", "watch_items"))
            rows.append(f"| {fp.stem} | {model} | {n_claims} | {sum(i['kind'] == '數字' for i in issues)} "
                        f"| {sum(i['kind'] == '期別' for i in issues)} | {rec['seconds']} | ${cost:.4f} |")
            details.append(f"## {fp.stem} · {model}\n\n```json\n{json.dumps(narr, ensure_ascii=False, indent=1)}\n```\n")
            details += [f"- **{i['kind']} `{i['token']}`**（{i['where']}）：{i['text']}" for i in issues] or ["- 沒有找不到出處的數字或期別"]
            details.append("")

    summary = ["| 事實表 | 模型 | 論點數 | 可疑數字 | 可疑期別 | 秒數 | 成本 |", "|---|---|---|---|---|---|---|", *rows,
               "", "可疑 = 事實表裡找不到（容許四捨五入、比率 ×100、正負號）。計算出來的數字也會列入，需人工判斷。", "", *details]
    (OUT_DIR / "summary.md").write_text("\n".join(summary))
    print("\n".join(rows) or "沒有可核對的輸出")
    print(f"\n詳細結果：{OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
