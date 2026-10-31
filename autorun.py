# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""每日自動執行：收盤後跑 update.py，資料日有前進才跑 llm_trader.py run，失敗發 macOS 通知。

前向模擬的樣本斷一天就補不回來（決策晚一天，訊號日就晚一天，不回填），所以排程每個平日自動跑。

用法:
  uv run autorun.py                 # 手動跑一次（排程跑的也是這個）
  uv run autorun.py --dry-run       # update 照跑，LLM 只估價不呼叫
  uv run autorun.py --install       # 寫入 ~/Library/LaunchAgents 並載入排程（平日 18:30）
  uv run autorun.py --uninstall     # 移除排程
  uv run autorun.py --status        # 看排程是否載入、最近幾次執行結果

每次執行:
  1. update.py：失敗或健康檢查不過 → 通知、停止，不在可能有問題的資料上讓 LLM 決策
  2. 資料日沒有比上次 LLM 決策的資料日新（假日、行情還沒公布）→ 不呼叫 LLM。
     否則 wait 的股票會在同一個資料日被重問，白花錢
  3. llm_trader.py run --dry-run 估價；超過 MAX_USD 就不跑、通知（防止候選異常暴增）
  4. llm_trader.py run（Haiku），結果與費用寫進 data/autorun_runs.jsonl，完成也發通知

launchd 用 StartCalendarInterval：電腦在排定時間睡眠，醒來後會補跑一次；關機則不會補跑，
所以每次執行都記錄，--status 和 autorun_runs.jsonl 可以看出哪幾天沒有跑。
"""
import argparse
import json
import plistlib
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
LOGS = ROOT / "logs"
RUNS = DATA / "autorun_runs.jsonl"
DECISIONS = DATA / "paper" / "llm_decisions.jsonl"
LABEL = "com.tw-stock-quant.autorun"
PLIST = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
HOUR, MINUTE = 18, 30           # update.py 17:30 後才抓得到當天行情
MAX_USD = 1.0                   # 單日 LLM 估價上限（Haiku 一批 120 檔約 US$0.25）
MODEL, MODEL_ID = "haiku", "claude-haiku-5-5"


# ---------------------------------------------------------------- 判斷（可單獨測試）

def parse_estimate(text: str) -> float | None:
    """llm_trader run --dry-run 的輸出 → 預估美元；「沒有要決策的股票」→ 0。"""
    if "沒有要決策的股票" in text:
        return 0.0
    m = re.search(r"費用約 US\$([\d.]+)", text)
    return float(m.group(1)) if m else None


def last_decision_asof(path: Path = DECISIONS, model: str = MODEL_ID) -> str | None:
    """LLM 決策紀錄裡這個模型最後一次決策的資料日（只讀）。"""
    if not path.exists():
        return None
    asofs = [r["asof"] for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip())
             if r.get("model") == model]
    return max(asofs) if asofs else None


def should_run_llm(data_date: str | None, last_asof: str | None) -> bool:
    """資料日要比上次決策的資料日新，才呼叫 LLM。"""
    return data_date is not None and (last_asof is None or data_date[:10] > last_asof[:10])


def plist_dict(uv: str, root: Path = ROOT) -> dict:
    return {
        "Label": LABEL,
        "ProgramArguments": [uv, "run", "-q", str(root / "autorun.py")],
        "WorkingDirectory": str(root),
        "StartCalendarInterval": [{"Weekday": d, "Hour": HOUR, "Minute": MINUTE} for d in range(1, 6)],
        "StandardOutPath": str(root / "logs" / "autorun.out"),
        "StandardErrorPath": str(root / "logs" / "autorun.err"),
        "EnvironmentVariables": {"PATH": f"{Path(uv).parent}:/usr/bin:/bin:/usr/sbin:/sbin"},
    }


# ---------------------------------------------------------------- 執行

def notify(title: str, msg: str) -> None:
    """macOS 通知；失敗不影響主流程。"""
    script = f'display notification {json.dumps(msg)} with title {json.dumps(title)}'
    subprocess.run(["osascript", "-e", script], capture_output=True)


def sh(args: list[str], log) -> tuple[int, str]:
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    log.write(f"\n$ {' '.join(args)}  → {p.returncode}\n{out}")
    log.flush()
    return p.returncode, out


def data_date() -> str | None:
    f = DATA / "update_status.json"
    return json.loads(f.read_text()).get("last_price_date") if f.exists() else None


def record(**kw) -> None:
    with RUNS.open("a") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **kw}, ensure_ascii=False) + "\n")


def run(dry: bool) -> int:
    LOGS.mkdir(exist_ok=True)
    uv = shutil.which("uv") or "uv"
    with (LOGS / f"autorun_{datetime.now():%Y%m%d}.log").open("a") as log:
        log.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        code, out = sh([uv, "run", "-q", "update.py"], log)
        dd = data_date()
        if code != 0:
            tail = " / ".join(out.strip().splitlines()[-2:])[:180]
            notify("台股研究：資料更新失敗", tail or f"update.py 結束碼 {code}")
            record(step="update", ok=False, data_date=dd, detail=tail)
            return 1
        last = last_decision_asof()
        if not should_run_llm(dd, last):
            record(step="llm", ok=True, data_date=dd, skipped=f"資料日沒有前進（上次決策 {last}）")
            return 0
        code, out = sh([uv, "run", "-q", "llm_trader.py", "run", "--model", MODEL, "--dry-run"], log)
        est = parse_estimate(out)
        if code != 0 or est is None:
            notify("台股研究：LLM 估價失敗", out.strip().splitlines()[-1][:180] if out.strip() else str(code))
            record(step="estimate", ok=False, data_date=dd)
            return 1
        if est > MAX_USD:
            notify("台股研究：LLM 費用超過上限", f"預估 US${est:.2f} > {MAX_USD}，今天沒有跑，請手動確認")
            record(step="estimate", ok=False, data_date=dd, estimate_usd=est, skipped="超過上限")
            return 1
        if dry or est == 0:
            record(step="llm", ok=True, data_date=dd, estimate_usd=est, skipped="dry-run" if dry else "沒有候選")
            return 0
        code, out = sh([uv, "run", "-q", "llm_trader.py", "run", "--model", MODEL], log)
        m = re.search(r"實際費用 US\$([\d.]+)", out)
        cost = float(m.group(1)) if m else None
        n = len(re.findall(r"^\s+\d{4} .+: (buy|skip|wait)", out, re.M))
        if code != 0:
            notify("台股研究：LLM 決策失敗", out.strip().splitlines()[-1][:180] if out.strip() else str(code))
            record(step="llm", ok=False, data_date=dd, estimate_usd=est, cost_usd=cost, n=n)
            return 1
        notify("台股研究：今日完成", f"資料日 {dd}，LLM 決策 {n} 檔，費用 US${cost or 0:.3f}")
        record(step="llm", ok=True, data_date=dd, estimate_usd=est, cost_usd=cost, n=n)
    return 0


def install() -> None:
    uv = shutil.which("uv")
    if not uv:
        sys.exit("找不到 uv")
    LOGS.mkdir(exist_ok=True)
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    PLIST.write_bytes(plistlib.dumps(plist_dict(uv)))
    subprocess.run(["launchctl", "bootout", f"gui/{_uid()}", str(PLIST)], capture_output=True)
    p = subprocess.run(["launchctl", "bootstrap", f"gui/{_uid()}", str(PLIST)], capture_output=True, text=True)
    print(f"已寫入 {PLIST}，平日 {HOUR}:{MINUTE:02d} 執行" if p.returncode == 0 else f"載入失敗：{p.stderr}")


def uninstall() -> None:
    subprocess.run(["launchctl", "bootout", f"gui/{_uid()}", str(PLIST)], capture_output=True)
    PLIST.unlink(missing_ok=True)
    print("已移除排程")


def status() -> None:
    p = subprocess.run(["launchctl", "print", f"gui/{_uid()}/{LABEL}"], capture_output=True, text=True)
    print("排程：" + ("已載入" if p.returncode == 0 else "沒有載入"))
    if RUNS.exists():
        print("最近執行：")
        for line in RUNS.read_text().splitlines()[-10:]:
            print("  " + line)


def _uid() -> int:
    import os
    return os.getuid()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--install", action="store_true")
    g.add_argument("--uninstall", action="store_true")
    g.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.install:
        install()
    elif a.uninstall:
        uninstall()
    elif a.status:
        status()
    else:
        sys.exit(run(a.dry_run))


if __name__ == "__main__":
    main()
