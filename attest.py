"""LLM 決策存證：證明決策紀錄在某個時間點已經是這個樣子，之後只有追加、沒有改寫。

用法:
  python attest.py stamp          # 算雜湊、附一筆到 attest/ledger.jsonl；有裝 ots 就順便蓋 OpenTimestamps 章
  python attest.py verify         # 重算現有檔案，對每筆存證檢查鏈結、前綴雜湊、輸入原文
  python attest.py push           # 把 attest/ 提交到獨立分支 attest（不碰 main）；加 --remote 才推到 origin

原理:
  - 每筆存證記下 llm_decisions.jsonl、llm_replay.jsonl 的整檔 sha256、位元組長度、行數。
    之後檔案只能追加：前 N 位元組的 sha256 必須還是當時記的值，否則 verify 失敗。
  - 每筆帶前一筆的雜湊（prev），整串成鏈，事後改舊紀錄會斷鏈。
  - 模型看到的輸入原文（llm_inputs/*.gz）靠決策紀錄裡的 input_sha256 逐檔核對，不另存一份清單。
  - 時間證明靠第三方：OpenTimestamps（比特幣區塊鏈，stamp 後約數小時要 ots upgrade）與 git push 的時間。
    只存雜湊，不含 TEJ 資料；決策原文留在 data/，不進 git。
  - 蓋章前的內容無法回溯證明，起點就是第一次 stamp。
"""
import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "paper"
ATTEST = ROOT / "attest"
LEDGER = ATTEST / "ledger.jsonl"
BRANCH = "attest"
GENESIS = "0" * 64
LOGS = ("llm_decisions.jsonl", "llm_replay.jsonl")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def file_info(path: Path) -> dict | None:
    if not path.exists():
        return None
    b = path.read_bytes()
    return {"sha256": sha256_bytes(b), "bytes": len(b), "lines": b.count(b"\n")}


def inputs_info(data: Path) -> dict:
    """所有輸入 gz 的「相對路徑 + sha256」排序後再雜湊，只當附註，核對靠決策紀錄裡的 input_sha256。"""
    root = data / "llm_inputs"
    files = sorted(root.rglob("*.gz")) if root.exists() else []
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(data).as_posix()}\t{sha256_bytes(f.read_bytes())}\n".encode())
    return {"count": len(files), "manifest_sha256": h.hexdigest()}


def record_hash(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "hash"}
    return sha256_bytes(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8"))


def load_ledger(ledger: Path = LEDGER) -> list[dict]:
    if not ledger.exists():
        return []
    return [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]


def git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def check_prefix(rec: dict, data: Path) -> list[str]:
    """這筆存證記的每個檔，目前前 bytes 位元組的雜湊是否還一樣。"""
    errs = []
    for name, info in rec["files"].items():
        if info is None:
            continue
        p = data / name
        if not p.exists():
            errs.append(f"#{rec['seq']} {name} 不見了")
            continue
        b = p.read_bytes()
        if len(b) < info["bytes"]:
            errs.append(f"#{rec['seq']} {name} 變短了（{len(b)} < {info['bytes']} 位元組），紀錄被刪過")
        elif sha256_bytes(b[:info["bytes"]]) != info["sha256"]:
            errs.append(f"#{rec['seq']} {name} 前 {info['bytes']} 位元組的雜湊不符，紀錄被改過")
    return errs


def check_inputs(data: Path) -> list[str]:
    """決策紀錄裡每筆的 input_sha256 對得上 gz 原文（只看還存在的檔，缺檔另外列出）。"""
    errs = []
    for name in LOGS:
        p = data / name
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "input_file" not in r:
                continue
            f = data.parent / r["input_file"]   # input_file 相對於 data/
            if not f.exists():
                errs.append(f"{name} 的輸入原文不見了：{r['input_file']}")
                continue
            text = gzip.decompress(f.read_bytes())
            if sha256_bytes(text)[:len(r["input_sha256"])] != r["input_sha256"]:
                errs.append(f"{name} 的輸入原文雜湊不符：{r['input_file']}")
    return errs


def verify(data: Path = DATA, ledger: Path = LEDGER) -> list[str]:
    recs = load_ledger(ledger)
    if not recs:
        return ["還沒有任何存證"]
    errs, prev = [], GENESIS
    for i, rec in enumerate(recs, 1):
        if rec["seq"] != i:
            errs.append(f"序號不連續：第 {i} 筆是 #{rec['seq']}")
        if rec["prev"] != prev:
            errs.append(f"#{rec['seq']} 的 prev 對不上前一筆，鏈斷了")
        if record_hash(rec) != rec["hash"]:
            errs.append(f"#{rec['seq']} 的內容和自己的雜湊不符")
        prev = rec["hash"]
        errs += check_prefix(rec, data)
    return errs + check_inputs(data)


def stamp(data: Path = DATA, ledger: Path = LEDGER, use_ots: bool = True) -> dict:
    """附一筆存證；現有檔案和前一筆對不上就拒絕（不自動修正），回傳新的一筆。"""
    problems = verify(data, ledger) if load_ledger(ledger) else check_inputs(data)
    if problems:
        raise SystemExit("拒絕存證，現有紀錄已經不對：\n  " + "\n  ".join(problems))
    recs = load_ledger(ledger)
    rec = {"seq": len(recs) + 1, "time": datetime.now().isoformat(timespec="seconds"), "git_head": git_head(),
           "files": {n: file_info(data / n) for n in LOGS}, "inputs": inputs_info(data),
           "prev": recs[-1]["hash"] if recs else GENESIS}
    rec["hash"] = record_hash(rec)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if use_ots:
        ots_stamp(rec, ledger.parent)
    return rec


def ots_stamp(rec: dict, folder: Path) -> None:
    if shutil.which("ots") is None:
        print("沒裝 ots，略過 OpenTimestamps（uv tool install opentimestamps-client）")
        return
    f = folder / f"{rec['seq']:04d}.sha256"
    f.write_text(rec["hash"] + "\n")
    r = subprocess.run(["ots", "stamp", str(f)], capture_output=True, text=True)
    print("OpenTimestamps 已蓋章，數小時後跑 `ots upgrade` 才會寫入區塊鏈" if r.returncode == 0
          else f"ots stamp 失敗：{r.stderr.strip()}")


def git(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


def push(remote: bool) -> None:
    """把 attest/ 提交到獨立分支 attest；不在 main，也不經 PR。"""
    if not LEDGER.exists():
        raise SystemExit("還沒有存證可提交")
    git("fetch", "origin", cwd=ROOT)
    has = lambda ref: subprocess.run(["git", "rev-parse", "--verify", "-q", ref], cwd=ROOT).returncode == 0
    tmp = Path(tempfile.mkdtemp(prefix="attest_wt_"))
    git("worktree", "add", "--detach", str(tmp), cwd=ROOT)
    try:
        if has(BRANCH):
            git("checkout", BRANCH, cwd=tmp)
        elif has(f"origin/{BRANCH}"):
            git("checkout", "-b", BRANCH, f"origin/{BRANCH}", cwd=tmp)
        else:
            git("checkout", "--orphan", BRANCH, cwd=tmp)
            subprocess.run(["git", "rm", "-rf", "-q", "."], cwd=tmp, capture_output=True)
        shutil.copytree(ATTEST, tmp / "attest", dirs_exist_ok=True)
        git("add", "attest", cwd=tmp)
        if not git("status", "--porcelain", cwd=tmp).stdout.strip():
            print("沒有新的存證要提交")
        else:
            last = load_ledger()[-1]
            git("commit", "-m", f"存證 {last['time'][:10]} 決策雜湊（第 {last['seq']} 筆）", cwd=tmp)
            print(f"已提交到本機分支 {BRANCH}")
        if remote:
            git("push", "origin", BRANCH, cwd=tmp)
            print(f"已推到 origin/{BRANCH}")
        else:
            print("還沒推到遠端；要推請加 --remote")
    finally:
        git("worktree", "remove", "--force", str(tmp), cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["stamp", "verify", "push"])
    ap.add_argument("--remote", action="store_true", help="push 時才真的推到 origin")
    a = ap.parse_args()
    if a.cmd == "stamp":
        rec = stamp()
        print(f"第 {rec['seq']} 筆：" + "、".join(f"{n} {i['lines']} 行" for n, i in rec["files"].items() if i)
              + f"，輸入原文 {rec['inputs']['count']} 檔，雜湊 {rec['hash'][:16]}")
    elif a.cmd == "verify":
        errs = verify()
        print("\n".join(errs) if errs else f"通過：{len(load_ledger())} 筆存證，鏈結、前綴、輸入原文都對得上")
        sys.exit(1 if errs else 0)
    else:
        push(a.remote)


if __name__ == "__main__":
    main()
