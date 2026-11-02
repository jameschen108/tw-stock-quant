"""決策存證：用暫存的合成紀錄測鏈結、只追加、被改寫偵測。"""
import gzip
import hashlib
import json

import pytest

import attest as T


def make_data(tmp_path, n=2):
    data = tmp_path / "paper"
    (data / "llm_inputs" / "m").mkdir(parents=True)
    rows = []
    for i in range(n):
        text = f"事實表 {i}"
        f = data / "llm_inputs" / "m" / f"{i}.txt.gz"
        f.write_bytes(gzip.compress(text.encode()))
        rows.append({"code": f"{1101 + i}", "decision": "buy", "input_file": f"paper/llm_inputs/m/{i}.txt.gz",
                     "input_sha256": hashlib.sha256(text.encode()).hexdigest()[:16]})
    (data / "llm_decisions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return data, tmp_path / "attest" / "ledger.jsonl"


def append_row(data, i):
    text = f"事實表 {i}"
    f = data / "llm_inputs" / "m" / f"{i}.txt.gz"
    f.write_bytes(gzip.compress(text.encode()))
    row = {"code": "9999", "decision": "skip", "input_file": f"paper/llm_inputs/m/{i}.txt.gz",
           "input_sha256": hashlib.sha256(text.encode()).hexdigest()[:16]}
    with (data / "llm_decisions.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_stamp_chain_and_append_only(tmp_path):
    data, ledger = make_data(tmp_path)
    r1 = T.stamp(data, ledger, use_ots=False)
    assert r1["prev"] == T.GENESIS and r1["files"]["llm_decisions.jsonl"]["lines"] == 2
    assert r1["files"]["llm_replay.jsonl"] is None
    append_row(data, 2)
    r2 = T.stamp(data, ledger, use_ots=False)
    assert r2["prev"] == r1["hash"] and r2["files"]["llm_decisions.jsonl"]["lines"] == 3
    assert T.verify(data, ledger) == []


def test_rewrite_detected(tmp_path):
    data, ledger = make_data(tmp_path)
    T.stamp(data, ledger, use_ots=False)
    p = data / "llm_decisions.jsonl"
    p.write_text(p.read_text(encoding="utf-8").replace("buy", "skip", 1), encoding="utf-8")
    assert any("被改過" in e for e in T.verify(data, ledger))
    with pytest.raises(SystemExit):
        T.stamp(data, ledger, use_ots=False)


def test_truncate_detected(tmp_path):
    data, ledger = make_data(tmp_path)
    T.stamp(data, ledger, use_ots=False)
    p = data / "llm_decisions.jsonl"
    p.write_bytes(p.read_bytes().splitlines(keepends=True)[0])
    assert any("變短" in e for e in T.verify(data, ledger))


def test_broken_chain_detected(tmp_path):
    data, ledger = make_data(tmp_path)
    T.stamp(data, ledger, use_ots=False)
    append_row(data, 2)
    T.stamp(data, ledger, use_ots=False)
    recs = T.load_ledger(ledger)
    recs[0]["time"] = "2020-01-01T00:00:00"          # 偷改第一筆的時間
    write = lambda: ledger.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    write()
    assert any("自己的雜湊不符" in e for e in T.verify(data, ledger))
    recs[0]["hash"] = T.record_hash(recs[0])         # 連雜湊一起重算，第二筆的 prev 就對不上
    write()
    assert any("鏈斷了" in e for e in T.verify(data, ledger))


def test_input_tamper_detected(tmp_path):
    data, ledger = make_data(tmp_path)
    T.stamp(data, ledger, use_ots=False)
    (data / "llm_inputs" / "m" / "0.txt.gz").write_bytes(gzip.compress("被改過".encode()))
    assert any("輸入原文雜湊不符" in e for e in T.verify(data, ledger))


def test_stamp_without_ots_is_ok(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(T.shutil, "which", lambda _: None)
    data, ledger = make_data(tmp_path)
    T.stamp(data, ledger)
    assert "沒裝 ots" in capsys.readouterr().out
