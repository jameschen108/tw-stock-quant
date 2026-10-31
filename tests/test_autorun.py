"""autorun.py：估價解析、資料日沒前進就不呼叫 LLM、launchd 設定內容。"""
import json

import autorun as A


def test_parse_estimate():
    assert A.parse_estimate("資料時點 2026-10-30；新候選 3 檔\n預估：5 次呼叫、輸入約 40k token，費用約 US$0.012（claude-haiku-5-5）") \
        == 0.012
    assert A.parse_estimate("資料時點 2026-10-30；新候選 0 檔，待重評 wait 0 檔\n沒有要決策的股票") == 0.0
    assert A.parse_estimate("Traceback ...") is None


def test_should_run_llm():
    assert A.should_run_llm("2026-10-30", "2026-10-22")
    assert not A.should_run_llm("2026-10-22", "2026-10-22")               # 假日或行情沒更新
    assert not A.should_run_llm("2026-10-21", "2026-10-22")
    assert A.should_run_llm("2026-10-30", None)
    assert not A.should_run_llm(None, "2026-10-22")
    assert A.should_run_llm("2026-10-30T00:00:00", "2026-10-22")


def test_last_decision_asof(tmp_path):
    f = tmp_path / "d.jsonl"
    rows = [{"model": A.MODEL_ID, "asof": "2026-10-22"}, {"model": "other", "asof": "2026-10-29"},
            {"model": A.MODEL_ID, "asof": "2026-10-28"}]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert A.last_decision_asof(f) == "2026-10-28"
    assert A.last_decision_asof(tmp_path / "missing.jsonl") is None


def test_plist_weekdays_after_close(tmp_path):
    d = A.plist_dict("/opt/homebrew/bin/uv", tmp_path)
    times = d["StartCalendarInterval"]
    assert [t["Weekday"] for t in times] == [1, 2, 3, 4, 5]
    assert all((t["Hour"], t["Minute"]) >= (17, 30) for t in times)
    assert d["ProgramArguments"][-1] == str(tmp_path / "autorun.py")
    assert d["EnvironmentVariables"]["PATH"].startswith("/opt/homebrew/bin")
