# 規則優先順序

- 本檔規則優先於個人全域設定（例如 `~/.claude/CLAUDE.md`）。兩者衝突時以本檔為準。

# 同步規則

- 動手改檔案之前（純問答不用），先執行 `git pull --ff-only` 同步目前分支的遠端版本。
- 在 feature branch 上，再執行 `git fetch origin main`，檢查分支落後 `main` 多少個 commit（`git rev-list --count HEAD..origin/main`）。落後時先告知使用者，不要自行合併。
- 若工作目錄有未 commit 的變更，或 pull 無法 fast-forward（有衝突或分歧），不要強行處理，先告知使用者再決定。

# 分支規則

- 不要直接 commit 或 push 到 `main`，一律開 feature branch，透過 PR 合併。
- 一個 PR 對應一條分支。PR 合併後這條分支就結束，不要接著用它做別的事；新工作從最新的 `origin/main` 切新分支（`git fetch origin` 後 `git switch -c <名稱> origin/main`）。
- 分支名稱用功能描述，加前綴 `feat/`、`fix/`、`docs/`，例如 `feat/theme-page`、`fix/delist-return`。名稱要和分支內容對得上。
- 分支要跟上 `main` 時用 merge（`git merge origin/main`），不要 rebase 已推送的分支。專案歷史本來就是 merge 為主。
- 不要 force push（含 `--force-with-lease`），除非使用者明確要求。
- PR 合併後，已合併的分支（本地和遠端）要清掉，但刪遠端分支前先問使用者；別人建立的分支一律不要自行刪除。
- 不要改寫已推送的 commit（amend、reset 後再推）。
- 合併衝突如果牽涉資料邏輯（回測、因子、策略設定），先說明衝突內容再處理，不要只求能編譯。

# Commit 前檢查

- 改了 `.py` 檔，commit 前先跑下面兩項，與 CI（`.github/workflows/ci.yml`）一致，任何一項失敗都不要 commit：
  - `uvx ruff check --select E9,F63,F7,F82 .`
  - `uv run --isolated --no-project --python 3.12 --with pytest --with pandas --with numpy --with pyarrow pytest tests -q`
- 改了選股、回測、因子、報酬計算等核心邏輯，要在 `tests/` 補或更新測試，測試用合成行情，不依賴 `data/`。
- 改了會影響回測結果的邏輯，在 commit／PR 說明對現有策略年化報酬的影響，不要只說「已修正」。
- 前端改動要實際開起來看過（`frontend/`），不能只靠測試通過。

# 不要碰的東西

- 不要 commit、顯示或貼出 `.env` 及任何 API key、token。`.env` 已在 `.gitignore`，不要改掉這條。
- `data/`、`logs/`、`.claude/` 不進 repo（TEJ 授權資料與對話紀錄）。`reports/` 底下以四位數股票代號開頭的個股報告含財報數字，也不進 repo。`git add` 時不要用 `-f` 繞過 `.gitignore`。
- 這些紀錄檔只追加、不改寫、不刪除、不重排，因為它們是樣本外證據：
  - `data/paper/llm_decisions.jsonl`、`data/paper/llm_replay.jsonl`、`data/paper/llm_inputs/`
  - `data/strategy/trials.jsonl`（回測嘗試紀錄）
  - `evals/analyst/holdout_runs.jsonl`
- 不要重跑已經記錄過的保留集評測（`--split holdout`）：每個 prompt 加模型只能跑一次。
- 會呼叫 Claude API 的指令（`llm_trader.py`、`llm_replay.py`、分析師評測、`report.py` 不加 `--no-llm`）會花錢：先估費用告知使用者，確認後再跑。預設用 Sonnet 或 Haiku，不要用 Opus。
- 不要為了讓測試或圖表好看而調整資料、參數或回測區間。結果不如預期就照實回報。

# Commit 規則

- Commit 標題使用繁體中文，以一句話簡潔描述本次變更。
- Description（詳細說明／本文）可有可無，有需要補充背景或細節時再填寫。
- 新增多個功能時，依功能分批 commit，每個 commit 只包含一個功能及其必要的相關變更。

範例：`修正回測報酬率計算錯誤`

# PR 規則

- PR 標題比照 commit 規則：繁體中文，一句話簡潔描述。
- PR 描述使用繁體中文，說明變更內容、動機與影響範圍；有需要時補充測試方式，條列即可，不必冗長。
