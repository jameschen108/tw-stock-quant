# 前端待辦清單

本機用的介面，把現在分散在 `reports/*.html` 和各腳本的東西集中到一個入口。現有的靜態報告保留，`update.py` 照舊產生。

## 啟動

在專案根目錄執行（設定檔 `.streamlit/config.toml` 綁定 `127.0.0.1`；含 TEJ 數字，不要用公開網址打開）：

```
uv run --with-requirements frontend/requirements.txt streamlit run frontend/app.py --server.address 127.0.0.1
```

## 限制

- **只在本機跑**：TEJ 資料授權不能再散布，所以不能部署到公開網址，也不能把含 TEJ 數字的頁面分享出去。綁定 `localhost` 就好。
- **只讀**：前端讀 `data/` 的 parquet 和 json，不改寫。例外有兩個：
  - 策略試跑會寫入 `data/strategy/trials.jsonl`（嘗試紀錄要誠實）。
  - 產生 LLM 報告會寫入 `reports/`。
- **免責文字**：每頁都要有「學術研究用。看漲 / 看跌是模型依歷史統計的分組，不保證未來報酬」。不要出現目標價、「建議買進」之類的字眼。
- **金鑰**：LLM 功能讀 `.env` 的 `ANTHROPIC_API_KEY`，不能顯示在畫面上，也不能寫進 log。個股報告預設用 Haiku，分析師問答預設用 Sonnet，每次呼叫前在畫面上標出大約的花費。
- **資料量**：`prices_daily.parquet` 有 740 萬列。讀取時用 `filters=[("code", "==", code)]` 或日期篩選，不要整檔載入；共用的大表要做快取（例如 Streamlit 的 `st.cache_data`）。

## 共用：資料狀態列

每頁頂端顯示一列資料狀態。

| 項目 | 來源 |
|---|---|
| 最新價格日、因子日、最近一次更新時間 | `data/update_status.json` 的 `last_price_date`、`factor_date`、`run_at` |
| 營收、財報新鮮度 | 同上，`rev_fresh_share`、`fin_fresh_share` |
| 更新問題 | 同上，`problems`、`fetch_ok` |
| 資料過舊提醒 | `report.freshness_warning()` 回傳字串時顯示 |
| 休市日 | `update.CLOSED`（例如 10/23 颱風休市） |

## 頁 1：總覽

| 功能 | 來源 |
|---|---|
| 觀察名單表：收盤、1 日 / 5 日報酬、短中長期標籤、綜合分數百分位與變化、外資持股、融資、法人 | `daily.watch_rows(codes, prev, first_run)`；名單用 `daily.read_watchlist()`，上次狀態在 `data/daily_state.json` |
| 提醒（漲跌超過 7%、綜合分數變化 15 分以上、外資 5 日變化 1 點以上、融資 5 日變化 15% 以上、標籤改變）與新公布 | 同上，每列的 `alerts`、`events` |
| 編輯觀察名單 | 讀寫 `watchlist.txt`（一行一個代號）。這是唯一允許前端寫入的設定檔 |
| 模型選股清單（短中長期都看漲） | `daily.model_picks()` → `(calib, DataFrame)` |
| 財報季速覽：已公布家數、各家營收 / 毛利率 / EPS 名次、依產業彙總 | `daily.season_rows(watch)` → `(ss, cur, ind)` |
| 多策略模擬摘要（見頁 4） | `paper.load_strategy_summaries()` |

注意：`daily.watch_rows` 會算出新的狀態，但只有 `daily.py` 正式執行時才寫回 `daily_state.json`。前端只讀不寫，否則「與上次比」會被前端打開的次數攪亂。

## 頁 2：個股

先輸入代號或名稱：`analyst.find_stock(query)`，或直接用 `analyst.names()` 做下拉選單。

| 區塊 | 來源 |
|---|---|
| 全部事實（一次取得） | `report.build_facts(code, asof)` → `(facts, series)`；`asof` 預設最新價格日，可以選過去日期做時點回溯 |
| 價格走勢、52 週高低、均線差距 | `series["px"]`、`facts["price"]` |
| 短中長期看法：標籤、百分位、樣本外歷史校準、可靠度 | `facts["outlook"]`（來自 `outlook.stock_outlook`）；歷史走勢用 `outlook.load_outlook(code)` |
| 綜合分數與四組百分位（營收動能、品質、價值、趨勢） | `facts["scores"]`、`series["trend"]` |
| 月營收 | `facts["revenue"]`、`series["rv"]` |
| 季財報 | `facts["financials"]`、`series["fn"]` |
| 法人買賣、外資持股 | `facts["institutional_flows"]`、`series["chips"]` |
| 融資融券 | `facts["margin_foreign_holding"]` |
| 重大訊息 | `facts["announcements"]`、`series["ann"]` |
| 除權息、減資、ETF 分割 | `facts["corporate_actions"]` |
| 同業比較 | `analyst.compare_peers(code)`（回傳 JSON 字串） |
| 區間報酬 | `analyst.price_performance(code, start, end)`（回傳 JSON 字串） |
| 產生 LLM 報告（按鈕） | `report.llm_narrative(facts)` → dict；HTML 用 `report.render(facts, series, narrative)`。Haiku，每份約 1 美分以內 |
| 既有報告 | `reports/{code}_{asof}.html` / `.json` |

## 頁 3：策略

| 功能 | 來源 |
|---|---|
| 策略清單 | `strategies/*.toml`，用 `strategy.load_config(path)` 讀 |
| 淨值疊圖（勾選多個策略 + 同池等權 + 0050）、回落圖 | `strategy.simulate(cfg, panel, rets)["daily"]`；`panel, rets = strategy.load_data()` 要快取 |
| 成績單：各期間年化、波動、Sharpe、最大回落、成本 / 年、換手、DSR、對 0050 的 DSR | `strategy.scorecard(...)` 的 `dsr` 與 `dsr_excess`，或讀 `data/strategy/results/<id>.json` |
| 每年報酬與對同池超額 | 同上，`daily` 依年份彙總 |
| 分段與相對 0050：超額、t、α、β | `robust.vs_etf(daily, start, end)`；分段期間用 `robust.SEGMENTS` |
| 參數微調結果（每個策略 9–12 個變體） | `robust.variants(cfg)` 加 `simulate`，或讀後端快取 |
| 嘗試紀錄（試過幾個設定、各自的 Sharpe） | `data/strategy/trials.jsonl`，一行一筆 JSON |
| 線上試跑：在頁面上改參數後回測 | 組出 cfg dict 後呼叫 `simulate`；記錄用 `strategy.log_trial(cfg, res)`，每次試跑都要記 |
| 現有報告 | `reports/strategy_*.html`、`reports/robust*.html` |

設定欄位說明在 `strategy.py` 檔頭的 docstring。主要有：

- `[select]`：sort、n、filters
- `[position]`：weighting、max_weight、max_industry、min_trade
- `[risk]`：regime、regime_freq、target_vol、stop_loss、dd_limit
- `[cost]`
- `lag`

速度：每次 `simulate` 約 3–5 秒，有每日出場規則的策略更久。疊多個策略時要快取結果，或讀後端快取。

## 頁 4：模擬持股

| 功能 | 來源 |
|---|---|
| 綜合分數前 30 名（原始的模擬持股）：本期持股、進場以來報酬、累積淨值對同池與 0050 | `paper.summary()`；淨值在 `data/paper/nav.parquet`（10/26 進場後才有） |
| 多策略：起點、期數、目標曝險、大盤濾網狀態（距均線多少 %）、累積報酬、0050、同池、最大回落、現金、出場次數 | `paper.load_strategy_summaries()`（讀 `data/paper/strat_summary.json`） |
| 各策略目前持股（市值、權重）或本期目標權重 | 同上，`holdings` / `targets` |
| 每期紀錄（訊號日、選了哪些、權重） | `data/paper/strat_ledger.parquet`：signal_date、strategy、hash、code、weight、in_pick、pick_order、regime_on |
| 追蹤哪些策略 | `paper_strategies.txt`（顯示即可；改了要等下個月才會記新策略） |
| 各策略淨值曲線 | `data/paper/strat_nav.parquet`（strategy、date、nav、pool、etf）。10/26 進場前是 0 列，頁面說明等待進場，不畫空圖 |

## 頁 5（可選）：分析師問答

`analyst.py` 已有工具和對話迴圈：`analyst.ask(client, messages)`，工具清單在 `analyst.TOOLS`。可以做成聊天頁。

- 預設用 Sonnet。不要預設 Opus（太貴）。
- 每輪對話顯示花費。
- 對話紀錄同 `analyst.log_turn`，存在 `logs/analyst/`。

## 後端已經提供

| 項目 | 前端怎麼用 |
|---|---|
| `data/paper/strat_nav.parquet`（strategy、date、nav、pool、etf，起點 = 1） | 頁 4「各策略淨值」。0 列時顯示進場前提示 |
| `data/strategy/results/<id>.parquet` / `.json` | 頁 3 優先讀這份。json 的 `hash`、`asof` 用來判斷快取舊不舊；雜湊不符就改現場回測 |
| `data/strategy/results/<id>_robust.json`（`variants`、`segments`，數字未格式化） | 頁 3「參數微調」直接畫表。另有 `reports/robust_all.html` |
| 成績單 `dsr_excess`（對 0050）；`dsr` 保留 | 兩張 KPI 都顯示。`scorecard` 沒傳超額變異數時會自己讀嘗試紀錄 |
| `frontend/requirements.txt` | 啟動見檔案開頭 |

## 建議順序

1. 先做資料狀態列和頁 1 總覽，每天都會用。
2. 頁 2 個股，`build_facts` 一次就拿到所有資料，最省事。
3. 頁 4 模擬持股（等後端存好淨值序列）。
4. 頁 3 策略（等後端快取和 JSON 完成）。
5. 頁 5 分析師問答。
