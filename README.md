# 台股量化研究

台股（上市、上櫃）的量化研究系統，學術研究用途。從月營收、財報、價格、籌碼資料算出因子，做月頻選股與策略回測，並提供本機前端和可用自然語言提問的分析師。

主要功能：

- 月頻因子表與橫斷面回測（rank IC、五分位投組）
- 綜合分數（營收動能、品質、價值、趨勢）與樣本內／樣本外回測
- 量化護城河分級、仿晨星的 1–5 星等、短中長期看漲／看跌標籤（每個標籤附歷史校準）
- 以 TOML 設定的策略回測、穩健性檢查、模擬持股（記錄真正的樣本外表現）
- 個股分析報告與每日速覽（HTML）
- 本機 Streamlit 前端，以及能呼叫專案資料回答問題的分析師問答（Claude）

> 輸出的選股清單、星等與看漲看跌標籤是依歷史統計的分組，不保證未來報酬，不構成投資建議。

## 畫面

本機 Streamlit 前端。

**總覽**：觀察名單、需要注意的變化、模型選股。

![總覽](docs/images/overview.png)

**個股**：價格、短中長期標籤、綜合分數、月營收、季財報、法人、重大訊息等分頁。

![個股](docs/images/stock.png)

**策略**：策略回測與 0050、同池等權比較，另有參數微調、線上試跑、嘗試紀錄。

![策略](docs/images/strategy.png)

## 需求

- Python 3.11 以上，用 [uv](https://docs.astral.sh/uv/) 執行。各腳本開頭已宣告依賴，`uv run` 會自動安裝，不必另建環境。
- 資料不在 repo 裡（`data/` 被 `.gitignore` 排除），需要另外取得，見下節。
- 只有分析師問答需要 Anthropic API key，其他功能不需要。

## 資料

所有資料放在 `data/` 底下，程式用相對路徑讀取：

| 位置 | 內容 | 來源 |
|---|---|---|
| `data/prices/{twse,tpex}/` | 每日全市場收盤行情 | 證交所、櫃買中心，`fetch_prices.py` |
| `data/tej/` | 月營收、季財報的 TEJ 匯出 CSV | TEJ（授權資料，需自備） |
| `data/raw/` | 現金流、三大法人、融資融券、外資持股、除權息、重大訊息、OpenAPI 快照 | 證交所、櫃買、公開資訊觀測站，各 `fetch_*.py` |
| `data/*.parquet`、`data/*.json` | 上述資料整理後的分析用表 | 由腳本產生 |

拿到資料包的話，照資料包裡的 README 解壓到上述位置即可。沒有資料包也可以自己從頭抓：

```bash
uv run fetch_prices.py --market twse && uv run fetch_prices.py --market tpex
uv run fetch_events.py --source twse && uv run fetch_events.py --source tpex
uv run fetch_chips.py --market twse && uv run fetch_chips.py --market tpex
uv run fetch_public.py
```

行情和籌碼從 2010 年抓起要好幾個小時，交易所會封鎖太頻繁的請求，腳本已內建間隔，被擋時等一陣子重跑會從中斷處接續。**TEJ 的月營收與財報沒有公開來源，要自己有授權並把匯出檔放進 `data/tej/`**；沒有的話，只能用證交所／櫃買 OpenAPI 快照累積的最新資料，歷史會很短。

## 快速開始

資料放好後，在專案根目錄依序執行：

```bash
uv run build_panel.py          # TEJ CSV → 月營收、季財報、月底面板
uv run build_prices.py         # 行情 + 除權息事件 → 還原股價
uv run fetch_chips.py --build  # 籌碼原始檔 → 三大法人、融資融券、外資持股表
uv run factors.py              # 月頻因子表
uv run composite.py            # 綜合分數與回測
```

之後每天更新資料並重算全部結果：

```bash
uv run update.py               # 抓新資料 → 重建 → 健康檢查 → 模擬持股 → 每日速覽
uv run update.py --skip-fetch  # 只重建與檢查
```

啟動前端（含 TEJ 數字，只綁本機）：

```bash
uv run --with-requirements frontend/requirements.txt streamlit run frontend/app.py --server.address 127.0.0.1
```

瀏覽器開 `http://127.0.0.1:8501`。頁面有總覽、個股、策略、模擬持股、分析師問答，說明見 [docs/frontend.md](docs/frontend.md)。

## 腳本一覽

| 腳本 | 用途 |
|---|---|
| `fetch_prices.py`、`fetch_events.py`、`fetch_chips.py`、`fetch_public.py`、`fetch_cashflow.py`、`fetch_mops.py` | 下載各類原始資料 |
| `build_panel.py`、`build_prices.py`、`public_data.py` | 整理成 parquet；OpenAPI 快照補 TEJ 之後的新資料 |
| `factors.py` | 月頻因子表（只用當時已公布的資料，避免偷看未來） |
| `backtest.py` | 單因子 rank IC 與五分位回測 |
| `composite.py` | 四組因子合成綜合分數，樣本內／外回測 |
| `moat.py`、`stars.py`、`outlook.py` | 護城河分級、星等、看漲看跌標籤 |
| `technicals.py`、`regime.py` | 技術面訊號與大盤多空、市場狀態指標 |
| `strategy.py`、`robust.py`、`strategies/*.toml` | 策略回測與穩健性檢查，策略以 TOML 設定 |
| `paper.py` | 模擬持股，每月記錄前 N 名並追蹤報酬 |
| `diagnose.py` | 拆解綜合分數輸給同池等權的原因 |
| `report.py`、`daily.py` | 個股報告、每日速覽，輸出到 `reports/` |
| `analyst.py`、`evals/analyst/` | 分析師問答（命令列）與評測題組 |
| `update.py` | 每日更新的總流程 |

各腳本開頭的說明文字有完整用法，例如 `uv run strategy.py strategies/top30.toml`。

## 分析師問答與費用

`analyst.py`、`report.py` 與前端的問答頁會呼叫 Anthropic API，**會產生費用**。在專案根目錄建立 `.env`：

```
ANTHROPIC_API_KEY=你的key
```

個股報告預設用較便宜的模型，問答預設用 Sonnet，可用 `--model haiku` 改。`.env` 已在 `.gitignore`，不會被提交。

## 使用限制

- TEJ 資料有授權限制，不得轉傳或公開，所以前端只能在本機執行，不要部署到公開網址。
- `reports/` 的 HTML 可能含 TEJ 數字，分享前請先確認。
- 證交所、櫃買、公開資訊觀測站的資料請依其使用條款取用，不要提高請求頻率。
