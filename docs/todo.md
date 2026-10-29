# 待辦事項

2026-10-28 整理，10-29 加入專題加分、新功能與改進項目。

## 新功能：形態辨識、訊號交叉比對、參考價

細節見 [memo_signals.md](memo_signals.md)。**不要動 `llm_trader.py` 的 prompt 和事實表**（`analyst.stock_facts` 也是它的輸入），新東西放 `report.py`、前端，要測 LLM 另開實驗組。

- [x] `technicals.py` 存最新一天的訊號值
- [x] `patterns.py`：帶量突破、箱型、W 底 / M 頭、回測均線、跳空缺口，附回測校準；沒通過的標「歷史上不可靠」
- [x] 參考價：支撐 / 壓力、收盤 − 2 × ATR(14) 停損參考、統計波動區間；回測區間覆蓋率是否接近 68%
- [x] 訊號矩陣（技術、籌碼、基本面、綜合）與矛盾清單；每試一種組合記進 `data/strategy/trials.jsonl`
- [x] 個股頁、報告顯示；`report.py` prompt 改成可引用參考價（仍不給目標價）
- [ ] 看結果決定要不要另開 LLM 實驗組（例如 `LLM篩選-haiku-形態`）

## 分析師評測（暫緩）

跑之前先報價，用 Sonnet。

- [ ] 3 題新籌碼題：credit_gw_margin、credit_tsmc_foreign_low、credit_0050_margin_drop（約 $0.25–0.4）
- [ ] 重評 edge_buy_advice（judge 改成可引用模型標籤）
- [ ] 重跑 lookup_tsmc_ytd（股票池排除 ETF 後答案改為 17.2）
- [ ] 跑完整開發集，確認 `outlook_picks`、`theme_summary` 工具與 prompt 的主題規則沒有影響舊題；另加 1–2 題主題題（例如「記憶體 IC 近 20 日表現」「南亞科屬於哪個主題」）
- [ ] 保留集 `evals/analyst/holdout.jsonl`（18 題）：同 prompt + 模型只能跑一次，等 prompt 和工具穩定再跑

## LLM 決策實驗

- [ ] 2026-12 初：9 月批 60 日報酬完整後，重跑 `llm_replay.py` 的 report
- [ ] 判讀前向模擬時記得：conviction 幾乎都是 3、量化全買組手續費偏高、第一批進場晚 2–3 週

## 專題加分（優先：前兩項）

- [ ] 前向紀錄自動化：launchd 每天收盤後跑 `update.py` 與 `llm_trader run`，失敗發通知；前向樣本斷一天就補不回來
- [ ] 決策存證：每次決策後把 `llm_decisions.jsonl` 雜湊 commit 進 git 或用 OpenTimestamps，證明事後沒改
- [ ] LLM 決策解釋（不花 API 錢）：用既有 buy／skip／wait 對事實表欄位跑邏輯迴歸或決策樹，看能不能被簡單規則重現；能的話把規則拿去回測 15 年；另比 `llm_inputs/` 裡 buy 與 skip 的理由用詞
- [ ] 檢定力分析：假設 LLM 每批真有 1% 超額，要累積幾批、到哪個月才有 80% 機率偵測到（目前 20 日 p=0.33）
- [ ] Demo 模式：沿用 `tests/conftest.py` 合成行情，加合成營收與財報，`--demo` 一鍵跑前端與回測，沒有 TEJ 也能看；加 `reproduce.sh` 從資料包重建所有報告
- [ ] 研究總結報告：事件研究、綜合分數樣本內外、策略 Deflated Sharpe、LLM 回放負面結果串成主線，附 `trials.jsonl` 嘗試次數表

## 新功能（候選）

- [ ] 財報事件研究：照營收事件研究做「毛利率創新高」「EPS 超過去年同季」後的超額報酬（Q3 財報 11-14 截止）
- [ ] 主題時點快照：每天存 `themes.toml` 成分與主題指數，累積沒有事後偏差的主題輪動資料
- [ ] 觀察名單提醒推播到手機（Telegram bot 或 email）
- [ ] 模擬持股與策略的產業、市值、因子曝險隨時間變化（例如 LLM 組是不是只偏小型股）
- [ ] 新 LLM 實驗組改成同批候選互相排序取前 K 名，取代沒有區分力的 conviction；另開組，不動原組
- [ ] 即時股價：`realtime.py` 打證交所 `mis.twse.com.tw/stock/api/getStockInfo.jsp`（上市 `tse_`、上櫃 `otc_`，一次查多檔），個股頁與觀察名單顯示即時價、漲跌、時間，盤中每 10 秒更新；只顯示，不寫進 `data/`、不進因子／回測／LLM

## 舊功能改進

- [ ] 時點測試：合成資料放一筆公布日晚於換股日的營收，確認 `factors.py` 沒用到；另補 `outlook.py`、`paper.py`、`llm_trader` 回應解析的測試
- [ ] CI 的 ruff 加 F401、F841
- [ ] 滑價敏感度：單筆超過日均成交值 X% 就加衝擊成本，放進 `robust.py`（小型股策略如 `mid_bull_invvol`）
- [ ] （低優先）根目錄約 40 個 .py 分成 `data/`、`signals/`、`backtest/`、`llm/` 等資料夾

## 小項目（可選）

- [ ] 報告 prompt 只要求檢查看漲端區分力；看跌組和中性組差不多時（例如短期 −0.18% vs −0.14%）也可以要求說明
- [ ] 個股頁（`frontend/p2_stock.py`）也顯示各標籤勝率
