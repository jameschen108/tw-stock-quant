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

- [x] 先修評分正負號：`evals/analyst/run_eval.py:programmatic` 用 `abs(abs(n) - abs(v))` 比對數字，答 +5% 對上 −5% 也算命中；修完再判斷既有分數是否受影響，下面各項重跑排在這之後（路線圖 P1 第 2 項）→ PR #13 待合併：改成看負號與方向詞，標準答案方向 = 數值正負 × 題目說明的方向詞（「減少」配正數）；已存開發集回答重算 numbers 不變（都是 1.000），但已存回答沒有「減少」題，新籌碼題跑時再確認
- [ ] 3 題新籌碼題：credit_gw_margin、credit_tsmc_foreign_low、credit_0050_margin_drop（約 $0.25–0.4）
- [ ] 重評 edge_buy_advice（judge 改成可引用模型標籤）
- [ ] 重跑 lookup_tsmc_ytd（股票池排除 ETF 後答案改為 17.2）
- [ ] 跑完整開發集，確認 `outlook_picks`、`theme_summary` 工具與 prompt 的主題規則沒有影響舊題；另加 1–2 題主題題（例如「記憶體 IC 近 20 日表現」「南亞科屬於哪個主題」）
- [ ] 保留集 `evals/analyst/holdout.jsonl`（18 題）：同 prompt + 模型只能跑一次，等 prompt 和工具穩定再跑

## LLM 決策實驗

- [ ] 2026-12 初：9 月批 60 日報酬完整後，重跑 `llm_replay.py` 的 report
- [ ] 判讀前向模擬時記得：conviction 幾乎都是 3、量化全買組手續費偏高、第一批進場晚 2–3 週

## 專題加分（優先：前兩項）

- [ ] 前向紀錄自動化：launchd 每天收盤後跑 `update.py` 與 `llm_trader run`，失敗發通知；前向樣本斷一天就補不回來。程式已寫好（`autorun.py`：資料日沒前進不呼叫 LLM、單日估價超過 US$1 不跑、失敗發 macOS 通知），**排程還沒安裝**，要用時 `uv run autorun.py --install`。2026-10-23～10-30 沒有執行，這段的決策時點補不回來；10-31 補資料時，候選與 wait 都還在有效期內（wait 15 檔，估 US$0.023），但 LLM 還沒跑
- [ ] 決策存證：每次決策後把 `llm_decisions.jsonl` 雜湊 commit 進 git 或用 OpenTimestamps，證明事後沒改
- [ ] LLM 決策解釋（不花 API 錢）：用既有 buy／skip／wait 對事實表欄位跑邏輯迴歸或決策樹，看能不能被簡單規則重現；能的話把規則拿去回測 15 年；另比 `llm_inputs/` 裡 buy 與 skip 的理由用詞；和路線圖 R01、R02（自訓選股模型＋SHAP 當 LLM 對照組）合併做
- [ ] LLM 決策事後檢討頁（不花 API 錢）：前端列出每筆 buy 的理由與 20／60 日後報酬，依理由類型分組看哪種說法比較準
- [ ] 檢定力分析：假設 LLM 每批真有 1% 超額，要累積幾批、到哪個月才有 80% 機率偵測到（目前 20 日 p=0.33）
- [ ] Demo 模式：沿用 `tests/conftest.py` 合成行情，加合成營收與財報，`--demo` 一鍵跑前端與回測，沒有 TEJ 也能看；加 `reproduce.sh` 從資料包重建所有報告
- [ ] 研究總結報告：事件研究、綜合分數樣本內外、策略 Deflated Sharpe、LLM 回放負面結果串成主線，附 `trials.jsonl` 嘗試次數表

## 期刊方法（回應「直接用 AI 生就好」的質疑）

2026-10-29 與教授討論後加入。原候選出處需逐項核對。另見已查核的 [文獻與功能擴充路線圖](literature_roadmap.md)：新增 19 篇文獻、資料缺口、實作入口與驗收方式；屬候選規劃，尚未實作，建議優先順序見路線圖（不需全做）。

- [x] 標明已用方法的出處：README 與報告寫明 Deflated Sharpe（Bailey & López de Prado 2014, JPM；`robust.py` 等）與 Newey-West 標準誤（Newey & West 1987, Econometrica；`patterns.py` 等）→ PR #14 待合併：README 新增「方法與文獻」段，報告還沒加
- [x] 形態辨識改用 Lo, Mamaysky & Wang (2000, JF)：核迴歸平滑後依論文定義辨識頭肩、雙重頂底等，檢定形態後報酬分布；`patterns.py` 原規則保留作基準比較，條件報酬不同不等於扣成本可獲利；K 線買賣點圖沿用 → PR #16 待合併（`lmw.py`）：CV 版（原文）頻寬一半落在下限、幾乎沒平滑，另加事前固定頻寬對照版（h = 1.25 日，只看價格選）。樣本外 H20 兩版都沒有可用的方向性超額（CV 版只有頭肩頂 t = −2.24，但頭肩底也是負的；固定版 t 都在 ±1.6 內），規則式有 7 個達「有統計支持」；KS 分布有差異但樣本相依。結論：不接進報告、個股頁，K 線買賣點圖改用規則式形態
- [ ] 多重檢定：照 Sullivan, Timmermann & White (1999, JF) 用 White (2000, Econometrica) Reality Check bootstrap 檢定最佳技術規則是否靠運氣，另評估 Bailey et al. (2017, J. Computational Finance) 的 PBO；先依設定 hash 保存各候選共同期間的逐期報酬，`trials.jsonl` 摘要不能直接當輸入；另見路線圖的 SPA 與 Romano–Wolf 候選
- [ ] 事件研究照 MacKinlay (1997, JEL)：已確認 `event_study.py` 有多種基準的持有期超額報酬，後續補市場模型 AR／CAR 與適當的相依性處理；財報事件研究加 Bernard & Thomas (1989) 的 PEAD 與 SUE
- [ ] 因子模型拆解：策略與 LLM 組報酬對 Fama-French (1993, 2015) 與 Carhart (1997) 因子迴歸，看扣掉曝險後的 alpha；和「因子曝險」項目合併做
- [ ] 趨勢區間用 GARCH (Bollerslev 1986, J. Econometrics)：和 `refprice.py` 現有統計波動區間比較，評分照路線圖 R12、R13 同時看覆蓋率、平均寬度、上下漏出與 interval score，不只看 68% 覆蓋率。評分已在 PR #17（待合併）做好：現有區間樣本外 H20 覆蓋 72.2%、寬度 21.8%、IS 36.7%，高波動組過寬（77%）、上下不對稱（超出上緣 16.7% vs 跌破 11.1%）、同日相依比值約 10；GARCH 要看能不能改善這幾點
- [ ] 籌碼因子的理論依據：引用 Richards (2005, JFQA) 外資行為、Barber, Lee, Liu & Odean (2009, RFS) 台灣散戶交易損失
- [ ] LLM 實驗定位成研究對象：引用 Lopez-Lira & Tang（JFE 2026, 184, 104335，卷期待核對）與 Glasserman & Lin（JFDS 2024, 6(1), 25–42，待核對），說明前向模擬的必要；出版來源見文獻路線圖。10-30 只查到兩篇的 arXiv 版本（2304.07619、2309.17322），PR #14 的 README 先當工作論文引用

## 新功能（候選）

- [ ] 財報事件研究：照營收事件研究做「毛利率創新高」「EPS 超過去年同季」後的超額報酬（Q3 財報 11-14 截止）
- [ ] 主題時點快照：每天存 `themes.toml` 成分與主題指數，累積沒有事後偏差的主題輪動資料
- [ ] 觀察名單提醒推播到手機（Telegram bot 或 email）
- [ ] 模擬持股與策略的產業、市值、因子曝險隨時間變化（例如 LLM 組是不是只偏小型股）；和期刊方法的因子模型拆解合併做
- [ ] 新 LLM 實驗組改成同批候選互相排序取前 K 名，取代沒有區分力的 conviction；另開組，不動原組
- [ ] K 線趨勢與買賣點圖：個股頁（`frontend/p2_stock.py`）K 線圖上標出可能的買入／賣出點，並畫未來趨勢區間；訊號沿用 `patterns.py`、`refprice.py`（支撐壓力、ATR 停損、波動區間），每個標記附歷史勝率，未通過校準的標「歷史上不可靠」；先回測標記點的後續報酬再上線，不給目標價；LMW 形態校準不如規則式，不採用（見期刊方法段）；區間附 PR #17 的覆蓋率與同日相依說明
- [ ] 買賣點跨週期確認：日 K 與週 K 訊號同向才標成強訊號，回測篩過後勝率有沒有提高
- [ ] 參考價前向命中追蹤：每天記錄 `refprice.py` 的支撐、壓力、停損參考價事後被觸及的比例；路徑觸價率和終點價格覆蓋率分開算，後者才和回測的 68% 覆蓋率比，看樣本外是否成立
- [ ] 策略失效監控：滾動 60 日超額報酬或 Sharpe 跌破回測區間下緣就提醒，接手機推播
- [ ] 資料新鮮度面板：前端顯示各資料源最後更新日、缺漏檔數，搭配前向紀錄自動化，斷檔馬上看得到
- [x] 即時股價：`realtime.py` 打證交所 `mis.twse.com.tw/stock/api/getStockInfo.jsp`（上市 `tse_`、上櫃 `otc_`，一次查多檔），個股頁與觀察名單顯示即時價、漲跌、時間，盤中每 10 秒更新；只顯示，不寫進 `data/`、不進因子／回測／LLM

## 舊功能改進

- [ ] 處置股、注意股、全額交割股排除：目前只有 `llm_trader.py` 處理，量化策略、`paper.py`、回測股票池也要排除，否則回測報酬偏高
- [ ] 時點測試：合成資料放一筆公布日晚於換股日的營收，確認 `factors.py` 沒用到；另補 `outlook.py`、`paper.py`、`llm_trader` 回應解析的測試
- [ ] CI 的 ruff 加 F401、F841
- [ ] 滑價敏感度：單筆超過日均成交值 X% 就加衝擊成本，放進 `robust.py`（小型股策略如 `mid_bull_invvol`）；和路線圖 R08（Amihud 流動性與交易容量）合併做，滑價係數標成情境假設
- [ ] （低優先）根目錄約 40 個 .py 分成 `data/`、`signals/`、`backtest/`、`llm/` 等資料夾

## 小項目（可選）

- [ ] 報告 prompt 只要求檢查看漲端區分力；看跌組和中性組差不多時（例如短期 −0.18% vs −0.14%）也可以要求說明
- [ ] 個股頁（`frontend/p2_stock.py`）也顯示各標籤勝率
