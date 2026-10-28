# 待辦事項

2026-10-28 整理。

## 新功能：形態辨識、訊號交叉比對、參考價

細節見 [memo_signals.md](memo_signals.md)。**不要動 `llm_trader.py` 的 prompt 和事實表**（`analyst.stock_facts` 也是它的輸入），新東西放 `report.py`、前端，要測 LLM 另開實驗組。

- [ ] `technicals.py` 存最新一天的訊號值
- [ ] `patterns.py`：帶量突破、箱型、W 底 / M 頭、回測均線、跳空缺口，附回測校準；沒通過的標「歷史上不可靠」
- [ ] 參考價：支撐 / 壓力、收盤 − 2 × ATR(14) 停損參考、統計波動區間；回測區間覆蓋率是否接近 68%
- [ ] 訊號矩陣（技術、籌碼、基本面、綜合）與矛盾清單；每試一種組合記進 `data/strategy/trials.jsonl`
- [ ] 個股頁、報告顯示；`report.py` prompt 改成可引用參考價（仍不給目標價）
- [ ] 看結果決定要不要另開 LLM 實驗組（例如 `LLM篩選-haiku-形態`）

## 分析師評測（暫緩）

跑之前先報價，用 Sonnet。

- [ ] 3 題新籌碼題：credit_gw_margin、credit_tsmc_foreign_low、credit_0050_margin_drop（約 $0.25–0.4）
- [ ] 重評 edge_buy_advice（judge 改成可引用模型標籤）
- [ ] 重跑 lookup_tsmc_ytd（股票池排除 ETF 後答案改為 17.2）
- [ ] 跑完整開發集，確認 `outlook_picks` 工具與 outlook 章節沒有影響舊題
- [ ] 保留集 `evals/analyst/holdout.jsonl`（18 題）：同 prompt + 模型只能跑一次，等 prompt 和工具穩定再跑

## LLM 決策實驗

- [ ] 2026-12 初：9 月批 60 日報酬完整後，重跑 `llm_replay.py` 的 report
- [ ] 判讀前向模擬時記得：conviction 幾乎都是 3、量化全買組手續費偏高、第一批進場晚 2–3 週

## 小項目（可選）

- [ ] 報告 prompt 只要求檢查看漲端區分力；看跌組和中性組差不多時（例如短期 −0.18% vs −0.14%）也可以要求說明
- [ ] 個股頁（`frontend/p2_stock.py`）也顯示各標籤勝率
