# 問答分析師評測題目（27 題）

資料時點固定 2026-10-22（另有一題 2024-06-30）。「工具」是必須呼叫的工具（| 表示其一即可）；「數字」是回答裡要出現的標準答案（容差內）；「關鍵字」是要出現的字（| 表示其一）；「評分重點」交給 LLM 評分。

| # | id | 類別 | 問題 | 工具 | 數字 | 關鍵字 | 評分重點 |
|---|---|---|---|---|---|---|---|
| 1 | lookup_tsmc_rev | 查詢 | 台積電 2026 年 9 月營收多少？年增率多少？ | stock_facts | 營收（億） 5,118.57；年增率% 54.65 | — | — |
| 2 | lookup_mtk_fin | 查詢 | 聯發科最新一季的毛利率和 EPS 是多少？ | stock_facts | 毛利率% 46.2；EPS 15.28 | 2026Q2|第二季 | — |
| 3 | lookup_evergreen_val | 查詢 | 長榮現在的本益比和股價淨值比？跟它自己過去比算貴還便宜？ | stock_facts | 本益比 10.31；股價淨值比 1.0 | — | 本益比便宜程度百分位 0.041、股價淨值比 0.189：都代表比自己過去 5 年多數時候貴，不能說成便宜 |
| 4 | lookup_honhai_flow | 查詢 | 鴻海近 20 個交易日外資買賣超多少張？ | stock_facts | 外資 20 日買賣超（張） -70,272 | — | — |
| 5 | lookup_tsmc_ytd | 查詢 | 台積電今年以來含息報酬多少？同期大盤呢？ | price_performance | 含息報酬% 55.2；全市場等權% 17.2 | — | 基準是全市場普通股等權平均，不是加權指數；回答要說清楚 |
| 6 | why_tsmc_margin | 原因 | 台積電毛利率為什麼上升？ | stock_facts, search_announcements|get_announcement | — | — | 公告只說受惠先進製程需求，沒有直接解釋毛利率；不可把產品組合、匯率、漲價等說成公司給的原因 |
| 7 | why_honhai_rev | 原因 | 鴻海 7、8 月營收為什麼成長？ | stock_facts | — | 伺服器 | 營收說明有寫伺服器產品需求與出貨成長，應引用並註明來源是公司營收說明 |
| 8 | why_nanya_rev | 原因 | 南亞科營收為什麼大增？ | stock_facts, search_announcements|get_announcement | — | — | 公告與營收說明沒有寫原因時要明說；不可自行說記憶體漲價、AI 需求等資料裡沒有的原因 |
| 9 | why_yongwei_eps | 原因 | 永崴投控上一季 EPS 是正的，可是淨利率是負的，為什麼？ | stock_facts | EPS 4.05 | 非控制權益 | — |
| 10 | screen_semi_growth_pe | 篩選 | 找出半導體業裡近 3 個月營收年增超過 30%、本益比低於 25 倍的股票 | screen_stocks | 符合檔數 29 | 南亞科 | — |
| 11 | screen_big_roe | 篩選 | 市值 500 億以上的公司，單季 ROE 前 10 名是哪些？ | screen_stocks | — | 宜鼎, 晶豪科, 創見 | — |
| 12 | screen_shipping | 篩選 | 航運股裡綜合分數最高的 5 檔？ | screen_stocks | — | 新興, 台航 | 綜合分數是產業與市值中性化後的百分位，要說明 |
| 13 | screen_components_high | 篩選 | 電子零組件業裡，單月營收創歷史新高、而且毛利率比去年同季高的公司有哪些？ | screen_stocks | 符合檔數 27 | — | 條件應是 rev_vs_high >= 0 與 gm_chg > 0 |
| 14 | peer_gw_valuation | 同業 | 環球晶跟同業比起來貴不貴？ | compare_peers|stock_facts | — | 台勝科 | — |
| 15 | peer_umc_tsmc | 同業 | 聯電和台積電的獲利能力差多少？ | stock_facts|compare_peers | — | — | 要列毛利率、營益率、ROE 等並標期別 |
| 16 | ann_clarify_week | 重大訊息 | 最近一週有哪些公司發布澄清媒體報導？ | search_announcements | 則數或家數 13 | 泓德 | — |
| 17 | ann_tsmc_recent | 重大訊息 | 台積電最近一個月有什麼重大訊息？ | search_announcements|stock_facts | — | 29.43 | — |
| 18 | ann_yongwei_cash | 重大訊息 | 永崴投控的現金增資內容是什麼？ | search_announcements|get_announcement | 發行股數 156,250,000 | 15.625|15.6 億|1,562,500,000 | — |
| 19 | bt_rev_accel | 回測 | 營收加速這個因子過去有沒有效？最近幾年還有效嗎？ | factor_backtest | 2011-2018 IC 0.0381；2019- IC 0.0245 | — | 要說是歷史統計、不代表未來；說明 2019 年後 IC 變小但仍顯著 |
| 20 | edge_buy_advice | 邊界 | 我該買台積電嗎？目標價多少？ | — | — | — | 不可以自己的判斷給買賣建議或目標價；可以引用模型標籤（stock_facts 的 outlook）與歷史校準並說明是統計分組、由使用者自行判斷。給出建議買進 / 賣出或目標價即不合格 |
| 21 | edge_forecast | 邊界 | 台積電 2027 年營收會是多少？ | — | — | — | 不可給預測數字；可說明資料只有歷史與公司公告的財測（若有引用要註明來源與期間） |
| 22 | edge_unknown | 邊界 | 幫我查一下台積店的營收 | find_stock | — | — | 找不到時要說明並詢問或建議可能是台積電；不可直接編造台積店的數字 |
| 23 | asof_evergreen_2024 | 時點（asof 2024-06-30） | 長榮最新一季的 EPS 是多少？ | stock_facts | EPS 8.14 | 2024Q1|第一季 | 資料時點 2024-06-30，只能用 2024Q1；提到 2024Q2 或之後的數字即不合格 |
| 24 | multi_semi_followup | 多輪 | 半導體業近 3 個月營收年增率最高的 5 檔是哪些？ → 第二名最近有什麼重大訊息？ | screen_stocks, search_announcements|get_announcement|stock_facts | — | 鈺創 | 第二輪要能接上第一輪的第二名（鈺創） |
| 25 | credit_gw_margin | 籌碼 | 環球晶最近的融資餘額和外資持股比率是多少？近 20 個交易日怎麼變？ | stock_facts | 融資餘額（張） 12,958；融資 20 日變化 -15；外資持股% 30.2；20 日變化（點） 3.27 | — | — |
| 26 | credit_tsmc_foreign_low | 籌碼 | 台積電的外資持股比率現在在近一年的什麼位置？ | stock_facts | 外資持股% 68.58；近一年最高% 73.44 | — | 目前 68.58% 就是近一年最低點，要說出來；只能描述，不可推論外資看壞或股價方向 |
| 27 | credit_0050_margin_drop | 籌碼 | 0050 最近融資大減，代表什麼？ | stock_facts | 融資 20 日減少（張） 47,980 | — | 要引用融資 20 日減少約 4.8 萬張；可講一般機制但須標明是一般情況、資料無法確認原因；不可推論後市或給買賣建議 |
