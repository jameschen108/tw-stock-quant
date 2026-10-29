# 文獻與功能擴充路線圖

查核日期：2026-10-29。新增整理 19 篇文獻：16 篇期刊、3 篇會議論文，不含原待辦已列的 Lo、DSR、Newey–West、Reality Check 等。此文件是研究與實作提案，所列功能尚未因此完成。

出版資訊與方法概要依期刊、學會、作者或大學提供的來源核對；部分僅能取得摘要與出版資訊，尚未逐式複現全文及附錄。下列台股功能、優先順序與驗收規格是本專案的設計建議，原論文的實證結果不能直接當成台股有效性的證據。正式實作前須讀全文、附錄與可取得的複現程式，固定公式、參數及資料口徑。

## 建議主線與實作順序

研究問題：在台股公告事件與量化選股中，經過時點控制、風險校準及成本評估後，LLM 能提供多少額外資訊？

| 階段 | 功能 | 使用者能看到什麼 | 文獻 | 資料與成本 | 建議 |
|---|---|---|---|---|---|
| 0 | 固定比較規格、保存每個實驗的逐期結果 | 可重建的研究成績單 | 下列各方法的共同前提 | 既有回測輸出，主要是工程工作 | 先做 |
| 1 | 自行訓練的選股模型與解釋 | 綜合分數、線性模型、樹模型、LLM 的比較 | R01、R02 | 既有因子；本機訓練不需 LLM API | 核心研究 |
| 1 | 流動性與交易容量分析 | 不同資金規模、成交參與率、滑價情境下的淨績效 | R08 | 日報酬與成交值；真實衝擊估計另需成交資料 | 核心驗證 |
| 1 | 區間評分與失準診斷 | 覆蓋率、寬度、漏出方向、失準是否集中 | R12、R13 | 既有股價與區間紀錄 | 容易接上現況 |
| 2 | 盈餘品質與財務體質面板 | 淨利與現金流差異、資產獲利能力、可追溯評分 | R03–R05 | 財報欄位與歷史公布日；完整 F-score 待補欄位 | 實用功能 |
| 2 | 多重比較後的訊號成績單 | 所有候選、校正前後結果與不確定性 | R10、R11 | 每個候選在共同期間的序列 | 保存實驗結果後做 |
| 2 | 報告逐句連結證據 | 點開理由即可看原數字、期別、公布日與公告原文 | R15 | 既有工具結果與公告；生成與語意評分可能有 API 費用 | 有展示價值 |
| 3 | 考慮相關性的持股權重 | 各股風險貢獻、集中度、權重法比較 | R06、R07 | 既有日報酬；本機計算 | 完整投組管理 |
| 3 | 公告情緒與文本變化 | 負面／不確定語氣、同類文件的內容變動 | R16–R18 | 中文標註與完整歷史文件需建置 | 資料準備較多 |
| 3 | 自適應區間、市場狀態、波動曝險研究 | 區間動態校準與風控方法比較 | R14、R19、R09 | 日報酬可支援起步，需處理延後到期的標籤 | 進階候選 |

每項都要允許得到「沒有改善」的結果；功能完整度與報酬提升分開驗收。

## 建議優先順序

依「回應教授的出處要求」優先、其次看成本與能否接上現有程式排序，含原待辦項目。

**P1：成本低，先做**

1. 補已用方法的出處：DSR、Newey–West，以及籌碼（Richards、Barber 等）與 LLM 前視偏差（Lopez-Lira & Tang、Glasserman & Lin）的引用。只改 README 與報告。Lopez-Lira & Tang 的 JFE 卷期須自行核對。
2. 修 `evals/analyst/run_eval.py:programmatic` 的正負號問題（比對時取了絕對值），修完再判斷既有評測分數是否受影響。
3. 區間評分與失準診斷（R12、R13）：在 `refprice.py` 的覆蓋率表加上寬度、上下漏出與 interval score，用既有資料即可。
4. Lo、Mamaysky & Wang (2000) 形態辨識，原規則保留作基準。

**P2：核心研究，挑一兩項**

5. 依設定 hash 保存逐期結果（階段 0）。P3 的多重檢定要以它為前提。
6. 流動性與交易容量（R08）。
7. 自行訓練選股模型與 SHAP（R01、R02），作為 LLM 的對照組。
8. 事件研究補市場模型 AR／CAR（MacKinlay）；Fama–French／Carhart 因子拆解。

**P3：有前置條件或工作量較大**

9. 多重檢定：Reality Check／SPA／Romano–Wolf／PBO（R10、R11），等第 5 項完成後再做，挑一種即可。
10. 盈餘品質與財務體質（R03–R05），F-score 需要先補欄位。
11. 報告逐句連結證據（R15）。
12. GARCH 區間，和第 3 項共用評分。

**P4：可選，有時間再看**

持股權重（R06、R07）、波動管理（R09）、公告情緒與文本變化（R16–R18）、自適應區間（R14）、市場狀態模型（R19）。

## 1. 自行訓練選股模型，建立 LLM 的強對照組

**R01｜Gu, S., Kelly, B., & Xiu, D. (2020).** *Empirical Asset Pricing via Machine Learning*. **The Review of Financial Studies, 33(5), 2223–2273**。[期刊與方法](https://academic.oup.com/rfs/article/33/5/2223/5758276)，[DOI](https://doi.org/10.1093/rfs/hhaa009)。

論文比較以公司特徵預測報酬的線性及非線性模型，並依時間安排訓練、驗證與測試。本專案可借用其比較框架，以台股現有因子做範圍較小的實證。

- **功能**：建立 Elastic Net 與梯度提升樹的報酬排序，與 `composite.py` 固定權重分數比較。另在相同公告候選中比較模型選股與 LLM 選股。
- **資料**：`factors.py` 已有營收、估值、品質、動能等欄位及下期報酬。原文的資料年數、變數與宏觀交互項遠多於本專案，應稱為「依其框架改編」，不能宣稱完整複現。
- **入口**：新增研究模組讀取 `data/factors_monthly.parquet`；輸出每個時點的預測、訓練截止日與特徵版本，再交給同一個交易模擬器。
- **驗收**：先固定少量模型與調參範圍，按時間滾動／擴張訓練。報告 rank IC、Top-K 扣成本報酬、換手率、回撤與分年穩定性。跨股票同月份須在同一個資料切分；缺值處理及標準化只能從訓練資料估計。
- **LLM 比較邊界**：每組使用相同候選、可用資訊、決策期限、資金與成交規則。LLM 的等待流程要比較整套決策結果，不能只事後挑出 buy。

**R02｜Lundberg, S. M., & Lee, S.-I. (2017).** *A Unified Approach to Interpreting Model Predictions*. **NeurIPS 30，會議論文**。[學會論文頁](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)。

- **功能**：對自己訓練的樹模型加入 SHAP，顯示哪些因子推高／壓低單次預測，及不同期間的重要度變化。
- **驗收**：背景資料限於當時可知的訓練集；檢查高度相關特徵下解釋是否穩定。SHAP 解釋的是模型預測，不能當成報酬因果，也不能用替代模型的 SHAP 宣稱知道 Claude 內部如何決策。
- **成本**：上述兩項可先用本機資料與運算；沿用既有 LLM 紀錄的分析不需重新呼叫 API。

## 2. 財務體質與盈餘品質，讓基本面評分有可核對的構成

**R03｜Sloan, R. G. (1996).** *Do Stock Prices Fully Reflect Information in Accruals and Cash Flows about Future Earnings?* **The Accounting Review, 71(3), 289–315**。[論文全文](https://www.cuhk.edu.hk/acy2/workshop/June2009Wasley/1996TAR%29.pdf)，[期刊該期](https://www.jstor.org/stable/i302556)。

- **功能**：增加盈餘品質分頁，拆解淨利、營業現金流與應計項目；對「獲利增加但現金流未跟上」顯示數值與歷史比較。
- **現況**：`factors.py` 已有 `accruals = (ni_ttm - cfo_ttm) / assets` 和 `cfo_ta`，可先把它們的構成、趨勢與校準呈現出來。
- **方法邊界**：現有現金流法、淨利範圍與期末資產分母不等於已完整實作 Sloan 原文口徑；須核對資產平均方式、原文應計衡量、合併／歸母及金融業適用性。高應計是研究訊號，不直接標成財報造假。
- **驗收**：同產業分組、樣本外排序結果，並分析是否只是價值、市值或產業曝險。

**R04｜Novy-Marx, R. (2013).** *The Other Side of Value: The Gross Profitability Premium*. **Journal of Financial Economics, 108(1), 1–28**。[期刊頁面](https://www.sciencedirect.com/science/article/pii/S0304405X13000044)。

- **功能**：補上「毛利／總資產」獲利能力因子，和現有毛利率、ROE 比較，形成價值×獲利能力的研究分組。
- **資料**：需核對毛利金額、資產及期間口徑。毛利率是毛利／營收，不能直接冒充毛利／資產；若從毛利率與營收推算，須確認兩者同期間、同合併範圍且可得時間一致。
- **驗收**：比較加入前後的 IC、組合績效與產業曝險；金融業另行處理。

**R05｜Piotroski, J. D. (2000).** *Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers*. **Journal of Accounting Research, 38, Supplement, 1–41**。[期刊出版紀錄](https://www.jstor.org/stable/2672906)，[全文](https://www.ivey.uwo.ca/media/3775523/value_investing_the_use_of_historical_financial_statement_information.pdf)。

- **功能**：新增 F-score 九項財務體質檢查，每項顯示原始值、比較期、公布日及得分依據。
- **資料缺口**：完整九項需要核對 ROA、營運現金流、槓桿、流動比率、股權發行、毛利率與資產周轉等欄位。當前因子表並未完整提供這些項目，不能只用現有欄位宣稱完整 F-score，也不能把缺值算成零分。
- **驗收**：先核對年度財報版，揭露九項資料覆蓋率；先在原文關注的高帳面市值比族群檢驗，再探索全市場。改成季度／TTM 是另外的版本，需獨立命名及驗證。
- **入口**：`factors.py`、`build_panel.py`、個股基本面頁；不改既有 LLM 事實表，先以獨立資料輸出供報告使用。

## 3. 考慮股票間相關性，補齊持股風險管理

**R06｜Ledoit, O., & Wolf, M. (2004).** *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*. **Journal of Multivariate Analysis, 88(2), 365–411**。[期刊頁面](https://doi.org/10.1016/S0047-259X(03)00096-4)，[作者全文](https://www.ledoit.net/Well-conditioned2004.pdf)。

**R07｜DeMiguel, V., Garlappi, L., & Uppal, R. (2009).** *Optimal Versus Naive Diversification: How Inefficient Is the 1/N Portfolio Strategy?* **The Review of Financial Studies, 22(5), 1915–1953**。[期刊頁面](https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901)。

前者提供協方差收縮估計；後者提示資產配置方法須認真比較等權基準，複雜配置可能受到估計誤差影響。

- **功能**：在相同選股名單下比較等權、既有逆波動、收縮協方差的受限最小變異權重；顯示個股風險貢獻與相關性。
- **入口**：`strategy.py:target_weights` 與風險估計；先新增設定選項及獨立結果，不覆蓋原策略。
- **約束**：長倉、單股／產業上限、換手限制與資料不足時退回等權。收縮估計量要標明目標矩陣；本條 R06 不是把所有 Ledoit–Wolf 論文與不同收縮目標混為同一實作。
- **驗收**：固定持股名單、成交成本與風險上限，分開比較預測波動與實現波動、風險集中、回撤及換手。權重優化不預設一定贏過等權。

## 4. 交易容量與滑價，檢查回測能否承受較大資金

**R08｜Amihud, Y. (2002).** *Illiquidity and Stock Returns: Cross-Section and Time-Series Effects*. **Journal of Financial Markets, 5(1), 31–56**。[期刊與衡量方法](https://www.sciencedirect.com/science/article/pii/S1386418101000246)。

- **功能**：用過去日絕對報酬／成交金額的平均建立非流動性指標；增加資金規模、單筆成交值占過去平均成交值比例、滑價情境的成績單。
- **資料**：日報酬與成交值可支援指標。幣別、成交值單位、零成交、停牌及極端值須固定處理；交易判斷只能使用成交前可知的流動性估計。
- **重要區分**：Amihud 指標是流動性代理，不能直接當成自己的訂單衝擊成本。沒有報價、委託簿或實際成交資料時，滑價係數應列為情境假設，不能聲稱已估出真實衝擊。
- **入口**：`strategy.py` 的交易與成本、`robust.py`、策略頁。
- **驗收**：以相同名單測試預先固定的資金規模與滑價情境，輸出淨績效、超過參與率限制的交易比例、未成交及延後成交結果。可先不新增任何 API 費用。

## 5. 比較現有目標波動與論文的波動管理

**R09｜Moreira, A., & Muir, T. (2017).** *Volatility-Managed Portfolios*. **The Journal of Finance, 72(4), 1611–1644**。[期刊頁面](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513)。

- **功能**：比較固定曝險、現有目標波動、依過去變異數調整曝險三種風控方式。
- **現況與差異**：`strategy.py:exposure` 已用 `target_vol / estimated_vol`，並把曝險限制在 0–1。原文核心是依過去變異數反向縮放，不能將目前逆標準差公式直接說成論文複現。
- **驗收**：標明縮放常數如何只用訓練期決定，並使用相同槓桿上限、現金報酬與交易成本。檢查改善是否只是平均曝險較低；另比較同風險或同平均曝險結果。
- **優先度**：中後段，因為專案已經有部分相關功能。

## 6. 同時試多種訊號後，哪些仍有證據？

**R10｜Hansen, P. R. (2005).** *A Test for Superior Predictive Ability*. **Journal of Business & Economic Statistics, 23(4), 365–380**。[期刊頁面](https://www.tandfonline.com/doi/abs/10.1198/073500105000000063)。

**R11｜Romano, J. P., & Wolf, M. (2005).** *Stepwise Multiple Testing as Formalized Data Snooping*. **Econometrica, 73(4), 1237–1282**。[期刊頁面](https://onlinelibrary.wiley.com/doi/10.1111/j.1468-0262.2005.00615.x)。

SPA 可評估一組候選相對基準是否有優越表現；Romano–Wolf 逐步法則可處理各項假設的多重比較，並利用檢定統計量的相依結構。兩者回答的問題不同。

- **功能**：訊號與策略成績單保留全部候選，並呈現預先定義檢定家族內的校正結果；不只列最好的 t 值。
- **必要工程**：`trials.jsonl` 主要是設定與摘要。`strategy.py:save_result` 雖存每日報酬，但依策略 id 命名，不能保證保留每個歷史變體。需要依設定 hash、資料快照與程式版本另存逐期結果，保留共同日曆、樣本遮罩、基準及成本版本。
- **訊號與策略分開處理**：形態條件均值和可交易策略報酬是不同 estimand。先決定檢定對象，不能把所有摘要 Sharpe 混在一起做 bootstrap。
- **驗收**：共用時間區塊抽樣以保留候選間相關性，說明平穩性、區塊長度與缺值處理。先用有時間及橫斷面相依的零訊號合成資料檢查誤報率，再測有訊號的檢定力。
- **邊界**：DSR、Reality Check、SPA、PBO 與逐項校正不能互相當成同一指標；不需要為了展示而全部實作。

## 7. 區間預測的評估與校準

**R12｜Christoffersen, P. F. (1998).** *Evaluating Interval Forecasts*. **International Economic Review, 39(4), 841–862**。[論文全文](https://users.ssc.wisc.edu/~bhansen/718/Christoffersen1998.pdf)。

**R13｜Gneiting, T., & Raftery, A. E. (2007).** *Strictly Proper Scoring Rules, Prediction, and Estimation*. **Journal of the American Statistical Association, 102(477), 359–378**。[期刊頁面](https://doi.org/10.1198/016214506000001437)，[作者全文](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)。

- **功能**：擴充 `refprice.py` 現有覆蓋率表，同時呈現平均區間寬度、漏出上／下界比例、失準群聚及 interval score。覆蓋率可以靠放寬區間提高，因此要同時考慮寬度與漏出懲罰。
- **比較**：先以現有波動區間作基準，再比較 GARCH 或歷史分位數等候選；各方法使用相同預測時點、期間與名目覆蓋率。
- **多期依賴**：20／60 日報酬視窗會重疊，不能直接把每日命中當成獨立 Bernoulli 套用檢定。應使用不重疊預測日作主檢定，或另建能處理相依性的重抽樣評估；多股票也要保留同日相依。
- **驗收**：在共同樣本報告整體、各年度及波動分組結果；新增 80%／95% 區間須另定義並校準，不能只改標籤。終點價格覆蓋率與路徑曾觸價率要分開。

**R14｜Gibbs, I., & Candès, E. (2021).** *Adaptive Conformal Inference Under Distribution Shift*. **NeurIPS 34，會議論文**。[學會論文頁](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html)。

- **進階功能**：依已觀察到的預測誤差，持續更新區間校準程度，觀察市場變動時是否改善長期覆蓋。
- **限制**：不能將一般 conformal 的交換性條件直接套在股價；ACI 的長期覆蓋敘述也不等於每檔、每天都有條件覆蓋保證，更不保證區間足夠窄。
- **20／60 日落地條件**：在 t 日只能用到已到期的預測誤差。延後回饋、多期間與多股票更新規則需另設計，不能直接沿用論文即時回饋公式並宣稱原理論保證仍成立。先從單一標的、單期預測原型開始。

## 8. 每個分析結論都能點開證據

**R15｜Gao, T., Yen, H., Yu, J., & Chen, D. (2023).** *Enabling Large Language Models to Generate Text with Citations*. **EMNLP，6465–6488，會議論文**。[ACL 論文與評估框架](https://aclanthology.org/2023.emnlp-main.398/)。

- **功能**：報告中的數字與重要理由連到本地事實表欄位或公告段落，顯示資料期別、公布時間、來源與計算方式。可在個股頁點選「查看依據」。
- **現況**：`analyst.py` 已有 `stock_facts`、公告查詢及工具紀錄，先替工具回傳內容建立穩定的 evidence id，即可做範圍明確的證據引用原型。
- **評估**：區分「有沒有引用」「引用是否真的支持這句話」「該有證據的主張是否都被支持」。數字、單位、正負號、期別與資料時間用程式檢查；語意支持以人工標註小樣本校準評分器。
- **已知入口**：`evals/analyst/run_eval.py:programmatic` 現有數字比對使用絕對值，正負相反也可能被算命中。這是需要補強的評測缺口；本次僅記錄，未修程式。
- **驗收**：先在開發題組加入「數字正確但期別錯」「引用相關但不支持」「沒有資料應說不足」等案例。保留集遵守既有一次性使用規則；LLM 評審不是唯一真值。
- **實驗隔離與成本**：先用於報告／問答，維持原 `llm_trader.py` prompt 及共享事實表介面；新決策版本另開組。重跑生成或評審可能產生 API 費用，本輪不執行。

## 9. 中文公告情緒，以及文件內容有何改變

**R16｜Loughran, T., & McDonald, B. (2011).** *When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks*. **The Journal of Finance, 66(1), 35–65**。[期刊頁面](https://doi.org/10.1111/j.1540-6261.2010.01625.x)，[作者詞典網站](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)。

**R17｜Tetlock, P. C. (2007).** *Giving Content to Investor Sentiment: The Role of Media in the Stock Market*. **The Journal of Finance, 62(3), 1139–1168**。[期刊頁面](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2007.01232.x)，[作者論文目錄](https://www.columbia.edu/~pt2238/papers/papers.html)。

- **功能**：把公告的負面、不確定語氣與事件類型做成可檢驗的文字訊號，對照營收與財務數字訊號。
- **適合的研究設計**：金融詞典／簡單文字模型／LLM 分析並列。這能接上先前 PTT 情緒分析的研究經驗，但 PTT、新聞與公司公告是不同資料來源，須分開定義樣本。
- **語言限制**：Loughran–McDonald 的英文金融詞典不能翻譯後就當成已驗證的中文模型。須建立中文標註準則與人工標註集，處理否定、制式聲明、同義詞及公司間模板差異。
- **現況**：`fetch_mops.py` 的解析器會保留公告時間與完整內文，但實際本地歷史涵蓋率仍待稽核；程式預設抓近 180 天，不代表已有十多年完整文本。
- **驗收**：先評估分類／情緒標註品質，再測訊號對報酬的增量資訊；以日期及公司作適當切分，避免重複公告與模板跨切分洩漏。盤後公告不能配到同日可交易報酬。

**R18｜Cohen, L., Malloy, C., & Nguyen, Q. (2020).** *Lazy Prices*. **The Journal of Finance, 75(3), 1371–1415**。[期刊頁面](https://onlinelibrary.wiley.com/doi/10.1111/jofi.12885)，[作者所屬大學版本](https://dash.harvard.edu/entities/publication/576a041c-19c4-47d8-b28d-2a5ef079f54f)。

- **進階功能**：同公司、同類型且可比較的財報文字，顯示新增／刪除的風險敘述、相似度變化及原文對照。
- **資料前提**：原研究以定期財報文件變動為核心。當前重大訊息檔不等於定期年報／季報語料，不能任意拿兩則不同公告算差異就宣稱複現。需另外保存完整文件、版本、章節與實際公布日，處理格式及 OCR 差異。
- **驗收**：人工核對文字差異品質，再研究是否對既有模型提供額外資訊。先做文件比較展示，取得足夠歷史後才做台股績效宣稱。

## 10. 市場狀態的機率化研究

**R19｜Hamilton, J. D. (1989).** *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*. **Econometrica, 57(2), 357–384**。[作者出版目錄](https://econweb.ucsd.edu/~jhamilto/pub_metrics)，[期刊 DOI](https://doi.org/10.2307/1912559)。

- **功能**：新增馬可夫轉換模型，估計當時處於不同報酬／波動狀態的機率，與 `regime.py` 的大型股領漲及均線濾網比較。
- **方法邊界**：原文以總體景氣分析為背景，套用台股風險狀態屬於延伸。狀態不要事後命名成必然的牛／熊，也不要任意增加狀態數直到回測最好。
- **避免前視**：回測訊號須使用當時可知資料算出的 filtered probability，不能使用事後全樣本的 smoothed probability；模型亦須依預定時間重新估計。
- **驗收**：相同曝險與成本限制下，評估波動校準、回撤、狀態切換穩定性及換手。歷史樣本短、估計不收斂時要揭露或退回既有規則。
- **優先度**：較後，先把現有濾網的比較與成本做好。

## 共同研究規格

1. **先凍結問題與主要指標**：每個功能記下假設、候選數量、主要持有期、基準、股票池與停止調參時間。失敗結果也保留。
2. **資料時點與版本**：記錄公布日、修訂版、資料快照與程式版本。財報有歷史值不代表當時市場已知該版本，仍需查核重編及回填。
3. **訓練標籤須到期**：在 t 日訓練模型，排除報酬區間延伸到 t 日之後的樣本；跨切分重疊視窗也須隔離。預測、校準與調參不得使用尚未到期結果。
4. **誠實標示樣本**：2019 年後資料已在既有開發中被反覆觀察，不能因為新模型沒有直接 fit 它就稱為全新保留集。區分歷史時間切分評估、開發驗證及真正前向紀錄。
5. **同條件比較 LLM**：相同候選、可用資訊、選股數／曝險、成本與資金限制；另評估等待、空手與再評估的影響。模型聲稱的知識截止日期不足以證明完全無污染，前向決策紀錄仍應保留。
6. **保存完整實驗**：每個設定以 hash 識別，保存訓練截止日、預測、逐期報酬、成交、成本及缺值遮罩；`trials.jsonl` 作索引，不取代序列資料。
7. **不混淆績效來源**：分開展示訊號預測、持股權重、風控、成本的效果；台股特徵分組只能稱為自建代理因子，未按規格建構時不可冒稱正式 Fama–French 因子。
8. **理論與改編分開**：引用理論動機、採用公式、局部重現、台股延伸各自寫清楚；公開資料缺欄位時揭露限制，不用 LLM 補造歷史數據。

## 可直接驗收的第一輪交付

- [ ] 實驗規格文件：固定模型、候選、主要指標、持有期、時間切分與成本。
- [ ] 實驗索引及依 hash 保存的逐期結果；確認能重建所有候選，避免同 id 輸出覆寫歷史變體。
- [ ] Elastic Net／樹模型與固定綜合分數的共同樣本比較，附真正的訓練截止時間。
- [ ] 流動性分層、資金規模與滑價情境成績單，標明假設與尚未估計的真實衝擊。
- [ ] 波動區間的覆蓋率＋寬度＋interval score，並處理多期重疊與到期延遲。
- [ ] 開發題組的數字正負號、單位、期別與證據支持測試；不重跑保留集。

## 與原有文獻待辦的銜接

- **Lo、Mamaysky & Wang (2000)**：仍適合做形態方法比較，保留原規則作基準；條件報酬分布不同，不直接等於扣成本可獲利。
- **DSR／Newey–West**：補出處與實作差異，清楚說明估計假設；已有函式名稱不等於完成整篇論文驗證。
- **事件研究**：`event_study.py` 已有多種基準的持有期超額報酬。補市場模型 AR／CAR 是另一種分析；現有月份聚合 t 值仍可能受到跨月持有期重疊影響。
- **Reality Check／PBO**：先補齊各候選共同期間的序列，再決定檢定，不直接餵入 `trials.jsonl` 摘要。
- **LLM 文獻出版狀態**：Lopez-Lira & Tang 已刊於 [Journal of Financial Economics, 184, 104335 (2026)](https://www.sciencedirect.com/science/article/pii/S0304405X26001066)；Glasserman & Lin 已刊於 Journal of Financial Data Science, 6(1), 25–42 (2024)，見[作者學術頁](https://business.columbia.edu/pglasserman)。這兩篇屬原待辦的更新，不計入新增 19 篇。

本輪只整理文獻與設計，未訓練模型、重跑回測、呼叫付費 LLM、抓取新資料或改動現有決策組。

## 備註

以上不需要全部做。這份清單的用途是在需要時有期刊依據可以挑選，不是必須完成的範圍。做完 P1，再從 P2 挑一兩項，就足以回應「方法要有出處」的要求；其餘項目等研究問題真的需要時再做。
