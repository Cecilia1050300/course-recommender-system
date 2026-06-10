# course-recommender-system
A comprehensive benchmark of course recommender systems using Collaborative Filtering, Matrix Factorization (PyTorch), and Random Walk with Restart (RWR) to address extreme data sparsity.

# 🎓 大學課程推薦系統：結合協同過濾、矩陣分解與隨機游走的混合式架構
An Advanced Academic Course Recommender System blending Memory-based CF, Matrix Factorization (PyTorch), and Random Walk with Restart (RWR).

## 🌟 專案亮點 (Key Features)
本專案專為解決**校園選課資料極端稀疏（Sparsity）**與**新課程冷啟動（Cold-Start）**問題而設計。透過嚴格的學術規格進行評估，包含以下核心實作：
- **嚴格評估與 5 分最大懲罰池機制**：實作 30% Masking 盲測，並杜絕「蹭全校平均分」的保底作弊漏洞，對於預測失效之極端稀疏樣本直接重罰最大誤差 5.0 分。
- **基於 PyTorch 的矩陣分解 (Matrix Factorization)**：利用隱含特徵空間（Latent Factor Space）捕捉全局修課規律，並完成多回合設定（Epochs）的收斂性測試（Convergence Test）。
- **圖論隨機游走 (Random Walk with Restart, RWR)**：將修課紀錄建模為二部圖（Bipartite Graph），並創新導入 **遮罩式橫向 Softmax (Masked Softmax)**，完美滿足轉移機率矩陣馬可夫鏈之橫向加總為 1.0 的數學定義。

---

## 📊 實驗結果大滿貫 (Experimental Results)
在引入嚴格的 5 分最大誤差懲罰機制後，各演算法在測試集上的對照成效如下：

| 推薦演算法 (Method) | 超參數設定 (K / Epoch) | MAE (↓) | RMSE (↓) | 失效預測次數 (NaN) | 學術定位與創新點 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Matrix_Factorization (MF)** | Epoch = 100 | **0.8051** | **1.0288** | 9 次 | **協同精準度之王**，背誦全局行為能力極強，但遭遇冷啟動盲區。 |
| **Item_Hybrid (本專題提出)** | K = 9 | **0.8546** | **1.0758** | 🏆 **0 次 (完美)** | **綜合表現冠軍**，引入 Word2Vec 語意特徵，徹底免疫冷啟動災難。 |
| **Item_Content** | K = 9 | 0.8820 | 1.1036 | 0 次 (完美) | 純語義模型，精準度稍遜於 Hybrid 模型。 |
| **Item_Rating** | K = 9 | 0.8855 | 1.1389 | 9 次 | 純行為項目 CF，受限於資料稀疏度。 |
| **User_Rating** | K = 9 | 0.9344 | 1.1959 | 9 次 | 傳統記憶型 User-based CF，表現中規中矩。 |

### 📈 圖論 RWR 橫向正規化策略交叉對比 (Top-5 for User 'U01')
本專案實作三種圖論轉移矩陣 W 的 Row-Normalization 策略。實驗發現，引入 **Masked Softmax** 能有效保留分數甜度相對權重，並具備高度的「推薦集中度」：
- **Strategy A (Binary)**: Top-1: `CS180`, Top-2: `CS140` (純結構關聯)
- **Strategy C (Masked Softmax)**: Top-1: `CS180`, Top-2: `CS200` (成功放大分數喜好偏好)

---

## 🛠️ 環境建置與執行說明 (Installation & Usage)

### 1. 安裝依賴套件
本專案使用 Python 3.8+ 開發，請先複製此 Repo 並執行以下指令安裝必要套件：
```bash
pip install -r requirements.txt
