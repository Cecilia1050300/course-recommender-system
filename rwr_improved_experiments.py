"""
=============================================================================
大學課程推薦系統 — RWR Softmax 三大純圖論改良實驗室
=============================================================================
學術對照亮點：
  1. 獨立於原始基礎策略，專注於解決 Softmax「贏家全拿、邊緣節點窒息」的缺陷。
  2. 整合三大純圖論改良機制於單一優化函式中：
     - 改良一：溫度係數（Temperature Scaling）-> 平滑極端機率
     - 改良二：Top-K 結構剪枝（Structural Pruning）-> 切斷全局大課噪聲
     - 改良三：度數逆向懲罰（Inverse Degree Penalty）-> 幫熱門明星課消磁
  3. 自動跑完四大對照組，輸出學術級 MAE / RMSE 指標大總表！
=============================================================================
"""

import numpy as np
import pandas as pd


# =============================================================================
# 1. 基礎圖論資料管線（完美對齊原有規格）
# =============================================================================

def load_benchmark_data(train_path: str, truth_path: str) -> tuple[np.ndarray, np.ndarray, list, list]:
    df_train = pd.read_csv(train_path, index_col=0)
    df_truth = pd.read_csv(truth_path, index_col=0)
    user_ids = list(df_train.index)
    item_ids = list(df_train.columns)
    R_train = df_train.values.astype(np.float64)
    R_truth = df_truth.values.astype(np.float64)
    return R_train, R_truth, user_ids, item_ids


def build_adjacency_matrix(R: np.ndarray) -> np.ndarray:
    M, N = R.shape
    total = M + N
    A = np.zeros((total, total), dtype=np.float64)
    A[:M, M:] = R
    A[M:, :M] = R.T
    return A


# =============================================================================
# 2. 核心：獨立封裝的大滿貫改良型 Softmax 轉移矩陣生成器
# =============================================================================

def build_improved_softmax_matrix(
    A: np.ndarray, 
    temperature: float = 1.0,      # 改良一：溫度係數（越高端越平滑趨近 Binary）
    top_k: int = None,              # 改良二：Top-K 局部剪枝（只保留前 K 強的通道）
    penalty_degree: bool = False   # 改良三：度數逆向懲罰（大課消磁機制）
) -> np.ndarray:
    """
    透過參數控制，自由切換、組合三大改良機制的密碼級圖論轉移矩陣函式。
    """
    # 複製一份原始鄰接矩陣進行修改，不破壞原始數據
    A_mod = A.copy()
    
    # ------ 【改良三：度數逆向懲罰（度數消磁）】 ------
    if penalty_degree:
        # 計算不包含自環時，每個節點原本的連通度度數（出度）
        degrees = (A_mod > 0).sum(axis=1, keepdims=True)
        # 保護孤立節點
        degrees_safe = np.where(degrees == 0, 1.0, degrees)
        # 被越多人修過的課，分數權重會被依比例壓低
        A_mod = A_mod / degrees_safe

    # ------ 【改良一：溫度係數調節】 ------
    # 在進入指數層前，將所有分數除以溫度常數 tau
    A_mod = A_mod / temperature

    # ------ 【改良二：Top-K 局部通道剪枝】 ------
    if top_k is not None:
        for i in range(A_mod.shape[0]):
            row = A_mod[i]
            # 如果這一列非零的通道數大於我們設定的 Top-K，就啟動殘酷剪枝
            if np.count_nonzero(row) > top_k:
                # 找出這一列中第 K 大的分數作為門檻
                threshold = np.partition(row, -top_k)[-top_k]
                # 低於門檻的通道，直接築牆封死（改成 0）
                A_mod[i] = np.where(row >= threshold, row, 0.0)

    # ------ 【標準遮罩式橫向 Softmax 計算】 ------
    mask = (A_mod > 0)
    A_masked = np.where(mask, A_mod, -1e9)  # 未連通通道設為 -inf
    
    row_max = A_masked.max(axis=1, keepdims=True)  # 數值穩定化平移
    exp_A = np.exp(A_masked - row_max)
    exp_A = np.where(mask, exp_A, 0.0)             # 強制遮罩外清零
    
    row_sums = exp_A.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    W = exp_A / row_sums_safe
    return W


# =============================================================================
# 3. RWR 核心迭代與盲測指標比對模組
# =============================================================================

def run_rwr_core(W: np.ndarray, target_idx: int, c: float = 0.15, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    total_nodes = W.shape[0]
    E = np.zeros(total_nodes, dtype=np.float64)
    E[target_idx] = 1.0
    P = E.copy()
    W_T = W.T
    
    for _ in range(max_iter):
        P_new = (1 - c) * W_T.dot(P) + c * E
        if np.linalg.norm(P_new - P, ord=2) < tol:
            P = P_new
            break
        P = P_new
    return P


def evaluate_metrics(P: np.ndarray, M: int, N: int, R_train: np.ndarray, R_truth: np.ndarray) -> tuple[float, float, str]:
    item_scores = P[M:]  # 抽出課程收斂機率
    
    # Min-Max 分數對齊映射
    max_val = item_scores.max()
    min_val = item_scores.min()
    if max_val != min_val:
        pred_ratings = 1.0 + 4.0 * ((item_scores - min_val) / (max_val - min_val))
    else:
        pred_ratings = np.ones(N) * 3.0
        
    # 抓出隱藏的蓋牌盲測集
    test_mask = (R_train[0] == 0) & (R_truth[0] > 0)
    
    mae, rmse = 0.0, 0.0
    if test_mask.sum() > 0:
        y_true = R_truth[0][test_mask]
        y_pred = pred_ratings[test_mask]
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        
    return mae, rmse, f"{pred_ratings[test_mask][0]:.2f}分 / {pred_ratings[test_mask][1]:.2f}分"


# =============================================================================
# 4. 實驗自動化控制入口
# =============================================================================

if __name__ == "__main__":
    TRAIN_CSV = "rating_matrix_train.csv"
    TRUTH_CSV = "rating_matrix_truth.csv"
    
    # 載入模擬蓋牌選課資料
    R_train, R_truth, user_ids, item_ids = load_benchmark_data(TRAIN_CSV, TRUTH_CSV)
    M, N = R_train.shape
    
    # 建立無向二部圖
    A = build_adjacency_matrix(R_train)
    
    # 宣告儲存實驗結果的字典
    results_summary = {}
    
    # -------------------------------------------------------------------------
    # 【實驗組一：標準原始 Softmax（對齊你剛剛爆掉的 3.22 數據）】
    # -------------------------------------------------------------------------
    W1 = build_improved_softmax_matrix(A, temperature=1.0, top_k=None, penalty_degree=False)
    P1 = run_rwr_core(W1, target_idx=0)
    mae1, rmse1, preds1 = evaluate_metrics(P1, M, N, R_train, R_truth)
    results_summary["1. 原始標準 Softmax"] = {"MAE": mae1, "RMSE": rmse1, "盲測兩課預測得分": preds1}
    
    # -------------------------------------------------------------------------
    # 【實驗組二：改良方案一 ➔ 啟動「高溫平滑化」（Temperature = 3.5）】
    # -------------------------------------------------------------------------
    W2 = build_improved_softmax_matrix(A, temperature=3.5, top_k=None, penalty_degree=False)
    P2 = run_rwr_core(W2, target_idx=0)
    mae2, rmse2, preds2 = evaluate_metrics(P2, M, N, R_train, R_truth)
    results_summary["2. 高溫平滑化 (tau=3.5)"] = {"MAE": mae2, "RMSE": rmse2, "盲測兩課預測得分": preds2}
    
    # -------------------------------------------------------------------------
    # 【實驗組三：改良方案二 ➔ 啟動「Top-3 局部通道剪枝」】
    # -------------------------------------------------------------------------
    W3 = build_improved_softmax_matrix(A, temperature=1.0, top_k=3, penalty_degree=False)
    P3 = run_rwr_core(W3, target_idx=0)
    mae3, rmse3, preds3 = evaluate_metrics(P3, M, N, R_train, R_truth)
    results_summary["3. 局部 Top-3 結構剪枝"] = {"MAE": mae3, "RMSE": rmse3, "盲測兩課預測得分": preds3}
    
    # -------------------------------------------------------------------------
    # 【實驗組四：改良方案三 ➔ 啟動「熱門大課度數消磁」】
    # -------------------------------------------------------------------------
    W4 = build_improved_softmax_matrix(A, temperature=1.0, top_k=None, penalty_degree=True)
    P4 = run_rwr_core(W4, target_idx=0)
    mae4, rmse4, preds4 = evaluate_metrics(P4, M, N, R_train, R_truth)
    results_summary["4. 明星課度數消磁懲罰"] = {"MAE": mae4, "RMSE": rmse4, "盲測兩課預測得分": preds4}
    
    # =============================================================================
    # 🏁 終極發榜報告：大總表輸出
    # =============================================================================
    print("\n" + "="*70)
    print("  🏆 【純圖論 Softmax 改良大滿貫對照總表】")
    print("="*70)
    df_report = pd.DataFrame(results_summary).T
    print(df_report.to_string())
    print("="*70 + "\n")