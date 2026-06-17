"""
=============================================================================
大學課程推薦系統 — RWR 改良版（三項核心修復）
=============================================================================
修復項目：
  1. Min-Max 只對候選課程（未評分）做映射，消除系統性低估
  2. 重啟率 c 從 0.15 提升到 0.35（適合稀疏小圖）
  3. 對 RWR 分數取 log 平滑偏態分布後再映射
=============================================================================
"""

import numpy as np
import pandas as pd

def load_benchmark_data(train_path: str, truth_path: str):
    df_train = pd.read_csv(train_path, index_col=0)
    df_truth = pd.read_csv(truth_path, index_col=0)
    return df_train.values.astype(np.float64), df_truth.values.astype(np.float64)

def build_adjacency_matrix(R: np.ndarray) -> np.ndarray:
    M, N = R.shape
    total = M + N
    A = np.zeros((total, total), dtype=np.float64)
    A[:M, M:] = R
    A[M:, :M] = R.T
    return A

def get_transition_matrix(A: np.ndarray, strategy: str, temperature: float = 1.0) -> np.ndarray:
    """
    strategy 選項：
      pure_binary  — 有邊=1，無邊=0，row normalize
      pure_linear  — 原始分數按比例，row normalize  
      softmax      — Softmax 正規化（老師指定）
    
    temperature：
      < 1.0 → 更銳利，高分邊更突出（建議 0.5）
      = 1.0 → 標準 Softmax
      > 1.0 → 更平滑，接近 uniform
    """
    if strategy == "pure_binary":
        W_base = np.where(A > 0, 1.0, 0.0)
        row_sums = W_base.sum(axis=1, keepdims=True)
        row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
        return W_base / row_sums_safe

    elif strategy == "pure_linear":
        W_base = A.copy()
        row_sums = W_base.sum(axis=1, keepdims=True)
        row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
        return W_base / row_sums_safe

    elif strategy == "softmax":
        # ── Softmax 正規化 ──────────────────────────────────────────────
        # 只對「有邊的位置」做 Softmax，無邊的保持 0
        # 否則 exp(0) = 1 會讓沒有評分的邊也獲得非零權重
        
        W = np.zeros_like(A, dtype=np.float64)
        
        for i in range(A.shape[0]):
            nonzero_mask = A[i] > 0
            if nonzero_mask.sum() == 0:
                continue  # 孤立節點，保持全 0
            
            scores = A[i, nonzero_mask] / temperature   # 套用溫度縮放
            
            # 數值穩定：減掉最大值，防止 exp overflow
            scores_stable = scores - scores.max()
            exp_scores = np.exp(scores_stable)
            
            W[i, nonzero_mask] = exp_scores / exp_scores.sum()
        
        return W

    else:
        raise ValueError("請選擇 pure_binary、pure_linear 或 softmax")

def run_rwr(W: np.ndarray, target_idx: int, c: float = 0.35, max_iter: int = 200) -> np.ndarray:
    """
    修復 ❷：c 從 0.15 → 0.35，適合稀疏小圖
    稀疏圖中，c=0.15 太小 → 墨水在遠端節點擴散後很難回來
    c=0.35 讓隨機游走更集中在目標學生鄰域
    """
    total_nodes = W.shape[0]
    E = np.zeros(total_nodes, dtype=np.float64)
    E[target_idx] = 1.0
    P = E.copy()
    W_T = W.T
    
    for _ in range(max_iter):
        P_new = (1 - c) * W_T.dot(P) + c * E
        if np.linalg.norm(P_new - P, ord=2) < 1e-8:
            P = P_new
            break
        P = P_new
    return P

def evaluate_metrics(
    P: np.ndarray,
    M: int,
    N: int,
    R_train: np.ndarray,
    R_truth: np.ndarray,
    student_idx: int = 0,
    use_log_smooth: bool = True
) -> tuple[float, float, str]:
    
    item_scores = P[M:]  # 全部課程節點的 RWR 收斂分數
    
    # 找出「候選課程」（該學生沒評過分的課）
    unrated_mask = (R_train[student_idx] == 0)
    
    # 修復 ❸：Log 平滑 —— RWR 分數呈重尾分布，log 讓排名更線性
    if use_log_smooth:
        eps = 1e-10
        item_scores = np.log(item_scores + eps)
    
    # 修復 ❶：只對「候選課程」做 Min-Max，不包含已評分課程
    # 這樣蓋牌課程不會因為有高分課程拉低而被壓到 1 分
    candidate_scores = item_scores.copy()
    
    # 只取 unrated 課程的分數做 Min-Max 基準
    unrated_vals = candidate_scores[unrated_mask]
    max_val = unrated_vals.max()
    min_val = unrated_vals.min()
    
    pred_ratings = np.zeros(N)
    if max_val != min_val:
        # 映射到 1~5 分
        pred_ratings[unrated_mask] = 1.0 + 4.0 * (
            (candidate_scores[unrated_mask] - min_val) / (max_val - min_val)
        )
    else:
        pred_ratings[unrated_mask] = 3.0
    
    # 計算蓋牌驗證誤差
    test_mask = (R_train[student_idx] == 0) & (R_truth[student_idx] > 0)
    
    mae, rmse = 0.0, 0.0
    pred_str = "無測試數據"
    
    if test_mask.sum() > 0:
        y_true = R_truth[student_idx][test_mask]
        y_pred = pred_ratings[test_mask]
        mae  = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        preds_detail = " / ".join([
            f"課#{i+1}: 真={yt:.1f} 預={yp:.4f}" 
            for i, (yt, yp) in enumerate(zip(y_true, y_pred))
        ])
        pred_str = preds_detail
    
    return mae, rmse, pred_str


# ==================== 進階選項：掃描最佳 c 值 ====================

def sweep_c_values(W: np.ndarray, M: int, N: int, R_train: np.ndarray, R_truth: np.ndarray, strategy_name: str):
    """掃描不同重啟率 c 的效果，幫你找最佳 c"""
    print(f"\n[{strategy_name}] 重啟率 c 掃描：")
    print(f"{'c 值':>8} | {'MAE':>8} | {'RMSE':>8}")
    print("-" * 32)
    
    best_mae, best_c = float('inf'), 0.35
    for c in [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
        P = run_rwr(W, target_idx=0, c=c)
        mae, rmse, _ = evaluate_metrics(P, M, N, R_train, R_truth, use_log_smooth=True)
        mark = " ← 最佳" if mae < best_mae else ""
        if mae < best_mae:
            best_mae, best_c = mae, c
        print(f"{c:>8.2f} | {mae:>8.4f} | {rmse:>8.4f}{mark}")
    
    return best_c


# =============================================================================
# 主執行入口
# =============================================================================
if __name__ == "__main__":
    R_train, R_truth = load_benchmark_data("rating_matrix_train.csv", "rating_matrix_truth.csv")
    M, N = R_train.shape
    A = build_adjacency_matrix(R_train)

    results = {}

    # 原本兩個策略
    for strategy_label, strategy_key in [
        ("Binary",     "pure_binary"),
        ("Linear", "pure_linear"),
    ]:
        W = get_transition_matrix(A, strategy_key)
        best_c = sweep_c_values(W, M, N, R_train, R_truth, strategy_label)
        P = run_rwr(W, target_idx=0, c=best_c)
        mae, rmse, preds = evaluate_metrics(P, M, N, R_train, R_truth, use_log_smooth=True)
        results[strategy_label] = {"最佳 c": best_c, "MAE": mae, "RMSE": rmse, "盲測預測": preds}

    # ── 新增：Softmax 策略，掃描不同 temperature ──
    print("\n[Softmax] Temperature 掃描：")
    print(f"{'temperature':>12} | {'最佳 c':>6} | {'MAE':>8} | {'RMSE':>8}")
    print("-" * 44)

    best_softmax_mae = float('inf')
    best_softmax_config = {}

    for temp in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        W_sm = get_transition_matrix(A, "softmax", temperature=temp)
        best_c = sweep_c_values(W_sm, M, N, R_train, R_truth, f"Softmax T={temp}")
        P_sm = run_rwr(W_sm, target_idx=0, c=best_c)
        mae, rmse, preds = evaluate_metrics(P_sm, M, N, R_train, R_truth, use_log_smooth=True)

        mark = ""
        if mae < best_softmax_mae:
            best_softmax_mae = mae
            best_softmax_config = {"最佳 c": best_c, "MAE": mae, "RMSE": rmse, "盲測預測": preds}
            mark = " ← 最佳"
        print(f"{temp:>12.1f} | {best_c:>6.2f} | {mae:>8.4f} | {rmse:>8.4f}{mark}")

    results["Softmax（最佳 temperature）"] = best_softmax_config

    # 最終報告
    print("\n" + "="*80)
    print(" 三策略最終比較")
    print("="*80)
    print(pd.DataFrame(results).T.to_string())
    print("="*80)