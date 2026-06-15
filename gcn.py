"""
=============================================================================
大學課程推薦系統 — 圖卷積網路 (Graph Convolutional Network, GCN)
=============================================================================
演算法架構：
  1. 讀取修課矩陣，建立包含自環（Self-loop）的全局無向鄰接矩陣 A_tilde
  2. 計算對角線度數矩陣 D_tilde，並實作對稱對角正規化 D^{-1/2} * A * D^{-1/2}
  3. 初始化節點特徵矩陣 X（使用隨機向量模擬課程與學生的初始特徵嵌入）
  4. 執行兩層 GCN 前向傳播（Message Passing & Aggregation）：
     Layer 1: H1 = ReLU( W_hat * X * W1 )
     Layer 2: H2 = W_hat * H1 * W2
  5. 解析出學生與課程的最終 Embedding，利用內積（Dot Product）預測分數
  6. 評估機制：針對盲測集計算 MAE / RMSE
=============================================================================
"""

import numpy as np
import pandas as pd

# =============================================================================
# 1. 資料載入與圖結構建置
# =============================================================================

def load_data(train_path, truth_path):
    df_train = pd.read_csv(train_path, index_col=0)
    df_truth = pd.read_csv(truth_path, index_col=0)
    
    user_ids = list(df_train.index)
    item_ids = list(df_train.columns)
    
    R_train = df_train.values.astype(np.float64)
    R_truth = df_truth.values.astype(np.float64)
    return R_train, R_truth, user_ids, item_ids

def build_gcn_normalized_adjacency(R_train):
    """
    實作 GCN 核心的對稱正規化鄰接矩陣: W_hat = D^{-1/2} * A_tilde * D^{-1/2}
    """
    M, N = R_train.shape
    total_nodes = M + N
    
    # 建立基礎二部圖鄰接矩陣
    A = np.zeros((total_nodes, total_nodes), dtype=np.float64)
    # 這裡將大於 0 的修課行為視為拓樸連線 (1)
    W_binary = (R_train > 0).astype(np.float64)
    A[:M, M:] = W_binary
    A[M:, :M] = W_binary.T
    
    # 💡 GCN 核心特點：加入自環 (Self-loop)，讓節點更新時也能保留自己的特徵
    A_tilde = A + np.eye(total_nodes)
    
    # 計算度數矩陣 D_tilde (每列的加總)
    row_sums = A_tilde.sum(axis=1)
    
    # 計算 D^{-1/2}
    d_inv_sqrt = np.power(row_sums, -0.5, where=row_sums > 0)
    d_inv_sqrt[row_sums == 0] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    # 對稱正規化 W_hat = D^{-1/2} * A_tilde * D^{-1/2}
    W_hat = D_inv_sqrt.dot(A_tilde).dot(D_inv_sqrt)
    
    print(f"[GCN 圖建置] 總節點數: {total_nodes} (學生: {M}, 課程: {N})")
    print(f"[GCN 圖建置] 正規化矩陣驗證 - 最大值: {W_hat.max():.4f}, 最小值: {W_hat.min():.4f}")
    return W_hat

# =============================================================================
# 2. GCN 模型前向傳播機制 (Forward Propagation)
# =============================================================================

def relu(x):
    return np.maximum(0, x)

def forward_gcn(W_hat, num_nodes, feature_dim=16, hidden_dim=8, out_dim=4, seed=42):
    """
    模擬 2-Layer GCN 的特徵聚合與變換過程
    """
    rng = np.random.default_rng(seed)
    
    # 🚀 步驟一：初始化節點特徵 X (長度 = 總節點數 x 初始特徵維度)
    # 現實中這可以用 Word2Vec 跑課程大綱，這裡使用標準常態分佈隨機初始化
    X = rng.normal(0, 0.1, size=(num_nodes, feature_dim))
    
    # 🚀 步驟二：定義兩層的神經網路權重矩陣 (Weights)
    W1 = rng.normal(0, 0.1, size=(feature_dim, hidden_dim))
    W2 = rng.normal(0, 0.1, size=(hidden_dim, out_dim))
    
    # 🚀 步驟三：Layer 1 卷積 -> 鄰居特徵聚合 + 線性變換 + ReLU 激活
    # H1 = ReLU( W_hat * X * W1 )
    Z1 = W_hat.dot(X).dot(W1)
    H1 = relu(Z1)
    
    # 🚀 步驟四：Layer 2 卷積 -> 得到最終降維後的節點 Embedding (空間表徵)
    # H2 = W_hat * H1 * W2
    H2 = W_hat.dot(H1).dot(W2)
    
    print(f"[GCN 前向傳播] 成功生成最終節點 Embedding，特徵維度壓縮軌跡: {feature_dim} -> {hidden_dim} -> {out_dim}")
    return H2

# =============================================================================
# 3. 分數預測、盲測評估與 Top-N 推薦
# =============================================================================

def predict_and_evaluate(H2, M, N, R_train, R_truth, target_user_idx, item_ids, top_n=5):
    # 抽離出學生 Embedding 與 課程 Embedding
    user_embeddings = H2[:M, :]  # Shape: (M, out_dim)
    item_embeddings = H2[M:, :]  # Shape: (N, out_dim)
    
    # 利用內積 (Dot Product) 計算目標學生對所有課程的原始預測得分
    target_user_emb = user_embeddings[target_user_idx] # (out_dim,)
    raw_predictions = item_embeddings.dot(target_user_emb) # (N,)
    
    # ➔ 將內積學到的神經網路分數，Min-Max 映射回 1.0 ~ 5.0 分的空間
    max_val = raw_predictions.max()
    min_val = raw_predictions.min()
    if max_val != min_val:
        pred_ratings = 1.0 + 4.0 * ((raw_predictions - min_val) / (max_val - min_val))
    else:
        pred_ratings = np.ones(N) * 3.0
        
    # --- 計算 MAE / RMSE (針對被挖洞的測試集) ---
    test_mask = (R_train[target_user_idx] == 0) & (R_truth[target_user_idx] > 0)
    test_count = int(test_mask.sum())
    
    mae, rmse = 0.0, 0.0
    if test_count > 0:
        y_true = R_truth[target_user_idx][test_mask]
        y_pred = pred_ratings[test_mask]
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        
        print(f"\n  🎯 [GCN 盲測效果評估] (測試樣本數: {test_count})")
        print(f"  🏆 【觀測結果】 GCN_MAE = {mae:.4f}  |  GCN_RMSE = {rmse:.4f}")
    
    # --- 輸出 Top-N 推薦結果 ---
    df = pd.DataFrame({
        "課程代碼": item_ids,
        "GCN預測評分": pred_ratings,
        "原始修課狀態": R_train[target_user_idx]
    })
    
    # 排除已修課程
    df_unvisited = df[df["原始修課狀態"] == 0].copy()
    df_unvisited = df_unvisited.sort_values("GCN預測評分", ascending=False).reset_index(drop=True)
    top_recommendations = df_unvisited.head(top_n)[["課程代碼", "GCN預測評分"]]
    
    print(f"\n  🎓 [GCN 推薦結果] 目標學生 {target_user_idx} 的 Top-{top_n} 課程：")
    for rank, (_, row) in enumerate(top_recommendations.iterrows(), 1):
        print(f"  第 {rank} 名：{row['課程代碼']:<15} 預測對齊評分 = {row['GCN預測評分']:.4f} 分")
        
    return mae, rmse

# =============================================================================
# 主程式入口
# =============================================================================

if __name__ == "__main__":
    # 使用你昨天跑出來的兩張模擬資料表進行基準測試
    TRAIN_CSV = "rating_matrix_train.csv"
    TRUTH_CSV = "rating_matrix_truth.csv"
    
    print("\n" + "="*60)
    print("  🚀 啟動最新階段：Graph Convolutional Network (GCN) 模型")
    print("="*60)
    
    # 1. 載入資料
    R_train, R_truth, user_ids, item_ids = load_data(TRAIN_CSV, TRUTH_CSV)
    M, N = R_train.shape
    
    # 2. 建立 GCN 特有的對稱歸一化圖拓樸結構
    W_hat = build_gcn_normalized_adjacency(R_train)
    
    # 3. 執行兩層空間域 GCN 特徵聚合傳播
    H2 = forward_gcn(W_hat, num_nodes=M+N, feature_dim=16, hidden_dim=8, out_dim=4)
    
    # 4. 計算指標與產出推薦
    predict_and_evaluate(H2, M, N, R_train, R_truth, target_user_idx=0, item_ids=item_ids, top_n=5)
    print("="*60 + "\n")