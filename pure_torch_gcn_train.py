"""
=============================================================================
大學課程推薦系統 — 純血 PyTorch 版 GCN 訓練架構 (免 PyG 依賴防封鎖版)
=============================================================================
學術創新亮點：
  1. 針對 Windows AppLocker/資安原則環境優化，完全剔除 PyG 外部 C++ 依賴。
  2. 使用純 PyTorch 矩陣張量算子，刻出經典 Thomas Kipf GCN 卷積層。
  3. 實作完整的反向傳播 (Backpropagation) 與 Adam 優化訓練迴圈。
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

# =============================================================================
# 1. 資料載入與純 PyTorch 圖結構對稱歸一化
# =============================================================================

def load_data(train_path, truth_path):
    df_train = pd.read_csv(train_path, index_col=0)
    df_truth = pd.read_csv(truth_path, index_col=0)
    user_ids = list(df_train.index)
    item_ids = list(df_train.columns)
    R_train = df_train.values.astype(np.float64)
    R_truth = df_truth.values.astype(np.float64)
    return R_train, R_truth, user_ids, item_ids

def build_pure_torch_gcn_matrix(R_train):
    """
    用純 PyTorch Tensor 實作 GCN 核心的對稱正規化鄰接矩陣: W_hat = D^{-1/2} * A_tilde * D^{-1/2}
    """
    M, N = R_train.shape
    total_nodes = M + N
    
    A = np.zeros((total_nodes, total_nodes), dtype=np.float64)
    W_binary = (R_train > 0).astype(np.float64)
    A[:M, M:] = W_binary
    A[M:, :M] = W_binary.T
    
    # 💡 GCN 核心：加入自環 (Self-loop)
    A_tilde = A + np.eye(total_nodes)
    
    # 計算度數矩陣 D_tilde
    row_sums = A_tilde.sum(axis=1)
    d_inv_sqrt = np.power(row_sums, -0.5, where=row_sums > 0)
    d_inv_sqrt[row_sums == 0] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    
    # W_hat = D^{-1/2} * A_tilde * D^{-1/2}
    W_hat_np = D_inv_sqrt.dot(A_tilde).dot(D_inv_sqrt)
    
    # 轉換成純 PyTorch FloatTensor (保證與神經網路相容)
    return torch.FloatTensor(W_hat_np)

# =============================================================================
# 2. 用純 PyTorch 刻出 GCNLayer 與 2-Layer 神經網路
# =============================================================================

class PureTorchGCNLayer(nn.Module):
    """ 純 PyTorch 刻出的 GCN 卷積層： H = W_hat * X * W """
    def __init__(self, in_dim, out_dim):
        super(PureTorchGCNLayer, self).__init__()
        # 宣告可以訓練的權重參數 (Weights)
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)  # 學術標準初始化

    def forward(self, x, W_hat):
        # 核心公式：先跟權重矩陣相乘，再跟對稱歸一化拓樸矩陣做訊息傳遞
        xw = torch.matmul(x, self.weight)
        out = torch.matmul(W_hat, xw)
        return out

class PureTorchGCNNet(nn.Module):
    """ 2-Layer GCN 網路 """
    def __init__(self, total_nodes, hidden_dim, out_dim):
        super(PureTorchGCNNet, self).__init__()
        # 初始特徵矩陣 X 直接使用 PyTorch 內建的 One-hot 恆等矩陣
        self.X_initial = nn.Parameter(torch.eye(total_nodes), requires_grad=False)
        
        self.gcn1 = PureTorchGCNLayer(total_nodes, hidden_dim)
        self.gcn2 = PureTorchGCNLayer(hidden_dim, out_dim)

    def forward(self, W_hat):
        # Layer 1 卷積 + ReLU 激活
        h1 = F.relu(self.gcn1(self.X_initial, W_hat))
        # Layer 2 卷積 -> 得到終極節點 Embedding
        h2 = self.gcn2(h1, W_hat)
        return h2

# =============================================================================
# 3. 評估與指標計算模組
# =============================================================================

def get_predictions_and_metrics(embeddings, M, N, R_train, R_truth):
    user_embs = embeddings[:M, :]
    item_embs = embeddings[M:, :]
    
    # 目標學生 U01 (索引為 0)
    target_user_emb = user_embs[0]
    raw_preds = torch.matmul(item_embs, target_user_emb).detach().numpy()
    
    # Min-Max 映射回 1~5 分
    max_val = raw_preds.max()
    min_val = raw_preds.min()
    if max_val != min_val:
        pred_ratings = 1.0 + 4.0 * ((raw_preds - min_val) / (max_val - min_val))
    else:
        pred_ratings = np.ones(N) * 3.0
        
    test_mask = (R_train[0] == 0) & (R_truth[0] > 0)
    y_true = R_truth[0][test_mask]
    y_pred = pred_ratings[test_mask]
    
    mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) > 0 else 0.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) > 0 else 0.0
    
    return mae, rmse, pred_ratings

# =============================================================================
# 4. 正式 Training Loop 啟動
# =============================================================================

if __name__ == "__main__":
    TRAIN_CSV = "rating_matrix_train.csv"
    TRUTH_CSV = "rating_matrix_truth.csv"
    
    R_train, R_truth, user_ids, item_ids = load_data(TRAIN_CSV, TRUTH_CSV)
    M, N = R_train.shape
    total_nodes = M + N
    
    # 建立歸一化矩陣並轉為 PyTorch 張量
    W_hat_tensor = build_pure_torch_gcn_matrix(R_train)
    
    # 宣告模型與 Adam 優化器
    model = PureTorchGCNNet(total_nodes=total_nodes, hidden_dim=12, out_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("\n" + "="*60)
    print("  🔥 啟動：純 PyTorch 自製版 GCN 權重反向傳播訓練迴圈")
    print("="*60)
    
    for epoch in range(1, 101):  # 跑 100 個 Epoch
        model.train()
        optimizer.zero_grad()
        
        # 前向傳播
        out_embeddings = model(W_hat_tensor)
        
        # 抽離嵌入向量計算協同損失
        user_embs = out_embeddings[:M, :]
        item_embs = out_embeddings[M:, :]
        
        # 找出訓練集裡所有正樣本（有修課的配對）
        train_user_idx, train_item_idx = np.where(R_train > 0)
        
        if len(train_user_idx) > 0:
            sel_user_embs = user_embs[train_user_idx]
            sel_item_embs = item_embs[train_item_idx]
            
            # 計算內積關聯度
            predictions = torch.sum(sel_user_embs * sel_item_embs, dim=1)
            target = torch.ones_like(predictions)
            
            # 計算 MSE 損失並進行反向傳播
            loss = F.mse_loss(predictions, target)
            loss.backward()
            optimizer.step()
            
        # 每 20 代觀測一次盲測指標是否下降（變聰明）
        if epoch % 20 == 0 or epoch == 1:
            mae, rmse, _ = get_predictions_and_metrics(out_embeddings, M, N, R_train, R_truth)
            print(f"  Epoch {epoch:3d} | 訓練 Loss: {loss.item():.4f} | 盲測 MAE: {mae:.4f} | 盲測 RMSE: {rmse:.4f}")

    # 最終發榜
    model.eval()
    final_embs = model(W_hat_tensor)
    mae, rmse, final_ratings = get_predictions_and_metrics(final_embs, M, N, R_train, R_truth)
    
    print("="*60)
    print(f"  🏁 【純 PyTorch 訓練完成最終報告】")
    print(f"  🏆 GCN 迭代優化後的真實 MAE  = {mae:.4f}")
    print(f"  🏆 GCN 迭代優化後的真實 RMSE = {rmse:.4f}")
    print("="*60)
    
    # 產出 Top-3 課程
    df_rec = pd.DataFrame({"課程代碼": item_ids, "評分": final_ratings, "已修": R_train[0]})
    df_rec = df_rec[df_rec["已修"] == 0].sort_values("評分", ascending=False).reset_index(drop=True)
    print(f"  🎓 [純 PyTorch 推薦結果] 目標學生 U01 的 Top-3 課程：")
    for i in range(3):
        print(f"  第 {i+1} 名：{df_rec.loc[i, '課程代碼']:<10} 預測對齊分數: {df_rec.loc[i, '評分']:.4f} 分")
    print("="*60 + "\n")