import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 使用 PyTorch 實作的 SGD Matrix Factorization
# ==========================================
def sgd_mf(train_df: pd.DataFrame, test_df: pd.DataFrame, epochs=100, lr=0.01, dim=64):
    n_users, n_items = train_df.shape
    user_list = train_df.index.tolist()
    item_list = train_df.columns.tolist()
    
    # ID 到索引的映射
    u_map = {id: i for i, id in enumerate(user_list)}
    i_map = {id: i for i, id in enumerate(item_list)}

    class MF(nn.Module):
        def __init__(self, n_users, n_items, k=64):
            super(MF, self).__init__()
            self.user_emb = nn.Embedding(n_users, k)
            self.item_emb = nn.Embedding(n_items, k)
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            # 全局平均值作為初始偏置
            self.global_bias = nn.Parameter(torch.tensor(train_df.values[train_df.values > 0].mean()))
            
            # 初始化參數，避免梯度爆炸
            nn.init.normal_(self.user_emb.weight, std=0.01)
            nn.init.normal_(self.item_emb.weight, std=0.01)
            nn.init.constant_(self.user_bias.weight, 0.0)
            nn.init.constant_(self.item_bias.weight, 0.0)

        def forward(self, u, i):
            u_e = self.user_emb(u)
            i_e = self.item_emb(i)
            # 矩陣分解核心：向量內積 + 偏置
            pred = (u_e * i_e).sum(dim=1, keepdim=True)
            pred += self.user_bias(u) + self.item_bias(i) + self.global_bias
            return pred.squeeze()

    # 準備訓練三元組
    triplets = []
    for u_id in user_list:
        for i_id in item_list:
            val = train_df.loc[u_id, i_id]
            if val > 0:
                triplets.append((u_map[u_id], i_map[i_id], val))
    
    us, is_, rs = zip(*triplets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    us_t = torch.tensor(us, dtype=torch.long).to(device)
    is_t = torch.tensor(is_, dtype=torch.long).to(device)
    rs_t = torch.tensor(rs, dtype=torch.float32).to(device)

    # 初始化模型與優化器
    model = MF(n_users, n_items, k=dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    # --- 開始訓練與記錄 Loss ---
    loss_history = []
    print("🚀 Training SGD MF Model...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(us_t, is_t)
        loss = loss_fn(output, rs_t)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1:3d}/{epochs}], Loss: {loss.item():.4f}")

    # --- 繪製 Loss Curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='#1f77b4', label='Training Loss')
    plt.title('Model Convergence: Loss Curve', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('loss_curve.png')
    print("✅ Loss Curve saved as 'loss_curve.png'")

    # --- 預測階段 ---
    model.eval()
    with torch.no_grad():
        P, Q = model.user_emb.weight, model.item_emb.weight
        bu, bi = model.user_bias.weight, model.item_bias.weight.T
        mu = model.global_bias
        res = (torch.matmul(P, Q.T) + bu + bi + mu).cpu().numpy()
    
    pred_df = pd.DataFrame(res, index=user_list, columns=item_list)

    # --- 繪製 Predicted vs Actual Scatter Plot ---
    actuals, preds = [], []
    for _, row in test_df.iterrows():
        u, i, true_s = str(row['Student_ID']), str(row['Course_ID']), row['Actual_Grade']
        if u in pred_df.index and i in pred_df.columns:
            actuals.append(true_s)
            preds.append(pred_df.loc[u, i])

    plt.figure(figsize=(6, 6))
    plt.scatter(actuals, preds, alpha=0.6, color='#ff7f0e', label='Data Points')
    # 畫出一條 y=x 的斜線作為基準
    limit = [min(actuals)-0.5, max(actuals)+0.5]
    plt.plot(limit, limit, 'k--', alpha=0.5, label='Ideal Predict')
    plt.title('Prediction Accuracy: Actual vs. Predicted', fontsize=14)
    plt.xlabel('Actual Grade')
    plt.ylabel('Predicted Grade')
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_scatter.png')
    print("✅ Scatter Plot saved as 'prediction_scatter.png'")

    return pred_df

# ==========================================
# 2. 執行流程
# ==========================================
if __name__ == "__main__":
    print("📂 讀取資料中...")
    if not os.path.exists('rating_matrix.csv') or not os.path.exists('test_set.csv'):
        print("❌ 錯誤：找不到 CSV 檔案，請確認路徑。")
    else:
        # 讀取訓練資料
        rating_df = pd.read_csv('rating_matrix.csv', index_col=0).fillna(0)
        rating_df.index = rating_df.index.astype(str)
        rating_df.columns = rating_df.columns.astype(str)
        
        # 讀取測試資料
        test_df = pd.read_csv('test_set.csv')
        
        # 執行 MF (傳入 test_df 用於繪圖)
        df_pred_sgd = sgd_mf(rating_df, test_df)

        # 評估 MAE
        errors = []
        for _, row in test_df.iterrows():
            u, i, true_s = str(row['Student_ID']), str(row['Course_ID']), row['Actual_Grade']
            if u in df_pred_sgd.index and i in df_pred_sgd.columns:
                pred = df_pred_sgd.loc[u, i]
                errors.append(abs(true_s - pred))

        mae = np.mean(errors)
        print("\n" + "="*40)
        print(f"📊 矩陣分解 (MF-SGD) 實驗結果報表")
        print(f"   最終 MAE: {mae:.4f}")
        print("="*40)

        # 存檔摘要
        pd.DataFrame([{"Method": "MF_SGD_PyTorch", "MAE": mae}]).to_csv('MF_MAE_Summary.csv', index=False)
        print("📁 摘要已存至 'MF_MAE_Summary.csv'")