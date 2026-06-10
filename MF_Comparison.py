import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. 矩陣分解模型定義
# ==========================================
class MF(nn.Module):
    def __init__(self, n_users, n_items, k=64, global_mean=0.0):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(n_users, k)
        self.item_emb = nn.Embedding(n_items, k)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(global_mean))
        
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

# ==========================================
# 2. 訓練與評估函數 (包含 Masking 與雙指標計算)
# ==========================================
def run_mf_experiment(rating_df, test_df, epochs=100, lr=0.01, dim=64):
    # --- A. 執行 Masking (挖洞) 預防資料外洩 ---
    print("🛡️  正在執行 Masking 動作，覆蓋訓練集中的測試資料...")
    train_df = rating_df.copy()
    mask_count = 0
    for _, row in test_df.iterrows():
        u, i = str(row['Student_ID']), str(row['Course_ID'])
        if u in train_df.index and i in train_df.columns:
            if train_df.loc[u, i] != 0:
                train_df.loc[u, i] = 0
                mask_count += 1
    print(f"✅ 已成功將 {mask_count} 筆測試資料從訓練矩陣中歸零 (Blind Test Mode)。")

    # --- B. 準備訓練資料 ---
    user_list = train_df.index.tolist()
    item_list = train_df.columns.tolist()
    u_map = {id: i for i, id in enumerate(user_list)}
    i_map = {id: i for i, id in enumerate(item_list)}

    triplets = []
    for u_id in user_list:
        for i_id in item_list:
            val = train_df.loc[u_id, i_id]
            if val > 0:
                triplets.append((u_map[u_id], i_map[i_id], val))
    
    if not triplets:
        print("❌ 錯誤：訓練集為空！")
        return None, None

    us, is_, rs = zip(*triplets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    us_t = torch.tensor(us, dtype=torch.long).to(device)
    is_t = torch.tensor(is_, dtype=torch.long).to(device)
    rs_t = torch.tensor(rs, dtype=torch.float32).to(device)

    # --- C. 初始化模型與優化器 ---
    g_mean = rs_t.mean().item()
    model = MF(len(user_list), len(item_list), k=dim, global_mean=g_mean).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    # --- D. 模型訓練 ---
    loss_history = []
    print(f"🚀 開始模型訓練 (Epochs: {epochs}, Dim: {dim})...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(us_t, is_t)
        loss = loss_fn(output, rs_t)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch [{epoch+1:3d}/{epochs}], Loss: {loss.item():.4f}")

    # --- E. 預測與多指標評估 (MAE & RMSE) ---
    model.eval()
    with torch.no_grad():
        P, Q = model.user_emb.weight, model.item_emb.weight
        bu, bi = model.user_bias.weight, model.item_bias.weight.T
        mu = model.global_bias
        prediction_matrix = (torch.matmul(P, Q.T) + bu + bi + mu).cpu().numpy()
    
    df_pred = pd.DataFrame(prediction_matrix, index=user_list, columns=item_list)

    errors = []
    sq_errors = []
    actuals, preds = [], []
    for _, row in test_df.iterrows():
        u, i, true_s = str(row['Student_ID']), str(row['Course_ID']), row['Actual_Grade']
        if u in df_pred.index and i in df_pred.columns:
            p_val = df_pred.loc[u, i]
            err = abs(true_s - p_val)
            errors.append(err)
            sq_errors.append(err ** 2)
            actuals.append(true_s)
            preds.append(p_val)

    # 最終計算指標
    final_mae = np.mean(errors)
    final_rmse = np.sqrt(np.mean(sq_errors))
    
    print("\n" + "="*40)
    print(f" 最終驗證結果 (K=3)")
    print(f"   測試集 MAE : {final_mae:.4f}")
    print(f"   測試集 RMSE: {final_rmse:.4f}")
    print("="*40)

    # --- F. 繪圖保存 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(loss_history, color='#1f77b4')
    ax1.set_title("Training Convergence (Loss Curve)")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("MSE Loss")
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(actuals, preds, alpha=0.5, color='#ff7f0e')
    ax2.plot([1, 5], [1, 5], 'k--', lw=2)
    ax2.set_title(f"MAE: {final_mae:.4f} | RMSE: {final_rmse:.4f}")
    ax2.set_xlabel("Actual Grade")
    ax2.set_ylabel("Predicted Grade")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Corrected_MF_Metrics.png')
    
    return final_mae, df_pred

# ==========================================
# 3. 主程式入口
# ==========================================
if __name__ == "__main__":
    if os.path.exists('rating_matrix.csv') and os.path.exists('test_set.csv'):
        # 讀取資料
        rating_data = pd.read_csv('rating_matrix.csv', index_col=0).fillna(0)
        rating_data.index = rating_data.index.astype(str)
        rating_data.columns = rating_data.columns.astype(str)
        
        test_data = pd.read_csv('test_set.csv')
        
        # 執行
        mae_val, pred_matrix = run_mf_experiment(rating_data, test_data)
        
        # 存檔摘要供 PPT 使用
        if mae_val:
            summary = pd.DataFrame([{"Method": "MF-SGD (Blind)", "MAE": mae_val}])
            summary.to_csv('Final_MF_Report.csv', index=False)
            print("📁 驗證圖表已儲存至 'Corrected_MF_Metrics.png'")
    else:
        print("❌ 錯誤：請確認 rating_matrix.csv 與 test_set.csv 存在於資料夾中。")