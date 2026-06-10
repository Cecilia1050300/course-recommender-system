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
        
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.constant_(self.user_bias.weight, 0.0)
        nn.init.constant_(self.item_bias.weight, 0.0)

    def forward(self, u, i):
        u_e = self.user_emb(u)
        i_e = self.item_emb(i)
        pred = (u_e * i_e).sum(dim=1, keepdim=True)
        pred += self.user_bias(u) + self.item_bias(i) + self.global_bias
        return pred.squeeze()

# ==========================================
# 2. 訓練與優化評估函數 (加入老師指定的 5 分懲罰池機制)
# ==========================================
def run_mf_tuning_experiment(rating_df, test_df, epochs_list=[10, 20, 50, 100], lr=0.01, dim=64):
    print("🛡️ 執行 30% Masking 盲測準備（擦除考試答案）...")
    train_df = rating_df.copy()
    
    # 統計每一門課在原始訓練集裡剩下多少人修
    # 用來判定這門課是不是「在訓練集修課人數為 0」的死角冷門課
    for _, row in test_df.iterrows():
        u, i = str(row['Student_ID']), str(row['Course_ID'])
        if u in train_df.index and i in train_df.columns:
            train_df.loc[u, i] = 0  # 嚴格挖洞歸零

    # 找出在訓練集裡「完全沒有任何修課紀錄」的極端冷門課 ID 名單
    # 只要測試集遇到這些課，就代表傳統純行為分解法根本沒看過它，必須給予懲罰
    zero_history_items = train_df.columns[(train_df > 0).sum() == 0].tolist()

    # 建立 ID 對齊映射表
    user_list = train_df.index.tolist()
    item_list = train_df.columns.tolist()
    u_map = {id: i for i, id in enumerate(user_list)}
    i_map = {id: i for i, id in enumerate(item_list)}

    # 抽出有分數的格子建立三元組 (Triplets)
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
    g_mean = rs_t.mean().item()

    tuning_results = []

    # 💡【實作老師意見 2】自動迴圈測試不同的回合數設定 (Epochs)
    for target_epoch in epochs_list:
        print(f"\n🔄 正在測試回合數設定: Epochs = {target_epoch}...")
        
        # 每次測試重新初始化模型，確保實驗公平獨立
        model = MF(len(user_list), len(item_list), k=dim, global_mean=g_mean).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        loss_fn = nn.MSELoss()

        # 模型訓練
        model.train()
        for epoch in range(target_epoch):
            optimizer.zero_grad()
            output = model(us_t, is_t)
            loss = loss_fn(output, rs_t)
            loss.backward()
            optimizer.step()

        # 預測推導
        model.eval()
        with torch.no_grad():
            P, Q = model.user_emb.weight, model.item_emb.weight
            bu, bi = model.user_bias.weight, model.item_bias.weight.T
            mu = model.global_bias
            prediction_matrix = (torch.matmul(P, Q.T) + bu + bi + mu).cpu().numpy()
        
        df_pred = pd.DataFrame(prediction_matrix, index=user_list, columns=item_list)

        # 多指標評估與懲罰池計算
        errors = []
        failed_count = 0
        
        for _, row in test_df.iterrows():
            u, i, true_s = str(row['Student_ID']), str(row['Course_ID']), row['Actual_Grade']
            
            # 💡【落實老師意見 1】核心懲罰機制：
            # 如果這門課在訓練集裡修課人數為 0，代表純行為模型完全沒線索
            # 不准蹭全校平均分，絕對誤差直接給予最大值 5.0 的重度懲罰！
            if i in zero_history_items:
                errors.append(5.0)
                failed_count += 1
            else:
                if u in df_pred.index and i in df_pred.columns:
                    p_val = df_pred.loc[u, i]
                    errors.append(abs(true_s - p_val))
                else:
                    errors.append(5.0) # 找不到對齊座標同樣給予 5 分懲罰
                    failed_count += 1

        final_mae = np.mean(errors)
        final_rmse = np.sqrt(np.mean(np.array(errors)**2))

        tuning_results.append({
            "Method": "Matrix_Factorization",
            "Epochs_Setting": target_epoch,
            "MAE": final_mae,
            "RMSE": final_rmse,
            "Failed_Predictions (NaN)": failed_count
        })
        print(f"✅ Epochs = {target_epoch} 觀測完畢。測試集 RMSE: {final_rmse:.4f}")

    # 彙整為 DataFrame
    report_df = pd.DataFrame(tuning_results)
    return report_df

# ==========================================
# 3. 主程式入口
# ==========================================
if __name__ == "__main__":
    if os.path.exists('rating_matrix.csv') and os.path.exists('test_set.csv'):
        rating_data = pd.read_csv('rating_matrix.csv', index_col=0).fillna(0)
        rating_data.index = rating_data.index.astype(str)
        rating_data.columns = rating_data.columns.astype(str)
        
        test_data = pd.read_csv('test_set.csv')
        
        # 執行自動調參實驗（回合數設定設定為 10, 20, 50, 100 觀察收斂）
        epochs_to_test = [10, 20, 50, 100]
        final_report = run_mf_tuning_experiment(rating_data, test_data, epochs_list=epochs_to_test)
        
        # 儲存為對照 CSV 檔案
        final_report.to_csv('MF_Epochs_Convergence_Report.csv', index=False, encoding='utf-8-sig')
        
        print("\n" + "="*65)
        print("📊 矩陣分解 (MF) 回合數設定與收斂觀測報告表")
        print("="*65)
        print(final_report.to_string(index=False))
        print("="*65)
        print("💾 任務成功！【MF_Epochs_Convergence_Report.csv】已成功儲存，快拿這份數據去報告！\n")
    else:
        print("❌ 錯誤：請確認 rating_matrix.csv 與 test_set.csv 存在於資料夾中。")