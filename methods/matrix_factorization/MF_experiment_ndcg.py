import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os

# ==========================================
# 1. 矩陣分解模型定義（不動）
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

# ★ 新增 NDCG 計算函數
def calculate_ndcg(user_records):
    if len(user_records) <= 1:
        return 1.0
    y_true = np.array([r['true'] for r in user_records])
    y_pred = np.array([r['pred'] for r in user_records])

    sorted_idx = np.argsort(y_pred)[::-1]
    dcg  = sum((2**y_true[i] - 1) / np.log2(rk + 2) for rk, i in enumerate(sorted_idx))
    idcg = sum((2**s - 1) / np.log2(rk + 2) for rk, s in enumerate(np.sort(y_true)[::-1]))
    return float(dcg / idcg) if idcg > 0 else 1.0

# ==========================================
# 2. 訓練與評估函數
# ==========================================
def run_mf_tuning_experiment(rating_df, test_df, epochs_list=[10, 20, 50, 100], lr=0.01, dim=64):
    print("🛡️ 執行 30% Masking 盲測準備...")
    train_df = rating_df.copy()

    for _, row in test_df.iterrows():
        u, i = str(row['Student_ID']), str(row['Course_ID'])
        if u in train_df.index and i in train_df.columns:
            train_df.loc[u, i] = 0

    zero_history_items = train_df.columns[(train_df > 0).sum() == 0].tolist()

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

    us, is_, rs = zip(*triplets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    us_t = torch.tensor(us, dtype=torch.long).to(device)
    is_t = torch.tensor(is_, dtype=torch.long).to(device)
    rs_t = torch.tensor(rs, dtype=torch.float32).to(device)
    g_mean = rs_t.mean().item()

    tuning_results = []

    for target_epoch in epochs_list:
        print(f"\n🔄 正在測試 Epochs = {target_epoch}...")

        model = MF(len(user_list), len(item_list), k=dim, global_mean=g_mean).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        loss_fn = nn.MSELoss()

        model.train()
        for epoch in range(target_epoch):
            optimizer.zero_grad()
            output = model(us_t, is_t)
            loss = loss_fn(output, rs_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            P, Q = model.user_emb.weight, model.item_emb.weight
            bu, bi = model.user_bias.weight, model.item_bias.weight.T
            mu = model.global_bias
            prediction_matrix = (torch.matmul(P, Q.T) + bu + bi + mu).cpu().numpy()

        df_pred = pd.DataFrame(prediction_matrix, index=user_list, columns=item_list)

        errors = []
        failed_count = 0
        ndcg_bundles = {}  # ★ 收集每位學生的預測紀錄

        for _, row in test_df.iterrows():
            u = str(row['Student_ID'])
            i = str(row['Course_ID'])
            true_s = row['Actual_Grade']

            if i in zero_history_items:
                errors.append(5.0)
                failed_count += 1
                pred_val = 0.0  # ★ 冷啟動給 0，排名會沉底，NDCG 會如實反映懲罰
            else:
                if u in df_pred.index and i in df_pred.columns:
                    pred_val = float(df_pred.loc[u, i])
                    errors.append(abs(true_s - pred_val))
                else:
                    errors.append(5.0)
                    failed_count += 1
                    pred_val = 0.0  # ★ 同上

            # ★ 收集進 NDCG bundle
            if u not in ndcg_bundles:
                ndcg_bundles[u] = []
            ndcg_bundles[u].append({'true': true_s, 'pred': pred_val})

        final_mae  = np.mean(errors)
        final_rmse = np.sqrt(np.mean(np.array(errors) ** 2))

        # ★ 計算平均 NDCG
        final_ndcg = np.mean([calculate_ndcg(records) for records in ndcg_bundles.values()])

        tuning_results.append({
            "Method"                  : "Matrix_Factorization",
            "Epochs_Setting"          : target_epoch,
            "MAE"                     : final_mae,
            "RMSE"                    : final_rmse,
            "NDCG"                    : final_ndcg,  # ★
            "Failed_Predictions (NaN)": failed_count
        })
        print(f"  ✅ Epochs={target_epoch} | RMSE: {final_rmse:.4f} | MAE: {final_mae:.4f} | NDCG: {final_ndcg:.4f}")

    return pd.DataFrame(tuning_results)

# ==========================================
# 3. 主程式
# ==========================================
if __name__ == "__main__":
    if os.path.exists('rating_matrix.csv') and os.path.exists('test_set.csv'):
        rating_data = pd.read_csv('old/rating_matrix - rating_matrix.csv', index_col=0).fillna(0)
        rating_data.index = rating_data.index.astype(str)
        rating_data.columns = rating_data.columns.astype(str)

        test_data = pd.read_csv('test_set.csv')
        test_data['Student_ID'] = test_data['Student_ID'].astype(str)
        test_data['Course_ID']  = test_data['Course_ID'].astype(str)

        final_report = run_mf_tuning_experiment(rating_data, test_data, epochs_list=[10, 20, 50, 100])
        final_report.to_csv('MF_Epochs_Convergence_Report.csv', index=False, encoding='utf-8-sig')

        pd.set_option('display.float_format', lambda x: f'{x:.4f}')
        print("\n" + "="*70)
        print("📊 MF 回合數收斂報告（含 NDCG）")
        print("="*70)
        print(final_report.to_string(index=False))
        print("="*70)
    else:
        print("❌ 請確認 rating_matrix.csv 與 test_set.csv 存在於資料夾中。")