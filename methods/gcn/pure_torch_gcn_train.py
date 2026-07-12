import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

# =============================================================================
# 1. 資料載入（換成真實大矩陣 + test_set.csv）
# =============================================================================
def load_data(matrix_path, test_path):
    df_raw = pd.read_csv(matrix_path, index_col=0)
    df_raw.index = df_raw.index.astype(str)
    df_raw.columns = df_raw.columns.astype(str)

    test_df = pd.read_csv(test_path)
    test_df['Student_ID'] = test_df['Student_ID'].astype(str)
    test_df['Course_ID']  = test_df['Course_ID'].astype(str)

    # 建立 masked train
    df_train = df_raw.copy()
    for _, row in test_df.iterrows():
        u, i = row['Student_ID'], row['Course_ID']
        if u in df_train.index and i in df_train.columns:
            df_train.loc[u, i] = 0.0

    user_ids = df_train.index.tolist()
    item_ids = df_train.columns.tolist()
    R_train  = df_train.values.astype(np.float64)
    R_raw    = df_raw.values.astype(np.float64)

    return R_train, R_raw, user_ids, item_ids, test_df

# =============================================================================
# 2. GCN 圖矩陣建構
# =============================================================================
def build_gcn_matrix(R_train):
    M, N = R_train.shape
    total = M + N
    A = np.zeros((total, total), dtype=np.float64)
    W_bin = (R_train > 0).astype(np.float64)
    A[:M, M:] = W_bin
    A[M:, :M] = W_bin.T
    A_tilde = A + np.eye(total)
    row_sums = A_tilde.sum(axis=1)
    d_inv = np.power(row_sums, -0.5, where=row_sums > 0)
    d_inv[row_sums == 0] = 0.0
    D = np.diag(d_inv)
    return torch.FloatTensor(D.dot(A_tilde).dot(D))

# =============================================================================
# 3. GCN 模型（不動）
# =============================================================================
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, W_hat):
        return torch.matmul(W_hat, torch.matmul(x, self.weight))

class GCNNet(nn.Module):
    def __init__(self, total_nodes, hidden_dim, out_dim):
        super().__init__()
        self.X = nn.Parameter(torch.eye(total_nodes), requires_grad=False)
        self.gcn1 = GCNLayer(total_nodes, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, out_dim)

    def forward(self, W_hat):
        h = F.dropout(F.relu(self.gcn1(self.X, W_hat)), p=0.1, training=self.training)
        return self.gcn2(h, W_hat)

# =============================================================================
# 4. NDCG
# =============================================================================
def calculate_ndcg(user_records):
    if len(user_records) <= 1:
        return 1.0
    y_true = np.array([r['true'] for r in user_records])
    y_pred = np.array([r['pred'] for r in user_records])
    sorted_idx = np.argsort(y_pred)[::-1]
    dcg  = sum((2**y_true[i] - 1) / np.log2(rk + 2) for rk, i in enumerate(sorted_idx))
    idcg = sum((2**s - 1) / np.log2(rk + 2) for rk, s in enumerate(np.sort(y_true)[::-1]))
    return float(dcg / idcg) if idcg > 0 else 1.0

# =============================================================================
# 5. 對所有學生評估（★ 核心修改）
# =============================================================================
def evaluate_all(embeddings, M, N, R_train, R_raw,
                 user_ids, item_ids, test_df):
    u_idx = {u: i for i, u in enumerate(user_ids)}
    i_idx = {i: j for j, i in enumerate(item_ids)}

    user_embs = embeddings[:M, :].detach().numpy()
    item_embs = embeddings[M:, :].detach().numpy()

    # ★ 預先算好每位出現在 test 的學生的預測向量
    pred_cache = {}
    for u_id in test_df['Student_ID'].unique():
        if u_id not in u_idx:
            continue
        uid = u_idx[u_id]
        raw = item_embs.dot(user_embs[uid])

        unrated_mask = (R_train[uid] == 0)
        pred = np.zeros(N)
        if unrated_mask.sum() > 0:
            vals = raw[unrated_mask]
            mn, mx = vals.min(), vals.max()
            if mx != mn:
                pred[unrated_mask] = 1.0 + 4.0 * (
                    (raw[unrated_mask] - mn) / (mx - mn))
            else:
                pred[unrated_mask] = 3.0
        pred_cache[u_id] = pred

    # ★ 逐筆計算誤差 + 收集 NDCG bundle
    errors = []
    ndcg_bundles = {}

    for _, row in test_df.iterrows():
        u, i = row['Student_ID'], row['Course_ID']
        true_s = row['Actual_Grade']

        if u not in pred_cache or i not in i_idx:
            errors.append(5.0)
            pred_val = 0.0
        else:
            pred_val = float(pred_cache[u][i_idx[i]])
            errors.append(abs(true_s - pred_val))

        if u not in ndcg_bundles:
            ndcg_bundles[u] = []
        ndcg_bundles[u].append({'true': true_s, 'pred': pred_val})

    mae  = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    ndcg = float(np.mean([calculate_ndcg(r) for r in ndcg_bundles.values()]))
    return mae, rmse, ndcg

# =============================================================================
# 6. 主程式
# =============================================================================
if __name__ == "__main__":
    R_train, R_raw, user_ids, item_ids, test_df = load_data(
        'old/rating_matrix - rating_matrix.csv',
        'test_set.csv'
    )
    M, N = R_train.shape
    W_hat = build_gcn_matrix(R_train)

    model = GCNNet(total_nodes=M+N, hidden_dim=12, out_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    print("\n" + "="*65)
    print("  🔥 GCN 訓練開始（20 epochs，全學生評估）")
    print("="*65)

    for epoch in range(1, 21):
        model.train()
        optimizer.zero_grad()
        embs = model(W_hat)
        u_e  = embs[:M]
        i_e  = embs[M:]
        tu, ti = np.where(R_train > 0)
        if len(tu) > 0:
            preds  = torch.sum(u_e[tu] * i_e[ti], dim=1)
            target = torch.FloatTensor(R_train[tu, ti])  # ★ 用真實分數當 target
            loss   = F.mse_loss(preds, target)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_embs = model(W_hat)
                mae, rmse, ndcg = evaluate_all(
                    val_embs, M, N, R_train, R_raw,
                    user_ids, item_ids, test_df)
            print(f"  Epoch {epoch:2d} | Loss: {loss.item():.4f} | "
                  f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | NDCG: {ndcg:.4f}")

    # 最終結果
    model.eval()
    with torch.no_grad():
        final_embs = model(W_hat)
        mae, rmse, ndcg = evaluate_all(
            final_embs, M, N, R_train, R_raw,
            user_ids, item_ids, test_df)

    print("="*65)
    print(f"  🏁 GCN 最終結果（第 20 代，全 {len(test_df)} 筆測試）")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  NDCG = {ndcg:.4f}")
    print("="*65)