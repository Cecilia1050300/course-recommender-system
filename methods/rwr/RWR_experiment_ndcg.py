import numpy as np
import pandas as pd

# ==========================================
# 1. 資料載入（換成真實大矩陣）
# ==========================================
def load_data(matrix_path: str, test_path: str):
    df_raw = pd.read_csv(matrix_path, index_col=0)
    df_raw.index = df_raw.index.astype(str)
    df_raw.columns = df_raw.columns.astype(str)

    test_df = pd.read_csv(test_path)
    test_df['Student_ID'] = test_df['Student_ID'].astype(str)
    test_df['Course_ID']  = test_df['Course_ID'].astype(str)

    # 建立 train（masked）和 truth
    df_train = df_raw.copy()
    df_truth = pd.DataFrame(0.0, index=df_raw.index, columns=df_raw.columns)

    for _, row in test_df.iterrows():
        u, i = row['Student_ID'], row['Course_ID']
        if u in df_train.index and i in df_train.columns:
            df_truth.loc[u, i] = df_raw.loc[u, i]
            df_train.loc[u, i] = 0.0

    return df_train, df_truth, test_df

# ==========================================
# 2. 圖建構
# ==========================================
def build_adjacency_matrix(R: np.ndarray) -> np.ndarray:
    M, N = R.shape
    A = np.zeros((M + N, M + N), dtype=np.float64)
    A[:M, M:] = R
    A[M:, :M] = R.T
    return A

def get_transition_matrix(A: np.ndarray, strategy: str, temperature: float = 1.0) -> np.ndarray:
    if strategy == "pure_binary":
        W = np.where(A > 0, 1.0, 0.0)
        s = W.sum(axis=1, keepdims=True)
        return W / np.where(s == 0, 1.0, s)

    elif strategy == "pure_linear":
        W = A.copy()
        s = W.sum(axis=1, keepdims=True)
        return W / np.where(s == 0, 1.0, s)

    elif strategy == "softmax":
        W = np.zeros_like(A, dtype=np.float64)
        for i in range(A.shape[0]):
            mask = A[i] > 0
            if mask.sum() == 0:
                continue
            scores = A[i, mask] / temperature
            scores -= scores.max()
            exp_s = np.exp(scores)
            W[i, mask] = exp_s / exp_s.sum()
        return W

    else:
        raise ValueError("請選擇 pure_binary、pure_linear 或 softmax")

# ==========================================
# 3. RWR
# ==========================================
def run_rwr(W: np.ndarray, target_idx: int, c: float = 0.35, max_iter: int = 200) -> np.ndarray:
    total = W.shape[0]
    E = np.zeros(total, dtype=np.float64)
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

# ==========================================
# 4. NDCG
# ==========================================
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
# 5. 對所有學生跑 RWR 並評估
# ==========================================
def evaluate_all_students(W, df_train, df_truth, test_df,
                           c=0.35, use_log_smooth=True):
    R_train = df_train.values.astype(np.float64)
    user_list = df_train.index.tolist()
    item_list = df_train.columns.tolist()
    M, N = R_train.shape

    u_idx_map = {u: i for i, u in enumerate(user_list)}
    i_idx_map = {i: j for j, i in enumerate(item_list)}

    # 只跑 test_df 裡出現的學生
    test_users = test_df['Student_ID'].unique()

    # 預先跑完每位學生的 RWR（避免重複計算）
    rwr_cache = {}
    for u_id in test_users:
        if u_id in u_idx_map:
            idx = u_idx_map[u_id]
            P = run_rwr(W, target_idx=idx, c=c)
            item_scores = P[M:].copy()

            if use_log_smooth:
                item_scores = np.log(item_scores + 1e-10)

            # Min-Max 只對未評分課程
            unrated_mask = (R_train[idx] == 0)
            pred_ratings = np.zeros(N)
            if unrated_mask.sum() > 0:
                vals = item_scores[unrated_mask]
                mn, mx = vals.min(), vals.max()
                if mx != mn:
                    pred_ratings[unrated_mask] = 1.0 + 4.0 * (
                        (item_scores[unrated_mask] - mn) / (mx - mn))
                else:
                    pred_ratings[unrated_mask] = 3.0

            rwr_cache[u_id] = pred_ratings

    # 逐筆計算誤差與收集 NDCG bundle
    errors = []
    ndcg_bundles = {}

    for _, row in test_df.iterrows():
        u, i = row['Student_ID'], row['Course_ID']
        true_s = row['Actual_Grade']

        if u not in rwr_cache or i not in i_idx_map:
            errors.append(5.0)
            pred_val = 0.0
        else:
            j = i_idx_map[i]
            pred_val = float(rwr_cache[u][j])
            errors.append(abs(true_s - pred_val))

        if u not in ndcg_bundles:
            ndcg_bundles[u] = []
        ndcg_bundles[u].append({'true': true_s, 'pred': pred_val})

    mae  = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    # 只對有 >=2 筆蓋牌測試資料的學生算 NDCG，單筆學生沒有排序可比較，
    # calculate_ndcg 對他們會硬回傳 1.0，混進平均會虛灌分數。
    ndcg_scores = [calculate_ndcg(records) for records in ndcg_bundles.values() if len(records) > 1]
    ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else float('nan')
    return mae, rmse, ndcg

# ==========================================
# 6. 掃描 c 值（含 NDCG）
# ==========================================
def sweep_c_values(W, df_train, df_truth, test_df, strategy_name):
    print(f"\n[{strategy_name}] 重啟率 c 掃描：")
    print(f"{'c 值':>8} | {'MAE':>8} | {'RMSE':>8} | {'NDCG':>8}")
    print("-" * 44)

    best_rmse, best_c = float('inf'), 0.35
    for c in [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
        mae, rmse, ndcg = evaluate_all_students(W, df_train, df_truth, test_df, c=c)
        mark = ""
        if rmse < best_rmse:
            best_rmse, best_c = rmse, c
            mark = " ← 最佳"
        print(f"{c:>8.2f} | {mae:>8.4f} | {rmse:>8.4f} | {ndcg:>8.4f}{mark}")

    return best_c

# ==========================================
# 7. 主程式
# ==========================================
if __name__ == "__main__":
    df_train, df_truth, test_df = load_data(
        'old/rating_matrix - rating_matrix.csv',
        'test_set.csv'
    )

    R_train = df_train.values.astype(np.float64)
    A = build_adjacency_matrix(R_train)

    results = []

    # Binary 和 Linear
    for label, key in [("Binary", "pure_binary"), ("Linear", "pure_linear")]:
        W = get_transition_matrix(A, key)
        best_c = sweep_c_values(W, df_train, df_truth, test_df, label)
        mae, rmse, ndcg = evaluate_all_students(W, df_train, df_truth, test_df, c=best_c)
        results.append({"Method": f"RWR_{label}", "Best_c": best_c,
                        "MAE": mae, "RMSE": rmse, "NDCG": ndcg})
        print(f"✅ {label} 完畢 | c={best_c} | MAE={mae:.4f} | RMSE={rmse:.4f} | NDCG={ndcg:.4f}")

    # Softmax + temperature 掃描
    print("\n[Softmax] Temperature 掃描：")
    print(f"{'temperature':>12} | {'best_c':>6} | {'MAE':>8} | {'RMSE':>8} | {'NDCG':>8}")
    print("-" * 52)

    best_softmax = {"RMSE": float('inf')}
    for temp in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        W_sm = get_transition_matrix(A, "softmax", temperature=temp)
        best_c = sweep_c_values(W_sm, df_train, df_truth, test_df, f"Softmax T={temp}")
        mae, rmse, ndcg = evaluate_all_students(
            W_sm, df_train, df_truth, test_df, c=best_c)
        mark = ""
        if rmse < best_softmax["RMSE"]:
            best_softmax = {"Method": f"RWR_Softmax(T={temp})", "Best_c": best_c,
                            "MAE": mae, "RMSE": rmse, "NDCG": ndcg}
            mark = " ← 最佳"
        print(f"{temp:>12.1f} | {best_c:>6.2f} | {mae:>8.4f} | {rmse:>8.4f} | {ndcg:>8.4f}{mark}")

    results.append(best_softmax)

    # 最終報告
    report_df = pd.DataFrame(results)[["Method", "Best_c", "MAE", "RMSE", "NDCG"]]
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print("\n" + "="*65)
    print("📊 RWR 三策略最終比較（含 NDCG）")
    print("="*65)
    print(report_df.to_string(index=False))
    print("="*65)