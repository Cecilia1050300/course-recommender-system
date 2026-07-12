import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. 讀取資料
# ==========================================
print("📂 資料讀取中...")
rating_df_raw = pd.read_csv('old/rating_matrix - rating_matrix.csv', index_col=0)

rating_df_raw.index = rating_df_raw.index.astype(str)
rating_df_raw.columns = rating_df_raw.columns.astype(str)

MIN_COUNT = 0
popular_items = rating_df_raw.columns[(rating_df_raw > 0).sum() >= MIN_COUNT].tolist()
rating_df_raw = rating_df_raw[popular_items]
rating_df_nan = rating_df_raw.replace(0, np.nan)

df_user_content = pd.read_csv('flattened.csv', index_col='feature').T
df_user_content.index = df_user_content.index.astype(str)
df_user_content = df_user_content.reindex(rating_df_raw.index).fillna(0)

df_item_content = pd.read_csv('item_content_100d.csv', index_col=0)
df_item_content.index = df_item_content.index.astype(str)
df_item_content = df_item_content.reindex(popular_items).fillna(0)

test_df = pd.read_csv('test_set.csv')
test_df['Student_ID'] = test_df['Student_ID'].astype(str)
test_df['Course_ID'] = test_df['Course_ID'].astype(str)

num_users = len(rating_df_raw.index)
num_items = len(rating_df_raw.columns)
total_ratings = (rating_df_raw > 0).sum().sum()
matrix_density = (total_ratings / (num_users * num_items)) * 100

print(f"\n{'='*40}")
print("📊 資料集基本統計")
print(f"{'='*40}")
print(f"  學生數  : {num_users}")
print(f"  課程數  : {num_items}")
print(f"  評分筆數: {total_ratings}")
print(f"  密度    : {matrix_density:.2f}%  稀疏度: {100-matrix_density:.2f}%")

# ==========================================
# 2. Masking
# ==========================================
print("\n🛡️  執行 30% Masking...")
rating_df_masked = rating_df_nan.copy()
for _, row in test_df.iterrows():
    u, i = row['Student_ID'], row['Course_ID']
    if u in rating_df_masked.index and i in rating_df_masked.columns:
        rating_df_masked.loc[u, i] = np.nan

global_mean = rating_df_masked.stack().mean()
if pd.isna(global_mean):
    global_mean = 3.5
print(f"   全局平均分 global_mean = {global_mean:.4f}")

# ==========================================
# 3. 相似度矩陣
# ==========================================
scaler = MinMaxScaler()
u_rating_part = rating_df_masked.fillna(0)
i_rating_part = rating_df_masked.T.fillna(0)

u_hybrid_features = np.concatenate(
    [scaler.fit_transform(u_rating_part), scaler.fit_transform(df_user_content)], axis=1)
i_hybrid_features = np.concatenate(
    [scaler.fit_transform(i_rating_part), scaler.fit_transform(df_item_content)], axis=1)

print("\n🔄 計算 6 種相似度矩陣...")
sim_user_rating  = pd.DataFrame(cosine_similarity(scaler.fit_transform(u_rating_part)),
                                 index=rating_df_raw.index, columns=rating_df_raw.index)
sim_user_content = pd.DataFrame(cosine_similarity(scaler.fit_transform(df_user_content)),
                                 index=rating_df_raw.index, columns=rating_df_raw.index)
sim_user_hybrid  = pd.DataFrame(cosine_similarity(u_hybrid_features),
                                 index=rating_df_raw.index, columns=rating_df_raw.index)
sim_item_rating  = pd.DataFrame(cosine_similarity(scaler.fit_transform(i_rating_part)),
                                 index=rating_df_raw.columns, columns=rating_df_raw.columns)
sim_item_content = pd.DataFrame(cosine_similarity(scaler.fit_transform(df_item_content)),
                                 index=rating_df_raw.columns, columns=rating_df_raw.columns)
sim_item_hybrid  = pd.DataFrame(cosine_similarity(i_hybrid_features),
                                 index=rating_df_raw.columns, columns=rating_df_raw.columns)

# ==========================================
# 4. 重構預測函數：三層 fallback，永遠回傳浮點數
# ==========================================
# ★ 回傳值從 (float, bool) 改為 (float, int)
#   tier 0 = CF 成功預測（有鄰居且加權有意義）
#   tier 1 = 軟著陸：改用課程平均（鄰居不足但課程有歷史評分）
#   tier 2 = 冷啟動：只能用全局平均（課程完全沒有歷史評分）
#
# ★ 關鍵：函數「永遠不回傳 NaN」，預測值始終是浮點數，
#   讓 NDCG 的排序能拿到真實數值，不會退化成全部 0。
#
# ★ 課程平均加入微小擾動（±0.001 以內的 hash 雜訊），
#   避免同學生的多筆盲測課程全部平手、NDCG 退化為隨機排序。

def _item_mean(i_id):
    """取課程在 masked matrix 下的平均，若無則回傳 NaN。"""
    col = rating_df_masked.get(i_id)
    if col is None:
        return np.nan
    m = col.mean()  # pandas mean() 自動略過 NaN
    return m if not pd.isna(m) else np.nan

def _jitter(i_id):
    """對同一課程產生穩定的微小擾動，讓排序有鑑別度但不影響誤差統計。"""
    # ★ 用課程 ID 的 hash 產生 [-0.001, +0.001] 內的固定偏移
    return (hash(i_id) % 1000 - 500) * 0.000002

def predict_v2(u_id, i_id, sim_matrix, m_type='user', k=10):
    u_id, i_id = str(u_id), str(i_id)

    # --- 共用 fallback 計算 ---
    item_mu = _item_mean(i_id)

    def soft_fallback():
        """tier 1：課程平均 + 微小擾動"""
        if not pd.isna(item_mu):
            return np.clip(item_mu + _jitter(i_id), 0, 5), 1
        return global_mean, 2  # tier 2：冷啟動

    try:
        if m_type == 'user':
            # ★ 邊界檢查：user 或 item 根本不在矩陣裡 → 軟著陸
            if u_id not in sim_matrix.index or i_id not in rating_df_masked.columns:
                return soft_fallback()

            # 找有評分此課程的其他用戶
            others = rating_df_masked[rating_df_masked[i_id].notnull()].index
            match  = [o for o in others if o in sim_matrix.index and o != u_id]

            if not match:
                return soft_fallback()  # ★ 無鄰居 → 軟著陸，不再硬懲罰

            top_k = sim_matrix.loc[u_id, match].sort_values(ascending=False).head(k)

            if top_k.sum() == 0:
                return soft_fallback()  # ★ 相似度全為 0 → 軟著陸

            pred = (top_k * rating_df_masked.loc[top_k.index, i_id]).sum() / top_k.sum()
            return np.clip(pred, 0, 5), 0  # ★ tier 0：CF 成功

        else:  # item-based
            if i_id not in sim_matrix.index or u_id not in rating_df_masked.index:
                return soft_fallback()

            items = rating_df_masked.loc[u_id].dropna().index
            match = [it for it in items if it in sim_matrix.index and it != i_id]

            if not match:
                return soft_fallback()

            top_k = sim_matrix.loc[i_id, match].sort_values(ascending=False).head(k)

            if top_k.sum() == 0:
                return soft_fallback()

            pred = (top_k * rating_df_masked.loc[u_id, top_k.index]).sum() / top_k.sum()
            return np.clip(pred, 0, 5), 0

    except Exception:
        return soft_fallback()

# ==========================================
# 5. NDCG 計算（不動邏輯，但確保輸入是真實浮點數）
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
# 6. 主實驗迴圈
# ==========================================
methods = [
    ("User_Rating",  "user", sim_user_rating),
    ("User_Content", "user", sim_user_content),
    ("User_Hybrid",  "user", sim_user_hybrid),
    ("Item_Rating",  "item", sim_item_rating),
    ("Item_Content", "item", sim_item_content),
    ("Item_Hybrid",  "item", sim_item_hybrid),
]
k_list = [1, 3, 5, 7, 9]
experiment_results = []

print("\n🚀 開始 6-Ways × 5 K值 盲測實驗...")
for name, m_type, sim_df in methods:
    for k_val in k_list:
        abs_errors = []
        # ★ 分三層統計，方便報告說明
        tier_counts = {0: 0, 1: 0, 2: 0}
        ndcg_bundles = {}

        for _, row in test_df.iterrows():
            uid    = str(row['Student_ID'])
            actual = row['Actual_Grade']

            pred, tier = predict_v2(row['Student_ID'], row['Course_ID'],
                                    sim_df, m_type, k=k_val)
            tier_counts[tier] += 1

            # ★ 誤差：tier 0/1 用真實差值；tier 2（冷啟動）才懲罰 5.0
            if tier == 2:
                abs_errors.append(5.0)
            else:
                abs_errors.append(abs(actual - pred))

            if uid not in ndcg_bundles:
                ndcg_bundles[uid] = []
            ndcg_bundles[uid].append({'true': actual, 'pred': pred})

        mae  = np.mean(abs_errors)
        rmse = np.sqrt(np.mean(np.array(abs_errors) ** 2))
        ndcg = np.mean([calculate_ndcg(records) for records in ndcg_bundles.values()])

        experiment_results.append({
            "Method"          : name,
            "K_Value"         : k_val,
            "MAE"             : mae,
            "RMSE"            : rmse,
            "NDCG"            : ndcg,
            "Tier0_CF"        : tier_counts[0],   # CF 真實預測筆數
            "Tier1_ItemMean"  : tier_counts[1],   # 軟著陸筆數
            "Tier2_ColdStart" : tier_counts[2],   # 冷啟動（才懲罰）
        })

    print(f"  ✅ {name} 完畢")

# ==========================================
# 7. 輸出黃金總表
# ==========================================
report_df = pd.DataFrame(experiment_results)

best_rows = []
for method_name, group in report_df.groupby("Method"):
    best_rows.append(group.loc[group["RMSE"].idxmin()])

best_df = pd.DataFrame(best_rows)[
    ["Method", "K_Value", "MAE", "RMSE", "NDCG",
     "Tier0_CF", "Tier1_ItemMean", "Tier2_ColdStart"]
]

pd.set_option('display.float_format', lambda x: f'{x:.4f}')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)

print("\n" + "="*95)
print("📊 各方法最佳 K 值統整表（三層 fallback，只有 Tier2 真冷啟動才計 5.0 懲罰）")
print("="*95)
print(best_df.to_string(index=False))
print("="*95)