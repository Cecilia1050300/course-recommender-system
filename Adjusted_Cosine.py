import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. 讀取資料
# ==========================================
print("📂 讀取資料中...")
rating_df_raw = pd.read_csv('rating_matrix.csv', index_col=0).fillna(0)
rating_df_raw.index = rating_df_raw.index.astype(str)
rating_df_raw.columns = rating_df_raw.columns.astype(str)
rating_df_nan = rating_df_raw.replace(0, np.nan)

df_user_content = pd.read_csv('flattened.csv', index_col='feature').T
df_user_content.index = df_user_content.index.astype(str)

df_item_content = pd.read_csv('item_content_100d.csv', index_col=0)
df_item_content.index = df_item_content.index.astype(str)

test_df = pd.read_csv('test_set.csv')
test_df['Student_ID'] = test_df['Student_ID'].astype(str)
test_df['Course_ID'] = test_df['Course_ID'].astype(str)

# ==========================================
# 2. 執行 Masking (30% 覆蓋測試)
# ==========================================
print("🛡️  執行 30% Masking...")
rating_df_masked = rating_df_nan.copy()
for _, row in test_df.iterrows():
    u, i = row['Student_ID'], row['Course_ID']
    if u in rating_df_masked.index and i in rating_df_masked.columns:
        rating_df_masked.loc[u, i] = np.nan

# 確定一個絕對不會出錯的全局平均值
FALLBACK_MEAN = rating_df_masked.stack().mean()
if np.isnan(FALLBACK_MEAN): FALLBACK_MEAN = 3.5 

# ==========================================
# 3. 計算 6 種相似度矩陣 (不縮減矩陣大小)
# ==========================================
scaler = MinMaxScaler()

def safe_cosine(df):
    scaled = scaler.fit_transform(df.fillna(0))
    return pd.DataFrame(cosine_similarity(scaled), index=df.index, columns=df.index)

# User 系列
sim_user_rating = safe_cosine(rating_df_masked)
# 為了對齊 460 人，沒 Content 的人用 0 向量補齊
u_content_full = df_user_content.reindex(rating_df_masked.index).fillna(0)
sim_user_content = safe_cosine(u_content_full)

u_hybrid_stack = np.concatenate([scaler.fit_transform(rating_df_masked.fillna(0)), 
                                 scaler.fit_transform(u_content_full)], axis=1)
sim_user_hybrid = pd.DataFrame(cosine_similarity(u_hybrid_stack), index=rating_df_masked.index, columns=rating_df_masked.index)

# Item 系列 (Adjusted Cosine)
def get_adj_cosine(matrix_nan):
    user_mean = matrix_nan.mean(axis=1)
    adj_matrix = matrix_nan.sub(user_mean, axis=0).fillna(0)
    return pd.DataFrame(cosine_similarity(adj_matrix.T), index=matrix_nan.columns, columns=matrix_nan.columns)

sim_item_rating_adj = get_adj_cosine(rating_df_masked)
i_content_full = df_item_content.reindex(rating_df_masked.columns).fillna(0)
sim_item_content = safe_cosine(i_content_full)

i_hybrid_stack = np.concatenate([scaler.fit_transform(sim_item_rating_adj), 
                                 scaler.fit_transform(i_content_full)], axis=1)
sim_item_hybrid = pd.DataFrame(cosine_similarity(i_hybrid_stack), index=rating_df_masked.columns, columns=rating_df_masked.columns)

# ==========================================
# 4. 終極防禦預測函數
# ==========================================
def predict_ultra_safe(u_id, i_id, sim_matrix, m_type='user', k=3):
    u_id, i_id = str(u_id), str(i_id)
    
    try:
        if m_type == 'user':
            if u_id not in sim_matrix.index: return FALLBACK_MEAN
            # 找到在訓練集中有修過這門課的人[cite: 1]
            valid_users = rating_df_masked[rating_df_masked[i_id].notnull()].index
            match_users = [o for o in valid_users if o in sim_matrix.index and o != u_id]
            
            if not match_users: return rating_df_masked[i_id].mean() or FALLBACK_MEAN
            
            scores = sim_matrix.loc[u_id, match_users].sort_values(ascending=False).head(k)
            if scores.sum() <= 0: return rating_df_masked[i_id].mean() or FALLBACK_MEAN
            return (scores * rating_df_masked.loc[scores.index, i_id]).sum() / scores.sum()
        else:
            if i_id not in sim_matrix.index: return FALLBACK_MEAN
            # 找到這學生在訓練集裡修過的課[cite: 1]
            valid_items = rating_df_masked.loc[u_id].dropna().index
            match_items = [it for it in valid_items if it in sim_matrix.index and it != i_id]
            
            if not match_items: return rating_df_masked.loc[u_id].mean() or FALLBACK_MEAN
            
            scores = sim_matrix.loc[i_id, match_items].sort_values(ascending=False).head(k)
            if scores.sum() <= 0: return rating_df_masked.loc[u_id].mean() or FALLBACK_MEAN
            return (scores * rating_df_masked.loc[u_id, scores.index]).sum() / scores.sum()
    except:
        return FALLBACK_MEAN

# ==========================================
# 5. 執行 6-Ways
# ==========================================
methods = [
    ("User_Rating", "user", sim_user_rating),
    ("User_Content", "user", sim_user_content),
    ("User_Hybrid", "user", sim_user_hybrid),
    ("Item_Rating_Adjusted", "item", sim_item_rating_adj),
    ("Item_Content", "item", sim_item_content),
    ("Item_Hybrid", "item", sim_item_hybrid)
]

print("\n🚀 計算 6-Ways 指標中...")
results = []
for name, m_type, sim_df in methods:
    all_preds = []
    all_actuals = []
    for _, row in test_df.iterrows():
        p = predict_ultra_safe(row['Student_ID'], row['Course_ID'], sim_df, m_type)
        # 萬一 p 還是 nan，強行填入保底值
        if np.isnan(p): p = FALLBACK_MEAN
        all_preds.append(p)
        all_actuals.append(row['Actual_Grade'])
    
    # 使用 np.abs 計算，並確保移除任何意外的 nan[cite: 1]
    errors = np.abs(np.array(all_actuals) - np.array(all_preds))
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    
    results.append({"Method": name, "MAE": mae, "RMSE": rmse})
    print(f"✅ {name} 完畢")

# ==========================================
# 6. 輸出表格
# ==========================================
report = pd.DataFrame(results)
print("\n" + "="*45)
for _, r in report.iterrows():
    print(f"{r['Method']:<25} | {r['MAE']:<8.4f} | {r['RMSE']:<8.4f}")
print("="*45)