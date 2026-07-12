import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. 讀取與數據預處理
# ==========================================
print("📂 資料讀取與前處理中...")
rating_df_raw = pd.read_csv('rating_matrix.csv', index_col=0).fillna(0)
rating_df_raw.index = rating_df_raw.index.astype(str)
rating_df_raw.columns = rating_df_raw.columns.astype(str)

# [方法 3] 過濾冷門課 (保留修課人數 >= 10)
popular_items = rating_df_raw.columns[(rating_df_raw > 0).sum() >= 10].tolist()
rating_df_raw = rating_df_raw[popular_items]
rating_df_nan = rating_df_raw.replace(0, np.nan)

# 讀取 Content 並「強制」與 rating 欄位順序完全一致
df_item_content = pd.read_csv('item_content_100d.csv', index_col=0)
df_item_content.index = df_item_content.index.astype(str)
df_item_content = df_item_content.reindex(popular_items).fillna(0)

df_user_content = pd.read_csv('flattened.csv', index_col='feature').T
df_user_content.index = df_user_content.index.astype(str)
df_user_content = df_user_content.reindex(rating_df_raw.index).fillna(0)

test_df = pd.read_csv('test_set.csv')
test_df['Student_ID'] = test_df['Student_ID'].astype(str)
test_df['Course_ID'] = test_df['Course_ID'].astype(str)

# ==========================================
# 2. 執行 30% Masking (嚴格盲測)
# ==========================================
print("🛡️  執行 30% Masking...")
rating_df_masked = rating_df_nan.copy()
for _, row in test_df.iterrows():
    u, i = row['Student_ID'], row['Course_ID']
    if u in rating_df_masked.index and i in rating_df_masked.columns:
        rating_df_masked.loc[u, i] = np.nan

# 建立全局平均保底，確保不再產生 nan
global_mean = rating_df_masked.stack().mean()
if np.isnan(global_mean): global_mean = 3.5

# ==========================================
# 3. 相似度計算 (強制索引對齊)
# ==========================================
scaler = MinMaxScaler()

def get_clean_sim(df, is_item=True):
    # 確保維度與索引完全一致
    data_scaled = scaler.fit_transform(df.fillna(0))
    sim_matrix = cosine_similarity(data_scaled)
    names = df.columns if is_item else df.index
    return pd.DataFrame(sim_matrix, index=names, columns=names).fillna(0)

# --- Adjusted Cosine (Item) ---
user_mean = rating_df_masked.mean(axis=1)
adj_matrix = rating_df_masked.sub(user_mean, axis=0).fillna(0)
sim_item_rating_adj = pd.DataFrame(cosine_similarity(adj_matrix.T), 
                                   index=popular_items, columns=popular_items).fillna(0)

# --- Item Content & Hybrid ---
i_rating_feat = scaler.fit_transform(sim_item_rating_adj.values)
i_content_feat = scaler.fit_transform(df_item_content.values)
i_hybrid_stack = np.hstack([i_rating_feat, i_content_feat]) # 修正變數名稱一致

sim_item_content = pd.DataFrame(cosine_similarity(i_content_feat), index=popular_items, columns=popular_items)
sim_item_hybrid = pd.DataFrame(cosine_similarity(i_hybrid_stack), index=popular_items, columns=popular_items)

# --- User 系列 ---
sim_user_rating = get_clean_sim(rating_df_masked, is_item=False)
sim_user_content = get_clean_sim(df_user_content, is_item=False)

# ==========================================
# 4. 預測函數 (絕對值防禦)
# ==========================================
def predict_safe(u_id, i_id, sim_matrix, m_type='user', k=10):
    u_id, i_id = str(u_id), str(i_id)
    try:
        if m_type == 'user':
            if u_id not in sim_matrix.index or i_id not in rating_df_masked.columns: 
                return rating_df_masked[i_id].mean() or global_mean
            
            others = rating_df_masked[rating_df_masked[i_id].notnull()].index
            match = [o for o in others if o in sim_matrix.index and o != u_id]
            if not match: return rating_df_masked[i_id].mean() or global_mean
            
            top_k = sim_matrix.loc[u_id, match].sort_values(ascending=False).head(k)
            if top_k.sum() <= 0: return rating_df_masked[i_id].mean() or global_mean
            
            pred = (top_k * rating_df_masked.loc[top_k.index, i_id]).sum() / top_k.sum()
        else:
            if i_id not in sim_matrix.index or u_id not in rating_df_masked.index:
                return rating_df_masked.loc[u_id].mean() or global_mean
            
            items = rating_df_masked.loc[u_id].dropna().index
            match = [it for it in items if it in sim_matrix.index and it != i_id]
            if not match: return rating_df_masked.loc[u_id].mean() or global_mean
            
            top_k = sim_matrix.loc[i_id, match].sort_values(ascending=False).head(k)
            if top_k.sum() <= 0: return rating_df_masked.loc[u_id].mean() or global_mean
            
            pred = (top_k * rating_df_masked.loc[u_id, top_k.index]).sum() / top_k.sum()
        
        if np.isnan(pred): return global_mean
        return np.clip(pred, 0, 5) # 截斷限制
    except:
        return global_mean

# ==========================================
# 5. 執行 5-Ways 評估
# ==========================================
methods = [
    ("User_Rating", "user", sim_user_rating),
    ("User_Content", "user", sim_user_content),
    ("Item_Rating_Adjusted", "item", sim_item_rating_adj),
    ("Item_Content", "item", sim_item_content),
    ("Item_Hybrid", "item", sim_item_hybrid)
]

print("\n🚀 計算 5-Ways 最終指標中...")
final_results = []
for name, m_type, sim_df in methods:
    errors = []
    for _, row in test_df.iterrows():
        p = predict_safe(row['Student_ID'], row['Course_ID'], sim_df, m_type)
        errors.append(abs(row['Actual_Grade'] - p))
    
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.array(errors)**2))
    final_results.append({"Method": name, "MAE": mae, "RMSE": rmse})
    print(f"✅ {name} 計算完成")

# ==========================================
# 6. 輸出表格
# ==========================================
report = pd.DataFrame(final_results)
print("\n" + "="*45)
for _, r in report.iterrows():
    print(f"{r['Method']:<25} | {r['MAE']:<8.4f} | {r['RMSE']:<8.4f}")
print("="*45)