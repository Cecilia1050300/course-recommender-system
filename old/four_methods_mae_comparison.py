import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. 讀取與處理資料 (確保型態一致)
# ==========================================
print("正在讀取資料...")

# (A) 學生特徵 (User Content) - 你的 flattened.csv
df_user_features = pd.read_csv('flattened.csv', index_col='feature').T
df_user_features.index = df_user_features.index.astype(str)

# (B) 成績矩陣 (Rating Matrix)
# 注意：程式會讀取 'rating_matrix.csv'
rating_df = pd.read_csv('rating_matrix.csv', index_col=0)
rating_df.index = rating_df.index.astype(str)
rating_df.columns = rating_df.columns.astype(str)
# 備份一個 NaN 版本用於預測計算，避免 0 分干擾
rating_df_nan = rating_df.replace(0, np.nan)

# (C) 測試集 (Test Set)
# 注意：程式會讀取 'test_set.csv'
test_df = pd.read_csv('test_set.csv')

# ==========================================
# 2. 計算相似度矩陣 (這就是老師要的四種核心)
# ==========================================
print("正在計算相似度...")

# --- User 相似度 ---
# 方法 1: 純 Rating 相似度 (Memory-based)
sim_user_rating = pd.DataFrame(cosine_similarity(rating_df.fillna(0)), 
                               index=rating_df.index, columns=rating_df.index)
# 方法 2: Content 相似度 (從 flattened.csv 算的)
sim_user_content = pd.DataFrame(cosine_similarity(df_user_features), 
                                index=df_user_features.index, columns=df_user_features.index)
# 方法 3: User Hybrid (混合：(Rating+Content)/2)
sim_user_hybrid = (sim_user_rating + sim_user_content) / 2

# --- Item 相似度 ---
# 方法 4: 純 Rating 相似度 (課程與課程的相似度)
sim_item_rating = pd.DataFrame(cosine_similarity(rating_df.T.fillna(0)), 
                               index=rating_df.columns, columns=rating_df.columns)
# 方法 5: Item Hybrid (由於還沒有學長資料，先用模擬數據佔位)
sim_item_content_dummy = sim_item_rating * 0.9 + 0.05
sim_item_hybrid = (sim_item_rating + sim_item_content_dummy) / 2

# ==========================================
# 3. 定義預測函數
# ==========================================

def predict_user_based(u_id, i_id, sim_matrix, k=10):
    u_id, i_id = str(u_id), str(i_id)
    if u_id not in sim_matrix.index or i_id not in rating_df_nan.columns:
        return np.nan
    others = rating_df_nan[rating_df_nan[i_id].notnull()].index
    valid_others = [o for o in others if o in sim_matrix.index]
    if not valid_others:
        return rating_df_nan[i_id].mean()
    top_k = sim_matrix.loc[u_id, valid_others].sort_values(ascending=False).head(k)
    if top_k.sum() == 0:
        return rating_df_nan.loc[top_k.index, i_id].mean()
    return (top_k * rating_df_nan.loc[top_k.index, i_id]).sum() / top_k.sum()

def predict_item_based(u_id, i_id, sim_matrix, k=10):
    u_id, i_id = str(u_id), str(i_id)
    if i_id not in sim_matrix.index or u_id not in rating_df_nan.index:
        return np.nan
    user_ratings = rating_df_nan.loc[u_id]
    items_rated = user_ratings[user_ratings.notnull()].index
    valid_items = [i for i in items_rated if i in sim_matrix.index]
    if not valid_items:
        return rating_df_nan.loc[u_id].mean()
    top_k = sim_matrix.loc[i_id, valid_items].sort_values(ascending=False).head(k)
    if top_k.sum() == 0:
        return user_ratings[top_k.index].mean()
    return (top_k * user_ratings[top_k.index]).sum() / top_k.sum()

# ==========================================
# 4. 執行實驗比較
# ==========================================
print("開始對照實驗...")
experiment_list = [
    ("User 純 Rating", "user", sim_user_rating),
    ("User 加 Content (Hybrid)", "user", sim_user_hybrid),
    ("Item 純 Rating", "item", sim_item_rating),
    ("Item 加 Content (Hybrid)", "item", sim_item_hybrid)
]

comparison_results = []

for name, m_type, sim_df in experiment_list:
    errors = []
    for _, row in test_df.iterrows():
        u, i, true_s = row['Student_ID'], row['Course_ID'], row['Actual_Grade']
        if m_type == "user":
            pred = predict_user_based(u, i, sim_df, k=10)
        else:
            pred = predict_item_based(u, i, sim_df, k=10)
        
        if not np.isnan(pred):
            errors.append(abs(true_s - pred))
    
    mae = np.mean(errors) if errors else np.nan
    comparison_results.append({"Method": name, "MAE": round(mae, 4)})

# ==========================================
# 5. 匯出比較結果 CSV
# ==========================================
final_df = pd.DataFrame(comparison_results)
output_name = 'four_methods_mae_comparison.csv'
final_df.to_csv(output_name, index=False, encoding='utf-8-sig')

print("\n" + "="*30)
print("實驗完成！結果摘要如下：")
print(final_df)
print("="*30)
print(f"結果已存入檔案: {output_name}")