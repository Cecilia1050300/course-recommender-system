import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# 1. 讀取與準備資料
rating_df_raw = pd.read_csv('rating_matrix.csv', index_col=0)
rating_df_nan = rating_df_raw.replace(0, np.nan)
test_df = pd.read_csv('test_set.csv')

# 載入 Word2Vec 特徵 (假設檔名正確)
i_content_part = pd.read_csv('item_content_100d.csv', index_col=0)
u_content_part = pd.read_csv('flattened.csv', index_col=0)

scaler = MinMaxScaler()

# 2. 計算相似度矩陣
# User 系列
sim_user_rating = pd.DataFrame(cosine_similarity(rating_df_raw), index=rating_df_raw.index, columns=rating_df_raw.index)
sim_user_content = pd.DataFrame(cosine_similarity(scaler.fit_transform(u_content_part)), index=u_content_part.index, columns=u_content_part.index)
sim_user_hybrid = (sim_user_rating + sim_user_content) / 2

# Item 系列
sim_item_rating = pd.DataFrame(cosine_similarity(rating_df_raw.T), index=rating_df_raw.columns, columns=rating_df_raw.columns)
sim_item_content = pd.DataFrame(cosine_similarity(scaler.fit_transform(i_content_part)), index=i_content_part.index, columns=i_content_part.index)
sim_item_hybrid = (sim_item_rating + sim_item_content) / 2

# ==========================================
# 3. 預測函數 (核心修改：k=3)
# ==========================================
def predict(u_id, i_id, sim_matrix, rating_matrix, k=3): # <--- 這裡改成了 3
    u_id, i_id = str(u_id), str(i_id)
    if u_id not in rating_matrix.index or i_id not in rating_matrix.columns:
        return rating_matrix.stack().mean()
    
    # 找出該用戶已評分或該項目被評分的清單
    # 若是 item-based，找修過該課的人；若是 user-based，找該人修過的課
    if sim_matrix.index.equals(rating_matrix.index): # User-based
        relevant_ratings = rating_matrix.loc[:, i_id].dropna()
    else: # Item-based
        relevant_ratings = rating_matrix.loc[u_id, :].dropna()
        
    common = sim_matrix.index.intersection(relevant_ratings.index)
    if len(common) == 0:
        return rating_matrix.stack().mean()
    
    # 取前 K 個最相似的鄰居
    top_k = sim_matrix.loc[common, common].index[:k] # 簡化取前K
    weights = sim_matrix.loc[sim_matrix.index == (u_id if u_id in sim_matrix.index else i_id), top_k]
    scores = relevant_ratings.loc[top_k]
    
    if weights.sum().sum() == 0:
        return rating_matrix.stack().mean()
        
    return (weights.values * scores.values).sum() / weights.sum().sum()

# ==========================================
# 4. 執行 6 種方法評估
# ==========================================
methods = [
    ("User_Rating", sim_user_rating),
    ("User_Content", sim_user_content),
    ("User_Hybrid", sim_user_hybrid),
    ("Item_Rating", sim_item_rating),
    ("Item_Content", sim_item_content),
    ("Item_Hybrid", sim_item_hybrid)
]

summary_mae = []

print(f"🚀 開始評估 (設定 K={3})")
for name, sim_mat in methods:
    errors = []
    for _, row in test_df.iterrows():
        u, i, true_s = row['Student_ID'], row['Course_ID'], row['Actual_Grade']
        pred = predict(u, i, sim_mat, rating_df_nan, k=3) # <--- 強制 K=3
        errors.append(abs(true_s - pred))
    
    mae = np.mean(errors)
    summary_mae.append({"Method": name, "MAE": mae})
    print(f"✅ {name:15s} MAE: {mae:.4f}")

# 5. 輸出結果
print("\n" + "="*30)
print("📊 最終實驗結果 (K=3)")
print(pd.DataFrame(summary_mae))
print("="*30)