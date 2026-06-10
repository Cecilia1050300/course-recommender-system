import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import os

# ==========================================
# 1. 讀取資料
# ==========================================
print("讀取資料中...")
# 評分矩陣
rating_df_raw = pd.read_csv('rating_matrix.csv', index_col=0)
rating_df_raw.index = rating_df_raw.index.astype(str)
rating_df_raw.columns = rating_df_raw.columns.astype(str)
rating_df_nan = rating_df_raw.replace(0, np.nan)

# 學生特質 (User Content)
df_user_content = pd.read_csv('flattened.csv', index_col='feature').T
df_user_content.index = df_user_content.index.astype(str)

# 課程 Word2Vec (Item Content)
df_item_content = pd.read_csv('item_content_100d.csv', index_col=0)
df_item_content.index = df_item_content.index.astype(str)

# 測試集
test_df = pd.read_csv('test_set.csv')

# ==========================================
# 2. 特徵對齊與標準化
# ==========================================
scaler = MinMaxScaler()

# 對齊 User ID
common_users = rating_df_raw.index.intersection(df_user_content.index)
u_rating_part = rating_df_raw.loc[common_users].fillna(0)
u_content_part = df_user_content.loc[common_users]

# 對齊 Item ID
common_items = rating_df_raw.columns.intersection(df_item_content.index)
i_rating_part = rating_df_raw.T.loc[common_items].fillna(0)
i_content_part = df_item_content.loc[common_items]

print(f"✅ 資料對齊完成：學生 {len(common_users)} 人，課程 {len(common_items)} 門")

# 串接 Hybrid 特徵 (Concatenation)
u_hybrid_features = np.concatenate([scaler.fit_transform(u_rating_part), scaler.fit_transform(u_content_part)], axis=1)
i_hybrid_features = np.concatenate([scaler.fit_transform(i_rating_part), scaler.fit_transform(i_content_part)], axis=1)

# ==========================================
# 3. 計算 6 種相似度矩陣
# ==========================================
print("計算 6 種相似度矩陣中...")

# User 系列
sim_user_rating = pd.DataFrame(cosine_similarity(scaler.fit_transform(u_rating_part)), index=common_users, columns=common_users)
sim_user_content = pd.DataFrame(cosine_similarity(scaler.fit_transform(u_content_part)), index=common_users, columns=common_users)
sim_user_hybrid = pd.DataFrame(cosine_similarity(u_hybrid_features), index=common_users, columns=common_users)

# Item 系列
sim_item_rating = pd.DataFrame(cosine_similarity(scaler.fit_transform(i_rating_part)), index=common_items, columns=common_items)
sim_item_content = pd.DataFrame(cosine_similarity(scaler.fit_transform(i_content_part)), index=common_items, columns=common_items)
sim_item_hybrid = pd.DataFrame(cosine_similarity(i_hybrid_features), index=common_items, columns=common_items)

# ==========================================
# 3.5 驗證 Word2Vec 語義品質 (加在相似度計算下方)
# ==========================================
print("\n" + "="*30)
print("🔎 Word2Vec 相似課程檢查 (Sanity Check)")

# 建立代碼到名稱的 map (方便我們看結果)
# 假設你的 Word2Vec.py 讀取的 course.csv 變數還能存取，或者重新讀一下
course_info = pd.read_csv(r'C:/Users/super/Desktop/Recommender Systems/old/Course Recommendation/course.csv', 
                          header=None, encoding='utf-8-sig').astype(str)
name_map = dict(zip(course_info[0], course_info[1])) # 序號 -> 課名
# 如果你的 common_items 是長代碼，請確保這裡的 map 是正確的。 
# 如果暫時對不起來，我們就先看代碼。

# 隨機抽 3 門課來看看
test_samples = list(sim_item_content.index[:3]) 

for target in test_samples:
    # 找出相似度前 5 名 (iloc[1:6] 避開自己)
    top_5 = sim_item_content.loc[target].sort_values(ascending=False).iloc[1:6]
    
    print(f"\n課程代碼 [{target}] 的最像鄰居：")
    for neighbor_id, score in top_5.items():
        print(f"  - {neighbor_id} (相似度: {score:.4f})")

print("="*30 + "\n")
# ==========================================
# 4. 預測函數 (邏輯強化)
# ==========================================
def predict(u_id, i_id, sim_matrix, m_type='user', k=10):
    u_id, i_id = str(u_id), str(i_id)
    global_mean = rating_df_nan.stack().mean()
    
    if m_type == 'user':
        if u_id not in sim_matrix.index or i_id not in rating_df_nan.columns: return global_mean
        # 找出修過這門課的人
        others = rating_df_nan[rating_df_nan[i_id].notnull()].index
        valid = [o for o in others if o in sim_matrix.index and o != u_id]
        if not valid: return rating_df_nan[i_id].mean() if i_id in rating_df_nan.columns else global_mean
        top_k = sim_matrix.loc[u_id, valid].sort_values(ascending=False).head(k)
        return (top_k * rating_df_nan.loc[top_k.index, i_id]).sum() / (top_k.sum() + 1e-9)
    else:
        if i_id not in sim_matrix.index or u_id not in rating_df_nan.index: return global_mean
        # 找出這個學生修過的課
        user_ratings = rating_df_nan.loc[u_id]
        items = user_ratings[user_ratings.notnull()].index
        valid = [it for it in items if it in sim_matrix.index and it != i_id]
        if not valid: return rating_df_nan.loc[u_id].mean() if u_id in rating_df_nan.index else global_mean
        top_k = sim_matrix.loc[i_id, valid].sort_values(ascending=False).head(k)
        return (top_k * user_ratings[top_k.index]).sum() / (top_k.sum() + 1e-9)

# ==========================================
# 5. 執行評估並產出 CSV
# ==========================================
methods = [
    ("User_Rating", "user", sim_user_rating),
    ("User_Content", "user", sim_user_content),
    ("User_Hybrid", "user", sim_user_hybrid),
    ("Item_Rating", "item", sim_item_rating),
    ("Item_Content", "item", sim_item_content),
    ("Item_Hybrid", "item", sim_item_hybrid)
]

summary_mae = []

print("\n--- 開始評估流程 ---")
for name, m_type, sim_df in methods:
    print(f"正在計算 {name}...", end=" ", flush=True)
    details = []
    for _, row in test_df.iterrows():
        u, i, true_s = str(row['Student_ID']), str(row['Course_ID']), row['Actual_Grade']
        pred = predict(u, i, sim_df, m_type)
        details.append({
            'Student_ID': u, 'Course_ID': i, 
            'Actual': true_s, 'Predicted': round(pred, 4),
            'Error': round(abs(true_s - pred), 4)
        })
    
    df_res = pd.DataFrame(details)
    mae = df_res['Error'].mean()
    summary_mae.append({"Method": name, "MAE": mae})
    
    # 產出詳細表
    df_res.to_csv(f'Detail_{name}.csv', index=False, encoding='utf-8-sig')
    print(f"完成! MAE: {mae:.4f}")

# ==========================================
# 6. 總表輸出
# ==========================================
summary_df = pd.DataFrame(summary_mae)
summary_df.to_csv('Final_MAE_Summary.csv', index=False, encoding='utf-8-sig')

print("\n" + "="*30)
print("✨ 實驗大功告成！")
print("請查看資料夾中的 Detail_*.csv 與 Final_MAE_Summary.csv")
print("="*30)
print(summary_df)