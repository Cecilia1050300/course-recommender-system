import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

# 1. 讀取與處理資料 (0 改為 NaN)
df = pd.read_csv('rating_matrix.csv', index_col=0)
original_matrix = df.replace(0, np.nan).values.astype(float)

# 2. 隨機覆蓋 30% 非空值
mask = ~np.isnan(original_matrix)
nonzero_coords = np.argwhere(mask)
np.random.seed(42)
np.random.shuffle(nonzero_coords)

num_to_mask = int(len(nonzero_coords) * 0.3)
mask_indices = nonzero_coords[:num_to_mask]

train_matrix = original_matrix.copy()
for r, c in mask_indices:
    train_matrix[r, c] = np.nan

# 3. 定義通用預測函數 (支援 User 與 Item 兩種模式)
def predict_cf(ratings, similarity, k=20, mode='item'):
    """
    mode='item': Item-based CF
    mode='user': User-based CF
    """
    if mode == 'user':
        # User-based: (Users, Users) 相似度
        n_users, n_items = ratings.shape
        pred = np.full((n_users, n_items), np.nan)
        
        # 算出每個 User 的平均分 (排除 NaN)
        user_mean = np.nanmean(ratings, axis=1)
        # 去中心化矩陣 (減去平均分)
        ratings_diff = ratings - user_mean[:, np.newaxis]
        
        for i in range(n_users):
            # 找最相似的 K 個 User (排除自己)
            sim_scores = similarity[i].copy()
            sim_scores[i] = -1
            top_k_users = np.argsort(sim_scores)[-k:]
            
            for j in range(n_items):
                user_sims = similarity[i, top_k_users]
                neighbor_ratings = ratings_diff[top_k_users, j]
                
                valid_idx = ~np.isnan(neighbor_ratings)
                if np.any(valid_idx):
                    sum_sim = np.sum(np.abs(user_sims[valid_idx]))
                    if sum_sim != 0:
                        # 預測值 = 該 User 平均分 + (鄰居偏差加權平均)
                        pred[i, j] = user_mean[i] + np.dot(user_sims[valid_idx], neighbor_ratings[valid_idx]) / sum_sim
        return pred

    else: # mode == 'item'
        n_users, n_items = ratings.shape
        pred = np.full((n_users, n_items), np.nan)
        
        # 算出每個 Item 的平均分
        item_mean = np.nanmean(ratings, axis=0)
        ratings_diff = ratings - item_mean[np.newaxis, :]
        
        for j in range(n_items):
            sim_scores = similarity[j].copy()
            sim_scores[j] = -1
            top_k_items = np.argsort(sim_scores)[-k:]
            
            for i in range(n_users):
                item_sims = similarity[j, top_k_items]
                neighbor_ratings = ratings_diff[i, top_k_items]
                
                valid_idx = ~np.isnan(neighbor_ratings)
                if np.any(valid_idx):
                    sum_sim = np.sum(np.abs(item_sims[valid_idx]))
                    if sum_sim != 0:
                        pred[i, j] = item_mean[j] + np.dot(item_sims[valid_idx], neighbor_ratings[valid_idx]) / sum_sim
        return pred

# 4. 分別計算相似度矩陣 (Pearson)
train_df = pd.DataFrame(train_matrix)
user_sim_mat = train_df.T.corr().fillna(0).values # User 相似度 (對轉置矩陣做 corr)
item_sim_mat = train_df.corr().fillna(0).values   # Item 相似度

# 5. 自動調參跑兩種模型
k_options = [5, 10, 20, 30]
print(f"{'K':<5} | {'User RMSE':<10} | {'Item RMSE':<10}")

for k in k_options:
    u_pred = predict_cf(train_matrix, user_sim_mat, k=k, mode='user')
    i_pred = predict_cf(train_matrix, item_sim_mat, k=k, mode='item')
    
    # 計算 RMSE (只針對 mask 區域)
    y_true = [original_matrix[r, c] for r, c in mask_indices]
    y_u = [u_pred[r, c] if not np.isnan(u_pred[r, c]) else 0 for r, c in mask_indices]
    y_i = [i_pred[r, c] if not np.isnan(i_pred[r, c]) else 0 for r, c in mask_indices]
    
    rmse_u = np.sqrt(mean_squared_error(y_true, y_u))
    rmse_i = np.sqrt(mean_squared_error(y_true, y_i))
    
    print(f"{k:<5} | {rmse_u:<10.4f} | {rmse_i:<10.4f}")

    # --- 6. 找出最佳 K 並匯出詳細報告 ---
# 假設我們拿最後一次跑的 K (或你可以手動指定一個最好的 K)
print(f"\n正在匯出最終結果 (使用 K={k})...")

# 建立比對清單 (只針對那 30% 被蓋掉的資料)
comparison_data = []
for r, c in mask_indices:
    actual = original_matrix[r, c]
    u_p = u_pred[r, c]
    i_p = i_pred[r, c]
    
    comparison_data.append({
        'Student_ID': df.index[r],
        'Course_ID': df.columns[c],
        'Actual_Grade': actual,
        'User_Pred': round(u_p, 2) if not np.isnan(u_p) else "無法預測",
        'Item_Pred': round(i_p, 2) if not np.isnan(i_p) else "無法預測"
    })

# 1. 匯出「30% 測試集比對報告」 (這是你要看「正不正確」的主要檔案)
report_df = pd.DataFrame(comparison_data)
report_df.to_csv('prediction_verification.csv', index=False, encoding='utf-8-sig')

# 2. 匯出「完整的 Item-based 預測矩陣」 (把原本的 0 補上預測值)
final_filled_df = df.copy().astype(float)
for r in range(df.shape[0]):
    for c in range(df.shape[1]):
        if df.iloc[r, c] == 0: # 如果原本沒修課
            pred_val = i_pred[r, c] # 以 Item-based 為例
            if not np.isnan(pred_val):
                final_filled_df.iloc[r, c] = round(max(0, min(5, pred_val)), 2)

final_filled_df.to_csv('final_item_filled_matrix.csv', encoding='utf-8-sig')

print("匯出完成！")
print("- 驗證報告：prediction_verification.csv (查看 30% 預測準度)")
print("- 填補後矩陣：final_item_filled_matrix.csv (原本是 0 的地方補上分數)")