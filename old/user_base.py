import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# 1. 讀取資料
df = pd.read_csv('rating_matrix.csv', index_col=0)
original_matrix = df.values.astype(float)

# 2. 隨機覆蓋 30% 既有評分
mask = (original_matrix > 0)
nonzero_coords = np.argwhere(mask)
np.random.seed(42) 
np.random.shuffle(nonzero_coords)

num_to_mask = int(len(nonzero_coords) * 0.3)
mask_indices = nonzero_coords[:num_to_mask]

train_matrix = original_matrix.copy()
for r, c in mask_indices:
    train_matrix[r, c] = 0

# 3. 定義預測函數 (優化計算速度)
def predict_top_k(ratings, similarity, k=5, type='user'):
    pred = np.zeros(ratings.shape)
    if type == 'user':
        for i in range(ratings.shape[0]):
            top_k_users = np.argsort(similarity[i])[:-k-1:-1]
            for j in range(ratings.shape[1]):
                sum_sim = np.sum(similarity[i, top_k_users])
                if sum_sim != 0:
                    pred[i, j] = similarity[i, top_k_users].dot(ratings[top_k_users, j]) / sum_sim
    else:
        for j in range(ratings.shape[1]):
            top_k_items = np.argsort(similarity[j])[:-k-1:-1]
            for i in range(ratings.shape[0]):
                sum_sim = np.sum(similarity[j, top_k_items])
                if sum_sim != 0:
                    pred[i, j] = ratings[i, top_k_items].dot(similarity[j, top_k_items]) / sum_sim
    return pred

# 計算相似度
user_sim = cosine_similarity(train_matrix)
item_sim = cosine_similarity(train_matrix.T)

# 4. 自動調參：測試不同的 K
k_options = [1, 3, 5, 10, 20, 30, 40, 50] # 你想測試的 K 範圍
best_user_rmse = float('inf')
best_item_rmse = float('inf')
best_user_k = 0
best_item_k = 0

print("開始自動調參...")
for k in k_options:
    # 預測
    u_pred = predict_top_k(train_matrix, user_sim, k=k, type='user')
    i_pred = predict_top_k(train_matrix, item_sim, k=k, type='item')
    
    # 提取被覆蓋處的預測值
    y_true = [original_matrix[r, c] for r, c in mask_indices]
    y_u = [u_pred[r, c] for r, c in mask_indices]
    y_i = [i_pred[r, c] for r, c in mask_indices]
    
    # 計算 RMSE
    rmse_u = np.sqrt(mean_squared_error(y_true, y_u))
    rmse_i = np.sqrt(mean_squared_error(y_true, y_i))
    
    print(f"K={k:2d} | User RMSE: {rmse_u:.4f} | Item RMSE: {rmse_i:.4f}")
    
    # 記錄最佳結果
    if rmse_u < best_user_rmse:
        best_user_rmse = rmse_u
        best_user_k = k
        final_user_pred = u_pred
        
    if rmse_i < best_item_rmse:
        best_item_rmse = rmse_i
        best_item_k = k
        final_item_pred = i_pred

print("-" * 30)
print(f"最佳結果：")
print(f"User-Based 最優 K={best_user_k}, RMSE={best_user_rmse:.4f}")
print(f"Item-Based 最優 K={best_item_k}, RMSE={best_item_rmse:.4f}")

# 5. 輸出最佳結果的檔案
pd.DataFrame(final_user_pred, index=df.index, columns=df.columns).to_csv('best_user_prediction.csv')
pd.DataFrame(final_item_pred, index=df.index, columns=df.columns).to_excel('best_item_prediction.xlsx')
print("\n最佳結果檔案已匯出！")