import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

# 1. 讀取資料
# 請確保這兩個檔案在同一個資料夾
df_features = pd.read_csv('flattened.csv', index_col='feature').T 
# 假設你的成績單檔名是 rating.csv，格式是：Row 是學生ID，Column 是課程ID
rating_df_raw = pd.read_csv('rating_matrix.csv', index_col=0)

# 2. 隨機抽出 30% 的非零分數作為測試集 (test_set)
# 我們要找出所有「有成績」的座標 (Row, Col)
records = []
for u_idx in rating_df_raw.index:
    for c_col in rating_df_raw.columns:
        score = rating_df_raw.loc[u_idx, c_col]
        if score > 0: # 只抓有成績的 (1-5分)
            records.append((u_idx, c_col, score))

# 隨機打亂並選取 30%
random.seed(42) # 固定隨機種子，方便你下次跑出一模一樣的結果
random.shuffle(records)
test_size = int(len(records) * 0.3)
test_set = records[:test_size]  # 這就是你要預測的 30%
train_set = records[test_size:] # 剩下的 70% 用來當參考資料

# 3. 建立訓練用成績矩陣 (把那 30% 變成 NaN)
rating_df = rating_df_raw.copy().astype(float)
for u_id, i_id, score in test_set:
    rating_df.loc[u_id, i_id] = np.nan
    
# 將原本就是 0 的地方也變成 NaN
rating_df = rating_df.replace(0, np.nan)

# 4. 計算學生特質相似度 (User-User Similarity)
user_sim = cosine_similarity(df_features)
user_sim_df = pd.DataFrame(user_sim, index=df_features.index, columns=df_features.index)

# 5. 預測函數
def predict_grade(user_id, item_id, k=10):
    if user_id not in user_sim_df.index or item_id not in rating_df.columns:
        return np.nan

    # 找出「訓練集」中對這門課有成績的同學
    others = rating_df[rating_df[item_id].notnull()].index
    if len(others) == 0:
        return rating_df[item_id].mean()
    
    # 找最像的 K 個鄰居
    similarities = user_sim_df.loc[user_id, others]
    top_k_users = similarities.sort_values(ascending=False).head(k)
    
    weights = top_k_users
    actual_ratings = rating_df.loc[top_k_users.index, item_id]
    
    if weights.sum() == 0:
        return actual_ratings.mean()
        
    return (weights * actual_ratings).sum() / weights.sum()

# 6. 執行預測並收集結果
final_results = []
for u_id, i_id, true_score in test_set:
    pred_score = predict_grade(u_id, i_id, k=10)
    final_results.append({
        'Student_ID': u_id,
        'Course_ID': i_id,
        'Actual_Grade': true_score,
        'Predicted_Grade': round(pred_score, 2) if not np.isnan(pred_score) else "N/A",
        'Error': abs(true_score - pred_score) if not np.isnan(pred_score) else np.nan
    })

# 7. 匯出 CSV
output_df = pd.DataFrame(final_results)
output_df.to_csv('prediction_test_results.csv', index=False, encoding='utf-8-sig')

print(f"完成！已預測 {len(final_results)} 筆資料，結果存於 prediction_test_results.csv")
print(f"平均絕對誤差 (MAE): {output_df['Error'].mean():.4f}")