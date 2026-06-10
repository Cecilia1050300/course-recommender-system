import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. 讀取與處理學生特質資料 (Flattened Data)
# ==========================================
print("正在讀取學生特質資料...")
# 讀取特質表，並確保所有欄位名稱（學生ID）都轉為字串
df_features_raw = pd.read_csv('flattened.csv', index_col='feature')
df_features_raw.columns = df_features_raw.columns.astype(str)

# 轉置：讓學生變成 Row，特徵變成 Column
df_features = df_features_raw.T 

# 計算學生之間的相似度
user_sim = cosine_similarity(df_features)
user_sim_df = pd.DataFrame(user_sim, index=df_features.index, columns=df_features.index)

# ==========================================
# 2. 讀取成績矩陣 (Rating Matrix)
# ==========================================
print("正在讀取成績矩陣...")
# 讀取時將第一欄設為 Index (學生ID)
rating_df = pd.read_csv('rating_matrix.csv', index_col=0)

# 【關鍵修正】強制將 Index(學生ID) 與 Columns(課程ID) 全部轉為字串
rating_df.index = rating_df.index.astype(str)
rating_df.columns = rating_df.columns.astype(str)

# 將 0 替換成 NaN (代表未修課或被蓋掉)
rating_df = rating_df.replace(0, np.nan)

# ==========================================
# 3. 讀取測試集 (Test Set)
# ==========================================
print("正在讀取測試集...")
test_df = pd.read_csv('test_set.csv')

# ==========================================
# 4. 定義預測函數 (安全加強版)
# ==========================================
def predict_grade(user_id, item_id, k=10):
    """
    預測學生 user_id 在課程 item_id 的分數
    """
    # 全部轉字串以利比對
    u_id_str = str(user_id)
    i_id_str = str(item_id)

    # 檢查學生是否存在於特質表/相似度矩陣中
    if u_id_str not in user_sim_df.index:
        return np.nan
    
    # 檢查課程是否存在於成績表的欄位中
    if i_id_str not in rating_df.columns:
        return np.nan
    
    # A. 找出對這門課有成績的同學 (非 NaN)
    target_item_scores = rating_df[i_id_str]
    others_with_scores = target_item_scores[target_item_scores.notnull()].index
    
    # B. 排除掉不在相似度矩陣中的同學，避免搜尋出錯
    valid_others = [o for o in others_with_scores if o in user_sim_df.index]
    
    if len(valid_others) == 0:
        return target_item_scores.mean() # 沒人修過就回傳該課平均
    
    # C. 從相似度矩陣提取相似度並找前 K 名
    similarities = user_sim_df.loc[u_id_str, valid_others]
    top_k_users = similarities.sort_values(ascending=False).head(k)
    
    # D. 取得這些鄰居的實際成績
    neighbor_ratings = rating_df.loc[top_k_users.index, i_id_str]
    
    # E. 加權平均公式
    if top_k_users.sum() == 0:
        return neighbor_ratings.mean()
        
    prediction = (top_k_users * neighbor_ratings).sum() / top_k_users.sum()
    return prediction

# ==========================================
# 5. 執行預測並收集結果
# ==========================================
print("開始執行預測...")
final_results = []

for index, row in test_df.iterrows():
    # 注意：這裡的欄位名稱需與 test_set.csv 一致
    u_id = row['Student_ID']   
    i_id = row['Course_ID']    
    true_score = row['Actual_Grade'] 
    
    # 執行預測
    pred_score = predict_grade(u_id, i_id, k=10)
    
    res = {
        'Student_ID': u_id,
        'Course_ID': i_id,
        'Actual_Grade': true_score,
        'Predicted_Grade': round(pred_score, 2) if not np.isnan(pred_score) else "N/A"
    }
    
    # 計算絕對誤差
    if not np.isnan(pred_score):
        res['Error'] = abs(true_score - pred_score)
    else:
        res['Error'] = np.nan
        
    final_results.append(res)

# ==========================================
# 6. 匯出結果 CSV
# ==========================================
output_df = pd.DataFrame(final_results)
output_filename = 'prediction_results_output.csv'
output_df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print("-" * 30)
print(f"完成！預測結果已匯出至: {output_filename}")

# 計算成效
valid_errors = output_df['Error'].dropna()
if len(valid_errors) > 0:
    print(f"測試集總筆數: {len(output_df)}")
    print(f"平均絕對誤差 (MAE): {valid_errors.mean():.4f}")
else:
    print("無法計算誤差，可能原因：預測值皆為 N/A。請檢查 Student_ID 或 Course_ID 是否對齊。")