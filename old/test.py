import pandas as pd

# 1. 讀取原始資料
df = pd.read_csv('transaction.csv', header=None, names=['student_id', 'course_id', 'col3', 'score', 'col5'])

# 2. 資料清理
df_filtered = df[['student_id', 'course_id', 'score']].copy()
df_filtered['score'] = pd.to_numeric(df_filtered['score'], errors='coerce') - 500

# 處理重複資料
df_filtered = df_filtered.drop_duplicates(subset=['student_id', 'course_id'], keep='last')

# 3. 建立評分矩陣並將沒修的課 (NaN) 填入 0
# 這裡多加了 .fillna(0)
rating_matrix = df_filtered.pivot(index='student_id', columns='course_id', values='score').fillna(0)

# 4. 匯出檔案
rating_matrix.to_csv('rating_matrix.csv')
rating_matrix.to_excel('rating_matrix.xlsx')

print("--- 成功！已將沒修的課程填入 0 並匯出檔案 ---")