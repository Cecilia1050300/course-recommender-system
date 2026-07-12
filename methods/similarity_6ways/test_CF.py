import pandas as pd
import numpy as np

# 重現你的 masked matrix
rating_df_raw = pd.read_csv('rating_matrix.csv', index_col=0)
rating_df_raw.index = rating_df_raw.index.astype(str)
rating_df_raw.columns = rating_df_raw.columns.astype(str)
rating_df_nan = rating_df_raw.replace(0, np.nan)

test_df = pd.read_csv('test_set.csv')
test_df['Student_ID'] = test_df['Student_ID'].astype(str)
test_df['Course_ID'] = test_df['Course_ID'].astype(str)

rating_df_masked = rating_df_nan.copy()
for _, row in test_df.iterrows():
    u, i = row['Student_ID'], row['Course_ID']
    if u in rating_df_masked.index and i in rating_df_masked.columns:
        rating_df_masked.loc[u, i] = np.nan

print("=== 【診斷一】test_set 的 Course_ID 在 rating_matrix 裡嗎？ ===")
test_courses = set(test_df['Course_ID'].unique())
matrix_courses = set(rating_df_raw.columns)
missing_courses = test_courses - matrix_courses
print(f"  test_set 課程數       : {len(test_courses)}")
print(f"  rating_matrix 課程數  : {len(matrix_courses)}")
print(f"  test_set 有但矩陣沒有 : {len(missing_courses)} 門")
if missing_courses:
    print(f"  範例 ID: {list(missing_courses)[:10]}")

print("\n=== 【診斷二】test_set 的 Student_ID 在 rating_matrix 裡嗎？ ===")
test_users = set(test_df['Student_ID'].unique())
matrix_users = set(rating_df_raw.index)
missing_users = test_users - matrix_users
print(f"  test_set 學生數       : {len(test_users)}")
print(f"  rating_matrix 學生數  : {len(matrix_users)}")
print(f"  test_set 有但矩陣沒有 : {len(missing_users)} 人")
if missing_users:
    print(f"  範例 ID: {list(missing_users)[:10]}")

print("\n=== 【診斷三】masking 後，test 課程在 masked matrix 還剩幾筆評分？ ===")
zero_left = 0
for cid in test_courses:
    if cid in rating_df_masked.columns:
        remaining = rating_df_masked[cid].notnull().sum()
        if remaining == 0:
            zero_left += 1
print(f"  masking 後完全空白的課程數: {zero_left} / {len(test_courses)}")

print("\n=== 【診斷四】30% masking 比例合理嗎？ ===")
total_ratings = rating_df_nan.notnull().sum().sum()
test_size = len(test_df)
print(f"  原始評分總筆數 : {total_ratings}")
print(f"  test_set 筆數  : {test_size}")
print(f"  實際 masking 率: {test_size / total_ratings * 100:.1f}%")
print(f"  => 若某課程只有 1 筆評分，masking 後該課程在矩陣裡就完全消失")