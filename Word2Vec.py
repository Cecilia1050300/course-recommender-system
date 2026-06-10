import pandas as pd
import numpy as np
import jieba
from gensim.models import Word2Vec

# ==========================================
# 1. 從 transaction 建立 ID 對照表
# ==========================================
# 假設你的 transaction 資料存成 transaction.csv
df_trans = pd.read_csv('transaction.csv', header=None).astype(str)

# 建立 序號(93) -> 真實代碼(40B0291) 的對應字典
# 我們只取第 2 欄 (真實 ID) 和 第 3 欄 (序號)
id_map = dict(zip(df_trans[2], df_trans[1]))

# ==========================================
# 2. 讀取課程名稱 (course.csv)
# ==========================================
file_path = r'C:/Users/super/Desktop/Recommender Systems/old/Course Recommendation/course.csv'
with open(file_path, 'r', encoding='utf-8-sig') as f:
    df_courses = pd.read_csv(f, header=None).astype(str)

# 抓取 序號(索引0) 和 名稱(索引1)
df_courses.columns = ['Seq_ID', 'Course_Name', 'Credits']

# 關鍵：利用 id_map 將 序號 換成 真實代碼
df_courses['Real_ID'] = df_courses['Seq_ID'].map(id_map)

# 剔除那些在 transaction 找不到對應代碼的課程 (避免報錯)
df_courses = df_courses.dropna(subset=['Real_ID'])

print(f"成功對應了 {len(df_courses)} 門課程！")

# ==========================================
# 3. 訓練 Word2Vec (邏輯不變)
# ==========================================
print("正在分詞...")
sentences = [list(jieba.cut(name)) for name in df_courses['Course_Name']]

print("訓練 Word2Vec (100維)...")
model = Word2Vec(sentences, vector_size=100, window=3, min_count=1)

def get_vector(word_list):
    vecs = [model.wv[word] for word in word_list if word in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

course_vectors = np.array([get_vector(s) for s in sentences])

# 4. 儲存：Index 改用對應後的 Real_ID
df_output = pd.DataFrame(course_vectors, index=df_courses['Real_ID'])
df_output.to_csv('item_content_100d.csv', encoding='utf-8-sig')

print("✅ 大功告成！item_content_100d.csv 現在使用的是真實課程代碼。")