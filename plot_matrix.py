import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 讀取與計算最原始資料統計 【任務二】
# ==========================================
print("📂 正在讀取原始評分矩陣並計算統計數據...")
rating_df_raw = pd.read_csv('rating_matrix.csv', index_col=0)
rating_df_nan = rating_df_raw.replace(0, np.nan)

num_users = rating_df_raw.shape[0]  # 學生數 (縱軸)
num_items = rating_df_raw.shape[1]  # 課程數 (橫軸)
total_cells = num_users * num_items

# 統計實際有修課/評分的總筆數
total_ratings = (rating_df_raw > 0).sum().sum()
raw_density = (total_ratings / total_cells) * 100
raw_sparsity = 100 - raw_density

print("\n" + "="*40)
print("📊 【最原始資料集】基本統計資料")
print("-" * 40)
print(f"👥 總學生數 (User)  : {num_users} 人")
print(f"📚 總課程數 (Item)  : {num_items} 門")
print(f"✍️  實際總評分筆數 (Data) : {total_ratings} 筆")
print(f"📈 原始矩陣密度 (Density): {raw_density:.4f}%")
print(f"📉 原始矩陣稀疏度(Sparsity): {raw_sparsity:.4f}%")
print("="*40 + "\n")

# 設定中文字型，防止 Windows/Mac 畫圖時中文變框框
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 圖一：全局評分矩陣熱點圖 (Heatmap)
# ==========================================
print("🎨 正在繪製圖一：評分矩陣熱點圖...")
plt.figure(figsize=(10, 8))
sns.heatmap(rating_df_nan, cmap="YlGnBu", cbar=True, 
            xticklabels=False, yticklabels=False, 
            cbar_kws={'label': '評分分數 (0 - 5 分)'})

plt.title(f'全校學生修課評分矩陣分佈圖 (Heatmap)\n(密度: {raw_density:.2f}% | 稀疏度: {raw_sparsity:.2f}%)', fontsize=14, fontweight='bold')
plt.xlabel(f'課程 (Items, 共 {num_items} 門) ───►', fontsize=12)
plt.ylabel(f'◄─── 學生 (Users, 共 {num_users} 人)', fontsize=12)
plt.tight_layout()
plt.savefig('Plot1_Rating_Matrix_Heatmap.png', dpi=300)
plt.close() # 關閉畫布釋放記憶體

# ==========================================
# 圖二：課程熱門度長尾分佈折線圖 (Long-Tail Plot)
# ==========================================
print("🎨 正在繪製圖二：課程長尾分佈折線圖...")
# 統計每門課有多少人修，並從大到小排序
course_popularity = (rating_df_raw > 0).sum(axis=0).sort_values(ascending=False).values

plt.figure(figsize=(10, 5))
plt.plot(course_popularity, color='royalblue', linewidth=3, label='每門課修課人數')
plt.fill_between(range(len(course_popularity)), course_popularity, color='royalblue', alpha=0.2)

plt.title('課程熱門度長尾分佈圖 (Sparsity Long-Tail)', fontsize=14, fontweight='bold')
plt.xlabel('課程 (依熱門度由左至右、由高至低排序)', fontsize=12)
plt.ylabel('修課學生人數', fontsize=12)
plt.xlim(0, len(course_popularity))
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('Plot2_Course_Long_Tail.png', dpi=300)
plt.close()

# ==========================================
# 圖三：雙向統計直方圖 (Distribution Histograms)
# ==========================================
print("🎨 正在繪製圖三：學生/課程修課數量直方圖...")
user_degrees = (rating_df_raw > 0).sum(axis=1) # 每個學生修了幾門課
item_degrees = (rating_df_raw > 0).sum(axis=0) # 每門課被幾個人修過

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左圖：學生視角
sns.histplot(user_degrees, bins=20, kde=True, color='teal', ax=ax1)
ax1.set_title('學生修課數量分佈直方圖', fontsize=12, fontweight='bold')
ax1.set_xlabel('修課門數 (個數)', fontsize=11)
ax1.set_ylabel('學生人數 (人)', fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# 右圖：課程視角
sns.histplot(item_degrees, bins=20, kde=True, color='crimson', ax=ax2)
ax2.set_title('課程被修人數分佈直方圖', fontsize=12, fontweight='bold')
ax2.set_xlabel('被修人數 (人)', fontsize=11)
ax2.set_ylabel('課程數量 (門)', fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle(f'資料集度數分佈統計 (Degree Distribution)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('Plot3_Degree_Distribution.png', dpi=300)
plt.close()

print("\n" + "="*40)
print("✨ 所有密度圖表生成大成功！")
print("💾 已在資料夾中儲存以下三張高畫質圖片：")
print("   1. Plot1_Rating_Matrix_Heatmap.png (全局矩陣熱點)")
print("   2. Plot2_Course_Long_Tail.png      (長尾分佈折線)")
print("   3. Plot3_Degree_Distribution.png   (修課數量直方)")
print("="*40)