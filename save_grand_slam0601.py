import pandas as pd

# =================================================================
# 建立終極大滿貫實驗總表數據 (已全數包含 5 分最大懲罰機制)
# =================================================================
data = [
    {
        "推薦演算法 (Method)": "Matrix_Factorization (MF)",
        "超參數設定 (Hyperparameters)": "Epoch = 100",
        "MAE (↓)": 0.8051,
        "RMSE (↓)": 1.0288,
        "失效預測次數 (Failed Predictions)": "9 次",
        "系統總評與學術定位 (Evaluation & Positioning)": "【協同精準度之王】 密碼背誦能力極強，但無法解決冷啟動。"
    },
    {
        "推薦演算法 (Method)": "Item_Hybrid (本專題提出)",
        "超參數設定 (Hyperparameters)": "K = 9",
        "MAE (↓)": 0.8546,
        "RMSE (↓)": 1.0758,
        "失效預測次數 (Failed Predictions)": "0 次 (完美)",
        "系統總評與學術定位 (Evaluation & Positioning)": "🌟 【全場綜合表現冠軍】 兼顧極高精準度與 100% 覆蓋率。"
    },
    {
        "推薦演算法 (Method)": "Item_Content",
        "超參數設定 (Hyperparameters)": "K = 9",
        "MAE (↓)": 0.8820,
        "RMSE (↓)": 1.1036,
        "失效預測次數 (Failed Predictions)": "0 次 (完美)",
        "系統總評與學術定位 (Evaluation & Positioning)": "純語義模型，雖不失效但精準度稍遜 Hybrid。"
    },
    {
        "推薦演算法 (Method)": "Item_Rating",
        "超參數設定 (Hyperparameters)": "K = 9",
        "MAE (↓)": 0.8855,
        "RMSE (↓)": 1.1389,
        "失效預測次數 (Failed Predictions)": "9 次",
        "系統總評與學術定位 (Evaluation & Positioning)": "純行為項目 CF，在稀疏矩陣與冷啟動下表現受限。"
    },
    {
        "推薦演算法 (Method)": "User_Rating",
        "超參數設定 (Hyperparameters)": "K = 9",
        "MAE (↓)": 0.9344,
        "RMSE (↓)": 1.1959,
        "失效預測次數 (Failed Predictions)": "9 次",
        "系統總評與學術定位 (Evaluation & Positioning)": "傳統 User-based CF，表現中規中矩。"
    },
    {
        "推薦演算法 (Method)": "User_Hybrid",
        "超參數設定 (Hyperparameters)": "K = 7",
        "MAE (↓)": 0.9390,
        "RMSE (↓)": 1.2018,
        "失效預測次數 (Failed Predictions)": "9 次",
        "系統總評與學術定位 (Evaluation & Positioning)": "User 端混合特徵，精準度提升有限。"
    },
    {
        "推薦演算法 (Method)": "User_Content",
        "超參數設定 (Hyperparameters)": "K = 9",
        "MAE (↓)": 1.0326,
        "RMSE (↓)": 1.2769,
        "失效預測次數 (Failed Predictions)": "9 次",
        "系統總評與學術定位 (Evaluation & Positioning)": "純學生背景特徵，與學業成績關聯性較弱。"
    }
]

# 轉換為 Pandas DataFrame
df = pd.DataFrame(data)

# 匯出為帶有 UTF-8 BOM 的 CSV 檔案，確保 Excel 開啟時中文不會變亂碼
output_filename = 'Grand_Slam_Experiment_Summary.csv'
df.to_csv(output_filename, index=False, encoding='utf-8-sig')

print("=" * 60)
print(f"🎉 任務成功！【{output_filename}】已順利生成。")
print("  提示：您可以直接用 Excel 開啟此檔案，並複製表格到 PPT 報告中使用！")
print("=" * 60)