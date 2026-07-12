"""
=============================================================================
大學課程推薦系統 — 隨機游走演算法 (Random Walk with Restart, RWR)
=============================================================================
演算法架構：
  1. 從修課評分矩陣建立二部圖（Bipartite Graph）
  2. 以三種策略將鄰接矩陣正規化為馬可夫鏈轉移機率矩陣 W
  3. 對目標學生執行 RWR 迭代直到收斂
  4. 過濾已修課程，輸出 Top-N 推薦結果

依賴套件：numpy, pandas
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Literal


# =============================================================================
# 工具函式
# =============================================================================

def load_rating_matrix(filepath: str) -> tuple[np.ndarray, list, list]:
    """
    讀取評分矩陣 CSV 檔案。

    格式假設：
      - 第一欄為學生 ID（作為 index）
      - 第一列為課程 ID（作為 columns）
      - 格子值 0~5，0 代表未修課

    回傳：
      R          : (num_users x num_items) 的 numpy 陣列（float64）
      user_ids   : 學生 ID 清單
      item_ids   : 課程 ID 清單
    """
    df = pd.read_csv(filepath, index_col=0)
    user_ids = list(df.index)
    item_ids = list(df.columns)
    R = df.values.astype(np.float64)
    print(f"[資料載入] 學生數 = {len(user_ids)}，課程數 = {len(item_ids)}")
    print(f"[資料載入] 評分矩陣大小 = {R.shape}，非零格子數 = {np.count_nonzero(R)}")
    return R, user_ids, item_ids


def build_adjacency_matrix(R: np.ndarray) -> np.ndarray:
    """
    建立二部圖（Bipartite Graph）的全局鄰接矩陣 A。

    節點排列：[ User_0, User_1, ..., User_M-1,  Item_0, Item_1, ..., Item_N-1 ]
    大小：(M+N) x (M+N)

    結構（Block Matrix）：
    ┌─────────────────┬──────────────────────┐
    │   0 (M x M)     │   R   (M x N)        │  ← User → Item 評分
    ├─────────────────┼──────────────────────┤
    │   R.T (N x M)   │   0   (N x N)        │  ← Item → User 評分
    └─────────────────┴──────────────────────┘

    因為是無向二部圖，右上角與左下角互為轉置，對角線區塊皆為 0。
    """
    M, N = R.shape          # M = 學生數, N = 課程數
    total = M + N

    # 初始化全零的大方陣
    A = np.zeros((total, total), dtype=np.float64)

    # 右上角：User → Item（學生修課評分）
    A[:M, M:] = R

    # 左下角：Item → User（課程被修評分，R 的轉置）
    A[M:, :M] = R.T

    print(f"\n[鄰接矩陣] 大小 = {A.shape}  (Total_Nodes = {total})")
    print(f"[鄰接矩陣] 非零元素數 = {np.count_nonzero(A)}")
    return A


# =============================================================================
# 核心：三種轉移機率矩陣正規化策略
# =============================================================================

def build_transition_matrix(
    A: np.ndarray,
    strategy: Literal["binary", "linear", "softmax"] = "linear"
) -> np.ndarray:
    """
    將全局鄰接矩陣 A 轉化為符合馬可夫鏈定義的橫向轉移機率矩陣 W。
    要求：W 的每一橫列（Row）加總完美等於 1.0。

    支援三種策略：
      "binary"   - 策略 A：二值化填 1 結構法
      "linear"   - 策略 B：分數線性轉機率法（預設）
      "softmax"  - 策略 C：遮罩式橫向 Softmax 法

    參數：
      A        : (Total_Nodes x Total_Nodes) 全局鄰接矩陣
      strategy : 正規化策略選擇

    回傳：
      W : (Total_Nodes x Total_Nodes) 橫向轉移機率矩陣
    """
    print(f"\n[轉移矩陣] 使用策略：{strategy.upper()}")

    if strategy == "binary":
        W = _strategy_binary(A)
    elif strategy == "linear":
        W = _strategy_linear(A)
    elif strategy == "softmax":
        W = _strategy_softmax(A)
    else:
        raise ValueError(f"不支援的策略：{strategy}，請選擇 'binary'、'linear' 或 'softmax'。")

    return W


def _strategy_binary(A: np.ndarray) -> np.ndarray:
    """
    策略 A：二值化填 1 結構法 (Binary)

    邏輯：
      - 只要評分 > 0（有修過/被修過），格子值一律視為 1
      - 再對每一橫列除以該列總和（即節點的出度 Degree）
      - 最終：W[i, j] = 1 / degree(i)，若存在邊；否則為 0
    """
    # 二值化：> 0 的格子變成 1，其餘保持 0
    W = (A > 0).astype(np.float64)

    # 計算每一列的度數（出度），並進行橫向正規化
    row_sums = W.sum(axis=1, keepdims=True)          # shape: (total, 1)

    # 處理孤立節點（degree = 0）：避免除以零，孤立節點列保持全零
    # 孤立節點不影響 RWR，因為重啟向量 E 會確保機率質量守恆
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    W = W / row_sums_safe

    return W


def _strategy_linear(A: np.ndarray) -> np.ndarray:
    """
    策略 B：分數線性轉機率法 (Linear Rating)

    邏輯：
      - 保留原始分數（0~5）作為邊的權重
      - 每一格除以該列所有分數的總和
      - W[i, j] = score(i, j) / sum_k(score(i, k))
      - 數學等價性：先除以 5 再除以列總和 ≡ 直接除以列總和（常數消去）
    """
    W = A.copy()

    # 每一橫列除以該列的分數總和
    row_sums = W.sum(axis=1, keepdims=True)

    # 孤立節點保護：degree = 0 的列維持全零
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    W = W / row_sums_safe

    return W


def _strategy_softmax(A: np.ndarray) -> np.ndarray:
    """
    策略 C：遮罩式橫向 Softmax 法 (Masked Softmax)

    關鍵問題：
      標準 Softmax 會對所有格子計算 exp()，包含原本為 0（未修課）的格子。
      由於 exp(0) = 1 ≠ 0，這會錯誤地給予「未修課」格子正的游走機率！

    解決方案（遮罩機制）：
      步驟 1. 建立遮罩 mask：原始分數 > 0 的格子為 True，其餘為 False
      步驟 2. 僅對 mask = True 的格子套用 Softmax
      步驟 3. mask = False 的格子強制保持為 0
      步驟 4. 橫向加總天然等於 1（因為 Softmax 特性）
    """
    # --- 步驟 1：建立遮罩（只標記有實際評分的格子）---
    mask = (A > 0)                                    # bool 矩陣，shape: (total, total)

    # --- 步驟 2：對有效格子計算 Softmax ---
    # 數值穩定性技巧：每一列減去該列最大值（不影響 Softmax 結果，防止 exp 溢位）
    # 注意：計算最大值時只看 mask = True 的格子
    # 對 mask = False 的格子填入一個極小值，使 exp 後趨近 0，之後再用遮罩清零
    A_masked = np.where(mask, A, -1e9)               # 未修課格子設為 -∞（實際用 -1e9 防溢位）

    # 每列最大值（用於數值穩定化）
    row_max = A_masked.max(axis=1, keepdims=True)    # shape: (total, 1)

    # 計算 exp（已平移，分子）
    exp_A = np.exp(A_masked - row_max)               # shape: (total, total)

    # --- 步驟 3：遮罩清零 —— 這是 Masked Softmax 的核心！---
    # 雖然 -1e9 的 exp 已很小，但為確保數學精確性，強制將遮罩外格子設為 0
    exp_A = np.where(mask, exp_A, 0.0)

    # --- 步驟 4：橫向正規化，完成 Softmax ---
    row_sums = exp_A.sum(axis=1, keepdims=True)      # 分母：每列有效格子的 exp 總和

    # 孤立節點保護
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    W = exp_A / row_sums_safe

    return W


# =============================================================================
# 嚴謹度自檢：驗證橫向加總是否完美等於 1.0
# =============================================================================

def validate_transition_matrix(W: np.ndarray, tol: float = 1e-9) -> bool:
    """
    驗證轉移機率矩陣 W 的馬可夫鏈性質：每一橫列加總必須等於 1.0。

    排除孤立節點（列全為零的節點，這些節點在二部圖中不影響收斂性）。

    回傳：
      True  - 所有非孤立節點的列加總均在誤差容忍範圍內等於 1.0
      False - 存在不合格的列
    """
    row_sums = W.sum(axis=1)                          # 計算每一列的加總

    # 找出非孤立節點（排除全零列）
    non_isolated = row_sums > tol
    valid_sums = row_sums[non_isolated]

    # 檢驗每一個非孤立節點的列加總是否在容忍誤差內等於 1.0
    is_valid = np.allclose(valid_sums, 1.0, atol=tol)

    print(f"\n{'='*60}")
    print(f"  🔍 [嚴謹度自檢] 橫向轉移機率矩陣驗證報告")
    print(f"{'='*60}")
    print(f"  總節點數           : {W.shape[0]}")
    print(f"  孤立節點數（全零列）: {W.shape[0] - non_isolated.sum()}")
    print(f"  待驗證非孤立節點數 : {non_isolated.sum()}")
    print(f"  最大列加總偏差     : {np.abs(valid_sums - 1.0).max():.2e}")
    print(f"  最小列加總值       : {valid_sums.min():.10f}")
    print(f"  最大列加總值       : {valid_sums.max():.10f}")
    print(f"  {'─'*56}")
    print(f"  每個橫向加總是否完美等於 1.0？  →  {is_valid}")
    print(f"{'='*60}\n")

    # 斷言：若驗證失敗，立即中止程式並報錯
    assert is_valid, (
        "❌ 矩陣驗證失敗！存在橫向加總不等於 1.0 的列，"
        "請檢查正規化邏輯。"
    )
    return is_valid


# =============================================================================
# RWR 核心迭代函式
# =============================================================================

def random_walk_with_restart(
    W: np.ndarray,
    target_user_idx: int,
    c: float = 0.15,
    max_iter: int = 100,
    tol: float = 1e-6
) -> np.ndarray:
    """
    執行隨機游走演算法（Random Walk with Restart, RWR）。

    迭代公式：
        P_{t+1} = (1 - c) * W^T * P_t + c * E

    其中：
      W^T  : 轉移機率矩陣的轉置（代表「從其他節點游走到當前節點」的機率）
      P_t  : 第 t 次迭代的機率分佈向量（長度 = Total_Nodes）
      E    : 個人化重啟向量（One-hot vector，目標學生位置為 1）
      c    : 重啟機率（Restart Probability），預設 0.15

    物理意義：
      每次迭代，螞蟻有 (1-c) 的機率沿著圖的邊游走，
      有 c 的機率「傳送」回到目標學生節點重新出發。
      收斂後的 P 即反映整張圖與目標學生相關性的排名分數。

    參數：
      W               : 橫向轉移機率矩陣
      target_user_idx : 目標學生在總節點中的索引（通常就是學生編號 0 ~ M-1）
      c               : 重啟機率
      max_iter        : 最大迭代次數
      tol             : 收斂閾值（歐幾里得距離）

    回傳：
      P : 收斂後的機率分佈向量（長度 = Total_Nodes）
    """
    total_nodes = W.shape[0]

    # --- 建立個人化重啟向量 E（One-hot Vector）---
    # E[target_user_idx] = 1，其餘 = 0
    E = np.zeros(total_nodes, dtype=np.float64)
    E[target_user_idx] = 1.0

    # --- 初始化機率分佈向量 P_0 = E（從目標學生出發）---
    P = E.copy()

    # --- 預計算 W 的轉置（一次性計算，加速迭代）---
    # 注意：公式中是 W^T * P_t，而非 W * P_t
    # W[i, j] = 從節點 i 游走到節點 j 的機率
    # W^T[j, i] = W[i, j]，代表「從 i 到 j」，反過來看就是「j 從 i 接收的機率流」
    W_T = W.T

    print(f"[RWR 迭代] 目標學生索引 = {target_user_idx}，重啟機率 c = {c}")

    for iteration in range(1, max_iter + 1):
        # RWR 核心公式：P_{t+1} = (1-c) * W^T * P_t + c * E
        P_new = (1 - c) * W_T.dot(P) + c * E

        # 計算收斂指標：前後兩次向量的歐幾里得距離
        delta = np.linalg.norm(P_new - P, ord=2)

        P = P_new

        # 每 10 次印一次進度
        if iteration % 10 == 0 or delta < tol:
            print(f"  迭代 {iteration:3d} 次，收斂指標（Δ）= {delta:.2e}")

        # 收斂條件：歐幾里得距離小於容忍閾值
        if delta < tol:
            print(f"  ✅ 已於第 {iteration} 次迭代收斂（Δ = {delta:.2e} < {tol}）")
            break
    else:
        print(f"  ⚠️  已達最大迭代次數 {max_iter}，強制停止（最終 Δ = {delta:.2e}）")

    return P


# =============================================================================
# 推薦結果輸出
# =============================================================================

def get_top_n_recommendations(
    P: np.ndarray,
    target_user_idx: int,
    R: np.ndarray,
    item_ids: list,
    top_n: int = 5
) -> pd.DataFrame:
    """
    從 RWR 收斂向量中提取目標學生的 Top-N 課程推薦。

    步驟：
      1. 取出所有課程節點對應的 RWR 分數（P 向量的後 N 個元素）
      2. 過濾掉目標學生「已修過的課程」（原始評分 > 0）
      3. 依 RWR 分數由高到低排序
      4. 輸出 Top-N 課程 ID 與分數

    參數：
      P               : RWR 收斂機率向量（長度 = Total_Nodes）
      target_user_idx : 目標學生索引（在 R 矩陣中的列索引）
      R               : 原始評分矩陣（M x N）
      item_ids        : 課程 ID 清單（長度 = N）
      top_n           : 推薦數量

    回傳：
      recommendations : DataFrame，欄位為 [課程代碼, RWR分數]
    """
    M, N = R.shape

    # --- 取出課程節點的 RWR 分數 ---
    # 在全局節點向量中，前 M 個是學生，後 N 個是課程
    item_scores = P[M:]                              # 長度 = N

    # --- 建立課程分數的 DataFrame ---
    df = pd.DataFrame({
        "課程代碼": item_ids,
        "RWR分數": item_scores,
        "原始評分": R[target_user_idx]               # 目標學生對各課程的原始評分
    })

    # --- 過濾已修過的課程（原始評分 > 0）---
    df_unvisited = df[df["原始評分"] == 0].copy()

    # --- 依 RWR 分數由高到低排序 ---
    df_unvisited = df_unvisited.sort_values("RWR分數", ascending=False).reset_index(drop=True)

    # 取 Top-N
    top_recommendations = df_unvisited.head(top_n)[["課程代碼", "RWR分數"]]

    print(f"\n{'='*60}")
    print(f"  🎓 [推薦結果] 目標學生索引 {target_user_idx} 的 Top-{top_n} 推薦課程")
    print(f"{'='*60}")
    print(f"  已修課程數：{(R[target_user_idx] > 0).sum()} 門（已從推薦候選中排除）")
    print(f"  未修課程數：{(R[target_user_idx] == 0).sum()} 門（推薦候選池）")
    print(f"  {'─'*56}")
    for rank, (_, row) in enumerate(top_recommendations.iterrows(), 1):
        print(f"  第 {rank} 名：{row['課程代碼']:<20} RWR 分數 = {row['RWR分數']:.8f}")
    print(f"{'='*60}\n")

    return top_recommendations


# =============================================================================
# 主程式：整合流程
# =============================================================================

def run_course_recommendation(
    filepath: str,
    target_user_id,                                   # 學生 ID（來自 CSV 的 index）
    strategy: Literal["binary", "linear", "softmax"] = "linear",
    restart_prob: float = 0.15,
    top_n: int = 5
) -> pd.DataFrame:
    """
    大學課程推薦系統主程式，整合完整的 RWR 推薦流程。

    參數：
      filepath        : 評分矩陣 CSV 路徑
      target_user_id  : 目標學生的 ID（CSV 中的 index 值）
      strategy        : 轉移矩陣正規化策略（"binary" / "linear" / "softmax"）
      restart_prob    : RWR 重啟機率 c（預設 0.15）
      top_n           : 推薦課程數量（預設 5）

    回傳：
      recommendations : Top-N 推薦課程的 DataFrame
    """
    print("\n" + "="*60)
    print("  大學課程推薦系統 — Random Walk with Restart (RWR)")
    print("="*60)

    # --- 步驟 1：讀取資料 ---
    R, user_ids, item_ids = load_rating_matrix(filepath)

    # 確認目標學生存在
    assert target_user_id in user_ids, (
        f"目標學生 '{target_user_id}' 不存在於資料集中！\n"
        f"可用學生 ID：{user_ids}"
    )
    target_user_idx = user_ids.index(target_user_id)
    print(f"[目標學生] ID = '{target_user_id}'，在矩陣中的索引 = {target_user_idx}")

    # --- 步驟 2：建立二部圖鄰接矩陣 ---
    A = build_adjacency_matrix(R)

    # --- 步驟 3：建立橫向轉移機率矩陣（依選定策略）---
    W = build_transition_matrix(A, strategy=strategy)

    # --- 步驟 4：嚴謹度自檢（斷言每列加總 = 1.0）---
    validate_transition_matrix(W)

    # --- 步驟 5：執行 RWR 迭代 ---
    P = random_walk_with_restart(
        W,
        target_user_idx=target_user_idx,
        c=restart_prob
    )

    # --- 步驟 6：輸出 Top-N 推薦結果 ---
    recommendations = get_top_n_recommendations(
        P, target_user_idx, R, item_ids, top_n=top_n
    )

    return recommendations





# =============================================================================
# 執行入口
# =============================================================================

if __name__ == "__main__":

    CSV_PATH = "rating_matrix.csv"   # ← 確認檔案路徑與你的 CSV 放在同一目錄
    TARGET_USER = "U01"              # ← 換成你 CSV 裡實際存在的學生 ID

    print("\n\n" + "★"*60)
    print("  策略 A：Binary（二值化填 1 結構法）")
    print("★"*60)
    rec_binary = run_course_recommendation(
        filepath=CSV_PATH,
        target_user_id=TARGET_USER,
        strategy="binary",
        restart_prob=0.15,
        top_n=5
    )

    print("\n\n" + "★"*60)
    print("  策略 B：Linear Rating（分數線性轉機率法）")
    print("★"*60)
    rec_linear = run_course_recommendation(
        filepath=CSV_PATH,
        target_user_id=TARGET_USER,
        strategy="linear",
        restart_prob=0.15,
        top_n=5
    )

    print("\n\n" + "★"*60)
    print("  策略 C：Masked Softmax（遮罩式橫向 Softmax 法）")
    print("★"*60)
    rec_softmax = run_course_recommendation(
        filepath=CSV_PATH,
        target_user_id=TARGET_USER,
        strategy="softmax",
        restart_prob=0.15,
        top_n=5
    )

    print("\n" + "="*60)
    print("  📊 三種策略推薦結果比較（Top-5）")
    print("="*60)
    comparison = pd.DataFrame({
        "策略A_Binary":  rec_binary["課程代碼"].values,
        "策略B_Linear":  rec_linear["課程代碼"].values,
        "策略C_Softmax": rec_softmax["課程代碼"].values,
    }, index=[f"第{i}名" for i in range(1, 6)])
    print(comparison.to_string())
    print("="*60)
