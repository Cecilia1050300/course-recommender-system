"""
人工抽查腳本 —— 從四個方法各自存的逐筆結果裡隨機抽樣，
把「真實成績 vs 預測值」列出來，方便手動核對預測合理性。

用法（從專案根目錄執行）：
    python shared/manual_verification.py

需要先跑過以下四個腳本，讓對應的 results/*_details.csv 存在：
    methods/gcn/gcn.py
    methods/rwr/RWR_experiment_ndcg.py
    methods/matrix_factorization/MF_experiment_ndcg.py
    methods/similarity_6ways/cf_experiment_ndcg.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PER_METHOD = 8
RANDOM_SEED = 42

SOURCES = {
    "GCN": PROJECT_ROOT / "methods" / "gcn" / "results" / "GCN_Pure_Torch_details.csv",
    "RWR": PROJECT_ROOT / "methods" / "rwr" / "results" / "RWR_details.csv",
    "MF": PROJECT_ROOT / "methods" / "matrix_factorization" / "results" / "MF_details.csv",
    "CF (similarity_6ways)": PROJECT_ROOT / "methods" / "similarity_6ways" / "results" / "CF_details.csv",
}


def sample_method(model_name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"⚠️  找不到 {path}，跳過 {model_name}（請先跑過對應的實驗腳本）")
        return pd.DataFrame()

    df = pd.read_csv(path)
    sample = df.sample(n=min(SAMPLE_PER_METHOD, len(df)), random_state=RANDOM_SEED)

    sample = sample.copy()
    sample.insert(0, "Model", model_name)
    if "Error" not in sample.columns:
        sample["Error"] = (sample["Actual"] - sample["Predicted"]).abs().round(4)

    keep_cols = ["Model", "Method", "Student_ID", "Course_ID", "Actual", "Predicted", "Error"]
    keep_cols = [c for c in keep_cols if c in sample.columns]
    return sample[keep_cols]


def main():
    samples = [sample_method(name, path) for name, path in SOURCES.items()]
    combined = pd.concat([s for s in samples if not s.empty], ignore_index=True)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    print("\n" + "=" * 90)
    print(f"隨機抽樣人工核對表（每個方法 {SAMPLE_PER_METHOD} 筆，seed={RANDOM_SEED}）")
    print("=" * 90)
    print(combined.to_string(index=False))
    print("=" * 90)
    print("\n請人工核對：Predicted 是否落在 1~5 合理範圍、Error 是否符合各方法回報的 MAE 量級。")

    out_path = Path(__file__).resolve().parent / "manual_verification_sample.csv"
    combined.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n已存檔：{out_path}")


if __name__ == "__main__":
    main()
