"""
Phase 5 报告资产: 生成 A6 (Hidden + top-16 head attn) 逐样本分析。

对比 hidden-only (A0s) 与 A6 在测试子集上的逐样本预测，
生成 a6_case_analysis.csv 和 a6_correction_matrix.json。

用法:
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llm_hallucination
    source ./.venv/bin/activate
    python -s scripts/report_assets/generate_a6_analysis.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("a6_analysis")

from src.data.preprocessing import load_processed_data
from src.utils.feature_cache import load_npz_cache
from src.utils.reproducibility import set_global_seed

# 子集大小（与 Phase 4 一致）
SUBSET_SIZES = {"train": 600, "val": 150, "test": 150}


def load_cached_features(cache_dir: Path) -> dict:
    """加载 Phase 4 缓存特征。"""
    logger.info("加载缓存特征...")

    h_train = load_npz_cache(cache_dir / "hidden_layer17_last_train.npz")
    X_h_train = h_train["features"]
    h_val = load_npz_cache(cache_dir / "hidden_layer17_last_val.npz")
    X_h_val = h_val["features"]
    h_test = load_npz_cache(cache_dir / "hidden_layer17_last_test.npz")
    X_h_test = h_test["features"]

    as_train = load_npz_cache(cache_dir / "attention_scores_train.npz")
    X_as_train = as_train["features"]
    as_val = load_npz_cache(cache_dir / "attention_scores_val.npz")
    X_as_val = as_val["features"]
    as_test = load_npz_cache(cache_dir / "attention_scores_test.npz")
    X_as_test = as_test["features"]

    logger.info("Hidden features: train=%s, val=%s, test=%s",
                X_h_train.shape, X_h_val.shape, X_h_test.shape)
    logger.info("Attention score features: train=%s, val=%s, test=%s",
                X_as_train.shape, X_as_val.shape, X_as_test.shape)

    return {
        "X_h_train": X_h_train, "X_h_val": X_h_val, "X_h_test": X_h_test,
        "X_as_train": X_as_train, "X_as_val": X_as_val, "X_as_test": X_as_test,
    }


def residualize_by_length_simple(
    train_X: np.ndarray,
    val_X: np.ndarray,
    test_X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对每个特征维度做序列长度线性残差化。"""
    from sklearn.linear_model import LinearRegression

    n_features = train_X.shape[1]
    # 第一列为 sequence_length
    length_train = train_X[:, 0].reshape(-1, 1)
    length_val = val_X[:, 0].reshape(-1, 1)
    length_test = test_X[:, 0].reshape(-1, 1)

    train_r = np.zeros_like(train_X)
    val_r = np.zeros_like(val_X)
    test_r = np.zeros_like(test_X)

    for j in range(n_features):
        reg = LinearRegression()
        reg.fit(length_train, train_X[:, j])
        a, b = float(reg.coef_[0]), float(reg.intercept_)
        train_r[:, j] = train_X[:, j] - (a * length_train[:, 0] + b)
        val_r[:, j] = val_X[:, j] - (a * length_val[:, 0] + b)
        test_r[:, j] = test_X[:, j] - (a * length_test[:, 0] + b)

    return train_r, val_r, test_r


def get_top_indices(head_selection_path: Path) -> list[int]:
    """从 head_selection.json 读取选中的特征索引。"""
    with open(head_selection_path, "r") as f:
        hs = json.load(f)
    return hs.get("selected_feature_indices", [])


def compute_probs(X_train, y_train, X_eval, seed: int = 42) -> np.ndarray:
    """训练 LR 并返回概率。"""
    set_global_seed(seed)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_eval_s = scaler.transform(X_eval)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    return clf.predict_proba(X_eval_s)[:, 1]


def main():
    set_global_seed(42)

    # ---- 1. 加载数据 ----
    logger.info("加载预处理数据...")
    train_ds, val_ds, test_ds = load_processed_data()

    # ---- 2. 加载缓存特征 ----
    cache_dir = Path("experiments/results/phase4/cache")
    features = load_cached_features(cache_dir)

    X_h_train = features["X_h_train"]
    X_h_val = features["X_h_val"]
    X_h_test = features["X_h_test"]
    X_as_train = features["X_as_train"]
    X_as_val = features["X_as_val"]
    X_as_test = features["X_as_test"]

    # ---- 3. 取子集（600/150/150） ----
    X_h_sub_train = X_h_train[:SUBSET_SIZES["train"]]
    X_h_sub_val = X_h_val[:SUBSET_SIZES["val"]]
    X_h_sub_test = X_h_test[:SUBSET_SIZES["test"]]

    X_as_sub_train = X_as_train[:SUBSET_SIZES["train"]]
    X_as_sub_val = X_as_val[:SUBSET_SIZES["val"]]
    X_as_sub_test = X_as_test[:SUBSET_SIZES["test"]]

    y_sub_train = np.array(train_ds.labels[:SUBSET_SIZES["train"]], dtype=np.int64)
    y_sub_val = np.array(val_ds.labels[:SUBSET_SIZES["val"]], dtype=np.int64)
    y_sub_test = np.array(test_ds.labels[:SUBSET_SIZES["test"]], dtype=np.int64)

    test_statements = test_ds.statements[:SUBSET_SIZES["test"]]

    # ---- 4. 残差化 attention score 特征 ----
    X_as_sub_train_r, X_as_sub_val_r, X_as_sub_test_r = residualize_by_length_simple(
        X_as_sub_train, X_as_sub_val, X_as_sub_test,
    )

    # ---- 5. 读取 top-16 head 特征索引 ----
    head_selection_path = Path("experiments/results/phase4/attention_head_selection.json")
    top_indices = get_top_indices(head_selection_path)
    logger.info("使用 %d 个 top-head attention 特征", len(top_indices))

    # ---- 6. 计算 A6 特征 = hidden + top-head attention ----
    X_a6_train = np.concatenate([X_h_sub_train, X_as_sub_train_r[:, top_indices]], axis=1)
    X_a6_val = np.concatenate([X_h_sub_val, X_as_sub_val_r[:, top_indices]], axis=1)
    X_a6_test = np.concatenate([X_h_sub_test, X_as_sub_test_r[:, top_indices]], axis=1)

    # ---- 7. 计算概率 ----
    logger.info("计算 hidden-only 概率...")
    hidden_probs = compute_probs(X_h_sub_train, y_sub_train, X_h_sub_test)
    hidden_preds = (hidden_probs >= 0.5).astype(int)

    logger.info("计算 A6 概率...")
    a6_probs = compute_probs(X_a6_train, y_sub_train, X_a6_test)
    a6_preds = (a6_probs >= 0.5).astype(int)

    # ---- 8. 逐样本分类 ----
    rows = []
    case_counts = {
        "hidden_correct_a6_correct": 0,
        "hidden_correct_a6_wrong": 0,
        "hidden_wrong_a6_correct": 0,
        "hidden_wrong_a6_wrong": 0,
    }

    for i, stmt in enumerate(test_statements):
        h_correct = hidden_preds[i] == y_sub_test[i]
        a6_correct = a6_preds[i] == y_sub_test[i]

        if h_correct and a6_correct:
            case_type = "hidden_correct_a6_correct"
        elif h_correct and not a6_correct:
            case_type = "hidden_correct_a6_wrong"
        elif not h_correct and a6_correct:
            case_type = "hidden_wrong_a6_correct"
        else:
            case_type = "hidden_wrong_a6_wrong"

        case_counts[case_type] += 1

        rows.append({
            "statement": stmt,
            "label": int(y_sub_test[i]),
            "hidden_prob": float(hidden_probs[i]),
            "hidden_pred": int(hidden_preds[i]),
            "a6_prob": float(a6_probs[i]),
            "a6_pred": int(a6_preds[i]),
            "case_type": case_type,
        })

    # ---- 9. 保存 CSV ----
    out_dir = Path("experiments/results/phase4")
    csv_path = out_dir / "a6_case_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["statement", "label", "hidden_prob", "hidden_pred",
                                                "a6_prob", "a6_pred", "case_type"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("A6 case analysis 已保存至 %s (%d 行)", csv_path, len(rows))

    # ---- 10. 生成 correction matrix JSON ----
    net = case_counts["hidden_wrong_a6_correct"] - case_counts["hidden_correct_a6_wrong"]
    correction_matrix = {
        "hidden_correct_a6_correct": case_counts["hidden_correct_a6_correct"],
        "hidden_correct_a6_wrong": case_counts["hidden_correct_a6_wrong"],
        "hidden_wrong_a6_correct": case_counts["hidden_wrong_a6_correct"],
        "hidden_wrong_a6_wrong": case_counts["hidden_wrong_a6_wrong"],
        "net_correction": net,
    }

    json_path = out_dir / "a6_correction_matrix.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(correction_matrix, f, indent=2, ensure_ascii=False)
    logger.info("A6 correction matrix 已保存至 %s", json_path)

    # ---- 11. 打印统计 ----
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    h_acc = accuracy_score(y_sub_test, hidden_preds)
    a6_acc = accuracy_score(y_sub_test, a6_preds)
    h_auroc = roc_auc_score(y_sub_test, hidden_probs)
    a6_auroc = roc_auc_score(y_sub_test, a6_probs)

    print("\n" + "=" * 60)
    print("  A6 vs Hidden-only 对比 (Test Subset 150)")
    print("=" * 60)
    print(f"  Hidden-only: Acc={h_acc:.4f}, AUROC={h_auroc:.4f}")
    print(f"  A6:          Acc={a6_acc:.4f}, AUROC={a6_auroc:.4f}")
    print(f"\n  Correction Matrix:")
    print(f"    H+A6+ : {case_counts['hidden_correct_a6_correct']}")
    print(f"    H+A6- : {case_counts['hidden_correct_a6_wrong']}")
    print(f"    H-A6+ : {case_counts['hidden_wrong_a6_correct']}")
    print(f"    H-A6- : {case_counts['hidden_wrong_a6_wrong']}")
    print(f"    Net correction: {net:+d}")
    print("=" * 60)

    # Check if there are improvement cases
    improvement = [r for r in rows if r["case_type"] == "hidden_wrong_a6_correct"]
    print(f"\n  Improvement cases (hidden wrong, A6 correct): {len(improvement)}")
    for r in improvement[:5]:
        print(f"    [{r['label']}] {r['statement'][:80]}... (h={r['hidden_prob']:.4f}, a6={r['a6_prob']:.4f})")


if __name__ == "__main__":
    main()
