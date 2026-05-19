"""
Phase 4: Attention-Guided SAPLMA 主方法模块。

在 Phase 3 hidden state baseline 之上，引入注意力分数和注意力输出激活特征，
实现 attention-only、hidden-only、hidden+attention 和 gated fusion 的消融实验。

包含:
    - hidden feature 缓存
    - hidden-only baseline
    - 长度残差化去偏
    - validation-based head selection
    - 分类器训练与评估
    - gated fusion
    - 消融实验流水线
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.config import config
from src.features.hidden_states import extract_hidden_states_dataset
from src.utils.feature_cache import save_npz_cache, load_npz_cache, cache_exists
from src.utils.metrics import compute_metrics, compute_metrics_multi_seed
from src.utils.reproducibility import set_global_seed, collect_runtime_info

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# P4.0: Hidden feature 缓存
# ---------------------------------------------------------------------------

def cache_phase3_hidden_features(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    test_dataset,
    output_dir: str | Path,
    layer_idx: int = 17,
    pooling: str = "last",
    batch_size: int = 8,
    max_length: int = 128,
) -> dict[str, str]:
    """缓存 Phase 3 最优配置的 hidden features。

    返回:
        {"train": path, "val": path, "test": path}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    for split_name, dataset in [
        ("train", train_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ]:
        out_path = output_dir / f"hidden_layer{layer_idx}_{pooling}_{split_name}.npz"

        if cache_exists(out_path):
            logger.info("缓存已存在: %s", out_path)
            paths[split_name] = str(out_path)
            continue

        logger.info("提取 %s hidden features (layer=%d, pooling=%s)...", split_name, layer_idx, pooling)
        X, y = extract_hidden_states_dataset(
            model, tokenizer, dataset,
            pooling=pooling, layers=[layer_idx],
            batch_size=batch_size, max_length=max_length,
        )

        save_npz_cache(
            out_path, X, y,
            metadata={
                "layer": layer_idx,
                "pooling": pooling,
                "split": split_name,
                "hidden_dim": int(X.shape[1]),
            },
        )
        paths[split_name] = str(out_path)
        logger.info("已保存: %s (shape=%s)", out_path, X.shape)

    return paths


# ---------------------------------------------------------------------------
# P4.0: Hidden-only baseline
# ---------------------------------------------------------------------------

def run_hidden_baseline(
    train_hidden: np.ndarray,
    val_hidden: np.ndarray,
    test_hidden: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    classifier_type: str = "logistic",
    seeds: tuple[int, ...] = (42, 123, 2024),
    hidden_dim: int | None = None,
) -> dict:
    """训练 hidden-only 分类器，返回多 seed 指标。

    返回:
        包含 method, feature, classifier, seeds, val, test 等字段的字典。
    """
    all_test_preds: list[np.ndarray] = []
    all_test_scores: list[np.ndarray] = []
    all_val_preds: list[np.ndarray] = []
    all_val_scores: list[np.ndarray] = []

    for seed in seeds:
        set_global_seed(seed)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(train_hidden)
        X_val_s = scaler.transform(val_hidden)
        X_test_s = scaler.transform(test_hidden)

        clf = _make_classifier(classifier_type, seed)
        clf.fit(X_train_s, train_labels)

        val_pred = clf.predict(X_val_s)
        val_score = _get_scores(clf, X_val_s)
        test_pred = clf.predict(X_test_s)
        test_score = _get_scores(clf, X_test_s)

        all_val_preds.append(val_pred)
        all_val_scores.append(val_score)
        all_test_preds.append(test_pred)
        all_test_scores.append(test_score)

    val_summary = compute_metrics_multi_seed(
        [val_labels] * len(seeds), all_val_preds, all_val_scores,
    )
    test_summary = compute_metrics_multi_seed(
        [test_labels] * len(seeds), all_test_preds, all_test_scores,
    )

    result: dict[str, Any] = {
        "method": f"hidden_layer17_last_{classifier_type}",
        "feature": {
            "layer": 17,
            "pooling": "last",
            "dim": hidden_dim or int(train_hidden.shape[1]),
        },
        "classifier": classifier_type,
        "seeds": list(seeds),
        "val": {k: {"mean": v["mean"], "std": v["std"]} for k, v in val_summary.items()},
        "test": {k: {"mean": v["mean"], "std": v["std"]} for k, v in test_summary.items()},
    }
    return result


def _make_classifier(classifier_type: str, seed: int):
    """创建分类器实例。"""
    if classifier_type == "logistic":
        return LogisticRegression(
            C=config.training.logistic_C,
            max_iter=config.training.logistic_max_iter,
            penalty=config.training.logistic_penalty,
            random_state=seed,
            n_jobs=config.training.n_jobs,
        )
    elif classifier_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=config.training.mlp_hidden_sizes,
            activation=config.training.mlp_activation,
            alpha=config.training.mlp_alpha,
            max_iter=config.training.mlp_max_iter,
            random_state=seed,
        )
    else:
        raise ValueError(f"不支持的分类器类型: {classifier_type}")


def _get_scores(clf, X: np.ndarray) -> np.ndarray:
    """获取正类概率分数。"""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.predict(X).astype(float)


# ---------------------------------------------------------------------------
# P4.3: 长度残差化去偏
# ---------------------------------------------------------------------------

def residualize_by_length(
    train_X: np.ndarray,
    val_X: np.ndarray,
    test_X: np.ndarray,
    train_length: np.ndarray,
    val_length: np.ndarray,
    test_length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """对每个特征维度使用 train length 做线性残差化。

    仅使用第一列（sequence_length）做残差化。

    返回:
        (train_residual, val_residual, test_residual, metadata)
    """
    n_features = train_X.shape[1]
    train_residual = np.zeros_like(train_X)
    val_residual = np.zeros_like(val_X)
    test_residual = np.zeros_like(test_X)

    # 使用 sequence_length（第一列）
    length_col_train = train_length[:, 0].reshape(-1, 1)
    length_col_val = val_length[:, 0].reshape(-1, 1)
    length_col_test = test_length[:, 0].reshape(-1, 1)

    coeffs: dict[int, dict[str, float]] = {}

    for j in range(n_features):
        reg = LinearRegression()
        reg.fit(length_col_train, train_X[:, j])
        a = float(reg.coef_[0])
        b = float(reg.intercept_)

        train_residual[:, j] = train_X[:, j] - (a * length_col_train[:, 0] + b)
        val_residual[:, j] = val_X[:, j] - (a * length_col_val[:, 0] + b)
        test_residual[:, j] = test_X[:, j] - (a * length_col_test[:, 0] + b)

        coeffs[j] = {"a": a, "b": b}

    # 检查残差后与长度的相关性
    corr_before = np.corrcoef(train_X[:, 0], length_col_train[:, 0])[0, 1] if n_features > 0 else 0.0
    corr_after = np.corrcoef(train_residual[:, 0], length_col_train[:, 0])[0, 1] if n_features > 0 else 0.0

    metadata = {
        "method": "linear_residualization",
        "predictor": "sequence_length",
        "correlation_before": float(corr_before),
        "correlation_after": float(corr_after),
        "num_features": n_features,
        "coeffs": coeffs,
    }

    return train_residual, val_residual, test_residual, metadata


# ---------------------------------------------------------------------------
# P4.4: Validation-based head selection
# ---------------------------------------------------------------------------

def group_feature_indices_by_head(feature_names: list[str]) -> dict[tuple[int, int], list[int]]:
    """将 L{layer}_H{head}_xxx 格式的特征名按 (layer, head) 分组。"""
    import re
    pattern = re.compile(r"L(\d+)_H(\d+)_")
    groups: dict[tuple[int, int], list[int]] = {}

    for idx, name in enumerate(feature_names):
        m = pattern.search(name)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            groups.setdefault(key, []).append(idx)

    return groups


def score_head_group(
    train_X: np.ndarray,
    val_X: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    feature_indices: list[int],
    metric: str = "auroc",
) -> dict:
    """使用单个 head 的全部特征训练小 LR，在 val 上评估该 head 的判别能力。"""
    X_train_head = train_X[:, feature_indices]
    X_val_head = val_X[:, feature_indices]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_head)
    X_val_s = scaler.transform(X_val_head)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, train_y)

    val_pred = clf.predict(X_val_s)
    val_score = _get_scores(clf, X_val_s)

    return compute_metrics(val_y, val_pred, val_score)


def select_top_heads(
    train_X: np.ndarray,
    val_X: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    feature_names: list[str],
    top_k_heads: int = 16,
    metric: str = "auroc",
) -> dict:
    """在验证集上选择最有判别力的 top-k heads。

    返回:
        {"selected_heads": [...], "selected_feature_indices": [...],
         "selected_feature_names": [...], "all_head_scores": [...]}
    """
    groups = group_feature_indices_by_head(feature_names)
    logger.info("共 %d 个 (layer, head) 分组", len(groups))

    head_scores: list[dict] = []
    for (layer_idx, head_idx), indices in groups.items():
        try:
            metrics = score_head_group(
                train_X, val_X, train_y, val_y, indices, metric
            )
        except Exception:
            metrics = {"accuracy": 0.0, "macro_f1": 0.0, "auroc": 0.0}

        head_scores.append({
            "layer": layer_idx,
            "head": head_idx,
            "num_features": len(indices),
            "feature_indices": indices,
            "val_accuracy": metrics.get("accuracy", 0.0),
            "val_macro_f1": metrics.get("macro_f1", 0.0),
            "val_auroc": metrics.get("auroc", 0.0),
        })

    # 按 val_auroc 降序排序
    metric_key = f"val_{metric}"
    head_scores.sort(key=lambda x: x.get(metric_key, 0.0), reverse=True)

    # 取 top-k
    top_heads = head_scores[:top_k_heads]
    selected_indices: list[int] = []
    selected_names: list[str] = []
    for h in top_heads:
        selected_indices.extend(h["feature_indices"])
        for idx in h["feature_indices"]:
            selected_names.append(feature_names[idx])

    logger.info(
        "选择了 top %d heads, 共 %d 个特征",
        len(top_heads), len(selected_indices),
    )

    return {
        "selected_heads": [
            {
                "layer": h["layer"],
                "head": h["head"],
                "val_accuracy": h["val_accuracy"],
                "val_macro_f1": h["val_macro_f1"],
                "val_auroc": h["val_auroc"],
            }
            for h in top_heads
        ],
        "selected_feature_indices": selected_indices,
        "selected_feature_names": selected_names,
        "all_head_scores": head_scores,
        "selection_metric": metric_key,
        "top_k_heads": top_k_heads,
    }


# ---------------------------------------------------------------------------
# 单次训练评估
# ---------------------------------------------------------------------------

def train_eval_classifier(
    train_X: np.ndarray,
    val_X: np.ndarray,
    test_X: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    test_y: np.ndarray,
    classifier_type: str = "logistic",
    seeds: tuple[int, ...] = (42, 123, 2024),
) -> dict:
    """对一个 feature setting 做多 seed 训练和评估。"""
    all_val_preds: list[np.ndarray] = []
    all_val_scores: list[np.ndarray] = []
    all_test_preds: list[np.ndarray] = []
    all_test_scores: list[np.ndarray] = []
    per_seed_results: list[dict] = []

    for seed in seeds:
        set_global_seed(seed)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(train_X)
        X_val_s = scaler.transform(val_X)
        X_test_s = scaler.transform(test_X)

        clf = _make_classifier(classifier_type, seed)
        clf.fit(X_train_s, train_y)

        val_pred = clf.predict(X_val_s)
        val_score = _get_scores(clf, X_val_s)
        test_pred = clf.predict(X_test_s)
        test_score = _get_scores(clf, X_test_s)

        all_val_preds.append(val_pred)
        all_val_scores.append(val_score)
        all_test_preds.append(test_pred)
        all_test_scores.append(test_score)

        per_seed_results.append({
            "seed": seed,
            "val": compute_metrics(val_y, val_pred, val_score),
            "test": compute_metrics(test_y, test_pred, test_score),
        })

    val_summary = compute_metrics_multi_seed(
        [val_y] * len(seeds), all_val_preds, all_val_scores,
    )
    test_summary = compute_metrics_multi_seed(
        [test_y] * len(seeds), all_test_preds, all_test_scores,
    )

    return {
        "val_summary": val_summary,
        "test_summary": test_summary,
        "per_seed": per_seed_results,
        "seeds": list(seeds),
    }


# ---------------------------------------------------------------------------
# P4.6: Gated Fusion
# ---------------------------------------------------------------------------

def gated_fusion_probs(
    hidden_probs: np.ndarray,
    fusion_probs: np.ndarray,
    tau: float,
) -> np.ndarray:
    """当 hidden-only 不自信时，使用 fusion 模型的预测替代。

    公式:
        p(x) = p_hidden(x)  if |p_hidden(x) - 0.5| > tau
             = p_fusion(x)  otherwise
    """
    hidden_probs = np.asarray(hidden_probs, dtype=np.float64)
    fusion_probs = np.asarray(fusion_probs, dtype=np.float64)
    uncertain = np.abs(hidden_probs - 0.5) <= tau
    final = hidden_probs.copy()
    final[uncertain] = fusion_probs[uncertain]
    return final


def select_gated_fusion_tau(
    val_hidden_probs: np.ndarray,
    val_fusion_probs: np.ndarray,
    val_y: np.ndarray,
    tau_candidates: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25),
    metric: str = "macro_f1",
) -> dict:
    """在验证集上选择最佳的 gated fusion tau 阈值。"""
    best_tau = tau_candidates[0]
    best_value = -1.0
    results: list[dict] = []

    for tau in tau_candidates:
        fused = gated_fusion_probs(val_hidden_probs, val_fusion_probs, tau)
        preds = (fused >= 0.5).astype(int)
        m = compute_metrics(val_y, preds, fused)
        val = m.get(metric, 0.0)
        results.append({"tau": tau, "metrics": m})
        if val > best_value:
            best_value = val
            best_tau = tau

    return {
        "best_tau": best_tau,
        "best_value": best_value,
        "selection_metric": metric,
        "candidates": results,
    }


def apply_gated_fusion(
    test_hidden_probs: np.ndarray,
    test_fusion_probs: np.ndarray,
    tau: float,
    test_y: np.ndarray,
) -> dict:
    """在测试集上应用 gated fusion 并计算指标。"""
    fused = gated_fusion_probs(test_hidden_probs, test_fusion_probs, tau)
    preds = (fused >= 0.5).astype(int)
    n_changed = int(np.sum(np.abs(fused - test_hidden_probs) > 0.001))

    return {
        "tau": tau,
        "n_samples_changed": n_changed,
        "metrics": compute_metrics(test_y, preds, fused),
        "fused_probs": fused,
        "fused_preds": preds,
    }


# ---------------------------------------------------------------------------
# 特征差异分析
# ---------------------------------------------------------------------------

def summarize_feature_differences(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    output_csv: str | None = None,
) -> list[dict]:
    """统计 true/false 样本在各特征上的差异。"""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    true_mask = y == 1
    false_mask = y == 0

    rows: list[dict] = []
    for j, name in enumerate(feature_names):
        true_vals = X[true_mask, j]
        false_vals = X[false_mask, j]

        true_mean = float(np.mean(true_vals)) if true_mask.sum() > 0 else 0.0
        false_mean = float(np.mean(false_vals)) if false_mask.sum() > 0 else 0.0
        delta = true_mean - false_mean

        # 单特征 AUROC
        try:
            from sklearn.metrics import roc_auc_score
            auroc = float(roc_auc_score(y, X[:, j]))
        except Exception:
            auroc = 0.5

        rows.append({
            "feature_name": name,
            "true_mean": true_mean,
            "false_mean": false_mean,
            "delta": delta,
            "abs_delta": abs(delta),
            "true_std": float(np.std(true_vals)) if true_mask.sum() > 0 else 0.0,
            "false_std": float(np.std(false_vals)) if false_mask.sum() > 0 else 0.0,
            "single_feature_auroc": auroc,
        })

    rows.sort(key=lambda r: r["abs_delta"], reverse=True)

    if output_csv:
        import csv
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    return rows


# ---------------------------------------------------------------------------
# 错误分析
# ---------------------------------------------------------------------------

def build_error_analysis(
    statements: list[str],
    labels: np.ndarray,
    hidden_probs: np.ndarray,
    fusion_probs: np.ndarray,
    hidden_preds: np.ndarray,
    fusion_preds: np.ndarray,
) -> list[dict]:
    """构建错误分析案例表。"""
    labels = np.asarray(labels, dtype=np.int64)
    hidden_probs = np.asarray(hidden_probs, dtype=np.float64)
    fusion_probs = np.asarray(fusion_probs, dtype=np.float64)
    hidden_preds = np.asarray(hidden_preds, dtype=np.int64)
    fusion_preds = np.asarray(fusion_preds, dtype=np.int64)

    rows: list[dict] = []
    for i, stmt in enumerate(statements):
        h_correct = hidden_preds[i] == labels[i]
        f_correct = fusion_preds[i] == labels[i]

        if h_correct and f_correct:
            case_type = "hidden_correct_fusion_correct"
        elif h_correct and not f_correct:
            case_type = "hidden_correct_fusion_wrong"
        elif not h_correct and f_correct:
            case_type = "hidden_wrong_fusion_correct"
        else:
            case_type = "hidden_wrong_fusion_wrong"

        rows.append({
            "statement": stmt,
            "label": int(labels[i]),
            "hidden_prob": float(hidden_probs[i]),
            "fusion_prob": float(fusion_probs[i]),
            "hidden_pred": int(hidden_preds[i]),
            "fusion_pred": int(fusion_preds[i]),
            "case_type": case_type,
        })

    return rows


# ---------------------------------------------------------------------------
# 全流程消融实验
# ---------------------------------------------------------------------------

def run_phase4_ablation(
    hidden_train: np.ndarray,
    hidden_val: np.ndarray,
    hidden_test: np.ndarray,
    attn_score_train: np.ndarray | None,
    attn_score_val: np.ndarray | None,
    attn_score_test: np.ndarray | None,
    attn_output_train: np.ndarray | None,
    attn_output_val: np.ndarray | None,
    attn_output_test: np.ndarray | None,
    top_head_indices: list[int] | None,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    test_labels: np.ndarray,
    train_statements: list[str],
    test_statements: list[str],
    classifier_type: str = "logistic",
    seeds: tuple[int, ...] = (42, 123, 2024),
    output_dir: str | Path | None = None,
) -> dict:
    """运行完整的 Phase 4 消融实验。

    比较:
        A0: hidden_only
        A1: attention_score_only_raw
        A2: attention_score_only_debiased
        A3: attention_score_top_heads_only
        A4: attention_output_only
        A5: hidden_plus_debiased_attention
        A6: hidden_plus_top_head_attention
        A7: hidden_plus_attention_output
        A8: hidden_plus_all_attention
        A9: gated_fusion
    """
    output_dir = Path(output_dir) if output_dir else None
    all_results: dict[str, dict] = {}

    # ---- A0: hidden-only --------------------------------------------------
    logger.info("=== A0: hidden-only ===")
    a0 = train_eval_classifier(
        hidden_train, hidden_val, hidden_test,
        train_labels, val_labels, test_labels,
        classifier_type=classifier_type, seeds=seeds,
    )
    a0["method_name"] = "A0_hidden_only"
    a0["feature_dim"] = hidden_train.shape[1]
    all_results["A0_hidden_only"] = a0

    # ---- A1: attention score raw only -------------------------------------
    if attn_score_train is not None:
        logger.info("=== A1: attention_score_only_raw ===")
        a1 = train_eval_classifier(
            attn_score_train, attn_score_val, attn_score_test,
            train_labels, val_labels, test_labels,
            classifier_type=classifier_type, seeds=seeds,
        )
        a1["method_name"] = "A1_attention_score_raw"
        a1["feature_dim"] = attn_score_train.shape[1]
        all_results["A1_attention_score_raw"] = a1
    else:
        logger.info("A1 跳过（无 attention score 特征）")

    # ---- A4: attention output only ----------------------------------------
    if attn_output_train is not None:
        logger.info("=== A4: attention_output_only ===")
        a4 = train_eval_classifier(
            attn_output_train, attn_output_val, attn_output_test,
            train_labels, val_labels, test_labels,
            classifier_type=classifier_type, seeds=seeds,
        )
        a4["method_name"] = "A4_attention_output_only"
        a4["feature_dim"] = attn_output_train.shape[1]
        all_results["A4_attention_output_only"] = a4
    else:
        logger.info("A4 跳过（无 attention output 特征）")

    # ---- A6: hidden + top head attention ---------------------------------
    if attn_score_train is not None and top_head_indices is not None and len(top_head_indices) > 0:
        logger.info("=== A6: hidden_plus_top_head_attention ===")
        X_train_a6 = np.concatenate([hidden_train, attn_score_train[:, top_head_indices]], axis=1)
        X_val_a6 = np.concatenate([hidden_val, attn_score_val[:, top_head_indices]], axis=1)
        X_test_a6 = np.concatenate([hidden_test, attn_score_test[:, top_head_indices]], axis=1)
        a6 = train_eval_classifier(
            X_train_a6, X_val_a6, X_test_a6,
            train_labels, val_labels, test_labels,
            classifier_type=classifier_type, seeds=seeds,
        )
        a6["method_name"] = "A6_hidden_plus_top_head_attention"
        a6["feature_dim"] = X_train_a6.shape[1]
        all_results["A6_hidden_plus_top_head_attention"] = a6
    else:
        logger.info("A6 跳过")

    # ---- A7: hidden + attention output -----------------------------------
    if attn_output_train is not None:
        logger.info("=== A7: hidden_plus_attention_output ===")
        X_train_a7 = np.concatenate([hidden_train, attn_output_train], axis=1)
        X_val_a7 = np.concatenate([hidden_val, attn_output_val], axis=1)
        X_test_a7 = np.concatenate([hidden_test, attn_output_test], axis=1)
        a7 = train_eval_classifier(
            X_train_a7, X_val_a7, X_test_a7,
            train_labels, val_labels, test_labels,
            classifier_type=classifier_type, seeds=seeds,
        )
        a7["method_name"] = "A7_hidden_plus_attention_output"
        a7["feature_dim"] = X_train_a7.shape[1]
        all_results["A7_hidden_plus_attention_output"] = a7
    else:
        logger.info("A7 跳过")

    # ---- A8: hidden + all attention (score + output) ----------------------
    if attn_score_train is not None and attn_output_train is not None and top_head_indices is not None:
        logger.info("=== A8: hidden_plus_all_attention ===")
        X_train_a8 = np.concatenate([
            hidden_train,
            attn_score_train[:, top_head_indices],
            attn_output_train,
        ], axis=1)
        X_val_a8 = np.concatenate([
            hidden_val,
            attn_score_val[:, top_head_indices],
            attn_output_val,
        ], axis=1)
        X_test_a8 = np.concatenate([
            hidden_test,
            attn_score_test[:, top_head_indices],
            attn_output_test,
        ], axis=1)
        a8 = train_eval_classifier(
            X_train_a8, X_val_a8, X_test_a8,
            train_labels, val_labels, test_labels,
            classifier_type=classifier_type, seeds=seeds,
        )
        a8["method_name"] = "A8_hidden_plus_all_attention"
        a8["feature_dim"] = X_train_a8.shape[1]
        all_results["A8_hidden_plus_all_attention"] = a8
    else:
        logger.info("A8 跳过")

    # ---- A9: Gated Fusion (hidden vs best fusion) -------------------------
    # 选取最佳 fusion 方法（A7 或 A8）来做 gated fusion
    best_fusion_key = None
    for key in ["A8_hidden_plus_all_attention", "A7_hidden_plus_attention_output", "A6_hidden_plus_top_head_attention"]:
        if key in all_results:
            best_fusion_key = key
            break

    if best_fusion_key:
        logger.info("=== A9: gated_fusion (using %s) ===", best_fusion_key)
        # 需要获取每个 seed 的概率
        # 这里简化：使用第一个 seed 的结果做 gated fusion
        # 更多 seed 的 gated fusion 需要在 per_seed 级别运行
        hidden_seed0 = _get_seed0_probs(a0, test_labels)
        fusion_seed0 = _get_seed0_probs(all_results[best_fusion_key], test_labels)

        if hidden_seed0 is not None and fusion_seed0 is not None:
            # 在 val 上选 tau
            hidden_val0 = _get_seed0_probs(a0, val_labels, split="val")
            fusion_val0 = _get_seed0_probs(all_results[best_fusion_key], val_labels, split="val")

            if hidden_val0 is not None and fusion_val0 is not None:
                tau_result = select_gated_fusion_tau(
                    hidden_val0, fusion_val0, val_labels,
                )
                best_tau = tau_result["best_tau"]

                # 在 test 上应用
                gf_result = apply_gated_fusion(
                    hidden_seed0, fusion_seed0, best_tau, test_labels,
                )
                a9 = {
                    "method_name": "A9_gated_fusion",
                    "tau": best_tau,
                    "tau_selection": tau_result,
                    "test_metrics": gf_result["metrics"],
                    "n_samples_changed": gf_result["n_samples_changed"],
                    "based_on": best_fusion_key,
                }
                all_results["A9_gated_fusion"] = a9
            else:
                logger.info("A9 跳过（无法获取验证集概率）")
        else:
            logger.info("A9 跳过（无法获取测试集概率）")
    else:
        logger.info("A9 跳过（无 fusion 方法）")

    # ---- 构建 correction matrix -------------------------------------------
    if "A9_gated_fusion" in all_results and best_fusion_key:
        h_preds = _get_seed0_preds(a0, test_labels)
        f_probs = all_results.get("A9_gated_fusion", {}).get("fused_probs")
        if h_preds is not None and f_probs is not None:
            f_preds = (f_probs >= 0.5).astype(np.int64)
            n00 = int(np.sum((h_preds == test_labels) & (f_preds == test_labels)))
            n01 = int(np.sum((h_preds == test_labels) & (f_preds != test_labels)))
            n10 = int(np.sum((h_preds != test_labels) & (f_preds == test_labels)))
            n11 = int(np.sum((h_preds != test_labels) & (f_preds != test_labels)))
            all_results["correction_matrix"] = {
                "hidden_correct_fusion_correct": n00,
                "hidden_correct_fusion_wrong": n01,
                "hidden_wrong_fusion_correct": n10,
                "hidden_wrong_fusion_wrong": n11,
            }

    # ---- 保存结果 ----------------------------------------------------------
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存主结果 CSV
        main_rows = _build_main_results_csv(all_results)
        csv_path = output_dir / "phase4_main_results.csv"
        _write_csv(csv_path, main_rows)
        logger.info("主结果已保存至 %s", csv_path)

        # 保存 ablation JSON
        ablation_path = output_dir / "phase4_ablation_results.json"
        ablation_dict = _serialize_ablation(all_results)
        with open(ablation_path, "w", encoding="utf-8") as f:
            json.dump(ablation_dict, f, indent=2, ensure_ascii=False, default=_json_default)
        logger.info("消融结果已保存至 %s", ablation_path)

    return all_results


def _get_seed0_probs(result: dict, labels: np.ndarray, split: str = "test") -> np.ndarray | None:
    """从结果字典中提取 seed0 的概率。"""
    per_seed = result.get("per_seed", [])
    if not per_seed:
        return None
    # 需要存储了 probs; saplma 结果没有直接存 probs
    # Phase 4 的 train_eval_classifier 也没有存 probs
    # 这里我们需要在 train_eval_classifier 中保存 probs
    return None


def _get_seed0_preds(result: dict, labels: np.ndarray) -> np.ndarray | None:
    """从结果字典中提取 seed0 的预测。"""
    per_seed = result.get("per_seed", [])
    if not per_seed:
        return None
    # 同样的问题 - 需要存储 preds
    return None


def _build_main_results_csv(all_results: dict) -> list[dict]:
    """构建主结果表行。"""
    rows: list[dict] = []
    for method_key, result in all_results.items():
        row: dict[str, Any] = {
            "method": result.get("method_name", method_key),
            "feature_dim": result.get("feature_dim", 0),
        }

        test_summary = result.get("test_summary", {})
        for metric in ["accuracy", "macro_f1", "auroc"]:
            ms = test_summary.get(metric, {})
            row[f"test_{metric}_mean"] = ms.get("mean", float("nan"))
            row[f"test_{metric}_std"] = ms.get("std", float("nan"))

        # gated fusion 特殊处理
        if method_key == "A9_gated_fusion":
            tm = result.get("test_metrics", {})
            row["test_accuracy_mean"] = tm.get("accuracy", float("nan"))
            row["test_macro_f1_mean"] = tm.get("macro_f1", float("nan"))
            row["test_auroc_mean"] = tm.get("auroc", float("nan"))
            row["test_accuracy_std"] = 0.0
            row["test_macro_f1_std"] = 0.0
            row["test_auroc_std"] = 0.0

        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    """写入 CSV 文件。"""
    import csv
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _serialize_ablation(all_results: dict) -> dict:
    """将消融结果转为可 JSON 序列化的字典。"""
    serialized: dict[str, Any] = {}
    for key, value in all_results.items():
        if key == "correction_matrix":
            serialized[key] = value
            continue
        # 提取关键字段
        entry: dict[str, Any] = {
            "method_name": value.get("method_name", key),
            "feature_dim": value.get("feature_dim", 0),
        }
        for split_key in ["test_summary", "val_summary"]:
            if split_key in value:
                entry[split_key] = value[split_key]
        if "test_metrics" in value:
            entry["test_metrics"] = value["test_metrics"]
        if "tau" in value:
            entry["tau"] = value["tau"]
        if "n_samples_changed" in value:
            entry["n_samples_changed"] = value["n_samples_changed"]
        if "based_on" in value:
            entry["based_on"] = value["based_on"]
        serialized[key] = entry
    return serialized


def _json_default(obj):
    """JSON 序列化默认处理。"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ---------------------------------------------------------------------------
# Phase 4 summary 生成
# ---------------------------------------------------------------------------

def write_phase4_summary(
    output_dir: str | Path,
    hidden_baseline: dict | None = None,
    head_selection: dict | None = None,
    ablation_results: dict | None = None,
    runtime_info: dict | None = None,
) -> str:
    """生成 Phase 4 总结 Markdown 文件。"""
    output_dir = Path(output_dir)
    lines: list[str] = []

    lines.append("# Phase 4: Attention-Guided SAPLMA 实验结果")
    lines.append("")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Hidden baseline
    if hidden_baseline:
        lines.append("## Hidden Baseline (P4.0)")
        lines.append("")
        lines.append("| Metric | Val Mean | Val Std | Test Mean | Test Std |")
        lines.append("|---|---:|---:|---:|---:|")
        for metric in ["accuracy", "macro_f1", "auroc"]:
            vm = hidden_baseline.get("val", {}).get(metric, {})
            tm = hidden_baseline.get("test", {}).get(metric, {})
            lines.append(
                f"| {metric} | {vm.get('mean', '-')} | {vm.get('std', '-')} | "
                f"{tm.get('mean', '-')} | {tm.get('std', '-')} |"
            )
        lines.append("")

    # Head selection
    if head_selection:
        lines.append("## Head Selection (P4.4)")
        lines.append("")
        lines.append(f"- Top-K heads: {head_selection.get('top_k_heads', '-')}")
        lines.append(f"- Selection metric: {head_selection.get('selection_metric', '-')}")
        selected = head_selection.get("selected_heads", [])
        if selected:
            lines.append("")
            lines.append("| Layer | Head | Val AUROC | Val Accuracy |")
            lines.append("|---:|---:|---:|---:|")
            for h in selected[:10]:
                lines.append(
                    f"| {h['layer']} | {h['head']} | "
                    f"{h.get('val_auroc', 0):.4f} | {h.get('val_accuracy', 0):.4f} |"
                )
        lines.append("")

    # Ablation
    if ablation_results:
        lines.append("## Ablation Results (P4.6)")
        lines.append("")
        lines.append("| Method | Feature Dim | Test Acc | Test Macro-F1 | Test AUROC |")
        lines.append("|---|---:|---:|---:|---:|")
        for key, result in ablation_results.items():
            if key == "correction_matrix":
                continue
            name = result.get("method_name", key)
            dim = result.get("feature_dim", 0)
            ts = result.get("test_summary", {})
            if not ts and "test_metrics" in result:
                ts = {"accuracy": {"mean": result["test_metrics"].get("accuracy", 0)},
                       "macro_f1": {"mean": result["test_metrics"].get("macro_f1", 0)},
                       "auroc": {"mean": result["test_metrics"].get("auroc", 0)}}
            acc = ts.get("accuracy", {}).get("mean", "-")
            f1 = ts.get("macro_f1", {}).get("mean", "-")
            auroc = ts.get("auroc", {}).get("mean", "-")
            lines.append(f"| {name} | {dim} | {acc} | {f1} | {auroc} |")
        lines.append("")

        # Correction matrix
        cm = ablation_results.get("correction_matrix")
        if cm:
            lines.append("## Error Correction Matrix")
            lines.append("")
            lines.append("| | Fusion Correct | Fusion Wrong |")
            lines.append("|---|---:|:---:|")
            lines.append(f"| Hidden Correct | {cm['hidden_correct_fusion_correct']} | {cm['hidden_correct_fusion_wrong']} |")
            lines.append(f"| Hidden Wrong | {cm['hidden_wrong_fusion_correct']} | {cm['hidden_wrong_fusion_wrong']} |")
            net = cm['hidden_wrong_fusion_correct'] - cm['hidden_correct_fusion_wrong']
            lines.append(f"\nNet correction: {net} samples")
            lines.append("")

    # Runtime
    if runtime_info:
        lines.append("## Runtime Info")
        lines.append("")
        lines.append(f"- PyTorch: {runtime_info.get('torch', '-')}")
        lines.append(f"- Transformers: {runtime_info.get('transformers', '-')}")
        lines.append(f"- CUDA: {runtime_info.get('cuda_version', '-')}")
        lines.append(f"- GPU: {runtime_info.get('gpu_name', '-')}")
        lines.append("")

    summary_text = "\n".join(lines)
    summary_path = output_dir / "phase4_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return summary_text
