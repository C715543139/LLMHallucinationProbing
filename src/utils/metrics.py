"""
评估指标模块。

提供统一的二分类评估接口：
    - Accuracy
    - Macro-F1
    - AUROC
    - 最优阈值搜索（基于验证集最大化 F1）
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """计算二分类常用指标。

    参数:
        y_true: 真实标签 (0/1), shape (N,).
        y_pred: 预测标签 (0/1), shape (N,).
        y_score: 预测分数（正类概率）, shape (N,). 若为 None 则 AUROC 返回 NaN.

    返回:
        包含 accuracy, macro_f1, auroc 的字典。
    """
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    if y_score is not None:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["auroc"] = float("nan")
    else:
        metrics["auroc"] = float("nan")

    return metrics


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """在验证集上搜索最优二分类阈值。

    参数:
        y_true: 真实标签 (0/1).
        y_score: 连续分数（数值越高越倾向正类）.
        metric: 优化目标 — "f1"（默认）或 "accuracy".

    返回:
        (best_threshold, best_value)
    """
    if metric == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        # 避免除零
        denom = precision + recall
        f1_scores = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
        best_idx = int(np.argmax(f1_scores))
        # precision_recall_curve 返回的 thresholds 比 scores 少一个元素
        if best_idx >= len(thresholds):
            best_idx = len(thresholds) - 1
        best_threshold = float(thresholds[best_idx])
        best_value = float(f1_scores[best_idx])
    elif metric == "accuracy":
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        # 寻找使 accuracy 最大的阈值（假设正负样本比例已知）
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))
        accuracies = (tpr * n_pos + (1 - fpr) * n_neg) / (n_pos + n_neg)
        best_idx = int(np.argmax(accuracies))
        best_threshold = float(thresholds[best_idx])
        best_value = float(accuracies[best_idx])
    else:
        raise ValueError(f"不支持的优化指标: {metric}，可选 'f1' 或 'accuracy'")

    logger.info("最优阈值=%.4f, 最佳 %s=%.4f", best_threshold, metric, best_value)
    return best_threshold, best_value


def evaluate_with_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    higher_is_positive: bool = True,
) -> Dict[str, float]:
    """使用给定阈值对连续分数做二分类并计算指标。

    参数:
        y_true: 真实标签 (0/1).
        y_score: 连续分数.
        threshold: 阈值.
        higher_is_positive: True 表示分数越高越倾向正类;
                           False 表示分数越低越倾向正类（如 PPL）.

    返回:
        指标字典。
    """
    if higher_is_positive:
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = (y_score <= threshold).astype(int)

    # AUROC 需要统一方向：正类分数应更高
    if not higher_is_positive:
        auroc_score_arr = -y_score
    else:
        auroc_score_arr = y_score

    return compute_metrics(y_true, y_pred, auroc_score_arr)


def compute_metrics_multi_seed(
    all_y_true: List[np.ndarray],
    all_y_pred: List[np.ndarray],
    all_y_score: List[np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """汇总多随机种子下的指标（均值 ± 标准差）。

    返回:
        {"accuracy": {"mean": ..., "std": ...}, "macro_f1": {...}, "auroc": {...}}
    """
    results: Dict[str, List[float]] = {"accuracy": [], "macro_f1": [], "auroc": []}

    for y_true, y_pred, y_score in zip(all_y_true, all_y_pred, all_y_score):
        m = compute_metrics(y_true, y_pred, y_score)
        for k in results:
            results[k].append(m[k])

    summary: Dict[str, Dict[str, float]] = {}
    for k, vals in results.items():
        arr = np.array(vals)
        summary[k] = {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0}

    return summary
