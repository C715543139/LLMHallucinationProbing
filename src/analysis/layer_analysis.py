"""Phase 3：分层隐藏状态分析。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Iterable, List

import numpy as np

from src.config import config
from src.features.hidden_states import extract_hidden_states_dataset
from src.methods.saplma import train_and_evaluate

if TYPE_CHECKING:
    from src.data.dataset import TrueFalseDataset

logger = logging.getLogger(__name__)


def _resolve_layers(num_hidden_layers: int, layers: Iterable[int] | None) -> list[int]:
    """将层索引列表解析为升序、去重后的正索引。"""
    if layers is None:
        return list(range(num_hidden_layers))

    resolved: list[int] = []
    for layer_idx in layers:
        normalized = num_hidden_layers + layer_idx if layer_idx < 0 else layer_idx
        if normalized < 0 or normalized >= num_hidden_layers:
            raise ValueError(f"层索引 {layer_idx} 超出范围 [0, {num_hidden_layers - 1}]")
        resolved.append(normalized)

    return sorted(set(resolved))


def _ensure_layer_major(features: np.ndarray) -> np.ndarray:
    """统一多层特征形状为 (num_layers, N, hidden_dim)。"""
    array = np.asarray(features)
    if array.ndim == 2:
        return array[np.newaxis, ...]
    return array


def _summarize_split_metrics(per_seed_results: list[Dict], split: str) -> Dict[str, Dict[str, float]]:
    """汇总某个数据划分上的多随机种子指标。"""
    metrics = ("accuracy", "macro_f1", "auroc")
    summary: Dict[str, Dict[str, float]] = {}

    for metric_name in metrics:
        values = np.asarray([result[split][metric_name] for result in per_seed_results], dtype=np.float64)
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        }

    return summary


def extract_layer_metric_curve(
    analysis_results: Dict,
    split: str = "test",
    metric: str = "accuracy",
) -> Dict[str, list[float]]:
    """从分层分析结果中提取可直接绘图的曲线数据。"""
    if split not in {"val", "test"}:
        raise ValueError(f"不支持的 split: {split}")
    if metric not in {"accuracy", "macro_f1", "auroc"}:
        raise ValueError(f"不支持的 metric: {metric}")

    layer_indices = [item["layer_idx"] for item in analysis_results["per_layer"]]
    means = [item[f"{split}_summary"][metric]["mean"] for item in analysis_results["per_layer"]]
    stds = [item[f"{split}_summary"][metric]["std"] for item in analysis_results["per_layer"]]

    return {
        "layer_indices": layer_indices,
        "means": means,
        "stds": stds,
    }


def analyze_layer_performance(
    model,
    tokenizer,
    train_dataset: TrueFalseDataset,
    val_dataset: TrueFalseDataset,
    test_dataset: TrueFalseDataset,
    classifier_type: str = "logistic",
    pooling: str = "last",
    layers: Iterable[int] | None = None,
    batch_size: int = 8,
    max_length: int = 128,
    seeds=None,
    selection_metric: str = "accuracy",
) -> Dict:
    """逐层训练分类器并汇总验证集 / 测试集结果。"""
    if seeds is None:
        seeds = config.training.random_seeds
    if selection_metric not in {"accuracy", "macro_f1", "auroc"}:
        raise ValueError(f"不支持的 selection_metric: {selection_metric}")

    num_layers = model.config.num_hidden_layers
    resolved_layers = _resolve_layers(num_layers, layers)

    logger.info("=" * 50)
    logger.info(
        "Layer Analysis: classifier=%s, pooling=%s, layers=%d",
        classifier_type,
        pooling,
        len(resolved_layers),
    )
    logger.info("=" * 50)

    logger.info("提取训练集全部目标层的隐藏状态...")
    X_train_all, y_train = extract_hidden_states_dataset(
        model,
        tokenizer,
        train_dataset,
        pooling=pooling,
        layers=resolved_layers,
        batch_size=batch_size,
        max_length=max_length,
    )
    logger.info("提取验证集全部目标层的隐藏状态...")
    X_val_all, y_val = extract_hidden_states_dataset(
        model,
        tokenizer,
        val_dataset,
        pooling=pooling,
        layers=resolved_layers,
        batch_size=batch_size,
        max_length=max_length,
    )
    logger.info("提取测试集全部目标层的隐藏状态...")
    X_test_all, y_test = extract_hidden_states_dataset(
        model,
        tokenizer,
        test_dataset,
        pooling=pooling,
        layers=resolved_layers,
        batch_size=batch_size,
        max_length=max_length,
    )

    X_train_all = _ensure_layer_major(X_train_all)
    X_val_all = _ensure_layer_major(X_val_all)
    X_test_all = _ensure_layer_major(X_test_all)

    per_layer_results: list[Dict] = []
    for layer_offset, layer_idx in enumerate(resolved_layers):
        logger.info("分析第 %d 层...", layer_idx)
        seed_runs: list[Dict] = []
        for seed in seeds:
            result = train_and_evaluate(
                X_train_all[layer_offset],
                y_train,
                X_val_all[layer_offset],
                y_val,
                X_test_all[layer_offset],
                y_test,
                classifier_type=classifier_type,
                random_seed=seed,
            )
            seed_runs.append(result)

        layer_result = {
            "layer_idx": layer_idx,
            "val_summary": _summarize_split_metrics(seed_runs, split="val"),
            "test_summary": _summarize_split_metrics(seed_runs, split="test"),
            "per_seed": [
                {
                    "seed": int(seed),
                    "val": result["val"],
                    "test": result["test"],
                }
                for seed, result in zip(seeds, seed_runs)
            ],
        }
        per_layer_results.append(layer_result)

    best_layer = max(
        per_layer_results,
        key=lambda item: item["val_summary"][selection_metric]["mean"],
    )

    logger.info(
        "最佳层: %d (val %s=%.4f)",
        best_layer["layer_idx"],
        selection_metric,
        best_layer["val_summary"][selection_metric]["mean"],
    )

    return {
        "analysis": "layer",
        "classifier_type": classifier_type,
        "pooling": pooling,
        "layers": resolved_layers,
        "num_layers": len(resolved_layers),
        "num_seeds": len(seeds),
        "seeds": list(seeds),
        "selection_metric": selection_metric,
        "per_layer": per_layer_results,
        "best_layer": {
            "layer_idx": best_layer["layer_idx"],
            "val_summary": best_layer["val_summary"],
            "test_summary": best_layer["test_summary"],
        },
    }


__all__ = [
    "analyze_layer_performance",
    "extract_layer_metric_curve",
]