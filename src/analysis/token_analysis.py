"""Phase 3：不同 token 表示方式分析。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Iterable

import numpy as np

from src.config import config
from src.features.hidden_states import extract_hidden_states_dataset
from src.methods.saplma import train_and_evaluate

if TYPE_CHECKING:
    from src.data.dataset import TrueFalseDataset

logger = logging.getLogger(__name__)


def _resolve_layer_idx(num_hidden_layers: int, layer_idx: int) -> int:
    """将单层索引解析为正索引。"""
    normalized = num_hidden_layers + layer_idx if layer_idx < 0 else layer_idx
    if normalized < 0 or normalized >= num_hidden_layers:
        raise ValueError(f"层索引 {layer_idx} 超出范围 [0, {num_hidden_layers - 1}]")
    return normalized


def _resolve_poolings(poolings: Iterable[str] | None) -> list[str]:
    """解析并校验 token pooling 策略列表。"""
    candidates = list(config.features.pooling_strategies if poolings is None else poolings)
    if not candidates:
        raise ValueError("poolings 不能为空")

    supported = {"last", "first", "mean"}
    for pooling in candidates:
        if pooling not in supported:
            raise ValueError(f"不支持的 pooling 策略: {pooling}")

    return list(dict.fromkeys(candidates))


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


def extract_token_metric_bars(
    analysis_results: Dict,
    split: str = "test",
    metric: str = "accuracy",
) -> Dict[str, list[float]]:
    """从 token 分析结果中提取柱状图数据。"""
    if split not in {"val", "test"}:
        raise ValueError(f"不支持的 split: {split}")
    if metric not in {"accuracy", "macro_f1", "auroc"}:
        raise ValueError(f"不支持的 metric: {metric}")

    poolings = [item["pooling"] for item in analysis_results["per_pooling"]]
    means = [item[f"{split}_summary"][metric]["mean"] for item in analysis_results["per_pooling"]]
    stds = [item[f"{split}_summary"][metric]["std"] for item in analysis_results["per_pooling"]]

    return {
        "poolings": poolings,
        "means": means,
        "stds": stds,
    }


def analyze_token_pooling(
    model,
    tokenizer,
    train_dataset: TrueFalseDataset,
    val_dataset: TrueFalseDataset,
    test_dataset: TrueFalseDataset,
    classifier_type: str = "logistic",
    layer_idx: int = -1,
    poolings: Iterable[str] | None = None,
    batch_size: int = 8,
    max_length: int = 128,
    seeds=None,
    selection_metric: str = "accuracy",
) -> Dict:
    """比较不同 token pooling 策略的真假检测效果。"""
    if seeds is None:
        seeds = config.training.random_seeds
    if selection_metric not in {"accuracy", "macro_f1", "auroc"}:
        raise ValueError(f"不支持的 selection_metric: {selection_metric}")

    resolved_poolings = _resolve_poolings(poolings)
    num_layers = model.config.num_hidden_layers
    resolved_layer_idx = _resolve_layer_idx(num_layers, layer_idx)

    logger.info("=" * 50)
    logger.info(
        "Token Analysis: classifier=%s, layer=%d, poolings=%s",
        classifier_type,
        resolved_layer_idx,
        ",".join(resolved_poolings),
    )
    logger.info("=" * 50)

    per_pooling_results: list[Dict] = []
    for pooling in resolved_poolings:
        logger.info("分析 pooling=%s...", pooling)
        X_train, y_train = extract_hidden_states_dataset(
            model,
            tokenizer,
            train_dataset,
            pooling=pooling,
            layers=[resolved_layer_idx],
            batch_size=batch_size,
            max_length=max_length,
        )
        X_val, y_val = extract_hidden_states_dataset(
            model,
            tokenizer,
            val_dataset,
            pooling=pooling,
            layers=[resolved_layer_idx],
            batch_size=batch_size,
            max_length=max_length,
        )
        X_test, y_test = extract_hidden_states_dataset(
            model,
            tokenizer,
            test_dataset,
            pooling=pooling,
            layers=[resolved_layer_idx],
            batch_size=batch_size,
            max_length=max_length,
        )

        seed_runs: list[Dict] = []
        for seed in seeds:
            result = train_and_evaluate(
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                classifier_type=classifier_type,
                random_seed=seed,
            )
            seed_runs.append(result)

        pooling_result = {
            "pooling": pooling,
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
        per_pooling_results.append(pooling_result)

    best_pooling = max(
        per_pooling_results,
        key=lambda item: item["val_summary"][selection_metric]["mean"],
    )

    logger.info(
        "最佳 pooling: %s (val %s=%.4f)",
        best_pooling["pooling"],
        selection_metric,
        best_pooling["val_summary"][selection_metric]["mean"],
    )

    return {
        "analysis": "token",
        "classifier_type": classifier_type,
        "layer_idx": resolved_layer_idx,
        "poolings": resolved_poolings,
        "num_seeds": len(seeds),
        "seeds": list(seeds),
        "selection_metric": selection_metric,
        "per_pooling": per_pooling_results,
        "best_pooling": {
            "pooling": best_pooling["pooling"],
            "val_summary": best_pooling["val_summary"],
            "test_summary": best_pooling["test_summary"],
        },
    }


__all__ = [
    "analyze_token_pooling",
    "extract_token_metric_bars",
]