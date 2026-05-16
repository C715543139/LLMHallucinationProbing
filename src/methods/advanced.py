"""Phase 4：基于注意力模式的增强幻觉检测方法。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Sequence

import numpy as np

from src.config import config
from src.features.attention import (
    extract_attention_features_dataset,
    summarize_attention_feature_differences,
)
from src.features.hidden_states import extract_hidden_states_dataset
from src.methods.saplma import train_and_evaluate

if TYPE_CHECKING:
    from src.data.dataset import TrueFalseDataset

logger = logging.getLogger(__name__)


def _resolve_layer_idx(num_hidden_layers: int, layer_idx: int) -> int:
    normalized = num_hidden_layers + layer_idx if layer_idx < 0 else layer_idx
    if normalized < 0 or normalized >= num_hidden_layers:
        raise ValueError(f"层索引 {layer_idx} 超出范围 [0, {num_hidden_layers - 1}]")
    return normalized


def _summarize_split_metrics(per_seed_results: list[Dict], split: str) -> Dict[str, Dict[str, float]]:
    metrics = ("accuracy", "macro_f1", "auroc")
    summary: Dict[str, Dict[str, float]] = {}

    for metric_name in metrics:
        values = np.asarray([result[split][metric_name] for result in per_seed_results], dtype=np.float64)
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        }

    return summary


def concatenate_feature_blocks(*feature_blocks: np.ndarray) -> np.ndarray:
    """按列拼接多个特征块，要求样本数一致。"""
    if not feature_blocks:
        raise ValueError("至少需要一个特征块")

    normalized = [np.asarray(block, dtype=np.float64) for block in feature_blocks]
    row_count = normalized[0].shape[0]
    for block in normalized:
        if block.ndim != 2:
            raise ValueError(f"特征块必须为二维矩阵，实际为 {block.shape}")
        if block.shape[0] != row_count:
            raise ValueError("所有特征块的样本数必须一致")

    return np.concatenate(normalized, axis=1)


def _run_multi_seed_feature_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier_type: str,
    seeds: Sequence[int],
) -> Dict[str, object]:
    per_seed_results: list[Dict] = []
    for seed in seeds:
        result = train_and_evaluate(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            classifier_type=classifier_type,
            random_seed=int(seed),
        )
        per_seed_results.append(result)

    return {
        "feature_dim": int(X_train.shape[1]),
        "val_summary": _summarize_split_metrics(per_seed_results, split="val"),
        "test_summary": _summarize_split_metrics(per_seed_results, split="test"),
        "per_seed": [
            {
                "seed": int(seed),
                "val": result["val"],
                "test": result["test"],
            }
            for seed, result in zip(seeds, per_seed_results)
        ],
    }


def run_attention_ablation_study(
    model,
    tokenizer,
    train_dataset: TrueFalseDataset,
    val_dataset: TrueFalseDataset,
    test_dataset: TrueFalseDataset,
    classifier_type: str = "logistic",
    hidden_layer_idx: int = -1,
    hidden_pooling: str = "last",
    batch_size: int = 4,
    max_length: int = 128,
    seeds: Optional[Iterable[int]] = None,
    selection_metric: str = "accuracy",
) -> Dict[str, object]:
    """运行 Phase 4 注意力特征消融：attention-only / hidden-only / fusion。"""
    if seeds is None:
        seeds = config.training.random_seeds
    seeds = tuple(int(seed) for seed in seeds)

    if selection_metric not in {"accuracy", "macro_f1", "auroc"}:
        raise ValueError(f"不支持的 selection_metric: {selection_metric}")

    resolved_hidden_layer = _resolve_layer_idx(model.config.num_hidden_layers, hidden_layer_idx)

    logger.info("=" * 50)
    logger.info(
        "Phase 4 Attention Study: classifier=%s, hidden_layer=%d, hidden_pooling=%s",
        classifier_type,
        resolved_hidden_layer,
        hidden_pooling,
    )
    logger.info("=" * 50)

    logger.info("提取隐藏状态基线特征...")
    X_hidden_train, y_train = extract_hidden_states_dataset(
        model,
        tokenizer,
        train_dataset,
        pooling=hidden_pooling,
        layers=[resolved_hidden_layer],
        batch_size=batch_size,
        max_length=max_length,
    )
    X_hidden_val, y_val = extract_hidden_states_dataset(
        model,
        tokenizer,
        val_dataset,
        pooling=hidden_pooling,
        layers=[resolved_hidden_layer],
        batch_size=batch_size,
        max_length=max_length,
    )
    X_hidden_test, y_test = extract_hidden_states_dataset(
        model,
        tokenizer,
        test_dataset,
        pooling=hidden_pooling,
        layers=[resolved_hidden_layer],
        batch_size=batch_size,
        max_length=max_length,
    )

    logger.info("提取注意力统计特征...")
    X_attention_train, y_train_attention, feature_names, _ = extract_attention_features_dataset(
        model,
        tokenizer,
        train_dataset,
        batch_size=batch_size,
        max_length=max_length,
    )
    X_attention_val, y_val_attention, feature_names_val, _ = extract_attention_features_dataset(
        model,
        tokenizer,
        val_dataset,
        batch_size=batch_size,
        max_length=max_length,
    )
    X_attention_test, y_test_attention, feature_names_test, _ = extract_attention_features_dataset(
        model,
        tokenizer,
        test_dataset,
        batch_size=batch_size,
        max_length=max_length,
    )

    if not np.array_equal(y_train, y_train_attention):
        raise ValueError("训练集隐藏状态标签与注意力标签不一致")
    if not np.array_equal(y_val, y_val_attention):
        raise ValueError("验证集隐藏状态标签与注意力标签不一致")
    if not np.array_equal(y_test, y_test_attention):
        raise ValueError("测试集隐藏状态标签与注意力标签不一致")
    if list(feature_names) != list(feature_names_val) or list(feature_names) != list(feature_names_test):
        raise ValueError("不同数据划分提取到的 attention feature names 不一致")

    attention_feature_summary = {
        "train": summarize_attention_feature_differences(X_attention_train, y_train, feature_names),
        "val": summarize_attention_feature_differences(X_attention_val, y_val, feature_names),
        "test": summarize_attention_feature_differences(X_attention_test, y_test, feature_names),
    }

    variants = {
        "attention_only": (X_attention_train, X_attention_val, X_attention_test),
        "hidden_only": (X_hidden_train, X_hidden_val, X_hidden_test),
        "hidden_plus_attention": (
            concatenate_feature_blocks(X_hidden_train, X_attention_train),
            concatenate_feature_blocks(X_hidden_val, X_attention_val),
            concatenate_feature_blocks(X_hidden_test, X_attention_test),
        ),
    }

    variant_results: dict[str, Dict[str, object]] = {}
    for variant_name, (X_train, X_val, X_test) in variants.items():
        logger.info("运行变体: %s (feature_dim=%d)", variant_name, X_train.shape[1])
        variant_results[variant_name] = _run_multi_seed_feature_experiment(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            classifier_type=classifier_type,
            seeds=seeds,
        )

    best_variant_name = max(
        variant_results,
        key=lambda name: variant_results[name]["val_summary"][selection_metric]["mean"],
    )
    best_variant = variant_results[best_variant_name]

    return {
        "method": "attention_enhanced",
        "classifier_type": classifier_type,
        "hidden_layer_idx": resolved_hidden_layer,
        "hidden_pooling": hidden_pooling,
        "num_seeds": len(seeds),
        "seeds": list(seeds),
        "selection_metric": selection_metric,
        "attention_feature_names": list(feature_names),
        "num_attention_features": len(feature_names),
        "variants": variant_results,
        "attention_feature_summary": attention_feature_summary,
        "best_variant": {
            "name": best_variant_name,
            "val_summary": best_variant["val_summary"],
            "test_summary": best_variant["test_summary"],
            "feature_dim": best_variant["feature_dim"],
        },
    }


__all__ = [
    "concatenate_feature_blocks",
    "run_attention_ablation_study",
]

