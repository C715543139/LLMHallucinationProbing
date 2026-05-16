"""Phase 3：分析结果可视化。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.layer_analysis import extract_layer_metric_curve
from src.analysis.token_analysis import extract_token_metric_bars


def _finalize_figure(fig, save_path: Optional[Union[str, Path]] = None):
    """根据需要保存图像并返回 Figure。"""
    fig.tight_layout()
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_layer_metric_curve(
    analysis_results: Dict,
    split: str = "test",
    metric: str = "accuracy",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    """绘制层深度-性能曲线。"""
    curve = extract_layer_metric_curve(analysis_results, split=split, metric=metric)
    x = np.asarray(curve["layer_indices"], dtype=np.int64)
    means = np.asarray(curve["means"], dtype=np.float64)
    stds = np.asarray(curve["stds"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, means, marker="o", linewidth=2)
    lower = np.clip(means - stds, 0.0, 1.0)
    upper = np.clip(means + stds, 0.0, 1.0)
    ax.fill_between(x, lower, upper, alpha=0.2)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Layer Depth vs {metric.upper()} ({split})")
    ax.grid(alpha=0.3)
    return _finalize_figure(fig, save_path=save_path), ax


def plot_token_metric_comparison(
    analysis_results: Dict,
    split: str = "test",
    metric: str = "accuracy",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    """绘制不同 pooling 策略的性能对比柱状图。"""
    bars = extract_token_metric_bars(analysis_results, split=split, metric=metric)
    labels = bars["poolings"]
    means = np.asarray(bars["means"], dtype=np.float64)
    stds = np.asarray(bars["stds"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(labels))
    ax.bar(positions, means, yerr=stds, capsize=4, alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Pooling Comparison ({split} {metric.upper()})")
    ax.grid(axis="y", alpha=0.3)
    return _finalize_figure(fig, save_path=save_path), ax


def plot_method_comparison(
    method_metrics: Mapping[str, Mapping[str, float]],
    metric: str = "accuracy",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    """绘制方法间指标对比图。"""
    labels = list(method_metrics.keys())
    values = np.asarray([method_metrics[label][metric] for label in labels], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(labels))
    ax.bar(positions, values, alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Method Comparison ({metric.upper()})")
    ax.grid(axis="y", alpha=0.3)
    return _finalize_figure(fig, save_path=save_path), ax


def plot_attention_variant_comparison(
    analysis_results: Dict,
    split: str = "test",
    metric: str = "accuracy",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    """绘制 Phase 4 中 attention-only / hidden-only / fusion 的对比图。"""
    if split not in {"val", "test"}:
        raise ValueError(f"不支持的 split: {split}")
    if metric not in {"accuracy", "macro_f1", "auroc"}:
        raise ValueError(f"不支持的 metric: {metric}")

    variants = analysis_results["variants"]
    labels = list(variants.keys())
    means = np.asarray([variants[name][f"{split}_summary"][metric]["mean"] for name in labels], dtype=np.float64)
    stds = np.asarray([variants[name][f"{split}_summary"][metric]["std"] for name in labels], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    positions = np.arange(len(labels))
    ax.bar(positions, means, yerr=stds, capsize=4, alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Attention Ablation ({split} {metric.upper()})")
    ax.grid(axis="y", alpha=0.3)
    return _finalize_figure(fig, save_path=save_path), ax


def plot_attention_feature_deltas(
    feature_summary: Dict,
    top_k: int = 8,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
):
    """绘制真/假样本之间注意力特征均值差异最大的若干特征。"""
    if top_k <= 0:
        raise ValueError("top_k 必须为正整数")

    top_features = list(feature_summary.get("top_features", []))[:top_k]
    if not top_features:
        raise ValueError("feature_summary 中不存在可绘制的 top_features")

    labels = [item["name"] for item in top_features][::-1]
    deltas = np.asarray([item["delta"] for item in top_features], dtype=np.float64)[::-1]
    colors = ["tab:blue" if value >= 0 else "tab:orange" for value in deltas]

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.55 * len(labels))))
    positions = np.arange(len(labels))
    ax.barh(positions, deltas, color=colors, alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("True Mean - False Mean")
    ax.set_title(title or "Top Attention Feature Deltas")
    ax.grid(axis="x", alpha=0.3)
    return _finalize_figure(fig, save_path=save_path), ax


__all__ = [
    "plot_attention_feature_deltas",
    "plot_attention_variant_comparison",
    "plot_layer_metric_curve",
    "plot_token_metric_comparison",
    "plot_method_comparison",
]