"""
Phase 4：注意力分析可视化模块。

提供:
    - layer × head AUROC heatmap
    - 特征差异箱线图
    - 方法对比柱状图
    - 修正矩阵可视化
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

import matplotlib.pyplot as plt
import numpy as np


def _finalize_figure(fig, save_path: str | Path | None = None):
    """保存或返回 Figure。"""
    fig.tight_layout()
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Layer × Head AUROC Heatmap
# ---------------------------------------------------------------------------

def plot_layer_head_auroc_heatmap(
    head_scores: list[dict],
    num_layers: int | None = None,
    num_heads: int | None = None,
    metric: str = "val_auroc",
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """绘制 layer × head 的判别能力 heatmap。

    参数:
        head_scores: 每个 head 的评分列表，每项含 layer, head, val_auroc 等。
        num_layers: 总层数（若为 None，自动推断）。
        num_heads: 总 head 数（若为 None，自动推断）。
        metric: 颜色映射的指标。
        title: 图标题。
        save_path: 保存路径。
    """
    if not head_scores:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No head scores available", ha="center", va="center")
        return _finalize_figure(fig, save_path), ax

    if num_layers is None:
        num_layers = max(h["layer"] for h in head_scores) + 1
    if num_heads is None:
        num_heads = max(h["head"] for h in head_scores) + 1

    # 构建矩阵
    matrix = np.full((num_layers, num_heads), np.nan)
    for h in head_scores:
        matrix[h["layer"], h["head"]] = h.get(metric, 0.0)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, aspect="auto", origin="upper", cmap="RdYlGn", vmin=0.4, vmax=0.9)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(title or f"Layer × Head {metric.upper()} Heatmap")
    plt.colorbar(im, ax=ax, label=metric.upper())
    return _finalize_figure(fig, save_path), ax


# ---------------------------------------------------------------------------
# 特征差异箱线图
# ---------------------------------------------------------------------------

def plot_feature_delta_boxplot(
    feature_summary: list[dict],
    top_n: int = 20,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """绘制 top-N 特征的 true/false 差异。

    参数:
        feature_summary: summarize_feature_differences 的输出。
        top_n: 展示前 N 个差异最大的特征。
        title: 图标题。
        save_path: 保存路径。
    """
    if not feature_summary:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No feature summary available", ha="center", va="center")
        return _finalize_figure(fig, save_path), ax

    top = feature_summary[:top_n]
    names = [f["feature_name"] for f in top]
    deltas = [f["delta"] for f in top]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    colors = ["#2ca02c" if d >= 0 else "#d62728" for d in deltas]
    y_pos = range(len(names))
    ax.barh(y_pos, deltas, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("True Mean - False Mean")
    ax.set_title(title or f"Top {top_n} Feature Differences (True vs False)")
    ax.invert_yaxis()
    return _finalize_figure(fig, save_path), ax


# ---------------------------------------------------------------------------
# 方法对比图
# ---------------------------------------------------------------------------

def plot_method_comparison(
    method_metrics: Mapping[str, Mapping[str, float]],
    metric: str = "accuracy",
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """绘制方法间指标对比柱状图。

    参数:
        method_metrics: {method_name: {metric_name: value}}.
        metric: 要展示的指标。
        title: 图标题。
        save_path: 保存路径。
    """
    labels = list(method_metrics.keys())
    values = np.asarray(
        [method_metrics[label].get(metric, 0.0) for label in labels],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(labels))
    bars = ax.bar(positions, values, alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Method Comparison ({metric.upper()})")
    ax.grid(axis="y", alpha=0.3)

    # 在柱上标注数值
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    return _finalize_figure(fig, save_path), ax


def plot_method_accuracy_comparison(
    method_metrics: Mapping[str, Mapping[str, float]],
    save_path: str | Path | None = None,
):
    """绘制方法 Accuracy 对比图。"""
    return plot_method_comparison(
        method_metrics,
        metric="accuracy",
        title="Method Comparison (Test Accuracy)",
        save_path=save_path,
    )


def plot_method_auroc_comparison(
    method_metrics: Mapping[str, Mapping[str, float]],
    save_path: str | Path | None = None,
):
    """绘制方法 AUROC 对比图。"""
    return plot_method_comparison(
        method_metrics,
        metric="auroc",
        title="Method Comparison (Test AUROC)",
        save_path=save_path,
    )


# ---------------------------------------------------------------------------
# 修正矩阵可视化
# ---------------------------------------------------------------------------

def plot_correction_matrix(
    correction_data: dict,
    title: str | None = None,
    save_path: str | Path | None = None,
):
    """绘制 hidden vs fusion 修正矩阵。

    参数:
        correction_data: {"hidden_correct_fusion_correct": n00, ...}
    """
    n00 = correction_data.get("hidden_correct_fusion_correct", 0)
    n01 = correction_data.get("hidden_correct_fusion_wrong", 0)
    n10 = correction_data.get("hidden_wrong_fusion_correct", 0)
    n11 = correction_data.get("hidden_wrong_fusion_wrong", 0)

    matrix = np.array([[n00, n01], [n10, n11]])
    row_labels = ["Hidden Correct", "Hidden Wrong"]
    col_labels = ["Fusion Correct", "Fusion Wrong"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="equal")

    # 标注数值
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=16)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(col_labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(row_labels)
    ax.set_title(title or "Error Correction Matrix")
    plt.colorbar(im, ax=ax)

    net = n10 - n01
    ax.text(
        0.5, -0.15,
        f"Net correction: {net:+d}",
        transform=ax.transAxes,
        ha="center",
        fontsize=12,
        fontweight="bold",
    )

    return _finalize_figure(fig, save_path), ax


# ---------------------------------------------------------------------------
# 综合可视化入口
# ---------------------------------------------------------------------------

def generate_phase4_figures(
    head_scores: list[dict] | None = None,
    feature_summary: list[dict] | None = None,
    method_metrics: dict[str, dict[str, float]] | None = None,
    correction_matrix: dict | None = None,
    output_dir: str | Path = "experiments/results/phase4/figures",
) -> dict[str, str]:
    """生成 Phase 4 所有标准图表。

    返回:
        {figure_name: saved_path}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}

    if head_scores:
        _, _ = plot_layer_head_auroc_heatmap(
            head_scores,
            save_path=output_dir / "layer_head_auroc_heatmap.png",
        )
        saved["layer_head_auroc_heatmap"] = str(output_dir / "layer_head_auroc_heatmap.png")

    if feature_summary:
        _, _ = plot_feature_delta_boxplot(
            feature_summary,
            save_path=output_dir / "feature_delta_boxplot.png",
        )
        saved["feature_delta_boxplot"] = str(output_dir / "feature_delta_boxplot.png")

    if method_metrics:
        _, _ = plot_method_accuracy_comparison(
            method_metrics,
            save_path=output_dir / "method_accuracy_comparison.png",
        )
        saved["method_accuracy_comparison"] = str(output_dir / "method_accuracy_comparison.png")

        _, _ = plot_method_auroc_comparison(
            method_metrics,
            save_path=output_dir / "method_auroc_comparison.png",
        )
        saved["method_auroc_comparison"] = str(output_dir / "method_auroc_comparison.png")

    if correction_matrix:
        _, _ = plot_correction_matrix(
            correction_matrix,
            save_path=output_dir / "correction_matrix.png",
        )
        saved["correction_matrix"] = str(output_dir / "correction_matrix.png")

    return saved
