"""测试 Phase 3 可视化模块。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")


def test_visualization_module_importable() -> None:
    import src.analysis.visualization  # noqa: F401


def test_plot_layer_metric_curve_saves_file(tmp_path: Path) -> None:
    from src.analysis.visualization import plot_layer_metric_curve

    results = {
        "per_layer": [
            {"layer_idx": 0, "test_summary": {"accuracy": {"mean": 0.55, "std": 0.02}}},
            {"layer_idx": 1, "test_summary": {"accuracy": {"mean": 0.72, "std": 0.03}}},
        ]
    }

    out_path = tmp_path / "layer_curve.png"
    fig, ax = plot_layer_metric_curve(results, save_path=out_path)
    assert out_path.exists()
    assert "Layer Depth" in ax.get_title()
    fig.clf()


def test_plot_token_metric_comparison_saves_file(tmp_path: Path) -> None:
    from src.analysis.visualization import plot_token_metric_comparison

    results = {
        "per_pooling": [
            {"pooling": "first", "test_summary": {"accuracy": {"mean": 0.51, "std": 0.01}}},
            {"pooling": "last", "test_summary": {"accuracy": {"mean": 0.74, "std": 0.02}}},
        ]
    }

    out_path = tmp_path / "token_bars.png"
    fig, ax = plot_token_metric_comparison(results, save_path=out_path)
    assert out_path.exists()
    assert "Pooling Comparison" in ax.get_title()
    fig.clf()


def test_plot_method_comparison_saves_file(tmp_path: Path) -> None:
    from src.analysis.visualization import plot_method_comparison

    out_path = tmp_path / "methods.png"
    fig, ax = plot_method_comparison(
        {
            "PPL": {"accuracy": 0.53},
            "SAPLMA-LR": {"accuracy": 0.74},
            "SAPLMA-MLP": {"accuracy": 0.77},
        },
        save_path=out_path,
    )
    assert out_path.exists()
    assert "Method Comparison" in ax.get_title()
    fig.clf()