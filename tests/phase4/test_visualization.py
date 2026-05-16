"""测试 Phase 4 可视化函数。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")


def test_phase4_visualization_functions_save_files(tmp_path: Path) -> None:
    from src.analysis.visualization import (
        plot_attention_feature_deltas,
        plot_attention_variant_comparison,
    )

    variant_out = tmp_path / "attention_variants.png"
    fig1, ax1 = plot_attention_variant_comparison(
        {
            "variants": {
                "attention_only": {"test_summary": {"accuracy": {"mean": 0.60, "std": 0.01}}},
                "hidden_only": {"test_summary": {"accuracy": {"mean": 0.72, "std": 0.02}}},
                "hidden_plus_attention": {"test_summary": {"accuracy": {"mean": 0.80, "std": 0.03}}},
            }
        },
        save_path=variant_out,
    )
    assert variant_out.exists()
    assert "Attention Ablation" in ax1.get_title()
    fig1.clf()

    delta_out = tmp_path / "attention_deltas.png"
    fig2, ax2 = plot_attention_feature_deltas(
        {
            "top_features": [
                {"name": "tail_attn_ratio", "delta": 0.18, "abs_delta": 0.18},
                {"name": "attn_entropy_mean", "delta": -0.12, "abs_delta": 0.12},
            ]
        },
        save_path=delta_out,
    )
    assert delta_out.exists()
    assert "Top Attention Feature Deltas" in ax2.get_title()
    fig2.clf()

