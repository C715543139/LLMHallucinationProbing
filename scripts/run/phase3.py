"""
Phase 3 运行脚本：执行层分析、token pooling 分析，或按子模式只运行其中之一。

用法:
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llm_hallucination
    source ./.venv/bin/activate
    python -s scripts/run/phase3.py
    python -s scripts/run/phase3.py layer
    python -s scripts/run/phase3.py token
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="运行 Phase 3 分析实验。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="full",
        choices=("full", "layer", "token"),
        help="选择运行全部、逐层分析、或 token pooling 分析",
    )
    return parser.parse_args()


def load_model_and_data():
    """加载模型与预处理数据。"""
    from src.config import config
    from src.data.preprocessing import load_processed_data
    from src.models.loader import load_model, print_device_info

    print_device_info()
    print(f"\n加载模型 (Qwen2-1.5B {config.models.primary_dtype})...")
    model, tokenizer = load_model()
    print(f"模型设备: {next(model.parameters()).device}")

    print("\n加载预处理数据...")
    train_ds, val_ds, test_ds = load_processed_data()
    print(train_ds.summary())
    return model, tokenizer, train_ds, val_ds, test_ds


def run_layer_analysis(model, tokenizer, train_ds, val_ds, test_ds, out_dir: Path, runtime_info: dict) -> None:
    """运行逐层分析。"""
    from src.analysis.layer_analysis import analyze_layer_performance
    from src.analysis.visualization import plot_layer_metric_curve

    print("\n" + "=" * 50)
    print("  Phase 3.1: 逐层分析")
    print("=" * 50)

    results = analyze_layer_performance(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        classifier_type="logistic",
        pooling="last",
        layers=None,
        batch_size=8,
        max_length=128,
    )

    summary = {**results, "runtime": runtime_info}
    out_path = out_dir / "layer_analysis_logistic_last.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=float)

    plot_layer_metric_curve(
        results,
        split="test",
        metric="accuracy",
        save_path=out_dir / "layer_accuracy_curve.png",
    )

    print(f"Layer analysis 结果已保存至 {out_path}")
    print(f"最佳层: {results['best_layer']['layer_idx']}")
    print(
        f"测试集 Accuracy: {results['best_layer']['test_summary']['accuracy']['mean']:.4f} ± "
        f"{results['best_layer']['test_summary']['accuracy']['std']:.4f}"
    )


def run_token_analysis(model, tokenizer, train_ds, val_ds, test_ds, out_dir: Path, runtime_info: dict) -> None:
    """运行 token pooling 分析。"""
    from src.analysis.token_analysis import analyze_token_pooling
    from src.analysis.visualization import plot_token_metric_comparison

    print("\n" + "=" * 50)
    print("  Phase 3.2: Token Pooling 分析")
    print("=" * 50)

    results = analyze_token_pooling(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        classifier_type="logistic",
        layer_idx=-1,
        poolings=("last", "first", "mean"),
        batch_size=8,
        max_length=128,
    )

    summary = {**results, "runtime": runtime_info}
    out_path = out_dir / "token_analysis_logistic_last_layer.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=float)

    plot_token_metric_comparison(
        results,
        split="test",
        metric="accuracy",
        save_path=out_dir / "token_accuracy_comparison.png",
    )

    print(f"Token analysis 结果已保存至 {out_path}")
    print(f"最佳 pooling: {results['best_pooling']['pooling']}")
    print(
        f"测试集 Accuracy: {results['best_pooling']['test_summary']['accuracy']['mean']:.4f} ± "
        f"{results['best_pooling']['test_summary']['accuracy']['std']:.4f}"
    )


def main() -> None:
    """脚本主入口。"""
    args = parse_args()

    from src.config import config
    from src.utils.reproducibility import collect_runtime_info

    t_start = time.time()
    print("=" * 60)
    print("  Phase 3: 分析实验")
    print("=" * 60)

    model, tokenizer, train_ds, val_ds, test_ds = load_model_and_data()
    runtime_info = collect_runtime_info(model)
    out_dir = config.paths.results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("full", "layer"):
        run_layer_analysis(model, tokenizer, train_ds, val_ds, test_ds, out_dir, runtime_info)

    if args.mode in ("full", "token"):
        run_token_analysis(model, tokenizer, train_ds, val_ds, test_ds, out_dir, runtime_info)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Phase 3 完成! 总耗时: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()