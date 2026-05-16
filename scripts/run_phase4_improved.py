
"""
Phase 4 改进版实验运行脚本。

用法:
    .\\.venv\\Scripts\\python.exe -s scripts/run_phase4_improved.py
    .\\.venv\\Scripts\\python.exe -s scripts/run_phase4_improved.py --classifier logistic --batch-size 8
    .\\.venv\\Scripts\\python.exe -s scripts/run_phase4_improved.py --classifier mlp --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run improved Phase 4 attention experiment")
    parser.add_argument("--classifier", choices=("logistic", "mlp"), default="logistic")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-layer", type=int, default=17)
    parser.add_argument("--selection-metric", choices=("accuracy", "macro_f1", "auroc"), default="accuracy")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    from src.analysis.visualization import plot_attention_feature_deltas, plot_attention_variant_comparison
    from src.config import config
    from src.data.preprocessing import load_processed_data
    from src.methods.advanced import run_attention_ablation_study
    from src.models.loader import load_model, print_device_info
    from src.utils.reproducibility import collect_runtime_info

    t_start = time.time()
    out_dir = config.paths.results_dir / "advanced"
    out_dir.mkdir(parents=True, exist_ok=True)

    print_device_info()
    print(f"\n加载模型 (Qwen2-1.5B {config.models.primary_dtype})...", flush=True)
    model, tokenizer = load_model()
    print(f"模型设备: {next(model.parameters()).device}", flush=True)

    print("\n加载预处理数据...", flush=True)
    train_ds, val_ds, test_ds = load_processed_data()
    print(train_ds.summary(), flush=True)

    hidden_layer_idx = min(args.hidden_layer, model.config.num_hidden_layers - 1)
    results = run_attention_ablation_study(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        classifier_type=args.classifier,
        hidden_layer_idx=hidden_layer_idx,
        hidden_pooling="last",
        batch_size=args.batch_size,
        max_length=args.max_length,
        selection_metric=args.selection_metric,
    )

    summary = {**results, "runtime": collect_runtime_info(model)}
    stem = f"attention_ablation_{args.classifier}_layer{hidden_layer_idx}_last_improved"
    result_path = out_dir / f"{stem}.json"
    result_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    plot_attention_variant_comparison(
        results,
        split="test",
        metric="accuracy",
        save_path=out_dir / f"{stem}_accuracy.png",
    )
    plot_attention_feature_deltas(
        results["attention_feature_summary"]["test"],
        top_k=8,
        save_path=out_dir / f"{stem}_feature_deltas.png",
    )

    elapsed = time.time() - t_start
    print(f"\n结果已保存至 {result_path}", flush=True)
    print(f"最佳变体: {results['best_variant']['name']}", flush=True)
    print(
        f"测试集 Accuracy: {results['best_variant']['test_summary']['accuracy']['mean']:.4f} ± "
        f"{results['best_variant']['test_summary']['accuracy']['std']:.4f}",
        flush=True,
    )
    print(f"总耗时: {elapsed:.0f}s ({elapsed / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()

