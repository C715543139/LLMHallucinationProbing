"""
Phase 2 运行脚本：执行 PPL 基线、SAPLMA 基线，或按子模式只运行其中之一。

用法:
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llm_hallucination
    source ./.venv/bin/activate
    python -s scripts/run/phase2.py
    python -s scripts/run/phase2.py ppl
    python -s scripts/run/phase2.py saplma
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase2")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="运行 Phase 2 基线实验。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="full",
        choices=("full", "ppl", "saplma"),
        help="选择运行全部、仅 PPL、或仅 SAPLMA",
    )
    return parser.parse_args()


def load_model_and_data():
    """加载模型和预处理数据。"""
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


def run_ppl(model, tokenizer, train_ds, val_ds, test_ds, out_dir: Path, runtime_info: dict) -> None:
    """运行并保存 PPL 基线。"""
    from src.methods.probability import evaluate_ppl_method

    print("\n" + "=" * 50)
    print("  P2.1-P2.2: PPL 方法")
    print("=" * 50)

    results = evaluate_ppl_method(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        batch_size=8,
        max_length=128,
        threshold_metric="f1",
    )

    summary = {
        key: value
        for key, value in results.items()
        if key in ("method", "threshold", "threshold_metric", "train", "val", "test")
    }
    summary["runtime"] = runtime_info

    out_path = out_dir / "ppl_results.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, default=float)

    print(f"PPL 结果已保存至 {out_path}")
    print(
        f"测试集: Acc={results['test']['accuracy']:.4f}, "
        f"Macro-F1={results['test']['macro_f1']:.4f}, "
        f"AUROC={results['test']['auroc']:.4f}"
    )


def run_saplma(model, tokenizer, train_ds, val_ds, test_ds, out_dir: Path, runtime_info: dict) -> None:
    """运行并保存 SAPLMA LR/MLP 基线。"""
    from src.methods.saplma import run_saplma_full

    print("\n" + "=" * 50)
    print("  P2.3-P2.5: SAPLMA 方法")
    print("=" * 50)

    results = run_saplma_full(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        layer_idx=-1,
        pooling="last",
        batch_size=8,
        max_length=128,
    )

    for clf_name, clf_result in results.items():
        summary = {
            "method": clf_result["method"],
            "layer_idx": clf_result["layer_idx"],
            "pooling": clf_result["pooling"],
            "num_seeds": clf_result["num_seeds"],
            "seeds": clf_result.get("seeds", []),
            "test_summary": clf_result["test_summary"],
            "runtime": runtime_info,
        }
        out_path = out_dir / f"saplma_{clf_name}_results.json"
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, default=float)

        print(f"SAPLMA ({clf_name}) 结果已保存至 {out_path}")
        test_summary = clf_result["test_summary"]
        print(f"  Accuracy:  {test_summary['accuracy']['mean']:.4f} ± {test_summary['accuracy']['std']:.4f}")
        print(f"  Macro-F1:  {test_summary['macro_f1']['mean']:.4f} ± {test_summary['macro_f1']['std']:.4f}")
        print(f"  AUROC:     {test_summary['auroc']['mean']:.4f} ± {test_summary['auroc']['std']:.4f}")


def main() -> None:
    """脚本主入口。"""
    args = parse_args()

    from src.config import config
    from src.utils.reproducibility import collect_runtime_info

    t_start = time.time()
    print("=" * 60)
    print("  Phase 2: 基础方法实现与评估")
    print("=" * 60)

    model, tokenizer, train_ds, val_ds, test_ds = load_model_and_data()
    runtime_info = collect_runtime_info(model)

    out_dir = config.paths.results_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("full", "ppl"):
        run_ppl(model, tokenizer, train_ds, val_ds, test_ds, out_dir, runtime_info)

    if args.mode in ("full", "saplma"):
        run_saplma(model, tokenizer, train_ds, val_ds, test_ds, out_dir, runtime_info)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Phase 2 完成! 总耗时: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()