"""
实验结果汇总脚本：统一读取 Phase 2、Phase 3、Phase 4 的现有结果文件并打印摘要。

用法:
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llm_hallucination
    source ./.venv/bin/activate
    python -s scripts/show_results.py
    python -s scripts/show_results.py --section phase4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="打印各阶段已有实验结果摘要。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--section",
        choices=("all", "phase2", "phase3", "phase4"),
        default="all",
        help="选择要打印的结果范围",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    """读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def print_phase2() -> None:
    """打印 Phase 2 结果。"""
    ppl = load_json(RESULTS_DIR / "baseline" / "ppl_results.json")
    sap_lr = load_json(RESULTS_DIR / "baseline" / "saplma_logistic_results.json")
    sap_mlp = load_json(RESULTS_DIR / "baseline" / "saplma_mlp_results_rerun_best.json")

    print("=== Phase 2 ===")
    print(f"PPL:        Acc={ppl['test']['accuracy']:.4f} F1={ppl['test']['macro_f1']:.4f} AUROC={ppl['test']['auroc']:.4f}")
    ts = sap_lr["test_summary"]
    print(f"SAPLMA LR:  Acc={ts['accuracy']['mean']:.4f}+/-{ts['accuracy']['std']:.4f} F1={ts['macro_f1']['mean']:.4f}+/-{ts['macro_f1']['std']:.4f} AUROC={ts['auroc']['mean']:.4f}+/-{ts['auroc']['std']:.4f}")
    ts = sap_mlp["test_summary"]
    print(f"SAPLMA MLP: Acc={ts['accuracy']['mean']:.4f}+/-{ts['accuracy']['std']:.4f} F1={ts['macro_f1']['mean']:.4f}+/-{ts['macro_f1']['std']:.4f} AUROC={ts['auroc']['mean']:.4f}+/-{ts['auroc']['std']:.4f}")
    print()


def print_phase3() -> None:
    """打印 Phase 3 结果。"""
    layer_analysis = load_json(RESULTS_DIR / "analysis" / "layer_analysis_logistic_last.json")
    token_analysis = load_json(RESULTS_DIR / "analysis" / "token_analysis_logistic_last_layer.json")

    print("=== Phase 3 ===")
    best_layer = layer_analysis["best_layer"]
    ts = best_layer["test_summary"]
    print(f"Best layer L{best_layer['layer_idx']}: Acc={ts['accuracy']['mean']:.4f}+/-{ts['accuracy']['std']:.4f} F1={ts['macro_f1']['mean']:.4f}+/-{ts['macro_f1']['std']:.4f} AUROC={ts['auroc']['mean']:.4f}+/-{ts['auroc']['std']:.4f}")
    best_pooling = token_analysis["best_pooling"]
    ts = best_pooling["test_summary"]
    print(f"Best pooling {best_pooling['pooling']}: Acc={ts['accuracy']['mean']:.4f}+/-{ts['accuracy']['std']:.4f} F1={ts['macro_f1']['mean']:.4f}+/-{ts['macro_f1']['std']:.4f} AUROC={ts['auroc']['mean']:.4f}+/-{ts['auroc']['std']:.4f}")
    print()


def print_phase4() -> None:
    """打印 Phase 4 结果。"""
    payload = load_json(RESULTS_DIR / "phase4" / "phase4_ablation_results.json")
    results = payload.get("ablation", payload if isinstance(payload, list) else [])

    print("=== Phase 4 ===")
    print(f"{'ID':<6} {'Method':<42} {'Test Acc':<16} {'Test F1':<16} {'Test AUROC':<14}")
    print("-" * 100)
    for result in results:
        print(
            f"{result['id']:<6} {result['method']:<42} "
            f"{result['test_accuracy_mean']:.4f}+/-{result['test_accuracy_std']:.4f}   "
            f"{result['test_macro_f1_mean']:.4f}+/-{result['test_macro_f1_std']:.4f}   "
            f"{result['test_auroc_mean']:.4f}+/-{result['test_auroc_std']:.4f}"
        )

    correction_matrix = payload.get("correction_matrix", {}) if isinstance(payload, dict) else {}
    if correction_matrix:
        print("-" * 100)
        print(
            f"Correction Matrix: H+F+={correction_matrix.get('n00', 0)} | "
            f"H+F-={correction_matrix.get('n01', 0)} | "
            f"H-F+={correction_matrix.get('n10', 0)} | H-F-={correction_matrix.get('n11', 0)}"
        )
        print(f"Net correction: {correction_matrix.get('n10', 0) - correction_matrix.get('n01', 0):+d} samples")
    print()


def main() -> None:
    """脚本主入口。"""
    args = parse_args()
    if args.section in ("all", "phase2"):
        print_phase2()
    if args.section in ("all", "phase3"):
        print_phase3()
    if args.section in ("all", "phase4"):
        print_phase4()


if __name__ == "__main__":
    main()