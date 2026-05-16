"""
LLM Hallucination Probing — 主入口。

用法:
    python -s main.py                  # 显示项目状态
    python -s main.py preprocess       # 运行数据预处理
    python -s main.py test-gpu         # 测试 GPU 与模型加载
    python -s main.py phase2           # 运行 Phase 2 全部实验
    python -s main.py phase2-ppl       # 仅运行 PPL 方法
    python -s main.py phase2-saplma    # 仅运行 SAPLMA 方法
    python -s main.py phase3           # 运行 Phase 3 层分析 + token 分析
    python -s main.py phase4           # 运行 Phase 4 注意力方法
    python -s main.py phase4-attention # 仅运行 Phase 4 注意力方法
    python -s main.py phase4-attention-mlp # 运行 Phase 4 MLP 融合版本
    python -s main.py phase4-attention-stacking # 运行 Phase 4 第三轮 stacking 融合版本

注意: 运行前必须依次激活环境:
    conda activate llm_hallucination
    .\\.venv\\Scripts\\activate.ps1
"""

from __future__ import annotations

import sys
from pathlib import Path


def status() -> None:
    """打印项目状态：环境、数据、模型。"""
    print("=" * 60)
    print("  LLM Hallucination Probing — Phase 1 状态检查")
    print("=" * 60)

    # Python
    print(f"\nPython: {sys.version}")

    # PyTorch / CUDA
    try:
        import torch as torch_mod
    except ImportError:
        print("PyTorch: NOT INSTALLED")
    else:
        print(f"PyTorch: {torch_mod.__version__}")
        print(f"CUDA available: {torch_mod.cuda.is_available()}")
        if torch_mod.cuda.is_available():
            print(f"GPU: {torch_mod.cuda.get_device_name(0)}")
            mem_total = torch_mod.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM: {mem_total:.1f} GB")

    # Transformers
    try:
        import transformers as transformers_mod
    except ImportError:
        print("Transformers: NOT INSTALLED")
    else:
        print(f"Transformers: {transformers_mod.__version__}")

    # 数据
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        pt_files = list(processed_dir.glob("*.pt"))
        print(f"\n预处理数据 ({len(pt_files)} 文件):")
        for f in sorted(pt_files):
            size_mb = f.stat().st_size / 1024**2
            print(f"  {f.name} ({size_mb:.1f} MB)")
    else:
        print("\n预处理数据: 尚未生成 (运行 preprocess)")

    # 模型缓存
    cache_dir = Path("models_cache")
    if cache_dir.exists():
        models = [d.name for d in cache_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        print(f"\n模型缓存: {models}")

    print("\n" + "=" * 60)
    print("  Phase 1 (P1.1-P1.8) 已完成 ✅")
    print("=" * 60)


def preprocess() -> None:
    """运行数据预处理流水线。"""
    from src.data.preprocessing import run_preprocessing
    run_preprocessing()


def test_gpu() -> None:
    """快速测试 GPU 与模型加载。"""
    from src.config import config
    from src.models.loader import print_device_info, load_model
    import torch

    print_device_info()

    print(f"\n加载模型 (Qwen2-1.5B {config.models.primary_dtype})...")
    model, tokenizer = load_model()
    print(f"模型设备: {next(model.parameters()).device}")

    # 简单前向传播测试
    test_input = "The sky is blue."
    inputs = tokenizer(test_input, return_tensors="pt")
    # 移至模型设备
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states
    num_layers = len(hidden) - 1  # 减去 embedding output
    print(f"隐藏状态层数: {num_layers} (不含 embedding)")
    print(f"隐藏维度: {hidden[-1].shape[-1]}")
    print(f"序列长度: {hidden[-1].shape[1]}")
    print("✅ 模型前向传播测试通过!")


# ===========================================================================
# Phase 2 实验
# ===========================================================================

def _load_model_and_data():
    """加载模型和数据集的通用辅助函数。"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.config import config
    from src.models.loader import load_model, print_device_info
    from src.data.preprocessing import load_processed_data

    print_device_info()

    print(f"\n加载模型 (Qwen2-1.5B {config.models.primary_dtype})...")
    model, tokenizer = load_model()
    print(f"模型设备: {next(model.parameters()).device}")

    print("\n加载预处理数据...")
    train_ds, val_ds, test_ds = load_processed_data()
    print(train_ds.summary())

    return model, tokenizer, train_ds, val_ds, test_ds


def phase2_ppl() -> None:
    """Phase 2: 运行 PPL 基线方法评估。"""
    model, tokenizer, train_ds, val_ds, test_ds = _load_model_and_data()

    from src.methods.probability import evaluate_ppl_method
    from src.utils.reproducibility import collect_runtime_info
    import json

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

    # 保存结果
    from src.config import config
    out_dir = config.paths.results_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存指标摘要
    summary = {
        "method": results["method"],
        "threshold": results["threshold"],
        "threshold_metric": results["threshold_metric"],
        "train": results["train"],
        "val": results["val"],
        "test": results["test"],
        "runtime": collect_runtime_info(model),
    }
    with open(out_dir / "ppl_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    print(f"\nPPL 结果已保存至 {out_dir / 'ppl_results.json'}")
    print(f"测试集: Acc={results['test']['accuracy']:.4f}, "
          f"Macro-F1={results['test']['macro_f1']:.4f}, "
          f"AUROC={results['test']['auroc']:.4f}")


def phase2_saplma() -> None:
    """Phase 2: 运行 SAPLMA 方法评估（LR + MLP）。"""
    model, tokenizer, train_ds, val_ds, test_ds = _load_model_and_data()

    from src.methods.saplma import run_saplma_full
    from src.utils.reproducibility import collect_runtime_info
    import json

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

    from src.config import config
    out_dir = config.paths.results_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_info = collect_runtime_info(model)

    # 保存摘要
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
        fname = f"saplma_{clf_name}_results.json"
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
        print(f"\nSAPLMA ({clf_name}) 结果已保存至 {out_dir / fname}")
        ts = clf_result["test_summary"]
        print(f"  Accuracy:  {ts['accuracy']['mean']:.4f} ± {ts['accuracy']['std']:.4f}")
        print(f"  Macro-F1:  {ts['macro_f1']['mean']:.4f} ± {ts['macro_f1']['std']:.4f}")
        print(f"  AUROC:     {ts['auroc']['mean']:.4f} ± {ts['auroc']['std']:.4f}")


def phase2() -> None:
    """Phase 2: 运行全部基线实验（PPL + SAPLMA LR + SAPLMA MLP）。"""
    import time
    import json
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    t_start = time.time()

    print("=" * 60)
    print("  Phase 2: 基础方法实现与评估")
    print("=" * 60)

    # 一次性加载模型和数据
    from src.models.loader import load_model, print_device_info
    from src.data.preprocessing import load_processed_data
    from src.config import config
    from src.utils.reproducibility import collect_runtime_info

    print_device_info()

    print(f"\n加载模型 (Qwen2-1.5B {config.models.primary_dtype})...")
    model, tokenizer = load_model()
    print(f"模型设备: {next(model.parameters()).device}")

    print("\n加载预处理数据...")
    train_ds, val_ds, test_ds = load_processed_data()
    print(train_ds.summary())

    out_dir = config.paths.results_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_info = collect_runtime_info(model)

    # ---- P2.1-P2.2: PPL 方法 ------------------------------------------------
    print("\n" + "=" * 50)
    print("  P2.1-P2.2: PPL 方法")
    print("=" * 50)

    from src.methods.probability import evaluate_ppl_method
    import numpy as np

    ppl_results = evaluate_ppl_method(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
        batch_size=8, max_length=128, threshold_metric="f1",
    )
    ppl_summary = {k: v for k, v in ppl_results.items()
                   if k in ("method", "threshold", "threshold_metric", "train", "val", "test")}
    ppl_summary["runtime"] = runtime_info
    with open(out_dir / "ppl_results.json", "w", encoding="utf-8") as f:
        json.dump(ppl_summary, f, indent=2, ensure_ascii=False, default=float)
    print(f"PPL 结果已保存至 {out_dir / 'ppl_results.json'}")
    print(f"测试集: Acc={ppl_results['test']['accuracy']:.4f}, "
          f"Macro-F1={ppl_results['test']['macro_f1']:.4f}, "
          f"AUROC={ppl_results['test']['auroc']:.4f}")

    # ---- P2.3-P2.5: SAPLMA 方法 ---------------------------------------------
    print("\n" + "=" * 50)
    print("  P2.3-P2.5: SAPLMA 方法")
    print("=" * 50)

    from src.methods.saplma import run_saplma_full

    saplma_results = run_saplma_full(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
        layer_idx=-1, pooling="last", batch_size=8, max_length=128,
    )
    for clf_name, clf_result in saplma_results.items():
        summary = {
            "method": clf_result["method"],
            "layer_idx": clf_result["layer_idx"],
            "pooling": clf_result["pooling"],
            "num_seeds": clf_result["num_seeds"],
            "seeds": clf_result.get("seeds", []),
            "test_summary": clf_result["test_summary"],
            "runtime": runtime_info,
        }
        fname = f"saplma_{clf_name}_results.json"
        with open(out_dir / fname, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
        print(f"SAPLMA ({clf_name}) 结果已保存至 {out_dir / fname}")
        ts = clf_result["test_summary"]
        print(f"  Accuracy:  {ts['accuracy']['mean']:.4f} ± {ts['accuracy']['std']:.4f}")
        print(f"  Macro-F1:  {ts['macro_f1']['mean']:.4f} ± {ts['macro_f1']['std']:.4f}")
        print(f"  AUROC:     {ts['auroc']['mean']:.4f} ± {ts['auroc']['std']:.4f}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Phase 2 全部完成! 总耗时: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")


# ===========================================================================
# Phase 3 实验
# ===========================================================================

def phase3_layer() -> None:
    """Phase 3: 逐层分析隐藏状态的检测性能。"""
    model, tokenizer, train_ds, val_ds, test_ds = _load_model_and_data()

    from src.analysis.layer_analysis import analyze_layer_performance
    from src.analysis.visualization import plot_layer_metric_curve
    from src.config import config
    from src.utils.reproducibility import collect_runtime_info
    import json

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

    out_dir = config.paths.results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        **results,
        "runtime": collect_runtime_info(model),
    }
    out_path = out_dir / "layer_analysis_logistic_last.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    plot_layer_metric_curve(
        results,
        split="test",
        metric="accuracy",
        save_path=out_dir / "layer_accuracy_curve.png",
    )

    print(f"\nLayer analysis 结果已保存至 {out_path}")
    print(f"最佳层: {results['best_layer']['layer_idx']}")
    print(
        f"测试集 Accuracy: {results['best_layer']['test_summary']['accuracy']['mean']:.4f} ± "
        f"{results['best_layer']['test_summary']['accuracy']['std']:.4f}"
    )


def phase3_token() -> None:
    """Phase 3: 分析不同 token pooling 的检测性能。"""
    model, tokenizer, train_ds, val_ds, test_ds = _load_model_and_data()

    from src.analysis.token_analysis import analyze_token_pooling
    from src.analysis.visualization import plot_token_metric_comparison
    from src.config import config
    from src.utils.reproducibility import collect_runtime_info
    import json

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

    out_dir = config.paths.results_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        **results,
        "runtime": collect_runtime_info(model),
    }
    out_path = out_dir / "token_analysis_logistic_last_layer.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    plot_token_metric_comparison(
        results,
        split="test",
        metric="accuracy",
        save_path=out_dir / "token_accuracy_comparison.png",
    )

    print(f"\nToken analysis 结果已保存至 {out_path}")
    print(f"最佳 pooling: {results['best_pooling']['pooling']}")
    print(
        f"测试集 Accuracy: {results['best_pooling']['test_summary']['accuracy']['mean']:.4f} ± "
        f"{results['best_pooling']['test_summary']['accuracy']['std']:.4f}"
    )


def phase3() -> None:
    """Phase 3: 运行层分析与 token pooling 分析。"""
    import time

    t_start = time.time()
    print("=" * 60)
    print("  Phase 3: 分析实验")
    print("=" * 60)

    phase3_layer()
    phase3_token()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Phase 3 全部完成! 总耗时: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")


# ===========================================================================
# Phase 4 实验
# ===========================================================================

def phase4_attention(
    classifier_type: str = "logistic",
    include_stacking_variant: bool = False,
    result_suffix: str = "improved",
) -> None:
    """Phase 4: 基于注意力模式的增强幻觉检测。"""
    model, tokenizer, train_ds, val_ds, test_ds = _load_model_and_data()

    from src.analysis.visualization import (
        plot_attention_feature_deltas,
        plot_attention_variant_comparison,
    )
    from src.config import config
    from src.methods.advanced import run_attention_ablation_study
    from src.utils.reproducibility import collect_runtime_info
    import json

    hidden_layer_idx = min(17, model.config.num_hidden_layers - 1)
    results = run_attention_ablation_study(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        classifier_type=classifier_type,
        hidden_layer_idx=hidden_layer_idx,
        hidden_pooling="last",
        batch_size=8,
        max_length=128,
        include_stacking_variant=include_stacking_variant,
    )

    out_dir = config.paths.results_dir / "advanced"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        **results,
        "runtime": collect_runtime_info(model),
    }
    result_stem = f"attention_ablation_{classifier_type}_layer{hidden_layer_idx}_last_{result_suffix}"
    out_path = out_dir / f"{result_stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    plot_attention_variant_comparison(
        results,
        split="test",
        metric="accuracy",
        save_path=out_dir / f"{result_stem}_accuracy.png",
    )
    plot_attention_feature_deltas(
        results["attention_feature_summary"]["test"],
        top_k=8,
        save_path=out_dir / f"{result_stem}_feature_deltas.png",
    )

    print(f"\nPhase 4 注意力方法结果已保存至 {out_path}")
    print(f"最佳变体: {results['best_variant']['name']}")
    print(
        f"测试集 Accuracy: {results['best_variant']['test_summary']['accuracy']['mean']:.4f} ± "
        f"{results['best_variant']['test_summary']['accuracy']['std']:.4f}"
    )


def phase4() -> None:
    """Phase 4: 运行当前已实现的注意力增强实验。"""
    import time

    t_start = time.time()
    print("=" * 60)
    print("  Phase 4: 基于注意力模式的幻觉检测")
    print("=" * 60)

    phase4_attention(classifier_type="logistic")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Phase 4 完成! 总耗时: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Hallucination Probing")
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=[
            "status", "preprocess", "test-gpu",
            "phase2", "phase2-ppl", "phase2-saplma",
            "phase3", "phase3-layer", "phase3-token",
                "phase4", "phase4-attention", "phase4-attention-mlp", "phase4-attention-stacking",
        ],
        help="要执行的命令 (默认: status)",
    )
    args = parser.parse_args()

    if args.command == "status":
        status()
    elif args.command == "preprocess":
        preprocess()
    elif args.command == "test-gpu":
        test_gpu()
    elif args.command == "phase2":
        phase2()
    elif args.command == "phase2-ppl":
        phase2_ppl()
    elif args.command == "phase2-saplma":
        phase2_saplma()
    elif args.command == "phase3":
        phase3()
    elif args.command == "phase3-layer":
        phase3_layer()
    elif args.command == "phase3-token":
        phase3_token()
    elif args.command == "phase4":
        phase4()
    elif args.command == "phase4-attention":
        phase4_attention(classifier_type="logistic")
    elif args.command == "phase4-attention-mlp":
        phase4_attention(classifier_type="mlp")
    elif args.command == "phase4-attention-stacking":
        phase4_attention(classifier_type="mlp", include_stacking_variant=True, result_suffix="stacking_v3")
