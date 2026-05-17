"""
LLM Hallucination Probing — 主入口。

用法:
    python -s main.py                  # 显示项目状态
    python -s main.py preprocess       # 运行数据预处理
    python -s main.py test-gpu         # 测试 GPU 与模型加载
    python -s main.py phase2           # 运行 Phase 2 全部实验
    python -s main.py phase2-ppl       # 仅运行 PPL 方法
    python -s main.py phase2-saplma    # 仅运行 SAPLMA 方法

注意: 运行前必须依次激活环境:
    conda activate llm_hallucination
    .\.venv\Scripts\activate.ps1
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
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM: {mem_total:.1f} GB")
    except ImportError:
        print("PyTorch: NOT INSTALLED")

    # Transformers
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: NOT INSTALLED")

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

def _load_model_eager():
    """加载模型（eager attention，确保 output_attentions 可用）。"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.config import config
    from src.models.loader import print_device_info, load_model_fp16

    print_device_info()

    print(f"\n加载模型 (Qwen2-1.5B {config.models.primary_dtype})...")
    model, tokenizer = load_model_fp16(
        model_path=str(config.paths.models_cache / "Qwen2-1.5B"),
        device_map=config.models.primary_device_map,
        torch_dtype=config.models.primary_dtype,
    )
    # 加载后设置 eager attention（避免 from_pretrained 时传参导致 NaN）
    model.set_attn_implementation("eager")
    print(f"模型设备: {next(model.parameters()).device} (eager attention)")

    print("\n加载预处理数据...")
    from src.data.preprocessing import load_processed_data
    train_ds, val_ds, test_ds = load_processed_data()
    print(train_ds.summary())

    return model, tokenizer, train_ds, val_ds, test_ds


def _get_phase4_output_dir():
    """获取 Phase 4 输出目录。"""
    from src.config import config
    out_dir = config.paths.results_dir / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    return out_dir


def phase4_cache_hidden() -> None:
    """P4.0: 缓存 hidden features。"""
    model, tokenizer, train_ds, val_ds, test_ds = _load_model_eager()
    out_dir = _get_phase4_output_dir()

    from src.methods.phase4_attention import cache_phase3_hidden_features

    paths = cache_phase3_hidden_features(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        output_dir=out_dir / "cache",
        layer_idx=17,
        pooling="last",
        batch_size=8,
        max_length=128,
    )
    print(f"\nHidden features 缓存完成: {paths}")


def phase4_hidden_baseline() -> None:
    """P4.0: 运行 hidden-only baseline。"""
    import json
    model, tokenizer, train_ds, val_ds, test_ds = _load_model_eager()
    out_dir = _get_phase4_output_dir()

    from src.methods.phase4_attention import run_hidden_baseline
    from src.features.hidden_states import extract_hidden_states_dataset
    from src.utils.reproducibility import collect_runtime_info

    print("\n提取 hidden features (layer=17, pooling=last)...")
    X_train, y_train = extract_hidden_states_dataset(
        model, tokenizer, train_ds, pooling="last", layers=[17], batch_size=8, max_length=128,
    )
    X_val, y_val = extract_hidden_states_dataset(
        model, tokenizer, val_ds, pooling="last", layers=[17], batch_size=8, max_length=128,
    )
    X_test, y_test = extract_hidden_states_dataset(
        model, tokenizer, test_ds, pooling="last", layers=[17], batch_size=8, max_length=128,
    )

    result = run_hidden_baseline(
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        classifier_type="logistic",
        seeds=(42, 123, 2024),
        hidden_dim=X_train.shape[1],
    )
    result["runtime"] = collect_runtime_info(model)

    with open(out_dir / "hidden_baseline.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nHidden baseline 结果已保存至 {out_dir / 'hidden_baseline.json'}")
    ts = result["test"]
    print(f"  Test Accuracy:  {ts['accuracy']['mean']:.4f} ± {ts['accuracy']['std']:.4f}")
    print(f"  Test Macro-F1:  {ts['macro_f1']['mean']:.4f} ± {ts['macro_f1']['std']:.4f}")
    print(f"  Test AUROC:     {ts['auroc']['mean']:.4f} ± {ts['auroc']['std']:.4f}")


def phase4_extract_attention_scores() -> None:
    """P4.2: 提取 attention score 特征。"""
    import json
    model, tokenizer, train_ds, val_ds, test_ds = _load_model_eager()
    out_dir = _get_phase4_output_dir()

    from src.features.attention_scores import (
        extract_attention_score_features_dataset,
        extract_length_metadata,
    )
    from src.utils.reproducibility import collect_runtime_info

    layers = [13, 14, 15, 16, 17, 18, 19, 20]
    print(f"提取 attention score 特征 (layers={layers})...")

    for split_name, dataset in [
        ("train", train_ds),
        ("val", val_ds),
        ("test", test_ds),
    ]:
        out_path = out_dir / "cache" / f"attention_scores_{split_name}.npz"
        print(f"  处理 {split_name}...")
        result = extract_attention_score_features_dataset(
            model, tokenizer, dataset,
            layers=layers, batch_size=1,
            output_path=str(out_path),
        )
        print(f"    特征维度: {result['features'].shape}")

    print(f"\nAttention score 特征已保存至 {out_dir / 'cache'}")


def phase4_extract_attention_outputs() -> None:
    """P4.5: 提取 attention output 特征。"""
    import json
    model, tokenizer, train_ds, val_ds, test_ds = _load_model_eager()
    out_dir = _get_phase4_output_dir()

    from src.features.attention_outputs import extract_attention_output_features_dataset
    from src.utils.reproducibility import collect_runtime_info

    layers = [13, 14, 15, 16, 17, 18, 19, 20]
    print(f"提取 attention output 特征 (layers={layers})...")

    for split_name, dataset in [
        ("train", train_ds),
        ("val", val_ds),
        ("test", test_ds),
    ]:
        out_path = out_dir / "cache" / f"attention_outputs_{split_name}.npz"
        print(f"  处理 {split_name}...")
        result = extract_attention_output_features_dataset(
            model, tokenizer, dataset,
            layers=layers, batch_size=1,
            output_path=str(out_path),
        )
        print(f"    特征维度: {result['features'].shape}")

    print(f"\nAttention output 特征已保存至 {out_dir / 'cache'}")


def phase4_select_heads() -> None:
    """P4.4: validation-based head selection。"""
    import json
    out_dir = _get_phase4_output_dir()

    from src.utils.feature_cache import load_npz_cache
    from src.methods.phase4_attention import select_top_heads
    from src.features.attention_scores import extract_length_metadata

    # 加载 attention score 特征
    print("加载 attention score 特征...")
    train_data = load_npz_cache(out_dir / "cache" / "attention_scores_train.npz")
    val_data = load_npz_cache(out_dir / "cache" / "attention_scores_val.npz")

    train_X = train_data["features"]
    val_X = val_data["features"]
    train_y = train_data["labels"]
    val_y = val_data["labels"]
    feature_names = train_data["feature_names"]

    print(f"  训练特征: {train_X.shape}, 验证特征: {val_X.shape}")

    # Head selection
    for top_k in [8, 16, 32]:
        print(f"\nSelecting top {top_k} heads...")
        result = select_top_heads(
            train_X, val_X, train_y, val_y,
            feature_names, top_k_heads=top_k,
            metric="auroc",
        )

        sel_path = out_dir / f"attention_head_selection_top{top_k}.json"
        # 清理不可序列化的字段
        save_result = {
            "selection_metric": result["selection_metric"],
            "top_k_heads": result["top_k_heads"],
            "selected_heads": result["selected_heads"],
            "all_head_scores": [
                {k: v for k, v in h.items() if k != "feature_indices"}
                for h in result["all_head_scores"]
            ],
        }
        with open(sel_path, "w", encoding="utf-8") as f:
            json.dump(save_result, f, indent=2, ensure_ascii=False, default=str)
        print(f"  结果已保存至 {sel_path}")


def phase4_ablation() -> None:
    """P4.6: 运行完整消融实验。"""
    import json
    out_dir = _get_phase4_output_dir()

    from src.utils.feature_cache import load_npz_cache
    from src.methods.phase4_attention import (
        run_phase4_ablation,
        train_eval_classifier,
        summarize_feature_differences,
    )
    from src.features.hidden_states import extract_hidden_states_dataset
    from src.utils.reproducibility import collect_runtime_info

    # 加载或计算 hidden features（从缓存）
    model, tokenizer, train_ds, val_ds, test_ds = _load_model_eager()

    print("\n提取 hidden features (layer=17, pooling=last)...")
    X_h_train, y_train = extract_hidden_states_dataset(
        model, tokenizer, train_ds, pooling="last", layers=[17], batch_size=8, max_length=128,
    )
    X_h_val, y_val = extract_hidden_states_dataset(
        model, tokenizer, val_ds, pooling="last", layers=[17], batch_size=8, max_length=128,
    )
    X_h_test, y_test = extract_hidden_states_dataset(
        model, tokenizer, test_ds, pooling="last", layers=[17], batch_size=8, max_length=128,
    )

    # 加载 attention score 特征
    print("\n加载 attention score 特征...")
    attn_score_train = None
    attn_score_val = None
    attn_score_test = None
    attn_score_names = None

    score_train_path = out_dir / "cache" / "attention_scores_train.npz"
    if score_train_path.exists():
        train_as = load_npz_cache(score_train_path)
        val_as = load_npz_cache(out_dir / "cache" / "attention_scores_val.npz")
        test_as = load_npz_cache(out_dir / "cache" / "attention_scores_test.npz")

        attn_score_train = train_as["features"]
        attn_score_val = val_as["features"]
        attn_score_test = test_as["features"]
        attn_score_names = train_as["feature_names"]
        print(f"  Attention score 特征维度: {attn_score_train.shape}")
    else:
        print("  ⚠ Attention score 特征未找到，将跳过相关实验。请先运行 phase4-extract-attention-scores")

    # 加载 attention output 特征
    print("\n加载 attention output 特征...")
    attn_output_train = None
    attn_output_val = None
    attn_output_test = None

    output_train_path = out_dir / "cache" / "attention_outputs_train.npz"
    if output_train_path.exists():
        train_ao = load_npz_cache(output_train_path)
        val_ao = load_npz_cache(out_dir / "cache" / "attention_outputs_val.npz")
        test_ao = load_npz_cache(out_dir / "cache" / "attention_outputs_test.npz")

        attn_output_train = train_ao["features"]
        attn_output_val = val_ao["features"]
        attn_output_test = test_ao["features"]
        print(f"  Attention output 特征维度: {attn_output_train.shape}")
    else:
        print("  ⚠ Attention output 特征未找到，将跳过相关实验。请先运行 phase4-extract-attention-outputs")

    # 加载 head selection
    print("\n加载 head selection...")
    top_head_indices = None
    head_selection_path = out_dir / "attention_head_selection_top16.json"
    if head_selection_path.exists():
        with open(head_selection_path, "r", encoding="utf-8") as f:
            hs_data = json.load(f)
        # 重建 feature_indices（从 feature_names 反查）
        if attn_score_names:
            selected_names = set()
            for h in hs_data.get("selected_heads", []):
                layer = h["layer"]
                head = h["head"]
                for i, name in enumerate(attn_score_names):
                    if f"L{layer}_H{head:02d}_" in name:
                        selected_names.add(i)
            top_head_indices = sorted(selected_names)
            print(f"  Selected {len(top_head_indices)} features from {len(hs_data.get('selected_heads', []))} heads")
    else:
        print("  ⚠ Head selection 未找到，将使用空列表。请先运行 phase4-select-heads")

    if top_head_indices is None:
        top_head_indices = []

    # 运行消融
    print("\n" + "=" * 60)
    print("  运行 Phase 4 消融实验")
    print("=" * 60)

    ablation_results = run_phase4_ablation(
        hidden_train=X_h_train,
        hidden_val=X_h_val,
        hidden_test=X_h_test,
        attn_score_train=attn_score_train,
        attn_score_val=attn_score_val,
        attn_score_test=attn_score_test,
        attn_output_train=attn_output_train,
        attn_output_val=attn_output_val,
        attn_output_test=attn_output_test,
        top_head_indices=top_head_indices if top_head_indices else None,
        train_labels=y_train,
        val_labels=y_val,
        test_labels=y_test,
        train_statements=train_ds.statements,
        test_statements=test_ds.statements,
        classifier_type="logistic",
        seeds=(42, 123, 2024),
        output_dir=out_dir,
    )

    # 特征差异分析
    if attn_score_train is not None and attn_score_names is not None:
        print("\n生成 attention score 特征差异分析...")
        summarize_feature_differences(
            attn_score_train, y_train, attn_score_names,
            output_csv=str(out_dir / "attention_score_feature_summary.csv"),
        )

    if attn_output_train is not None:
        ao_data = load_npz_cache(out_dir / "cache" / "attention_outputs_train.npz")
        ao_names = ao_data.get("feature_names", [])
        if ao_names:
            print("生成 attention output 特征差异分析...")
            summarize_feature_differences(
                attn_output_train, y_train, ao_names,
                output_csv=str(out_dir / "attention_output_feature_summary.csv"),
            )

    # 生成 Phase 4 总结
    print("\n生成 Phase 4 总结...")
    from src.methods.phase4_attention import write_phase4_summary
    runtime_info = collect_runtime_info(model)

    # 确保 hs_data 始终有定义
    head_selection_data = None
    head_selection_path = out_dir / "attention_head_selection_top16.json"
    if head_selection_path.exists():
        import json as _json
        with open(head_selection_path, "r", encoding="utf-8") as _f:
            head_selection_data = _json.load(_f)

    write_phase4_summary(
        output_dir=out_dir,
        hidden_baseline=None,  # 可选
        head_selection=head_selection_data,
        ablation_results=ablation_results,
        runtime_info=runtime_info,
    )

    print(f"\nPhase 4 消融实验完成！结果保存在 {out_dir}")


def phase4_visualize() -> None:
    """P4.7: 生成所有 Phase 4 图表。"""
    import json
    out_dir = _get_phase4_output_dir()

    from src.analysis.phase4_analysis import generate_phase4_figures, plot_layer_head_auroc_heatmap

    # 收集 head scores
    head_scores = None
    hs_path = out_dir / "attention_head_selection_top16.json"
    if hs_path.exists():
        with open(hs_path, "r", encoding="utf-8") as f:
            hs_data = json.load(f)
        head_scores = hs_data.get("all_head_scores", [])

    # 收集方法指标
    method_metrics = {}
    results_csv = out_dir / "phase4_main_results.csv"
    if results_csv.exists():
        import csv
        with open(results_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("method", "unknown")
                method_metrics[name] = {
                    "accuracy": float(row.get("test_accuracy_mean", 0)),
                    "macro_f1": float(row.get("test_macro_f1_mean", 0)),
                    "auroc": float(row.get("test_auroc_mean", 0)),
                }

    # 收集 correction matrix
    correction_matrix = None
    ablation_path = out_dir / "phase4_ablation_results.json"
    if ablation_path.exists():
        with open(ablation_path, "r", encoding="utf-8") as f:
            ablation_data = json.load(f)
        correction_matrix = ablation_data.get("correction_matrix")

    saved = generate_phase4_figures(
        head_scores=head_scores,
        feature_summary=None,
        method_metrics=method_metrics if method_metrics else None,
        correction_matrix=correction_matrix,
        output_dir=out_dir / "figures",
    )
    print(f"已生成 {len(saved)} 张图表:")
    for name, path in saved.items():
        print(f"  {name}: {path}")


def phase4() -> None:
    """Phase 4: 一键运行完整流程。"""
    import time
    t_start = time.time()

    print("=" * 60)
    print("  Phase 4: Attention-Guided SAPLMA")
    print("=" * 60)

    # Step 1: Cache hidden features
    print("\n[P4.0] 缓存 hidden features...")
    phase4_cache_hidden()

    # Step 2: Hidden baseline
    print("\n[P4.0] 运行 hidden baseline...")
    phase4_hidden_baseline()

    # Step 3: Extract attention scores
    print("\n[P4.2] 提取 attention score 特征...")
    phase4_extract_attention_scores()

    # Step 4: Extract attention outputs
    print("\n[P4.5] 提取 attention output 特征...")
    phase4_extract_attention_outputs()

    # Step 5: Head selection
    print("\n[P4.4] 运行 head selection...")
    phase4_select_heads()

    # Step 6: Ablation
    print("\n[P4.6] 运行消融实验...")
    phase4_ablation()

    # Step 7: Visualize
    print("\n[P4.7] 生成可视化图表...")
    phase4_visualize()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Phase 4 全部完成! 总耗时: {elapsed:.0f}s ({elapsed/60:.1f} min)")
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
            "phase4", "phase4-cache-hidden", "phase4-hidden-baseline",
            "phase4-extract-attention-scores", "phase4-select-heads",
            "phase4-extract-attention-outputs", "phase4-ablation",
            "phase4-visualize",
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
    elif args.command == "phase4-cache-hidden":
        phase4_cache_hidden()
    elif args.command == "phase4-hidden-baseline":
        phase4_hidden_baseline()
    elif args.command == "phase4-extract-attention-scores":
        phase4_extract_attention_scores()
    elif args.command == "phase4-select-heads":
        phase4_select_heads()
    elif args.command == "phase4-extract-attention-outputs":
        phase4_extract_attention_outputs()
    elif args.command == "phase4-ablation":
        phase4_ablation()
    elif args.command == "phase4-visualize":
        phase4_visualize()
