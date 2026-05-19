"""
Phase 4 主运行脚本：在稳定的 bfloat16 + eager attention 路径上执行完整 A0-A9 评估，
也支持基于已有缓存重跑消融或直接打印当前结果。

用法:
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llm_hallucination
    source ./.venv/bin/activate
    PHASE4_DTYPE=bfloat16 python -s scripts/run_phase4_full.py
    python -s scripts/run_phase4_full.py --use-cache
    python -s scripts/run_phase4_full.py --summary-only
    python -s scripts/run_phase4_full.py --use-cache --cache-dir experiments/results/phase4/1

说明:
    - 需要逐阶段运行时，优先使用 main.py 中的 phase4-* 子命令。
    - --use-cache 会跳过模型前向，只从已有缓存重跑 A0-A9。
    - --summary-only 只读取 phase4_ablation_results.json 并打印汇总表。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase4_full")

from src.config import config
from src.data.dataset import TrueFalseDataset
from src.data.preprocessing import load_processed_data
from src.features.attention_outputs import extract_attention_output_features_dataset
from src.features.attention_scores import extract_attention_score_features_dataset
from src.features.hidden_states import extract_hidden_states_dataset
from src.methods.phase4_attention import (
    apply_gated_fusion,
    build_error_analysis,
    residualize_by_length,
    run_hidden_baseline,
    select_gated_fusion_tau,
    select_top_heads,
    summarize_feature_differences,
    train_eval_classifier,
)
from src.models.loader import load_model_fp16, print_device_info
from src.utils.feature_cache import load_npz_cache, save_npz_cache
from src.utils.reproducibility import collect_runtime_info, set_global_seed

SEEDS = (42, 123, 2024)
CANDIDATE_LAYERS = [13, 14, 15, 16, 17, 18, 19, 20]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="运行或复用缓存重跑 Phase 4 A0-A9 实验。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dtype",
        default=os.environ.get("PHASE4_DTYPE", config.models.primary_dtype),
        choices=("float16", "bfloat16", "float32", "auto"),
        help="完整运行时使用的模型精度",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "experiments" / "results" / "phase4"),
        help="结果输出目录",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="特征缓存目录，默认使用 output-dir/cache",
    )
    parser.add_argument("--subset-train", type=int, default=600, help="attention 特征 train 子集大小")
    parser.add_argument("--subset-val", type=int, default=150, help="attention 特征 val 子集大小")
    parser.add_argument("--subset-test", type=int, default=150, help="attention 特征 test 子集大小")
    parser.add_argument("--top-k-heads", type=int, default=16, help="validation head selection 的 top-k")
    parser.add_argument("--use-cache", action="store_true", help="跳过模型前向，直接读取缓存重跑 A0-A9")
    parser.add_argument("--summary-only", action="store_true", help="只打印现有 Phase 4 结果，不执行训练")
    args = parser.parse_args()

    if args.use_cache and args.summary_only:
        parser.error("--use-cache 与 --summary-only 不能同时使用")

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"
    args.output_dir = output_dir
    args.cache_dir = cache_dir
    return args


def subset_dataset(dataset, size: int) -> TrueFalseDataset:
    """按顺序截取数据子集。"""
    size = min(size, len(dataset))
    domains = dataset.domains[:size] if dataset.domains else None
    return TrueFalseDataset(dataset.statements[:size], dataset.labels[:size], domains)


def sanitize_features(name: str, features: np.ndarray) -> np.ndarray:
    """统一清理 NaN/Inf。"""
    array = np.asarray(features)
    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())
    if nan_count or inf_count:
        logger.warning("%s 含有 NaN=%d, Inf=%d，已替换为 0", name, nan_count, inf_count)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array


def compute_length_proxy(features: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """使用 attention sink 近似序列长度，供残差化使用。"""
    sink_cols = [idx for idx, name in enumerate(feature_names) if "attention_sink_mass" in name]
    if not sink_cols:
        return np.ones((features.shape[0], 6), dtype=np.float64)

    sink_proxy = features[:, sink_cols].mean(axis=1, keepdims=True)
    zeros = np.zeros((features.shape[0], 5), dtype=np.float64)
    return np.concatenate([sink_proxy, zeros], axis=1)


def load_json(path: Path) -> dict:
    """读取 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    """保存 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def print_results_table(results: list[dict]) -> None:
    """打印 Phase 4 结果表。"""
    if not results:
        print("未找到可打印的 Phase 4 结果。")
        return

    print("\n" + "=" * 100)
    print("                     Phase 4 消融实验结果 (A0-A9)")
    print("=" * 100)
    print(f"{'ID':<6} {'Method':<42} {'Dim':<8} {'Test Acc':<16} {'Test F1':<16} {'Test AUROC':<14}")
    print("-" * 100)
    for result in results:
        print(
            f"{result['id']:<6} {result['method']:<42} {str(result['feature_dim']):<8} "
            f"{result['test_accuracy_mean']:.4f} +/- {result['test_accuracy_std']:.4f}   "
            f"{result['test_macro_f1_mean']:.4f} +/- {result['test_macro_f1_std']:.4f}   "
            f"{result['test_auroc_mean']:.4f} +/- {result['test_auroc_std']:.4f}"
        )
    print("-" * 100)


def get_probs(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, seed: int = 42) -> np.ndarray:
    """训练二分类 LR 并返回评估集正类概率。"""
    set_global_seed(seed)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    return clf.predict_proba(X_eval_scaled)[:, 1]


def append_result_from_summary(
    results: list[dict],
    method_id: str,
    method_name: str,
    feature_dim: int | str,
    summary: dict,
    note: str = "",
) -> None:
    """将已有 summary 结果写入统一表。"""
    results.append({
        "id": method_id,
        "method": method_name,
        "feature_dim": feature_dim,
        "test_accuracy_mean": summary["accuracy"]["mean"],
        "test_accuracy_std": summary["accuracy"]["std"],
        "test_macro_f1_mean": summary["macro_f1"]["mean"],
        "test_macro_f1_std": summary["macro_f1"]["std"],
        "test_auroc_mean": summary["auroc"]["mean"],
        "test_auroc_std": summary["auroc"]["std"],
        "note": note,
    })


def run_and_record(
    results: list[dict],
    method_id: str,
    method_name: str,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    note: str = "",
) -> None:
    """训练单个 ablation 并记录结果。"""
    print(f"  [{method_id}] {method_name}...", end=" ", flush=True)
    result = train_eval_classifier(X_train, X_val, X_test, y_train, y_val, y_test, "logistic", SEEDS)
    summary = result["test_summary"]
    print(
        f"Acc={summary['accuracy']['mean']:.4f} "
        f"F1={summary['macro_f1']['mean']:.4f} "
        f"AUROC={summary['auroc']['mean']:.4f}"
    )
    append_result_from_summary(results, method_id, method_name, X_train.shape[1], summary, note)


def load_model_and_datasets(dtype_name: str):
    """加载模型、tokenizer 与处理后的数据。"""
    print("=" * 60)
    print("  Phase 4 Full Evaluation")
    print("=" * 60)
    print_device_info()

    t0 = time.time()
    model, tokenizer = load_model_fp16(
        model_path=str(config.paths.models_cache / "Qwen2-1.5B"),
        torch_dtype=dtype_name,
    )
    model.set_attn_implementation("eager")
    print(f"Model loaded (eager attention, dtype={dtype_name}) in {time.time() - t0:.0f}s")

    train_ds, val_ds, test_ds = load_processed_data()
    print(train_ds.summary())
    runtime_info = collect_runtime_info(model)
    return model, tokenizer, train_ds, val_ds, test_ds, runtime_info


def extract_feature_bundle(
    model,
    tokenizer,
    train_ds,
    val_ds,
    test_ds,
    cache_dir: Path,
    subset_sizes: dict[str, int],
) -> dict:
    """完整提取 hidden、attention score 和 attention output 特征。"""
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Step 1/3] Extracting full hidden features...")
    X_h_train, y_train_full = extract_hidden_states_dataset(
        model, tokenizer, train_ds, pooling="last", layers=[17], batch_size=8, max_length=128,
    )
    X_h_val, y_val_full = extract_hidden_states_dataset(
        model, tokenizer, val_ds, pooling="last", layers=[17], batch_size=8, max_length=128,
    )
    X_h_test, y_test_full = extract_hidden_states_dataset(
        model, tokenizer, test_ds, pooling="last", layers=[17], batch_size=8, max_length=128,
    )

    X_h_train = sanitize_features("hidden_train", X_h_train)
    X_h_val = sanitize_features("hidden_val", X_h_val)
    X_h_test = sanitize_features("hidden_test", X_h_test)

    save_npz_cache(cache_dir / "hidden_layer17_last_train.npz", X_h_train, y_train_full)
    save_npz_cache(cache_dir / "hidden_layer17_last_val.npz", X_h_val, y_val_full)
    save_npz_cache(cache_dir / "hidden_layer17_last_test.npz", X_h_test, y_test_full)
    print(f"  Hidden features cached to: {cache_dir}")

    print(f"\n[Step 2/3] Extracting attention features on subset {subset_sizes}...")
    train_sub = subset_dataset(train_ds, subset_sizes["train"])
    val_sub = subset_dataset(val_ds, subset_sizes["val"])
    test_sub = subset_dataset(test_ds, subset_sizes["test"])
    print(f"  Subset sizes: train={len(train_sub)}, val={len(val_sub)}, test={len(test_sub)}")

    score_train = extract_attention_score_features_dataset(
        model, tokenizer, train_sub, layers=CANDIDATE_LAYERS, batch_size=1,
        output_path=str(cache_dir / "attention_scores_train.npz"),
    )
    score_val = extract_attention_score_features_dataset(
        model, tokenizer, val_sub, layers=CANDIDATE_LAYERS, batch_size=1,
        output_path=str(cache_dir / "attention_scores_val.npz"),
    )
    score_test = extract_attention_score_features_dataset(
        model, tokenizer, test_sub, layers=CANDIDATE_LAYERS, batch_size=1,
        output_path=str(cache_dir / "attention_scores_test.npz"),
    )

    output_train = extract_attention_output_features_dataset(
        model, tokenizer, train_sub, layers=CANDIDATE_LAYERS, batch_size=1,
        output_path=str(cache_dir / "attention_outputs_train.npz"),
    )
    output_val = extract_attention_output_features_dataset(
        model, tokenizer, val_sub, layers=CANDIDATE_LAYERS, batch_size=1,
        output_path=str(cache_dir / "attention_outputs_val.npz"),
    )
    output_test = extract_attention_output_features_dataset(
        model, tokenizer, test_sub, layers=CANDIDATE_LAYERS, batch_size=1,
        output_path=str(cache_dir / "attention_outputs_test.npz"),
    )

    X_as_train = sanitize_features("attention_scores_train", score_train["features"])
    X_as_val = sanitize_features("attention_scores_val", score_val["features"])
    X_as_test = sanitize_features("attention_scores_test", score_test["features"])
    X_ao_train = sanitize_features("attention_outputs_train", output_train["features"])
    X_ao_val = sanitize_features("attention_outputs_val", output_val["features"])
    X_ao_test = sanitize_features("attention_outputs_test", output_test["features"])

    print(f"  Attention scores: train={X_as_train.shape}, val={X_as_val.shape}, test={X_as_test.shape}")
    print(f"  Attention outputs: train={X_ao_train.shape}, val={X_ao_val.shape}, test={X_ao_test.shape}")

    return {
        "cache_dir": cache_dir,
        "train_sub": train_sub,
        "val_sub": val_sub,
        "test_sub": test_sub,
        "X_h_train": X_h_train,
        "X_h_val": X_h_val,
        "X_h_test": X_h_test,
        "y_train_full": y_train_full,
        "y_val_full": y_val_full,
        "y_test_full": y_test_full,
        "X_h_sub_train": X_h_train[:len(train_sub)],
        "X_h_sub_val": X_h_val[:len(val_sub)],
        "X_h_sub_test": X_h_test[:len(test_sub)],
        "X_as_train": X_as_train,
        "X_as_val": X_as_val,
        "X_as_test": X_as_test,
        "X_ao_train": X_ao_train,
        "X_ao_val": X_ao_val,
        "X_ao_test": X_ao_test,
        "as_names": list(score_train["feature_names"]),
        "ao_names": list(output_train["feature_names"]),
        "y_sub_train": score_train["labels"],
        "y_sub_val": score_val["labels"],
        "y_sub_test": score_test["labels"],
    }


def load_cached_bundle(cache_dir: Path, train_ds, val_ds, test_ds) -> dict:
    """从缓存重建完整的 Phase 4 特征包。"""
    required_paths = [
        cache_dir / "hidden_layer17_last_train.npz",
        cache_dir / "hidden_layer17_last_val.npz",
        cache_dir / "hidden_layer17_last_test.npz",
        cache_dir / "attention_scores_train.npz",
        cache_dir / "attention_scores_val.npz",
        cache_dir / "attention_scores_test.npz",
        cache_dir / "attention_outputs_train.npz",
        cache_dir / "attention_outputs_val.npz",
        cache_dir / "attention_outputs_test.npz",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少以下缓存文件:\n- " + "\n- ".join(missing))

    hidden_train = load_npz_cache(cache_dir / "hidden_layer17_last_train.npz")
    hidden_val = load_npz_cache(cache_dir / "hidden_layer17_last_val.npz")
    hidden_test = load_npz_cache(cache_dir / "hidden_layer17_last_test.npz")
    score_train = load_npz_cache(cache_dir / "attention_scores_train.npz")
    score_val = load_npz_cache(cache_dir / "attention_scores_val.npz")
    score_test = load_npz_cache(cache_dir / "attention_scores_test.npz")
    output_train = load_npz_cache(cache_dir / "attention_outputs_train.npz")
    output_val = load_npz_cache(cache_dir / "attention_outputs_val.npz")
    output_test = load_npz_cache(cache_dir / "attention_outputs_test.npz")

    score_sizes = (len(score_train["labels"]), len(score_val["labels"]), len(score_test["labels"]))
    output_sizes = (len(output_train["labels"]), len(output_val["labels"]), len(output_test["labels"]))
    if score_sizes != output_sizes:
        raise ValueError(f"attention score/output 缓存样本数不一致: score={score_sizes}, output={output_sizes}")

    train_sub = subset_dataset(train_ds, score_sizes[0])
    val_sub = subset_dataset(val_ds, score_sizes[1])
    test_sub = subset_dataset(test_ds, score_sizes[2])

    X_h_train = sanitize_features("hidden_train", hidden_train["features"])
    X_h_val = sanitize_features("hidden_val", hidden_val["features"])
    X_h_test = sanitize_features("hidden_test", hidden_test["features"])
    X_as_train = sanitize_features("attention_scores_train", score_train["features"])
    X_as_val = sanitize_features("attention_scores_val", score_val["features"])
    X_as_test = sanitize_features("attention_scores_test", score_test["features"])
    X_ao_train = sanitize_features("attention_outputs_train", output_train["features"])
    X_ao_val = sanitize_features("attention_outputs_val", output_val["features"])
    X_ao_test = sanitize_features("attention_outputs_test", output_test["features"])

    return {
        "cache_dir": cache_dir,
        "train_sub": train_sub,
        "val_sub": val_sub,
        "test_sub": test_sub,
        "X_h_train": X_h_train,
        "X_h_val": X_h_val,
        "X_h_test": X_h_test,
        "y_train_full": hidden_train["labels"],
        "y_val_full": hidden_val["labels"],
        "y_test_full": hidden_test["labels"],
        "X_h_sub_train": X_h_train[:len(train_sub)],
        "X_h_sub_val": X_h_val[:len(val_sub)],
        "X_h_sub_test": X_h_test[:len(test_sub)],
        "X_as_train": X_as_train,
        "X_as_val": X_as_val,
        "X_as_test": X_as_test,
        "X_ao_train": X_ao_train,
        "X_ao_val": X_ao_val,
        "X_ao_test": X_ao_test,
        "as_names": list(score_train["feature_names"]),
        "ao_names": list(output_train["feature_names"]),
        "y_sub_train": score_train["labels"],
        "y_sub_val": score_val["labels"],
        "y_sub_test": score_test["labels"],
    }


def run_phase4_pipeline(bundle: dict, output_dir: Path, runtime_info: dict, top_k_heads: int) -> list[dict]:
    """基于完整 bundle 运行 head selection、A0-A9、错误分析与结果保存。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cache").mkdir(parents=True, exist_ok=True)

    X_as_train = bundle["X_as_train"]
    X_as_val = bundle["X_as_val"]
    X_as_test = bundle["X_as_test"]
    X_ao_train = bundle["X_ao_train"]
    X_ao_val = bundle["X_ao_val"]
    X_ao_test = bundle["X_ao_test"]
    X_h_train = bundle["X_h_train"]
    X_h_val = bundle["X_h_val"]
    X_h_test = bundle["X_h_test"]
    X_h_sub_train = bundle["X_h_sub_train"]
    X_h_sub_val = bundle["X_h_sub_val"]
    X_h_sub_test = bundle["X_h_sub_test"]
    y_train_full = bundle["y_train_full"]
    y_val_full = bundle["y_val_full"]
    y_test_full = bundle["y_test_full"]
    y_sub_train = bundle["y_sub_train"]
    y_sub_val = bundle["y_sub_val"]
    y_sub_test = bundle["y_sub_test"]
    as_names = bundle["as_names"]
    ao_names = bundle["ao_names"]
    test_sub = bundle["test_sub"]

    print("\n[Step 3/3] Running hidden baseline, head selection, and A0-A9...")
    hidden_baseline = run_hidden_baseline(
        X_h_train, X_h_val, X_h_test,
        y_train_full, y_val_full, y_test_full,
        classifier_type="logistic",
        seeds=SEEDS,
        hidden_dim=X_h_train.shape[1],
    )
    save_json(output_dir / "hidden_baseline.json", hidden_baseline)

    length_train = compute_length_proxy(X_as_train, as_names)
    length_val = compute_length_proxy(X_as_val, as_names)
    length_test = compute_length_proxy(X_as_test, as_names)
    X_as_train_r, X_as_val_r, X_as_test_r, residual_meta = residualize_by_length(
        X_as_train, X_as_val, X_as_test, length_train, length_val, length_test,
    )
    print(
        f"  Residualization: corr before={residual_meta['correlation_before']:.4f}, "
        f"after={residual_meta['correlation_after']:.4f}"
    )

    head_selection = select_top_heads(
        X_as_train_r, X_as_val_r, y_sub_train, y_sub_val,
        as_names, top_k_heads=top_k_heads, metric="auroc",
    )
    top_indices = head_selection["selected_feature_indices"]
    print(f"  Selected {len(head_selection['selected_heads'])} heads, {len(top_indices)} features")

    head_selection_to_save = {key: value for key, value in head_selection.items() if key != "all_head_scores"}
    head_selection_to_save["all_head_scores"] = [
        {key: value for key, value in row.items() if key != "feature_indices"}
        for row in head_selection["all_head_scores"]
    ]
    save_json(output_dir / "attention_head_selection.json", head_selection_to_save)

    results: list[dict] = []
    append_result_from_summary(
        results,
        "A0",
        "Hidden-only (L17 last, LR)",
        X_h_train.shape[1],
        hidden_baseline["test"],
        "Full dataset baseline",
    )
    run_and_record(
        results,
        "A0s",
        "Hidden-only (subset, LR)",
        X_h_sub_train,
        X_h_sub_val,
        X_h_sub_test,
        y_sub_train,
        y_sub_val,
        y_sub_test,
        f"Subset sizes: train={len(bundle['train_sub'])}, val={len(bundle['val_sub'])}, test={len(bundle['test_sub'])}",
    )
    run_and_record(results, "A1", "Attn-score only (raw)", X_as_train, X_as_val, X_as_test, y_sub_train, y_sub_val, y_sub_test)
    run_and_record(
        results,
        "A2",
        "Attn-score only (debiased)",
        X_as_train_r,
        X_as_val_r,
        X_as_test_r,
        y_sub_train,
        y_sub_val,
        y_sub_test,
        "Length-residualized",
    )
    if top_indices:
        run_and_record(
            results,
            "A3",
            f"Attn-score (top-{top_k_heads} heads)",
            X_as_train_r[:, top_indices],
            X_as_val_r[:, top_indices],
            X_as_test_r[:, top_indices],
            y_sub_train,
            y_sub_val,
            y_sub_test,
            f"{len(top_indices)} features",
        )
    run_and_record(results, "A4", "Attn-output only", X_ao_train, X_ao_val, X_ao_test, y_sub_train, y_sub_val, y_sub_test)
    run_and_record(
        results,
        "A5",
        "Hidden + debiased attn-score",
        np.concatenate([X_h_sub_train, X_as_train_r], axis=1),
        np.concatenate([X_h_sub_val, X_as_val_r], axis=1),
        np.concatenate([X_h_sub_test, X_as_test_r], axis=1),
        y_sub_train,
        y_sub_val,
        y_sub_test,
    )
    if top_indices:
        run_and_record(
            results,
            "A6",
            f"Hidden + top-{top_k_heads} head attn",
            np.concatenate([X_h_sub_train, X_as_train_r[:, top_indices]], axis=1),
            np.concatenate([X_h_sub_val, X_as_val_r[:, top_indices]], axis=1),
            np.concatenate([X_h_sub_test, X_as_test_r[:, top_indices]], axis=1),
            y_sub_train,
            y_sub_val,
            y_sub_test,
        )
    run_and_record(
        results,
        "A7",
        "Hidden + attn-output",
        np.concatenate([X_h_sub_train, X_ao_train], axis=1),
        np.concatenate([X_h_sub_val, X_ao_val], axis=1),
        np.concatenate([X_h_sub_test, X_ao_test], axis=1),
        y_sub_train,
        y_sub_val,
        y_sub_test,
    )
    if top_indices:
        run_and_record(
            results,
            "A8",
            "Hidden + top-head + output",
            np.concatenate([X_h_sub_train, X_as_train_r[:, top_indices], X_ao_train], axis=1),
            np.concatenate([X_h_sub_val, X_as_val_r[:, top_indices], X_ao_val], axis=1),
            np.concatenate([X_h_sub_test, X_as_test_r[:, top_indices], X_ao_test], axis=1),
            y_sub_train,
            y_sub_val,
            y_sub_test,
            "Full fusion",
        )

    print("  [A9] Gated Fusion...", end=" ", flush=True)
    hidden_val_probs = get_probs(X_h_sub_train, y_sub_train, X_h_sub_val)
    hidden_test_probs = get_probs(X_h_sub_train, y_sub_train, X_h_sub_test)
    fusion_train = np.concatenate([X_h_sub_train, X_ao_train], axis=1)
    fusion_val = np.concatenate([X_h_sub_val, X_ao_val], axis=1)
    fusion_test = np.concatenate([X_h_sub_test, X_ao_test], axis=1)
    fusion_val_probs = get_probs(fusion_train, y_sub_train, fusion_val)
    fusion_test_probs = get_probs(fusion_train, y_sub_train, fusion_test)

    tau_result = select_gated_fusion_tau(hidden_val_probs, fusion_val_probs, y_sub_val)
    best_tau = tau_result["best_tau"]
    gated_fusion = apply_gated_fusion(hidden_test_probs, fusion_test_probs, best_tau, y_sub_test)
    fusion_metrics = gated_fusion["metrics"]
    print(
        f"tau={best_tau:.2f} Acc={fusion_metrics['accuracy']:.4f} "
        f"F1={fusion_metrics['macro_f1']:.4f} AUROC={fusion_metrics['auroc']:.4f} "
        f"changed={gated_fusion['n_samples_changed']}"
    )
    results.append({
        "id": "A9",
        "method": f"Gated Fusion (tau={best_tau:.2f})",
        "feature_dim": "-",
        "test_accuracy_mean": fusion_metrics["accuracy"],
        "test_accuracy_std": 0.0,
        "test_macro_f1_mean": fusion_metrics["macro_f1"],
        "test_macro_f1_std": 0.0,
        "test_auroc_mean": fusion_metrics["auroc"],
        "test_auroc_std": 0.0,
        "note": f"Samples changed: {gated_fusion['n_samples_changed']}",
    })

    summarize_feature_differences(
        X_as_train_r, y_sub_train, as_names,
        output_csv=str(output_dir / "attention_score_feature_summary.csv"),
    )
    summarize_feature_differences(
        X_ao_train, y_sub_train, ao_names,
        output_csv=str(output_dir / "attention_output_feature_summary.csv"),
    )

    hidden_preds = (hidden_test_probs >= 0.5).astype(np.int64)
    fusion_preds = (gated_fusion["fused_probs"] >= 0.5).astype(np.int64)
    error_rows = build_error_analysis(
        test_sub.statements,
        y_sub_test,
        hidden_test_probs,
        gated_fusion["fused_probs"],
        hidden_preds,
        fusion_preds,
    )
    correction_matrix = {
        "n00": sum(1 for row in error_rows if row["case_type"] == "hidden_correct_fusion_correct"),
        "n01": sum(1 for row in error_rows if row["case_type"] == "hidden_correct_fusion_wrong"),
        "n10": sum(1 for row in error_rows if row["case_type"] == "hidden_wrong_fusion_correct"),
        "n11": sum(1 for row in error_rows if row["case_type"] == "hidden_wrong_fusion_wrong"),
    }
    with open(output_dir / "phase4_error_analysis.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=error_rows[0].keys())
        writer.writeheader()
        writer.writerows(error_rows)

    with open(output_dir / "phase4_main_results.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    save_json(
        output_dir / "phase4_ablation_results.json",
        {
            "hidden_baseline": hidden_baseline,
            "head_selection": head_selection_to_save,
            "ablation": results,
            "correction_matrix": correction_matrix,
            "runtime": runtime_info,
        },
    )

    print_results_table(results)
    print(
        f"\nCorrection Matrix: H+F+={correction_matrix['n00']} | H+F-={correction_matrix['n01']} "
        f"| H-F+={correction_matrix['n10']} | H-F-={correction_matrix['n11']}"
    )
    print(f"Net correction: {correction_matrix['n10'] - correction_matrix['n01']:+d} samples")
    print(f"\nResults saved to: {output_dir}")
    return results


def print_existing_summary(output_dir: Path) -> None:
    """读取现有结果并打印。"""
    result_path = output_dir / "phase4_ablation_results.json"
    if not result_path.exists():
        raise FileNotFoundError(f"未找到结果文件: {result_path}")

    payload = load_json(result_path)
    if isinstance(payload, list):
        results = payload
        correction_matrix = {}
    else:
        results = payload.get("ablation", [])
        correction_matrix = payload.get("correction_matrix", {})

    print_results_table(results)
    if correction_matrix:
        print(
            f"\nCorrection Matrix: H+F+={correction_matrix.get('n00', 0)} | "
            f"H+F-={correction_matrix.get('n01', 0)} | "
            f"H-F+={correction_matrix.get('n10', 0)} | H-F-={correction_matrix.get('n11', 0)}"
        )
        print(f"Net correction: {correction_matrix.get('n10', 0) - correction_matrix.get('n01', 0):+d} samples")


def load_existing_runtime(output_dir: Path, cache_dir: Path) -> dict:
    """缓存模式下复用已有 runtime 信息。"""
    result_path = output_dir / "phase4_ablation_results.json"
    if result_path.exists():
        payload = load_json(result_path)
        if isinstance(payload, dict) and "runtime" in payload:
            return payload["runtime"]
    return {"mode": "cache-only", "cache_dir": str(cache_dir)}


def main() -> None:
    """脚本主入口。"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_only:
        print_existing_summary(args.output_dir)
        return

    train_ds, val_ds, test_ds = load_processed_data()

    if args.use_cache:
        print("=" * 60)
        print("  Phase 4 Cached Re-run")
        print("=" * 60)
        print(f"Using cache directory: {args.cache_dir}")
        bundle = load_cached_bundle(args.cache_dir, train_ds, val_ds, test_ds)
        runtime_info = load_existing_runtime(args.output_dir, args.cache_dir)
    else:
        model, tokenizer, train_ds, val_ds, test_ds, runtime_info = load_model_and_datasets(args.dtype)
        bundle = extract_feature_bundle(
            model,
            tokenizer,
            train_ds,
            val_ds,
            test_ds,
            args.cache_dir,
            {
                "train": args.subset_train,
                "val": args.subset_val,
                "test": args.subset_test,
            },
        )

    run_phase4_pipeline(bundle, args.output_dir, runtime_info, args.top_k_heads)


if __name__ == "__main__":
    main()
