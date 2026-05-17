"""
Phase 4 快速评估脚本：运行所有实验并收集结果。

用法:
    conda activate llm_hallucination
    .\.venv\Scripts\activate.ps1
    python -s scripts/run_phase4_eval.py
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path

import numpy as np

# 添加项目根到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.config import config
from src.models.loader import load_model_fp16, print_device_info
from src.data.preprocessing import load_processed_data
from src.features.hidden_states import extract_hidden_states_dataset
from src.methods.saplma import train_and_evaluate
from src.methods.phase4_attention import (
    run_hidden_baseline,
    train_eval_classifier,
    residualize_by_length,
    select_top_heads,
    summarize_feature_differences,
    gated_fusion_probs,
    select_gated_fusion_tau,
    apply_gated_fusion,
    build_error_analysis,
    write_phase4_summary,
)
from src.utils.reproducibility import collect_runtime_info, set_global_seed
from src.utils.metrics import compute_metrics, compute_metrics_multi_seed

OUTPUT_DIR = config.paths.results_dir / "phase4"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "cache").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
SEEDS = (42, 123, 2024)

# ---------------------------------------------------------------------------
# Step 1: 加载模型和数据
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Phase 4 快速评估")
print("=" * 60)

print_device_info()

print(f"\n加载模型 (Qwen2-1.5B float16, eager attention)...")
model, tokenizer = load_model_fp16(
    model_path=str(config.paths.models_cache / "Qwen2-1.5B"),
    device_map=config.models.primary_device_map,
    torch_dtype=config.models.primary_dtype,
    attn_implementation="eager",
)
print(f"模型设备: {next(model.parameters()).device}")

print("\n加载预处理数据...")
train_ds, val_ds, test_ds = load_processed_data()
print(train_ds.summary())

runtime_info = collect_runtime_info(model)
all_results = {}

# ---------------------------------------------------------------------------
# Step 2: P4.0 Hidden Baseline (layer 17, last token, LR)
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("  [P4.0] Hidden Baseline: layer=17, pooling=last, LR")
print("=" * 50)

X_h_train, y_train = extract_hidden_states_dataset(
    model, tokenizer, train_ds, pooling="last", layers=[17], batch_size=8, max_length=128,
)
X_h_val, y_val = extract_hidden_states_dataset(
    model, tokenizer, val_ds, pooling="last", layers=[17], batch_size=8, max_length=128,
)
X_h_test, y_test = extract_hidden_states_dataset(
    model, tokenizer, test_ds, pooling="last", layers=[17], batch_size=8, max_length=128,
)

hidden_baseline = run_hidden_baseline(
    X_h_train, X_h_val, X_h_test,
    y_train, y_val, y_test,
    classifier_type="logistic",
    seeds=SEEDS,
    hidden_dim=X_h_train.shape[1],
)
hidden_baseline["runtime"] = runtime_info

with open(OUTPUT_DIR / "hidden_baseline.json", "w", encoding="utf-8") as f:
    json.dump(hidden_baseline, f, indent=2, ensure_ascii=False, default=str)

print("\n[Hidden Baseline 结果]")
ts = hidden_baseline["test"]
print(f"  Test Accuracy:  {ts['accuracy']['mean']:.4f} ± {ts['accuracy']['std']:.4f}")
print(f"  Test Macro-F1:  {ts['macro_f1']['mean']:.4f} ± {ts['macro_f1']['std']:.4f}")
print(f"  Test AUROC:     {ts['auroc']['mean']:.4f} ± {ts['auroc']['std']:.4f}")

all_results["hidden_only"] = {
    "method": "Hidden-only (L17 last, LR)",
    "test_accuracy": ts["accuracy"]["mean"],
    "test_macro_f1": ts["macro_f1"]["mean"],
    "test_auroc": ts["auroc"]["mean"],
}

# ---------------------------------------------------------------------------
# Step 3: P4.2 / P4.5 提取 Attention 特征 (仅用前 500 样本加速)
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("  [P4.2] Extraction: Attention Scores (subset for speed)")
print("=" * 50)

CANDIDATE_LAYERS = [13, 14, 15, 16, 17, 18, 19, 20]
MAX_SAMPLES = 800  # 用子集加速，完整的建议用 phase4-extract-attention-scores

from src.data.dataset import TrueFalseDataset

def subset_dataset(ds, n: int):
    """取数据集前 n 个样本。"""
    n = min(n, len(ds))
    return TrueFalseDataset(ds.statements[:n], ds.labels[:n], ds.domains[:n] if ds.domains else None)

train_sub = subset_dataset(train_ds, MAX_SAMPLES)
val_sub = subset_dataset(val_ds, max(200, MAX_SAMPLES // 4))
test_sub = subset_dataset(test_ds, max(200, MAX_SAMPLES // 4))

from src.features.attention_scores import (
    extract_attention_score_features_dataset,
    extract_length_metadata,
)

print(f"  使用 {len(train_sub)} train / {len(val_sub)} val / {len(test_sub)} test 样本...")

as_train_data = extract_attention_score_features_dataset(
    model, tokenizer, train_sub, layers=CANDIDATE_LAYERS, batch_size=1,
    output_path=str(OUTPUT_DIR / "cache" / "attention_scores_train.npz"),
)
as_val_data = extract_attention_score_features_dataset(
    model, tokenizer, val_sub, layers=CANDIDATE_LAYERS, batch_size=1,
    output_path=str(OUTPUT_DIR / "cache" / "attention_scores_val.npz"),
)
as_test_data = extract_attention_score_features_dataset(
    model, tokenizer, test_sub, layers=CANDIDATE_LAYERS, batch_size=1,
    output_path=str(OUTPUT_DIR / "cache" / "attention_scores_test.npz"),
)

X_as_train = as_train_data["features"]
X_as_val = as_val_data["features"]
X_as_test = as_test_data["features"]
as_names = as_train_data["feature_names"]
y_train_sub = as_train_data["labels"]
y_val_sub = as_val_data["labels"]
y_test_sub = as_test_data["labels"]

print(f"  Attention score 特征维度: {X_as_train.shape}")

# ---------------------------------------------------------------------------
# Step 4: P4.3 Length residualization
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("  [P4.3] Length Residualization")
print("=" * 50)

len_train = extract_length_metadata(model, tokenizer, train_sub, layers=CANDIDATE_LAYERS, batch_size=1)
len_val = extract_length_metadata(model, tokenizer, val_sub, layers=CANDIDATE_LAYERS, batch_size=1)
len_test = extract_length_metadata(model, tokenizer, test_sub, layers=CANDIDATE_LAYERS, batch_size=1)

X_as_train_r, X_as_val_r, X_as_test_r, resid_meta = residualize_by_length(
    X_as_train, X_as_val, X_as_test,
    len_train, len_val, len_test,
)
print(f"  残差化前相关系数: {resid_meta['correlation_before']:.4f}")
print(f"  残差化后相关系数: {resid_meta['correlation_after']:.4f}")

# ---------------------------------------------------------------------------
# Step 5: P4.4 Head Selection
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("  [P4.4] Head Selection")
print("=" * 50)

head_sel = select_top_heads(
    X_as_train_r, X_as_val_r, y_train_sub, y_val_sub,
    as_names, top_k_heads=16, metric="auroc",
)

# 保存 head selection
sel_save = {
    "selection_metric": head_sel["selection_metric"],
    "top_k_heads": head_sel["top_k_heads"],
    "selected_heads": head_sel["selected_heads"],
    "all_head_scores": [
        {k: v for k, v in h.items() if k != "feature_indices"}
        for h in head_sel["all_head_scores"]
    ],
}
with open(OUTPUT_DIR / "attention_head_selection.json", "w", encoding="utf-8") as f:
    json.dump(sel_save, f, indent=2, ensure_ascii=False, default=str)

top_indices = head_sel["selected_feature_indices"]
print(f"  Selected {len(head_sel['selected_heads'])} heads, {len(top_indices)} features")

# ---------------------------------------------------------------------------
# Step 6: P4.5 Attention Output Features
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("  [P4.5] Attention Output Features")
print("=" * 50)

from src.features.attention_outputs import extract_attention_output_features_dataset

ao_train_data = extract_attention_output_features_dataset(
    model, tokenizer, train_sub, layers=CANDIDATE_LAYERS, batch_size=1,
    output_path=str(OUTPUT_DIR / "cache" / "attention_outputs_train.npz"),
)
ao_val_data = extract_attention_output_features_dataset(
    model, tokenizer, val_sub, layers=CANDIDATE_LAYERS, batch_size=1,
    output_path=str(OUTPUT_DIR / "cache" / "attention_outputs_val.npz"),
)
ao_test_data = extract_attention_output_features_dataset(
    model, tokenizer, test_sub, layers=CANDIDATE_LAYERS, batch_size=1,
    output_path=str(OUTPUT_DIR / "cache" / "attention_outputs_test.npz"),
)

X_ao_train = ao_train_data["features"]
X_ao_val = ao_val_data["features"]
X_ao_test = ao_test_data["features"]
ao_names = ao_train_data["feature_names"]

print(f"  Attention output 特征维度: {X_ao_train.shape}")

# ---------------------------------------------------------------------------
# Step 7: P4.6 Ablation Experiments
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("  [P4.6] Ablation Experiments")
print("=" * 50)

# 为 ablation 准备 hidden 子集
h_train_sub = extract_hidden_states_dataset(
    model, tokenizer, train_sub, pooling="last", layers=[17], batch_size=8, max_length=128,
)[0]
h_val_sub = extract_hidden_states_dataset(
    model, tokenizer, val_sub, pooling="last", layers=[17], batch_size=8, max_length=128,
)[0]
h_test_sub = extract_hidden_states_dataset(
    model, tokenizer, test_sub, pooling="last", layers=[17], batch_size=8, max_length=128,
)[0]

ablation_entries = []

# ---- A0: Hidden-only -----------------------------------------------
print("\n--- A0: Hidden-only ---")
a0 = train_eval_classifier(
    h_train_sub, h_val_sub, h_test_sub,
    y_train_sub, y_val_sub, y_test_sub,
    classifier_type="logistic", seeds=SEEDS,
)
_print_result("A0", "Hidden-only (L17 last, LR)", a0)
ablation_entries.append(_make_entry("A0", "Hidden-only (L17 last, LR)", a0, h_train_sub.shape[1]))

# ---- A1: Raw attention score only ----------------------------------
print("\n--- A1: Attention score (raw) ---")
a1 = train_eval_classifier(
    X_as_train, X_as_val, X_as_test,
    y_train_sub, y_val_sub, y_test_sub,
    classifier_type="logistic", seeds=SEEDS,
)
_print_result("A1", "Attention score (raw)", a1)
ablation_entries.append(_make_entry("A1", "Attention score only (raw)", a1, X_as_train.shape[1]))

# ---- A2: Debiased attention score only -----------------------------
print("\n--- A2: Attention score (debiased) ---")
a2 = train_eval_classifier(
    X_as_train_r, X_as_val_r, X_as_test_r,
    y_train_sub, y_val_sub, y_test_sub,
    classifier_type="logistic", seeds=SEEDS,
)
_print_result("A2", "Attention score (debiased)", a2)
ablation_entries.append(_make_entry("A2", "Attention score only (debiased)", a2, X_as_train_r.shape[1]))

# ---- A3: Top-head attention only -----------------------------------
if top_indices:
    print("\n--- A3: Attention score (top heads only) ---")
    a3 = train_eval_classifier(
        X_as_train_r[:, top_indices], X_as_val_r[:, top_indices], X_as_test_r[:, top_indices],
        y_train_sub, y_val_sub, y_test_sub,
        classifier_type="logistic", seeds=SEEDS,
    )
    _print_result("A3", "Attention score (top heads)", a3)
    ablation_entries.append(_make_entry("A3", "Attention score (top heads)", a3, len(top_indices)))

# ---- A4: Attention output only -------------------------------------
print("\n--- A4: Attention output only ---")
a4 = train_eval_classifier(
    X_ao_train, X_ao_val, X_ao_test,
    y_train_sub, y_val_sub, y_test_sub,
    classifier_type="logistic", seeds=SEEDS,
)
_print_result("A4", "Attention output only", a4)
ablation_entries.append(_make_entry("A4", "Attention output only", a4, X_ao_train.shape[1]))

# ---- A5: Hidden + debiased attention --------------------------------
print("\n--- A5: Hidden + debiased attention ---")
X_a5_train = np.concatenate([h_train_sub, X_as_train_r], axis=1)
X_a5_val = np.concatenate([h_val_sub, X_as_val_r], axis=1)
X_a5_test = np.concatenate([h_test_sub, X_as_test_r], axis=1)
a5 = train_eval_classifier(
    X_a5_train, X_a5_val, X_a5_test,
    y_train_sub, y_val_sub, y_test_sub,
    classifier_type="logistic", seeds=SEEDS,
)
_print_result("A5", "Hidden + debiased attention", a5)
ablation_entries.append(_make_entry("A5", "Hidden + debiased attention", a5, X_a5_train.shape[1]))

# ---- A6: Hidden + top-head attention --------------------------------
if top_indices:
    print("\n--- A6: Hidden + top-head attention ---")
    X_a6_train = np.concatenate([h_train_sub, X_as_train_r[:, top_indices]], axis=1)
    X_a6_val = np.concatenate([h_val_sub, X_as_val_r[:, top_indices]], axis=1)
    X_a6_test = np.concatenate([h_test_sub, X_as_test_r[:, top_indices]], axis=1)
    a6 = train_eval_classifier(
        X_a6_train, X_a6_val, X_a6_test,
        y_train_sub, y_val_sub, y_test_sub,
        classifier_type="logistic", seeds=SEEDS,
    )
    _print_result("A6", "Hidden + top-head attention", a6)
    ablation_entries.append(_make_entry("A6", "Hidden + top-head attention", a6, X_a6_train.shape[1]))

# ---- A7: Hidden + attention output ----------------------------------
print("\n--- A7: Hidden + attention output ---")
X_a7_train = np.concatenate([h_train_sub, X_ao_train], axis=1)
X_a7_val = np.concatenate([h_val_sub, X_ao_val], axis=1)
X_a7_test = np.concatenate([h_test_sub, X_ao_test], axis=1)
a7 = train_eval_classifier(
    X_a7_train, X_a7_val, X_a7_test,
    y_train_sub, y_val_sub, y_test_sub,
    classifier_type="logistic", seeds=SEEDS,
)
_print_result("A7", "Hidden + attention output", a7)
ablation_entries.append(_make_entry("A7", "Hidden + attention output", a7, X_a7_train.shape[1]))

# ---- A8: Hidden + all attention ------------------------------------
if top_indices:
    print("\n--- A8: Hidden + all attention ---")
    X_a8_train = np.concatenate([h_train_sub, X_as_train_r[:, top_indices], X_ao_train], axis=1)
    X_a8_val = np.concatenate([h_val_sub, X_as_val_r[:, top_indices], X_ao_val], axis=1)
    X_a8_test = np.concatenate([h_test_sub, X_as_test_r[:, top_indices], X_ao_test], axis=1)
    a8 = train_eval_classifier(
        X_a8_train, X_a8_val, X_a8_test,
        y_train_sub, y_val_sub, y_test_sub,
        classifier_type="logistic", seeds=SEEDS,
    )
    _print_result("A8", "Hidden + all attention", a8)
    ablation_entries.append(_make_entry("A8", "Hidden + all attention", a8, X_a8_train.shape[1]))

# ---- A9: Gated Fusion -----------------------------------------------
# 使用 A0 hidden-only 和 A7 hidden+attn_output 做 gated fusion
print("\n--- A9: Gated Fusion ---")
# 手动提取 seed0 的概率
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def _get_probs(X_train, y_train, X_eval, seed=42):
    set_global_seed(seed)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_ev = scaler.transform(X_eval)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, n_jobs=-1)
    clf.fit(X_tr, y_train)
    return clf.predict_proba(X_ev)[:, 1]

# Hidden-only probs
h_val_probs = _get_probs(h_train_sub, y_train_sub, h_val_sub, seed=42)
h_test_probs = _get_probs(h_train_sub, y_train_sub, h_test_sub, seed=42)

# Fusion (A7: hidden+ao) probs
f_val_probs = _get_probs(X_a7_train, y_train_sub, X_a7_val, seed=42)
f_test_probs = _get_probs(X_a7_train, y_train_sub, X_a7_test, seed=42)

# Select tau
tau_res = select_gated_fusion_tau(
    h_val_probs, f_val_probs, y_val_sub,
    tau_candidates=(0.05, 0.10, 0.15, 0.20, 0.25),
    metric="macro_f1",
)
best_tau = tau_res["best_tau"]
print(f"  Best tau: {best_tau}")

# Apply
gf_res = apply_gated_fusion(h_test_probs, f_test_probs, best_tau, y_test_sub)
test_m = gf_res["metrics"]
print(f"  Gated Fusion: Acc={test_m['accuracy']:.4f}, F1={test_m['macro_f1']:.4f}, AUROC={test_m['auroc']:.4f}")
print(f"  Samples changed: {gf_res['n_samples_changed']}")

ablation_entries.append({
    "id": "A9",
    "method": "Gated Fusion (Hidden + AO, τ={:.2f})".format(best_tau),
    "feature_dim": "-",
    "test_accuracy_mean": test_m["accuracy"],
    "test_accuracy_std": 0.0,
    "test_macro_f1_mean": test_m["macro_f1"],
    "test_macro_f1_std": 0.0,
    "test_auroc_mean": test_m["auroc"],
    "test_auroc_std": 0.0,
    "note": f"Samples changed: {gf_res['n_samples_changed']}",
})

# ---------------------------------------------------------------------------
# Step 8: Feature summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 50)
print("  Feature Analysis")
print("=" * 50)

summarize_feature_differences(
    X_as_train_r, y_train_sub, as_names,
    output_csv=str(OUTPUT_DIR / "attention_score_feature_summary.csv"),
)
summarize_feature_differences(
    X_ao_train, y_train_sub, ao_names,
    output_csv=str(OUTPUT_DIR / "attention_output_feature_summary.csv"),
)

# ---------------------------------------------------------------------------
# Step 9: Error Analysis (hidden vs A7 fusion)
# ---------------------------------------------------------------------------
h_test_preds = (h_test_probs >= 0.5).astype(np.int64)
f_test_preds = (gf_res["fused_preds"]).astype(np.int64)

error_rows = build_error_analysis(
    test_sub.statements,
    y_test_sub,
    h_test_probs,
    gf_res["fused_probs"],
    h_test_preds,
    f_test_preds,
)

# Correction matrix
n00 = sum(1 for r in error_rows if r["case_type"] == "hidden_correct_fusion_correct")
n01 = sum(1 for r in error_rows if r["case_type"] == "hidden_correct_fusion_wrong")
n10 = sum(1 for r in error_rows if r["case_type"] == "hidden_wrong_fusion_correct")
n11 = sum(1 for r in error_rows if r["case_type"] == "hidden_wrong_fusion_wrong")

correction_matrix = {
    "hidden_correct_fusion_correct": n00,
    "hidden_correct_fusion_wrong": n01,
    "hidden_wrong_fusion_correct": n10,
    "hidden_wrong_fusion_wrong": n11,
}

# 保存错误分析
import csv
error_csv_path = OUTPUT_DIR / "phase4_error_analysis.csv"
with open(error_csv_path, "w", newline="", encoding="utf-8") as f:
    if error_rows:
        writer = csv.DictWriter(f, fieldnames=error_rows[0].keys())
        writer.writeheader()
        writer.writerows(error_rows)

# ---------------------------------------------------------------------------
# Step 10: Save All Results
# ---------------------------------------------------------------------------

# 保存主结果 CSV
csv_path = OUTPUT_DIR / "phase4_main_results.csv"
fieldnames = [
    "id", "method", "feature_dim",
    "test_accuracy_mean", "test_accuracy_std",
    "test_macro_f1_mean", "test_macro_f1_std",
    "test_auroc_mean", "test_auroc_std", "note",
]
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for entry in ablation_entries:
        row = {k: entry.get(k, "") for k in fieldnames}
        writer.writerow(row)

# 保存 ablation JSON
ablation_json = {
    "hidden_baseline": hidden_baseline,
    "head_selection": sel_save,
    "ablation_entries": ablation_entries,
    "correction_matrix": correction_matrix,
    "runtime": runtime_info,
}
with open(OUTPUT_DIR / "phase4_ablation_results.json", "w", encoding="utf-8") as f:
    json.dump(ablation_json, f, indent=2, ensure_ascii=False, default=str)

# 写 summary
write_phase4_summary(
    output_dir=OUTPUT_DIR,
    hidden_baseline=hidden_baseline,
    head_selection=sel_save,
    ablation_results={"ablation_entries": ablation_entries, "correction_matrix": correction_matrix},
    runtime_info=runtime_info,
)

# ---------------------------------------------------------------------------
# Step 11: Print Final Summary Table
# ---------------------------------------------------------------------------
print("\n\n" + "=" * 80)
print("                     Phase 4 消融实验结果汇总")
print("=" * 80)
print(f"{'ID':<6} {'Method':<40} {'Dim':<8} {'Test Acc':<12} {'Test F1':<12} {'Test AUROC':<12}")
print("-" * 80)
for entry in ablation_entries:
    print(
        f"{entry['id']:<6} {entry['method']:<40} {str(entry['feature_dim']):<8} "
        f"{entry['test_accuracy_mean']:.4f}±{entry['test_accuracy_std']:.4f}   "
        f"{entry['test_macro_f1_mean']:.4f}±{entry['test_macro_f1_std']:.4f}   "
        f"{entry['test_auroc_mean']:.4f}±{entry['test_auroc_std']:.4f}"
    )
print("-" * 80)
print(f"\n修正矩阵: Hidden Correct & Fusion Correct: {n00} | Hidden Correct & Fusion Wrong: {n01}")
print(f"           Hidden Wrong & Fusion Correct: {n10} | Hidden Wrong & Fusion Wrong: {n11}")
print(f"           净修正: {n10 - n01:+d} 样本")
print(f"\n所有结果已保存至: {OUTPUT_DIR}")
print("=" * 80)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_result(aid, name, result):
    ts = result["test_summary"]
    print(f"  {aid} {name}:")
    print(f"    Acc={ts['accuracy']['mean']:.4f}±{ts['accuracy']['std']:.4f}  "
          f"F1={ts['macro_f1']['mean']:.4f}±{ts['macro_f1']['std']:.4f}  "
          f"AUROC={ts['auroc']['mean']:.4f}±{ts['auroc']['std']:.4f}")


def _make_entry(aid, name, result, dim):
    ts = result["test_summary"]
    return {
        "id": aid,
        "method": name,
        "feature_dim": dim,
        "test_accuracy_mean": ts["accuracy"]["mean"],
        "test_accuracy_std": ts["accuracy"]["std"],
        "test_macro_f1_mean": ts["macro_f1"]["mean"],
        "test_macro_f1_std": ts["macro_f1"]["std"],
        "test_auroc_mean": ts["auroc"]["mean"],
        "test_auroc_std": ts["auroc"]["std"],
        "note": "",
    }
