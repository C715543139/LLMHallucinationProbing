"""
Phase 4 精简评估：使用已有缓存数据直接训练分类器并对比。

前提：必须先运行过 scripts/run_phase4_eval.py 缓存特征，
     或手动运行 phase4-cache-hidden、phase4-extract-attention-scores、
     phase4-extract-attention-outputs。

用法:
    conda activate llm_hallucination
    .\.venv\Scripts\activate.ps1
    python -s scripts/run_phase4_quick.py
"""

from __future__ import annotations

import json
import sys
import csv
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import config
from src.methods.phase4_attention import (
    train_eval_classifier,
    residualize_by_length,
    select_top_heads,
    summarize_feature_differences,
    gated_fusion_probs,
    select_gated_fusion_tau,
    apply_gated_fusion,
    build_error_analysis,
)
from src.utils.feature_cache import load_npz_cache
from src.utils.reproducibility import set_global_seed
from src.features.hidden_states import extract_hidden_states_dataset

OUTPUT_DIR = config.paths.results_dir / "phase4"
CACHE_DIR = OUTPUT_DIR / "cache"
SEEDS = (42, 123, 2024)

# ---------------------------------------------------------------------------
# 1. 加载缓存的特征
# ---------------------------------------------------------------------------
print("Loading cached features...")

# Hidden states (from cache - if not exist, extract from model)
h_cache_train = CACHE_DIR / "hidden_layer17_last_train.npz"
if h_cache_train.exists():
    h_train = load_npz_cache(h_cache_train)
    h_val = load_npz_cache(CACHE_DIR / "hidden_layer17_last_val.npz")
    h_test = load_npz_cache(CACHE_DIR / "hidden_layer17_last_test.npz")
    X_h_train = h_train["features"]
    X_h_val = h_val["features"]
    X_h_test = h_test["features"]
    y_train = h_train["labels"]
    y_val = h_val["labels"]
    y_test = h_test["labels"]
    print(f"  Hidden features loaded: train={X_h_train.shape}, val={X_h_val.shape}, test={X_h_test.shape}")
else:
    print("  Hidden cache not found. Please run phase4-cache-hidden first.")
    sys.exit(1)

# Attention scores (from cache)
as_cache_train = CACHE_DIR / "attention_scores_train.npz"
ATTN_AVAILABLE = as_cache_train.exists()
if ATTN_AVAILABLE:
    as_train = load_npz_cache(as_cache_train)
    as_val = load_npz_cache(CACHE_DIR / "attention_scores_val.npz")
    as_test = load_npz_cache(CACHE_DIR / "attention_scores_test.npz")
    X_as_train = as_train["features"]
    X_as_val = as_val["features"]
    X_as_test = as_test["features"]
    as_names = list(as_train["feature_names"])
    y_as_train = as_train["labels"]
    y_as_val = as_val["labels"]
    y_as_test = as_test["labels"]
    print(f"  Attention score features loaded: train={X_as_train.shape}")
else:
    print("  Attention score cache not found. Attention ablation will be limited.")

# Attention outputs (from cache) 
ao_cache_train = CACHE_DIR / "attention_outputs_train.npz"
AO_AVAILABLE = ao_cache_train.exists()
if AO_AVAILABLE:
    ao_train = load_npz_cache(ao_cache_train)
    ao_val = load_npz_cache(CACHE_DIR / "attention_outputs_val.npz")
    ao_test = load_npz_cache(CACHE_DIR / "attention_outputs_test.npz")
    X_ao_train = ao_train["features"]
    X_ao_val = ao_val["features"]
    X_ao_test = ao_test["features"]
    ao_names = list(ao_train["feature_names"])
    y_ao_train = ao_train["labels"]
    y_ao_val = ao_val["labels"]
    y_ao_test = ao_test["labels"]
    print(f"  Attention output features loaded: train={X_ao_train.shape}")
else:
    print("  Attention output cache not found. Output ablation will be limited.")

# ---------------------------------------------------------------------------
# 2. 运行消融实验
# ---------------------------------------------------------------------------
results = []

def run_and_record(method_id, method_name, X_tr, X_va, X_te, y_tr, y_va, y_te, note=""):
    """训练并记录结果。"""
    print(f"\n  [{method_id}] {method_name}...")
    result = train_eval_classifier(
        X_tr, X_va, X_te, y_tr, y_va, y_te,
        classifier_type="logistic", seeds=SEEDS,
    )
    ts = result["test_summary"]
    vs = result["val_summary"]
    print(f"    Test:  Acc={ts['accuracy']['mean']:.4f}±{ts['accuracy']['std']:.4f}  "
          f"F1={ts['macro_f1']['mean']:.4f}±{ts['macro_f1']['std']:.4f}  "
          f"AUROC={ts['auroc']['mean']:.4f}±{ts['auroc']['std']:.4f}")
    results.append({
        "id": method_id,
        "method": method_name,
        "feature_dim": X_tr.shape[1],
        "test_accuracy_mean": ts["accuracy"]["mean"],
        "test_accuracy_std": ts["accuracy"]["std"],
        "test_macro_f1_mean": ts["macro_f1"]["mean"],
        "test_macro_f1_std": ts["macro_f1"]["std"],
        "test_auroc_mean": ts["auroc"]["mean"],
        "test_auroc_std": ts["auroc"]["std"],
        "val_accuracy_mean": vs["accuracy"]["mean"],
        "val_macro_f1_mean": vs["macro_f1"]["mean"],
        "val_auroc_mean": vs["auroc"]["mean"],
        "note": note,
    })
    return result

# ---- A0: Hidden-only baseline ----
a0 = run_and_record("A0", "Hidden-only (L17 last, LR)", X_h_train, X_h_val, X_h_test, y_train, y_val, y_test)

# ---- A1-A4: Attention-only ----
if ATTN_AVAILABLE:
    # Length residualization
    from src.features.attention_scores import extract_length_metadata
    # For cached data, we need length metadata; use a simple fallback
    # Extract seq_len from the data or use a dummy approach
    # Since we're using cached features from subset, match alignment
    n_train_attn = X_as_train.shape[0]
    n_val_attn = X_as_val.shape[0]
    n_test_attn = X_as_test.shape[0]

    # Align labels (use first N samples of hidden labels)
    y_tr_attn = y_train[:n_train_attn] if len(y_train) >= n_train_attn else y_train
    y_va_attn = y_val[:n_val_attn] if len(y_val) >= n_val_attn else y_val
    y_te_attn = y_test[:n_test_attn] if len(y_test) >= n_test_attn else y_test

    # Align hidden features
    X_h_tr_attn = X_h_train[:n_train_attn]
    X_h_va_attn = X_h_val[:n_val_attn]
    X_h_te_attn = X_h_test[:n_test_attn]

    a1 = run_and_record("A1", "Attention score only (raw)", X_as_train, X_as_val, X_as_test, y_tr_attn, y_va_attn, y_te_attn)

    # Simple length debias using attention sink as proxy for length
    # Use column index for attention_sink_mass as a length proxy
    sink_cols = [i for i, n in enumerate(as_names) if "attention_sink_mass" in n]
    if sink_cols:
        sink_proxy_train = X_as_train[:, sink_cols].mean(axis=1, keepdims=True)
        sink_proxy_val = X_as_val[:, sink_cols].mean(axis=1, keepdims=True)
        sink_proxy_test = X_as_test[:, sink_cols].mean(axis=1, keepdims=True)

        # Create length arrays with sink_proxy as first column + zeros for others
        len_tr = np.concatenate([sink_proxy_train, np.zeros((n_train_attn, 5))], axis=1)
        len_va = np.concatenate([sink_proxy_val, np.zeros((n_val_attn, 5))], axis=1)
        len_te = np.concatenate([sink_proxy_test, np.zeros((n_test_attn, 5))], axis=1)

        X_as_train_r, X_as_val_r, X_as_test_r, _ = residualize_by_length(
            X_as_train, X_as_val, X_as_test, len_tr, len_va, len_te,
        )
        a2 = run_and_record("A2", "Attention score only (debiased)", X_as_train_r, X_as_val_r, X_as_test_r, y_tr_attn, y_va_attn, y_te_attn)
    else:
        X_as_train_r, X_as_val_r, X_as_test_r = X_as_train, X_as_val, X_as_test
        print("  [A2] Skipped: no sink columns for debiasing")

    # Head selection
    head_sel = select_top_heads(
        X_as_train_r, X_as_val_r, y_tr_attn, y_va_attn,
        as_names, top_k_heads=16, metric="auroc",
    )
    top_indices = head_sel["selected_feature_indices"]
    print(f"  Selected {len(head_sel['selected_heads'])} heads, {len(top_indices)} features")

    if top_indices:
        a3 = run_and_record("A3", "Attention score (top heads)", X_as_train_r[:, top_indices], X_as_val_r[:, top_indices], X_as_test_r[:, top_indices], y_tr_attn, y_va_attn, y_te_attn)

    # ---- A5: Hidden + debiased attention ----
    a5 = run_and_record("A5", "Hidden + debiased attention",
        np.concatenate([X_h_tr_attn, X_as_train_r], axis=1),
        np.concatenate([X_h_va_attn, X_as_val_r], axis=1),
        np.concatenate([X_h_te_attn, X_as_test_r], axis=1),
        y_tr_attn, y_va_attn, y_te_attn)

    # ---- A6: Hidden + top-head attention ----
    if top_indices:
        a6 = run_and_record("A6", "Hidden + top-head attention",
            np.concatenate([X_h_tr_attn, X_as_train_r[:, top_indices]], axis=1),
            np.concatenate([X_h_va_attn, X_as_val_r[:, top_indices]], axis=1),
            np.concatenate([X_h_te_attn, X_as_test_r[:, top_indices]], axis=1),
            y_tr_attn, y_va_attn, y_te_attn)

if AO_AVAILABLE:
    n_train_ao = X_ao_train.shape[0]
    n_val_ao = X_ao_val.shape[0]
    n_test_ao = X_ao_test.shape[0]

    y_tr_ao = y_train[:n_train_ao] if len(y_train) >= n_train_ao else y_train
    y_va_ao = y_val[:n_val_ao] if len(y_val) >= n_val_ao else y_val
    y_te_ao = y_test[:n_test_ao] if len(y_test) >= n_test_ao else y_test

    X_h_tr_ao = X_h_train[:n_train_ao]
    X_h_va_ao = X_h_val[:n_val_ao]
    X_h_te_ao = X_h_test[:n_test_ao]

    a4 = run_and_record("A4", "Attention output only", X_ao_train, X_ao_val, X_ao_test, y_tr_ao, y_va_ao, y_te_ao)

    a7 = run_and_record("A7", "Hidden + attention output",
        np.concatenate([X_h_tr_ao, X_ao_train], axis=1),
        np.concatenate([X_h_va_ao, X_ao_val], axis=1),
        np.concatenate([X_h_te_ao, X_ao_test], axis=1),
        y_tr_ao, y_va_ao, y_te_ao)

    # ---- A8: Hidden + all ----
    if ATTN_AVAILABLE and top_indices:
        # Align all three feature sets
        n_common = min(n_train_attn, n_train_ao)
        a8 = run_and_record("A8", "Hidden + top-head attn + output",
            np.concatenate([X_h_train[:n_common], X_as_train_r[:n_common][:, top_indices], X_ao_train[:n_common]], axis=1),
            np.concatenate([X_h_val[:min(n_val_attn, n_val_ao)], X_as_val_r[:min(n_val_attn, n_val_ao)][:, top_indices], X_ao_val[:min(n_val_attn, n_val_ao)]], axis=1),
            np.concatenate([X_h_test[:min(n_test_attn, n_test_ao)], X_as_test_r[:min(n_test_attn, n_test_ao)][:, top_indices], X_ao_test[:min(n_test_attn, n_test_ao)]], axis=1),
            y_train[:n_common], y_val[:min(n_val_attn, n_val_ao)], y_test[:min(n_test_attn, n_test_ao)],
            note="Combined features")

    # ---- A9: Gated Fusion ----
    print("\n  [A9] Gated Fusion...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    def get_probs(X_tr, y_tr, X_ev, seed=42):
        set_global_seed(seed)
        sc = StandardScaler()
        Xt = sc.fit_transform(X_tr)
        Xe = sc.transform(X_ev)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, n_jobs=-1)
        clf.fit(Xt, y_tr)
        return clf.predict_proba(Xe)[:, 1]

    h_val_p = get_probs(X_h_tr_ao, y_tr_ao, X_h_va_ao)
    h_test_p = get_probs(X_h_tr_ao, y_tr_ao, X_h_te_ao)
    f_val_p = get_probs(
        np.concatenate([X_h_tr_ao, X_ao_train], axis=1), y_tr_ao,
        np.concatenate([X_h_va_ao, X_ao_val], axis=1))
    f_test_p = get_probs(
        np.concatenate([X_h_tr_ao, X_ao_train], axis=1), y_tr_ao,
        np.concatenate([X_h_te_ao, X_ao_test], axis=1))

    tau_res = select_gated_fusion_tau(h_val_p, f_val_p, y_va_ao)
    bt = tau_res["best_tau"]
    gf = apply_gated_fusion(h_test_p, f_test_p, bt, y_te_ao)
    tm = gf["metrics"]
    print(f"    Best tau={bt:.2f}, Test: Acc={tm['accuracy']:.4f}, F1={tm['macro_f1']:.4f}, AUROC={tm['auroc']:.4f}")
    results.append({
        "id": "A9", "method": f"Gated Fusion (τ={bt:.2f})", "feature_dim": "-",
        "test_accuracy_mean": tm["accuracy"], "test_accuracy_std": 0.0,
        "test_macro_f1_mean": tm["macro_f1"], "test_macro_f1_std": 0.0,
        "test_auroc_mean": tm["auroc"], "test_auroc_std": 0.0,
        "val_accuracy_mean": 0.0, "val_macro_f1_mean": 0.0, "val_auroc_mean": 0.0,
        "note": f"τ={bt:.2f}, changed {gf['n_samples_changed']} samples",
    })

# ---------------------------------------------------------------------------
# 3. 保存结果
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("  保存结果...")

# CSV
csv_path = OUTPUT_DIR / "phase4_main_results.csv"
fieldnames = list(results[0].keys()) if results else []
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
print(f"  CSV: {csv_path}")

# JSON
json_path = OUTPUT_DIR / "phase4_ablation_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"  JSON: {json_path}")

# ---------------------------------------------------------------------------
# 4. 打印汇总表
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("                        Phase 4 消融实验结果汇总表")
print("=" * 90)
header = f"{'ID':<6} {'Method':<38} {'Dim':<8} {'Test Acc':<16} {'Test F1':<16} {'Test AUROC':<14}"
print(header)
print("-" * 90)
for r in results:
    print(
        f"{r['id']:<6} {r['method']:<38} {str(r['feature_dim']):<8} "
        f"{r['test_accuracy_mean']:.4f} ± {r['test_accuracy_std']:.4f}   "
        f"{r['test_macro_f1_mean']:.4f} ± {r['test_macro_f1_std']:.4f}   "
        f"{r['test_auroc_mean']:.4f} ± {r['test_auroc_std']:.4f}"
    )
print("-" * 90)
print(f"\n{'=' * 90}")
print("  完成！")
print(f"{'=' * 90}")
