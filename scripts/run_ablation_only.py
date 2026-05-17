"""从缓存加载特征，运行 A0-A9 消融并更新结果。"""
import json, csv, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.methods.phase4_attention import (
    train_eval_classifier, residualize_by_length,
    select_top_heads, summarize_feature_differences,
    gated_fusion_probs, select_gated_fusion_tau, apply_gated_fusion,
    build_error_analysis, run_hidden_baseline,
)
from src.utils.feature_cache import load_npz_cache
from src.utils.reproducibility import set_global_seed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

OUTPUT_DIR = Path("experiments/results/phase4")
CACHE_DIR = OUTPUT_DIR / "cache"
SEEDS = (42, 123, 2024)

# ---- Load cached features ----
print("Loading cached features...")

h_train = load_npz_cache(CACHE_DIR / "hidden_layer17_last_train.npz")
h_val = load_npz_cache(CACHE_DIR / "hidden_layer17_last_val.npz")
h_test = load_npz_cache(CACHE_DIR / "hidden_layer17_last_test.npz")
X_h_train, y_train_full = h_train["features"], h_train["labels"]
X_h_val, y_val_full = h_val["features"], h_val["labels"]
X_h_test, y_test_full = h_test["features"], h_test["labels"]
print(f"  Hidden: train={X_h_train.shape}, NaN={np.isnan(X_h_train).sum()}")

as_train = load_npz_cache(CACHE_DIR / "attention_scores_train.npz")
as_val = load_npz_cache(CACHE_DIR / "attention_scores_val.npz")
as_test = load_npz_cache(CACHE_DIR / "attention_scores_test.npz")
X_as_train = np.nan_to_num(as_train["features"], nan=0.0)
X_as_val = np.nan_to_num(as_val["features"], nan=0.0)
X_as_test = np.nan_to_num(as_test["features"], nan=0.0)
as_names = as_train["feature_names"]
y_sub_train = as_train["labels"]
y_sub_val = as_val["labels"]
y_sub_test = as_test["labels"]
print(f"  Attn scores: {X_as_train.shape}, NaN_before={np.isnan(as_train['features']).sum()}")

ao_train = load_npz_cache(CACHE_DIR / "attention_outputs_train.npz")
ao_val = load_npz_cache(CACHE_DIR / "attention_outputs_val.npz")
ao_test = load_npz_cache(CACHE_DIR / "attention_outputs_test.npz")
X_ao_train = np.nan_to_num(ao_train["features"], nan=0.0)
X_ao_val = np.nan_to_num(ao_val["features"], nan=0.0)
X_ao_test = np.nan_to_num(ao_test["features"], nan=0.0)
ao_names = ao_train["feature_names"]
print(f"  Attn outputs: {X_ao_train.shape}, NaN_before={np.isnan(ao_train['features']).sum()}")

# Align subset
n_sub = len(y_sub_train)
X_h_sub_train = X_h_train[:n_sub]
X_h_sub_val = X_h_val[:len(y_sub_val)]
X_h_sub_test = X_h_test[:len(y_sub_test)]

# ---- Hidden baseline on FULL ----
print("\n=== Hidden Baseline (FULL dataset) ===")
hb = run_hidden_baseline(
    X_h_train, X_h_val, X_h_test, y_train_full, y_val_full, y_test_full,
    classifier_type="logistic", seeds=SEEDS, hidden_dim=X_h_train.shape[1])
ts = hb["test"]
print(f"  Test: Acc={ts['accuracy']['mean']:.4f} F1={ts['macro_f1']['mean']:.4f} AUROC={ts['auroc']['mean']:.4f}")

# ---- Residualization ----
sink_cols = [i for i, n in enumerate(as_names) if "attention_sink_mass" in n]
len_tr = np.concatenate([X_as_train[:, sink_cols].mean(axis=1, keepdims=True), np.zeros((n_sub, 5))], axis=1) if sink_cols else np.ones((n_sub, 6))
len_va = np.concatenate([X_as_val[:, sink_cols].mean(axis=1, keepdims=True), np.zeros((len(y_sub_val), 5))], axis=1) if sink_cols else np.ones((len(y_sub_val), 6))
len_te = np.concatenate([X_as_test[:, sink_cols].mean(axis=1, keepdims=True), np.zeros((len(y_sub_test), 5))], axis=1) if sink_cols else np.ones((len(y_sub_test), 6))
X_as_train_r, X_as_val_r, X_as_test_r, _ = residualize_by_length(
    X_as_train, X_as_val, X_as_test, len_tr, len_va, len_te)

# ---- Head Selection ----
print("\n=== Head Selection ===")
head_sel = select_top_heads(
    X_as_train_r, X_as_val_r, y_sub_train, y_sub_val,
    as_names, top_k_heads=16, metric="auroc")
top_indices = head_sel["selected_feature_indices"]
print(f"  Selected {len(head_sel['selected_heads'])} heads, {len(top_indices)} features")
for h in head_sel["selected_heads"][:5]:
    print(f"    L{h['layer']}_H{h['head']:02d}: AUROC={h['val_auroc']:.4f}")

# ---- A0-A9 Ablation ----
print("\n=== Ablation Experiments ===")
all_results = []

def run(mid, name, X_tr, X_va, X_te, y_tr, y_va, y_te, note=""):
    print(f"  [{mid}] {name}...", end=" ", flush=True)
    t0 = time.time()
    r = train_eval_classifier(X_tr, X_va, X_te, y_tr, y_va, y_te, "logistic", SEEDS)
    ts = r["test_summary"]
    print(f"Acc={ts['accuracy']['mean']:.4f} F1={ts['macro_f1']['mean']:.4f} AUROC={ts['auroc']['mean']:.4f} ({time.time()-t0:.1f}s)")
    all_results.append({
        "id": mid, "method": name, "feature_dim": X_tr.shape[1],
        "test_accuracy_mean": ts["accuracy"]["mean"], "test_accuracy_std": ts["accuracy"]["std"],
        "test_macro_f1_mean": ts["macro_f1"]["mean"], "test_macro_f1_std": ts["macro_f1"]["std"],
        "test_auroc_mean": ts["auroc"]["mean"], "test_auroc_std": ts["auroc"]["std"],
        "note": note,
    })

run("A0", "Hidden-only (L17 last, LR, FULL)", X_h_train, X_h_val, X_h_test,
    y_train_full, y_val_full, y_test_full, "Full 5047 samples")
run("A0s", "Hidden-only (L17 last, LR, subset)", X_h_sub_train, X_h_sub_val, X_h_sub_test,
    y_sub_train, y_sub_val, y_sub_test, f"Subset {n_sub} samples")
run("A1", "Attn-score only (raw)", X_as_train, X_as_val, X_as_test,
    y_sub_train, y_sub_val, y_sub_test)
run("A2", "Attn-score only (debiased)", X_as_train_r, X_as_val_r, X_as_test_r,
    y_sub_train, y_sub_val, y_sub_test, "Length-residualized")
if top_indices:
    run("A3", "Attn-score (top-16 heads)", 
        X_as_train_r[:, top_indices], X_as_val_r[:, top_indices], X_as_test_r[:, top_indices],
        y_sub_train, y_sub_val, y_sub_test)
run("A4", "Attn-output only", X_ao_train, X_ao_val, X_ao_test,
    y_sub_train, y_sub_val, y_sub_test)
run("A5", "Hidden + debiased attn-score",
    np.concatenate([X_h_sub_train, X_as_train_r], axis=1),
    np.concatenate([X_h_sub_val, X_as_val_r], axis=1),
    np.concatenate([X_h_sub_test, X_as_test_r], axis=1),
    y_sub_train, y_sub_val, y_sub_test)
if top_indices:
    run("A6", "Hidden + top-16 head attn",
        np.concatenate([X_h_sub_train, X_as_train_r[:, top_indices]], axis=1),
        np.concatenate([X_h_sub_val, X_as_val_r[:, top_indices]], axis=1),
        np.concatenate([X_h_sub_test, X_as_test_r[:, top_indices]], axis=1),
        y_sub_train, y_sub_val, y_sub_test)
run("A7", "Hidden + attn-output",
    np.concatenate([X_h_sub_train, X_ao_train], axis=1),
    np.concatenate([X_h_sub_val, X_ao_val], axis=1),
    np.concatenate([X_h_sub_test, X_ao_test], axis=1),
    y_sub_train, y_sub_val, y_sub_test)
if top_indices:
    run("A8", "Hidden + top-head + output",
        np.concatenate([X_h_sub_train, X_as_train_r[:, top_indices], X_ao_train], axis=1),
        np.concatenate([X_h_sub_val, X_as_val_r[:, top_indices], X_ao_val], axis=1),
        np.concatenate([X_h_sub_test, X_as_test_r[:, top_indices], X_ao_test], axis=1),
        y_sub_train, y_sub_val, y_sub_test, "Full fusion")

# Gated Fusion
print("  [A9] Gated Fusion...", end=" ", flush=True)
def get_probs(X_tr, y_tr, X_ev, seed=42):
    set_global_seed(seed)
    sc = StandardScaler(); Xt = sc.fit_transform(X_tr); Xe = sc.transform(X_ev)
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, n_jobs=-1)
    clf.fit(Xt, y_tr); return clf.predict_proba(Xe)[:, 1]

h_val_p = get_probs(X_h_sub_train, y_sub_train, X_h_sub_val)
h_test_p = get_probs(X_h_sub_train, y_sub_train, X_h_sub_test)
X_a7_tr = np.concatenate([X_h_sub_train, X_ao_train], axis=1)
X_a7_va = np.concatenate([X_h_sub_val, X_ao_val], axis=1)
X_a7_te = np.concatenate([X_h_sub_test, X_ao_test], axis=1)
f_val_p = get_probs(X_a7_tr, y_sub_train, X_a7_va)
f_test_p = get_probs(X_a7_tr, y_sub_train, X_a7_te)

tau_res = select_gated_fusion_tau(h_val_p, f_val_p, y_sub_val)
bt = tau_res["best_tau"]
gf = apply_gated_fusion(h_test_p, f_test_p, bt, y_sub_test)
tm = gf["metrics"]
print(f"tau={bt:.2f} Acc={tm['accuracy']:.4f} F1={tm['macro_f1']:.4f} AUROC={tm['auroc']:.4f} changed={gf['n_samples_changed']}")
all_results.append({
    "id": "A9", "method": f"Gated Fusion (tau={bt:.2f})", "feature_dim": "-",
    "test_accuracy_mean": tm["accuracy"], "test_accuracy_std": 0.0,
    "test_macro_f1_mean": tm["macro_f1"], "test_macro_f1_std": 0.0,
    "test_auroc_mean": tm["auroc"], "test_auroc_std": 0.0,
    "note": f"Changed {gf['n_samples_changed']} samples",
})

# Error analysis
h_preds = (h_test_p >= 0.5).astype(np.int64)
f_preds = (gf["fused_probs"] >= 0.5).astype(np.int64)
error_rows = build_error_analysis(
    [f"s{i}" for i in range(len(y_sub_test))], y_sub_test, h_test_p, gf["fused_probs"], h_preds, f_preds)
n00 = sum(1 for r in error_rows if "correct_fusion_correct" in r["case_type"])
n01 = sum(1 for r in error_rows if "correct_fusion_wrong" in r["case_type"])
n10 = sum(1 for r in error_rows if "wrong_fusion_correct" in r["case_type"])
n11 = sum(1 for r in error_rows if "wrong_fusion_wrong" in r["case_type"])

# ---- Save ----
print("\n=== Saving Results ===")
# CSV
with open(OUTPUT_DIR / "phase4_main_results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=all_results[0].keys())
    w.writeheader(); w.writerows(all_results)

# JSON
with open(OUTPUT_DIR / "phase4_ablation_results.json", "w") as f:
    json.dump({
        "hidden_baseline": hb,
        "head_selection": {k: v for k, v in head_sel.items() if k != "all_head_scores"},
        "ablation": all_results,
        "correction_matrix": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
    }, f, indent=2, ensure_ascii=False, default=str)

# Feature summaries
summarize_feature_differences(X_as_train_r, y_sub_train, as_names,
    output_csv=str(OUTPUT_DIR / "attention_score_feature_summary.csv"))
summarize_feature_differences(X_ao_train, y_sub_train, ao_names,
    output_csv=str(OUTPUT_DIR / "attention_output_feature_summary.csv"))

# ---- Final Table ----
print("\n" + "=" * 95)
print("                     Phase 4 完整消融实验结果 (A0-A9)")
print("=" * 95)
print(f"{'ID':<6} {'Method':<42} {'Dim':<8} {'Test Acc':<16} {'Test F1':<16} {'Test AUROC':<14}")
print("-" * 95)
for r in all_results:
    print(f"{r['id']:<6} {r['method']:<42} {str(r['feature_dim']):<8} "
          f"{r['test_accuracy_mean']:.4f} +/- {r['test_accuracy_std']:.4f}   "
          f"{r['test_macro_f1_mean']:.4f} +/- {r['test_macro_f1_std']:.4f}   "
          f"{r['test_auroc_mean']:.4f} +/- {r['test_auroc_std']:.4f}")
print("-" * 95)
print(f"\nCorrection Matrix: H+F+={n00} | H+F-={n01} | H-F+={n10} | H-F-={n11}")
print(f"Net correction: {n10 - n01:+d} samples")
print(f"\nSaved to: {OUTPUT_DIR}")
