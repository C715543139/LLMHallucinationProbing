"""
Phase 4 一键评估：提取特征 + 运行所有消融实验 + 生成对比表。

用法:
    conda activate llm_hallucination
    .\.venv\Scripts\activate.ps1
    python -s scripts/run_phase4_full.py
"""
from __future__ import annotations

import json, csv, sys, time, logging
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("phase4_full")

# ---------- config ----------
SEEDS = (42, 123, 2024)
CANDIDATE_LAYERS = [13, 14, 15, 16, 17, 18, 19, 20]
SUBSET_SIZES = {"train": 600, "val": 150, "test": 150}
OUTPUT_DIR = Path("experiments/results/phase4")
CACHE_DIR = OUTPUT_DIR / "cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- imports ----------
from src.config import config
from src.models.loader import load_model_fp16, print_device_info
from src.data.preprocessing import load_processed_data
from src.data.dataset import TrueFalseDataset
from src.features.hidden_states import extract_hidden_states_dataset
from src.features.attention_scores import extract_attention_score_features_dataset
from src.features.attention_outputs import extract_attention_output_features_dataset
from src.methods.phase4_attention import (
    run_hidden_baseline, train_eval_classifier, residualize_by_length,
    select_top_heads, summarize_feature_differences,
    gated_fusion_probs, select_gated_fusion_tau, apply_gated_fusion,
)
from src.utils.feature_cache import save_npz_cache, load_npz_cache
from src.utils.reproducibility import collect_runtime_info, set_global_seed
from src.utils.metrics import compute_metrics

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------- 1. Load model + data ----------
print("=" * 60)
print("  Phase 4 Full Evaluation")
print("=" * 60)
print_device_info()

t0 = time.time()
model, tokenizer = load_model_fp16(
    model_path=str(config.paths.models_cache / "Qwen2-1.5B"),
)
# Set eager attention AFTER loading to avoid NaN issues
model.set_attn_implementation("eager")
print(f"Model loaded (eager attention) in {time.time()-t0:.0f}s")

train_ds, val_ds, test_ds = load_processed_data()
print(train_ds.summary())

runtime_info = collect_runtime_info(model)

# ---------- 2. Extract hidden states (FULL) ----------
print("\n[Step 1/5] Extracting FULL hidden states (layer=17, last)...")
t0 = time.time()

X_h_train, y_train_full = extract_hidden_states_dataset(
    model, tokenizer, train_ds, pooling="last", layers=[17], batch_size=8, max_length=128)
X_h_val, y_val_full = extract_hidden_states_dataset(
    model, tokenizer, val_ds, pooling="last", layers=[17], batch_size=8, max_length=128)
X_h_test, y_test_full = extract_hidden_states_dataset(
    model, tokenizer, test_ds, pooling="last", layers=[17], batch_size=8, max_length=128)

# Check for NaN
for name, arr in [("train", X_h_train), ("val", X_h_val), ("test", X_h_test)]:
    nan_count = np.isnan(arr).sum()
    if nan_count > 0:
        logger.warning(f"  {name}: {nan_count} NaN values found, replacing with 0")
        arr = np.nan_to_num(arr, nan=0.0)
        if name == "train": X_h_train = arr
        elif name == "val": X_h_val = arr
        else: X_h_test = arr

print(f"  Hidden features: train={X_h_train.shape}, val={X_h_val.shape}, test={X_h_test.shape}")
print(f"  Time: {time.time()-t0:.0f}s")

# Save clean cache
save_npz_cache(CACHE_DIR / "hidden_layer17_last_train.npz", X_h_train, y_train_full)
save_npz_cache(CACHE_DIR / "hidden_layer17_last_val.npz", X_h_val, y_val_full)
save_npz_cache(CACHE_DIR / "hidden_layer17_last_test.npz", X_h_test, y_test_full)
print("  Cached.")

# ---------- 3. Hidden Baseline ----------
print("\n[Step 2/5] Hidden Baseline (L17 last, LR)...")
hidden_baseline = run_hidden_baseline(
    X_h_train, X_h_val, X_h_test,
    y_train_full, y_val_full, y_test_full,
    classifier_type="logistic", seeds=SEEDS, hidden_dim=X_h_train.shape[1],
)
ts = hidden_baseline["test"]
vs = hidden_baseline["val"]
print(f"  Val:  Acc={vs['accuracy']['mean']:.4f} F1={vs['macro_f1']['mean']:.4f} AUROC={vs['auroc']['mean']:.4f}")
print(f"  Test: Acc={ts['accuracy']['mean']:.4f} F1={ts['macro_f1']['mean']:.4f} AUROC={ts['auroc']['mean']:.4f}")

with open(OUTPUT_DIR / "hidden_baseline.json", "w") as f:
    json.dump(hidden_baseline, f, indent=2, ensure_ascii=False, default=str)

# ---------- 4. Extract attention features on SUBSET ----------
print(f"\n[Step 3/5] Extracting attention features on subset ({SUBSET_SIZES})...")
t0 = time.time()

def subset(ds, n):
    n = min(n, len(ds))
    return TrueFalseDataset(ds.statements[:n], ds.labels[:n], ds.domains[:n] if ds.domains else None)

train_sub = subset(train_ds, SUBSET_SIZES["train"])
val_sub = subset(val_ds, SUBSET_SIZES["val"])
test_sub = subset(test_ds, SUBSET_SIZES["test"])
print(f"  Subset: {len(train_sub)}/{len(val_sub)}/{len(test_sub)}")

# Attention scores
print("  Extracting attention scores...")
as_train = extract_attention_score_features_dataset(
    model, tokenizer, train_sub, layers=CANDIDATE_LAYERS, batch_size=1,
    output_path=str(CACHE_DIR / "attention_scores_train.npz"))
as_val = extract_attention_score_features_dataset(
    model, tokenizer, val_sub, layers=CANDIDATE_LAYERS, batch_size=1,
    output_path=str(CACHE_DIR / "attention_scores_val.npz"))
as_test = extract_attention_score_features_dataset(
    model, tokenizer, test_sub, layers=CANDIDATE_LAYERS, batch_size=1,
    output_path=str(CACHE_DIR / "attention_scores_test.npz"))

X_as_train = np.nan_to_num(as_train["features"], nan=0.0)
X_as_val = np.nan_to_num(as_val["features"], nan=0.0)
X_as_test = np.nan_to_num(as_test["features"], nan=0.0)
as_names = as_train["feature_names"]
y_sub_train = as_train["labels"]
y_sub_val = as_val["labels"]
y_sub_test = as_test["labels"]

# Align hidden subset
X_h_sub_train = X_h_train[:len(train_sub)]
X_h_sub_val = X_h_val[:len(val_sub)]
X_h_sub_test = X_h_test[:len(test_sub)]

print(f"  Attention scores: {X_as_train.shape}")

# Attention outputs
print("  Extracting attention outputs...")
ao_train = extract_attention_output_features_dataset(
    model, tokenizer, train_sub, layers=CANDIDATE_LAYERS, batch_size=1,
    output_path=str(CACHE_DIR / "attention_outputs_train.npz"))
ao_val = extract_attention_output_features_dataset(
    model, tokenizer, val_sub, layers=CANDIDATE_LAYERS, batch_size=1,
    output_path=str(CACHE_DIR / "attention_outputs_val.npz"))
ao_test = extract_attention_output_features_dataset(
    model, tokenizer, test_sub, layers=CANDIDATE_LAYERS, batch_size=1,
    output_path=str(CACHE_DIR / "attention_outputs_test.npz"))

X_ao_train = np.nan_to_num(ao_train["features"], nan=0.0)
X_ao_val = np.nan_to_num(ao_val["features"], nan=0.0)
X_ao_test = np.nan_to_num(ao_test["features"], nan=0.0)
ao_names = ao_train["feature_names"]

print(f"  Attention outputs: {X_ao_train.shape}")
print(f"  Total extraction time: {time.time()-t0:.0f}s")

# ---------- 5. Length residualization ----------
print("\n[Step 4/5] Length residualization + Head selection...")

# Use attention sink as length proxy
sink_cols = [i for i, n in enumerate(as_names) if "attention_sink_mass" in n]
n_sub = len(train_sub)
len_tr = np.concatenate([X_as_train[:, sink_cols].mean(axis=1, keepdims=True), np.zeros((n_sub, 5))], axis=1) if sink_cols else np.ones((n_sub, 6))
len_va = np.concatenate([X_as_val[:, sink_cols].mean(axis=1, keepdims=True), np.zeros((len(val_sub), 5))], axis=1) if sink_cols else np.ones((len(val_sub), 6))
len_te = np.concatenate([X_as_test[:, sink_cols].mean(axis=1, keepdims=True), np.zeros((len(test_sub), 5))], axis=1) if sink_cols else np.ones((len(test_sub), 6))

X_as_train_r, X_as_val_r, X_as_test_r, resid_meta = residualize_by_length(
    X_as_train, X_as_val, X_as_test, len_tr, len_va, len_te)
print(f"  Residualization: corr before={resid_meta['correlation_before']:.4f}, after={resid_meta['correlation_after']:.4f}")

# Head selection
head_sel = select_top_heads(
    X_as_train_r, X_as_val_r, y_sub_train, y_sub_val,
    as_names, top_k_heads=16, metric="auroc")
top_indices = head_sel["selected_feature_indices"]
print(f"  Selected {len(head_sel['selected_heads'])} heads, {len(top_indices)} features")

sel_save = {k: v for k, v in head_sel.items() if k != "all_head_scores"}
sel_save["all_head_scores"] = [{k: v for k, v in h.items() if k != "feature_indices"} for h in head_sel["all_head_scores"]]
with open(OUTPUT_DIR / "attention_head_selection.json", "w") as f:
    json.dump(sel_save, f, indent=2, ensure_ascii=False, default=str)

# ---------- 6. Ablation ----------
print("\n[Step 5/5] Running ablation experiments...")

all_results = []

def run(mid, name, X_tr, X_va, X_te, y_tr, y_va, y_te, note=""):
    print(f"  [{mid}] {name}...", end=" ", flush=True)
    r = train_eval_classifier(X_tr, X_va, X_te, y_tr, y_va, y_te, "logistic", SEEDS)
    ts = r["test_summary"]
    print(f"Acc={ts['accuracy']['mean']:.4f} F1={ts['macro_f1']['mean']:.4f} AUROC={ts['auroc']['mean']:.4f}")
    all_results.append({
        "id": mid, "method": name, "feature_dim": X_tr.shape[1],
        "test_accuracy_mean": ts["accuracy"]["mean"], "test_accuracy_std": ts["accuracy"]["std"],
        "test_macro_f1_mean": ts["macro_f1"]["mean"], "test_macro_f1_std": ts["macro_f1"]["std"],
        "test_auroc_mean": ts["auroc"]["mean"], "test_auroc_std": ts["auroc"]["std"],
        "note": note,
    })
    return r

# A0: hidden-only (full)
run("A0", "Hidden-only (L17 last, LR)", X_h_train, X_h_val, X_h_test,
    y_train_full, y_val_full, y_test_full, "Full dataset baseline")

# A0s: hidden-only (subset)
run("A0s", "Hidden-only (subset, LR)", X_h_sub_train, X_h_sub_val, X_h_sub_test,
    y_sub_train, y_sub_val, y_sub_test, f"Subset {SUBSET_SIZES}")

# A1: attention score raw
run("A1", "Attn-score only (raw)", X_as_train, X_as_val, X_as_test,
    y_sub_train, y_sub_val, y_sub_test)

# A2: attention score debiased
run("A2", "Attn-score only (debiased)", X_as_train_r, X_as_val_r, X_as_test_r,
    y_sub_train, y_sub_val, y_sub_test, "Length-residualized")

# A3: top-head attention only
if top_indices:
    run("A3", "Attn-score (top-16 heads)", X_as_train_r[:, top_indices], X_as_val_r[:, top_indices], X_as_test_r[:, top_indices],
        y_sub_train, y_sub_val, y_sub_test, f"{len(top_indices)} features")

# A4: attention output only
run("A4", "Attn-output only", X_ao_train, X_ao_val, X_ao_test,
    y_sub_train, y_sub_val, y_sub_test)

# A5: hidden + debiased attention
run("A5", "Hidden + debiased attn-score",
    np.concatenate([X_h_sub_train, X_as_train_r], axis=1),
    np.concatenate([X_h_sub_val, X_as_val_r], axis=1),
    np.concatenate([X_h_sub_test, X_as_test_r], axis=1),
    y_sub_train, y_sub_val, y_sub_test)

# A6: hidden + top-head attention
if top_indices:
    run("A6", "Hidden + top-16 head attn",
        np.concatenate([X_h_sub_train, X_as_train_r[:, top_indices]], axis=1),
        np.concatenate([X_h_sub_val, X_as_val_r[:, top_indices]], axis=1),
        np.concatenate([X_h_sub_test, X_as_test_r[:, top_indices]], axis=1),
        y_sub_train, y_sub_val, y_sub_test)

# A7: hidden + attention output
run("A7", "Hidden + attn-output",
    np.concatenate([X_h_sub_train, X_ao_train], axis=1),
    np.concatenate([X_h_sub_val, X_ao_val], axis=1),
    np.concatenate([X_h_sub_test, X_ao_test], axis=1),
    y_sub_train, y_sub_val, y_sub_test)

# A8: hidden + all
if top_indices:
    run("A8", "Hidden + top-head + output",
        np.concatenate([X_h_sub_train, X_as_train_r[:, top_indices], X_ao_train], axis=1),
        np.concatenate([X_h_sub_val, X_as_val_r[:, top_indices], X_ao_val], axis=1),
        np.concatenate([X_h_sub_test, X_as_test_r[:, top_indices], X_ao_test], axis=1),
        y_sub_train, y_sub_val, y_sub_test, "Full fusion")

# A9: Gated fusion (hidden + AO)
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
    "note": f"Samples changed: {gf['n_samples_changed']}",
})

# ---------- 7. Feature summaries ----------
print("\nGenerating feature summaries...")
summarize_feature_differences(X_as_train_r, y_sub_train, as_names,
    output_csv=str(OUTPUT_DIR / "attention_score_feature_summary.csv"))
summarize_feature_differences(X_ao_train, y_sub_train, ao_names,
    output_csv=str(OUTPUT_DIR / "attention_output_feature_summary.csv"))

# ---------- 8. Save results ----------
# Save CSV
csv_path = OUTPUT_DIR / "phase4_main_results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)

# Save JSON
with open(OUTPUT_DIR / "phase4_ablation_results.json", "w") as f:
    json.dump({
        "hidden_baseline": hidden_baseline,
        "head_selection": sel_save,
        "ablation": all_results,
        "runtime": runtime_info,
    }, f, indent=2, ensure_ascii=False, default=str)

# ---------- 9. Print final table ----------
print("\n" + "=" * 95)
print("                        Phase 4 消融实验结果汇总")
print("=" * 95)
print(f"{'ID':<6} {'Method':<40} {'Dim':<8} {'Test Acc':<16} {'Test F1':<16} {'Test AUROC':<14}")
print("-" * 95)
for r in all_results:
    print(f"{r['id']:<6} {r['method']:<40} {str(r['feature_dim']):<8} "
          f"{r['test_accuracy_mean']:.4f} +/- {r['test_accuracy_std']:.4f}   "
          f"{r['test_macro_f1_mean']:.4f} +/- {r['test_macro_f1_std']:.4f}   "
          f"{r['test_auroc_mean']:.4f} +/- {r['test_auroc_std']:.4f}")
print("-" * 95)
print(f"\nResults saved to: {OUTPUT_DIR}")
print("=" * 95)
