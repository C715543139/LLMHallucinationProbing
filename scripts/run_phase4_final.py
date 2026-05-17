"""Phase 4 评估：SDPA模式（跳过attention特征）+ 完整对比表生成。

用法:
    conda activate llm_hallucination
    .\.venv\Scripts\activate.ps1
    python -s scripts/run_phase4_final.py
"""
from __future__ import annotations
import json, csv, sys, time, logging
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

from src.config import config
from src.models.loader import load_model_fp16, print_device_info
from src.data.preprocessing import load_processed_data
from src.features.hidden_states import extract_hidden_states_dataset
from src.methods.phase4_attention import run_hidden_baseline, train_eval_classifier
from src.utils.reproducibility import collect_runtime_info, set_global_seed
from src.utils.feature_cache import save_npz_cache

OUTPUT_DIR = Path("experiments/results/phase4")
CACHE_DIR = OUTPUT_DIR / "cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 123, 2024)

# ====== 1. Load model (SDPA - stable) ======
print("=" * 60)
print("  Phase 4 Evaluation (SDPA mode)")
print("=" * 60)
print_device_info()

t0 = time.time()
model, tokenizer = load_model_fp16(model_path=str(config.paths.models_cache / "Qwen2-1.5B"))
print(f"Model loaded in {time.time()-t0:.0f}s")
print(f"Attention implementation: {getattr(model.config, 'attn_implementation', 'default')}")

train_ds, val_ds, test_ds = load_processed_data()
print(train_ds.summary())
runtime_info = collect_runtime_info(model)

# ====== 2. Extract hidden states ======
print("\n[1/3] Extracting hidden states (layer=17, last token)...")
t0 = time.time()

X_h_train, y_train = extract_hidden_states_dataset(
    model, tokenizer, train_ds, pooling="last", layers=[17], batch_size=8, max_length=128)
X_h_val, y_val = extract_hidden_states_dataset(
    model, tokenizer, val_ds, pooling="last", layers=[17], batch_size=8, max_length=128)
X_h_test, y_test = extract_hidden_states_dataset(
    model, tokenizer, test_ds, pooling="last", layers=[17], batch_size=8, max_length=128)

for name, arr in [("train", X_h_train), ("val", X_h_val), ("test", X_h_test)]:
    nc = np.isnan(arr).sum()
    if nc > 0:
        print(f"  WARNING: {name} has {nc} NaN, replacing with 0")
        arr[np.isnan(arr)] = 0.0
        if name == "train": X_h_train = arr
        elif name == "val": X_h_val = arr
        else: X_h_test = arr

print(f"  Features: train={X_h_train.shape}, val={X_h_val.shape}, test={X_h_test.shape}")
print(f"  Time: {time.time()-t0:.0f}s")

# Cache
save_npz_cache(CACHE_DIR / "hidden_layer17_last_train.npz", X_h_train, y_train)
save_npz_cache(CACHE_DIR / "hidden_layer17_last_val.npz", X_h_val, y_val)
save_npz_cache(CACHE_DIR / "hidden_layer17_last_test.npz", X_h_test, y_test)

# ====== 3. Hidden Baseline ======
print("\n[2/3] Hidden Baseline (L17 last, LR, 3 seeds)...")
hidden_baseline = run_hidden_baseline(
    X_h_train, X_h_val, X_h_test, y_train, y_val, y_test,
    classifier_type="logistic", seeds=SEEDS, hidden_dim=X_h_train.shape[1])
ts = hidden_baseline["test"]
vs = hidden_baseline["val"]
print(f"  Val:  Acc={vs['accuracy']['mean']:.4f} F1={vs['macro_f1']['mean']:.4f} AUROC={vs['auroc']['mean']:.4f}")
print(f"  Test: Acc={ts['accuracy']['mean']:.4f} F1={ts['macro_f1']['mean']:.4f} AUROC={ts['auroc']['mean']:.4f}")

with open(OUTPUT_DIR / "hidden_baseline.json", "w") as f:
    json.dump(hidden_baseline, f, indent=2, ensure_ascii=False, default=str)

# ====== 4. Also run MLP ======
print("\n[3/3] Hidden Baseline (L17 last, MLP, 3 seeds)...")
mlp_result = run_hidden_baseline(
    X_h_train, X_h_val, X_h_test, y_train, y_val, y_test,
    classifier_type="mlp", seeds=SEEDS, hidden_dim=X_h_train.shape[1])
tsm = mlp_result["test"]
print(f"  Test: Acc={tsm['accuracy']['mean']:.4f} F1={tsm['macro_f1']['mean']:.4f} AUROC={tsm['auroc']['mean']:.4f}")

# ====== 5. Read Phase 2 & 3 existing results ======
print("\n" + "=" * 60)
print("  Compiling comparison table...")
print("=" * 60)

results_dir = Path("experiments/results")

def read_json(path):
    with open(path) as f:
        return json.load(f)

# Phase 2
ppl = read_json(results_dir / "baseline/ppl_results.json")
sap_lr = read_json(results_dir / "baseline/saplma_logistic_results.json")
sap_mlp = read_json(results_dir / "baseline/saplma_mlp_results_rerun_best.json")

# Phase 3
la = read_json(results_dir / "analysis/layer_analysis_logistic_last.json")
ta = read_json(results_dir / "analysis/token_analysis_logistic_last_layer.json")

# ====== 6. Build comparison table ======
def fmt_mean_std(d, metric):
    m = d.get("test_summary", d.get("test", {}))
    if metric in m:
        if isinstance(m[metric], dict):
            return m[metric]["mean"], m[metric]["std"]
        else:
            return m[metric], 0.0
    return float("nan"), 0.0

def fmt_single(d, metric):
    return d.get(metric, float("nan")), 0.0

rows = []

# P2: PPL
a, a_s = fmt_single(ppl["test"], "accuracy")
f, f_s = fmt_single(ppl["test"], "macro_f1")
r, r_s = fmt_single(ppl["test"], "auroc")
rows.append({"Phase": "Phase 2", "Method": "PPL (sequence probability)", "Classifier": "Threshold (F1 opt)",
             "Feature": "Perplexity", "Test Acc": f"{a:.4f}", "Test F1": f"{f:.4f}", "Test AUROC": f"{r:.4f}"})

# P2: SAPLMA LR (last layer)
a, a_s = fmt_mean_std(sap_lr, "accuracy")
f, f_s = fmt_mean_std(sap_lr, "macro_f1")
r, r_s = fmt_mean_std(sap_lr, "auroc")
rows.append({"Phase": "Phase 2", "Method": "SAPLMA (Logistic Regression)", "Classifier": "LR",
             "Feature": "L27 last hidden", "Test Acc": f"{a:.4f}±{a_s:.4f}", "Test F1": f"{f:.4f}±{f_s:.4f}", "Test AUROC": f"{r:.4f}±{r_s:.4f}"})

# P2: SAPLMA MLP (last layer)
a, a_s = fmt_mean_std(sap_mlp, "accuracy")
f, f_s = fmt_mean_std(sap_mlp, "macro_f1")
r, r_s = fmt_mean_std(sap_mlp, "auroc")
rows.append({"Phase": "Phase 2", "Method": "SAPLMA (MLP)", "Classifier": "MLP",
             "Feature": "L27 last hidden", "Test Acc": f"{a:.4f}±{a_s:.4f}", "Test F1": f"{f:.4f}±{f_s:.4f}", "Test AUROC": f"{r:.4f}±{r_s:.4f}"})

# P3: Layer Analysis (best = L17)
best_la = la["best_layer"]
a, a_s = fmt_mean_std({"test_summary": best_la["test_summary"]}, "accuracy")
f, f_s = fmt_mean_std({"test_summary": best_la["test_summary"]}, "macro_f1")
r, r_s = fmt_mean_std({"test_summary": best_la["test_summary"]}, "auroc")
rows.append({"Phase": "Phase 3", "Method": f"Layer Analysis (best L{best_la['layer_idx']})", "Classifier": "LR",
             "Feature": f"L{best_la['layer_idx']} last hidden", "Test Acc": f"{a:.4f}±{a_s:.4f}", "Test F1": f"{f:.4f}±{f_s:.4f}", "Test AUROC": f"{r:.4f}±{r_s:.4f}"})

# P3: Token Analysis (best = last)
best_ta = ta["best_pooling"]
a, a_s = fmt_mean_std({"test_summary": best_ta["test_summary"]}, "accuracy")
f, f_s = fmt_mean_std({"test_summary": best_ta["test_summary"]}, "macro_f1")
r, r_s = fmt_mean_std({"test_summary": best_ta["test_summary"]}, "auroc")
rows.append({"Phase": "Phase 3", "Method": f"Token Analysis (best: {best_ta['pooling']})", "Classifier": "LR",
             "Feature": f"L27 {best_ta['pooling']} token", "Test Acc": f"{a:.4f}±{a_s:.4f}", "Test F1": f"{f:.4f}±{f_s:.4f}", "Test AUROC": f"{r:.4f}±{r_s:.4f}"})

# P3: Layer 17 specific
for layer_r in la["per_layer"]:
    if layer_r["layer_idx"] == 17:
        a, a_s = fmt_mean_std({"test_summary": layer_r["test_summary"]}, "accuracy")
        f, f_s = fmt_mean_std({"test_summary": layer_r["test_summary"]}, "macro_f1")
        r, r_s = fmt_mean_std({"test_summary": layer_r["test_summary"]}, "auroc")
        rows.append({"Phase": "Phase 3", "Method": "Layer 17 (specific)", "Classifier": "LR",
                     "Feature": "L17 last hidden", "Test Acc": f"{a:.4f}±{a_s:.4f}", "Test F1": f"{f:.4f}±{f_s:.4f}", "Test AUROC": f"{r:.4f}±{r_s:.4f}"})
        break

# P4: Hidden Baseline (LR)
a, a_s = ts["accuracy"]["mean"], ts["accuracy"]["std"]
f, f_s = ts["macro_f1"]["mean"], ts["macro_f1"]["std"]
r, r_s = ts["auroc"]["mean"], ts["auroc"]["std"]
rows.append({"Phase": "Phase 4", "Method": "Hidden-only (L17 last, LR) ★ Baseline", "Classifier": "LR",
             "Feature": "L17 last hidden", "Test Acc": f"{a:.4f}±{a_s:.4f}", "Test F1": f"{f:.4f}±{f_s:.4f}", "Test AUROC": f"{r:.4f}±{r_s:.4f}"})

# P4: Hidden Baseline (MLP)
a, a_s = tsm["accuracy"]["mean"], tsm["accuracy"]["std"]
f, f_s = tsm["macro_f1"]["mean"], tsm["macro_f1"]["std"]
r, r_s = tsm["auroc"]["mean"], tsm["auroc"]["std"]
rows.append({"Phase": "Phase 4", "Method": "Hidden-only (L17 last, MLP) ★ Baseline", "Classifier": "MLP",
             "Feature": "L17 last hidden", "Test Acc": f"{a:.4f}±{a_s:.4f}", "Test F1": f"{f:.4f}±{f_s:.4f}", "Test AUROC": f"{r:.4f}±{r_s:.4f}"})

# P4: Attention methods (note: not available due to eager attention incompatibility)
rows.append({"Phase": "Phase 4", "Method": "Attention-Guided SAPLMA", "Classifier": "-",
             "Feature": "Attention scores + hidden", "Test Acc": "N/A", "Test F1": "N/A", "Test AUROC": "N/A"})

# ====== 7. Print and save ======
print("\n" + "=" * 110)
print("                              完整实验对比结果")
print("=" * 110)
header = f"{'Phase':<8} {'Method':<42} {'Classifier':<12} {'Feature':<22} {'Test Acc':<16} {'Test F1':<16} {'Test AUROC':<14}"
print(header)
print("-" * 110)
for row in rows:
    print(f"{row['Phase']:<8} {row['Method']:<42} {row['Classifier']:<12} {row['Feature']:<22} {row['Test Acc']:<16} {row['Test F1']:<16} {row['Test AUROC']:<14}")
print("-" * 110)

# Save CSV
csv_path = OUTPUT_DIR / "phase4_main_results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Save JSON
json_path = OUTPUT_DIR / "phase4_full_comparison.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False, default=str)

print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"  {csv_path}")
print(f"  {json_path}")
print("=" * 110)
print("\nNote: Attention-guided methods (A1-A9) skipped due to eager attention")
print("incompatibility with Qwen2-1.5B on this transformers version.")
print("The attention feature extraction architecture is implemented and ready")
print("for environments where eager attention works correctly.")
