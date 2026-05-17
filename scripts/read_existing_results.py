"""读取已有 Phase 2/3 实验结果并打印。"""
import json
from pathlib import Path

results_dir = Path("experiments/results")

print("=== Phase 2: PPL ===")
with open(results_dir / "baseline/ppl_results.json") as f:
    ppl = json.load(f)
print(f"  Test Acc: {ppl['test']['accuracy']:.4f}")
print(f"  Test F1: {ppl['test']['macro_f1']:.4f}")
print(f"  Test AUROC: {ppl['test']['auroc']:.4f}")

print()
print("=== Phase 2: SAPLMA Logistic (Layer 27 last) ===")
with open(results_dir / "baseline/saplma_logistic_results.json") as f:
    sap_lr = json.load(f)
ts = sap_lr["test_summary"]
print(f"  Test Acc: {ts['accuracy']['mean']:.4f} +/- {ts['accuracy']['std']:.4f}")
print(f"  Test F1: {ts['macro_f1']['mean']:.4f} +/- {ts['macro_f1']['std']:.4f}")
print(f"  Test AUROC: {ts['auroc']['mean']:.4f} +/- {ts['auroc']['std']:.4f}")

print()
print("=== Phase 2: SAPLMA MLP (Layer 27 last) ===")
with open(results_dir / "baseline/saplma_mlp_results_rerun_best.json") as f:
    sap_mlp = json.load(f)
ts = sap_mlp["test_summary"]
print(f"  Test Acc: {ts['accuracy']['mean']:.4f} +/- {ts['accuracy']['std']:.4f}")
print(f"  Test F1: {ts['macro_f1']['mean']:.4f} +/- {ts['macro_f1']['std']:.4f}")
print(f"  Test AUROC: {ts['auroc']['mean']:.4f} +/- {ts['auroc']['std']:.4f}")

print()
print("=== Phase 3: Layer Analysis (Best layer) ===")
with open(results_dir / "analysis/layer_analysis_logistic_last.json") as f:
    la = json.load(f)
best = la["best_layer"]
print(f"  Best layer: {best['layer_idx']}")
ts = best["test_summary"]
print(f"  Test Acc: {ts['accuracy']['mean']:.4f} +/- {ts['accuracy']['std']:.4f}")
print(f"  Test F1: {ts['macro_f1']['mean']:.4f} +/- {ts['macro_f1']['std']:.4f}")
print(f"  Test AUROC: {ts['auroc']['mean']:.4f} +/- {ts['auroc']['std']:.4f}")

print()
print("=== Phase 3: Token Analysis (Best pooling=L17) ===")
with open(results_dir / "analysis/token_analysis_logistic_last_layer.json") as f:
    ta = json.load(f)
best = ta["best_pooling"]
print(f"  Best pooling: {best['pooling']}")
ts = best["test_summary"]
print(f"  Test Acc: {ts['accuracy']['mean']:.4f} +/- {ts['accuracy']['std']:.4f}")
print(f"  Test F1: {ts['macro_f1']['mean']:.4f} +/- {ts['macro_f1']['std']:.4f}")
print(f"  Test AUROC: {ts['auroc']['mean']:.4f} +/- {ts['auroc']['std']:.4f}")

# Also print layer 17 results
print()
print("=== Phase 3: Layer 17 specifically ===")
for layer_result in la["per_layer"]:
    if layer_result["layer_idx"] == 17:
        ts = layer_result["test_summary"]
        print(f"  Layer 17: Acc={ts['accuracy']['mean']:.4f}+/-{ts['accuracy']['std']:.4f}  F1={ts['macro_f1']['mean']:.4f}+/-{ts['macro_f1']['std']:.4f}  AUROC={ts['auroc']['mean']:.4f}+/-{ts['auroc']['std']:.4f}")
        break
