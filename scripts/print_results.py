"""Print Phase 4 ablation results."""
import json
d = json.load(open("experiments/results/phase4/phase4_ablation_results.json"))
abl = d.get("ablation", [])

print("ID    | Method                                        | Dim     | Test Acc          | Test F1           | Test AUROC")
print("-" * 105)
for r in abl:
    print(f"{r['id']:<6}| {r['method']:<46}| {str(r['feature_dim']):<8}| {r['test_accuracy_mean']:.4f} +/- {r['test_accuracy_std']:.4f}  | {r['test_macro_f1_mean']:.4f} +/- {r['test_macro_f1_std']:.4f}  | {r['test_auroc_mean']:.4f} +/- {r['test_auroc_std']:.4f}")

cm = d.get("correction_matrix", {})
n00, n01, n10, n11 = cm.get("n00", 0), cm.get("n01", 0), cm.get("n10", 0), cm.get("n11", 0)
print(f"\nCorrection Matrix: H+F+={n00} | H+F-={n01} | H-F+={n10} | H-F-={n11}")
print(f"Net correction: {n10 - n01:+d} samples")
