"""快速运行 hidden baseline 并打印结果。"""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.feature_cache import load_npz_cache
from src.methods.phase4_attention import run_hidden_baseline

cache = Path("experiments/results/phase4/cache")
h_train = load_npz_cache(cache / "hidden_layer17_last_train.npz")
h_val = load_npz_cache(cache / "hidden_layer17_last_val.npz")
h_test = load_npz_cache(cache / "hidden_layer17_last_test.npz")

result = run_hidden_baseline(
    h_train["features"], h_val["features"], h_test["features"],
    h_train["labels"], h_val["labels"], h_test["labels"],
    classifier_type="logistic", seeds=(42, 123, 2024),
    hidden_dim=h_train["features"].shape[1],
)
ts = result["test"]
vs = result["val"]
print(f"Hidden Baseline (L17 last, LR):")
print(f"  Val:  Acc={vs['accuracy']['mean']:.4f}+/-{vs['accuracy']['std']:.4f}  F1={vs['macro_f1']['mean']:.4f}+/-{vs['macro_f1']['std']:.4f}  AUROC={vs['auroc']['mean']:.4f}+/-{vs['auroc']['std']:.4f}")
print(f"  Test: Acc={ts['accuracy']['mean']:.4f}+/-{ts['accuracy']['std']:.4f}  F1={ts['macro_f1']['mean']:.4f}+/-{ts['macro_f1']['std']:.4f}  AUROC={ts['auroc']['mean']:.4f}+/-{ts['auroc']['std']:.4f}")

with open(Path("experiments/results/phase4") / "hidden_baseline.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
print("Saved to hidden_baseline.json")
