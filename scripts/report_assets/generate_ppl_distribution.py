"""
Phase 5 报告资产: 生成 PPL 逐样本分数与分布图。

用法:
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llm_hallucination
    source ./.venv/bin/activate
    python -s scripts/report_assets/generate_ppl_distribution.py
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ppl_distribution")

from src.config import config
from src.data.preprocessing import load_processed_data
from src.models.loader import load_model_fp16
from src.methods.probability import compute_ppl_scores
from src.utils.reproducibility import set_global_seed


def main():
    set_global_seed(42)
    t_start = time.time()

    # ---- 1. 加载模型 (bfloat16 + eager) ----
    logger.info("加载模型 (bfloat16)...")
    model, tokenizer = load_model_fp16(torch_dtype="bfloat16")
    # 设置 eager attention（Phase 4 稳定配置）
    try:
        model.config._attn_implementation = "eager"
    except Exception:
        pass
    model.eval()
    logger.info("模型加载完成，dtype: %s", next(model.parameters()).dtype)

    # ---- 2. 加载数据 ----
    logger.info("加载预处理数据...")
    train_ds, val_ds, test_ds = load_processed_data()
    logger.info("Train: %d, Val: %d, Test: %d", len(train_ds), len(val_ds), len(test_ds))

    # ---- 3. 计算逐样本 PPL ----
    out_dir = config.paths.results_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for split_name, dataset in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        logger.info("计算 %s 集 PPL (%d 样本)...", split_name, len(dataset))
        ppl_scores = compute_ppl_scores(
            model, tokenizer,
            dataset.statements,
            batch_size=1,
            max_length=128,
        )
        for i in range(len(dataset)):
            all_rows.append({
                "split": split_name,
                "statement": dataset.statements[i],
                "label": int(dataset.labels[i]),
                "ppl_score": float(ppl_scores[i]),
            })

    # ---- 4. 保存 CSV ----
    csv_path = out_dir / "ppl_score_distribution.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "statement", "label", "ppl_score"])
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info("PPL 逐样本分数已保存至 %s", csv_path)

    # ---- 5. 绘制分布图 ----
    test_rows = [r for r in all_rows if r["split"] == "test"]
    true_ppls = [r["ppl_score"] for r in test_rows if r["label"] == 1]
    false_ppls = [r["ppl_score"] for r in test_rows if r["label"] == 0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- 直方图 ----
    ax = axes[0]
    bins = np.linspace(0, np.percentile([r["ppl_score"] for r in test_rows], 99), 40)
    ax.hist(true_ppls, bins=bins, alpha=0.6, label=f"True (n={len(true_ppls)})", color="#2ca02c", edgecolor="white")
    ax.hist(false_ppls, bins=bins, alpha=0.6, label=f"False (n={len(false_ppls)})", color="#d62728", edgecolor="white")
    ax.axvline(x=232.43, color="black", linestyle="--", linewidth=1.5, label=f"Threshold=232.43")
    ax.set_xlabel("PPL Score")
    ax.set_ylabel("Count")
    ax.set_title("PPL Distribution: True vs False (Test, Histogram)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # ---- KDE ----
    ax = axes[1]
    from scipy.stats import gaussian_kde
    for ppls, label, color in [(true_ppls, "True", "#2ca02c"), (false_ppls, "False", "#d62728")]:
        ppls_clean = [p for p in ppls if p > 0 and not np.isinf(p) and not np.isnan(p)]
        if len(ppls_clean) > 3:
            clipped = np.clip(ppls_clean, np.percentile(ppls_clean, 0.5), np.percentile(ppls_clean, 99.5))
            kde = gaussian_kde(clipped)
            x_range = np.linspace(min(clipped), max(clipped), 200)
            ax.plot(x_range, kde(x_range), color=color, linewidth=2, label=f"{label} (n={len(ppls_clean)})")
            ax.fill_between(x_range, kde(x_range), alpha=0.15, color=color)
    ax.axvline(x=232.43, color="black", linestyle="--", linewidth=1.5, label="Threshold=232.43")
    ax.set_xlabel("PPL Score")
    ax.set_ylabel("Density")
    ax.set_title("PPL Distribution: True vs False (Test, KDE)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    png_path = out_dir / "ppl_score_distribution.png"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("PPL 分布图已保存至 %s", png_path)

    # ---- 6. 统计汇总 ----
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    test_labels = np.array([r["label"] for r in test_rows])
    test_ppls_arr = np.array([r["ppl_score"] for r in test_rows])
    # PPL 越低越倾向 true，所以取负号
    auroc = roc_auc_score(test_labels, -test_ppls_arr)
    # 使用阈值 232.43
    preds = (test_ppls_arr <= 232.43).astype(int)
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="macro")

    logger.info("Test PPL Stats: mean_true=%.2f, mean_false=%.2f, AUROC=%.4f, Acc=%.4f, F1=%.4f",
                np.mean(true_ppls), np.mean(false_ppls), auroc, acc, f1)

    elapsed = time.time() - t_start
    logger.info("PPL 分布生成完成! 耗时: %.0fs (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
