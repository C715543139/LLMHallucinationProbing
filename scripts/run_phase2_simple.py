"""
Phase 2 实验运行脚本 (简化版 — 不使用文件日志避免潜在问题)
"""
import sys
import os
import time
import json
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

# 使用 print 代替 logging
def log(msg, *args):
    if args:
        msg = msg % args
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

t_start = time.time()

log("=" * 60)
log("Phase 2: 基础方法实现与评估")
log("=" * 60)

# ---- 加载模型和数据 -------------------------------------------------
log("正在导入模块...")
from src.models.loader import load_model, print_device_info
from src.data.preprocessing import load_processed_data
log("模块导入完成")

print_device_info()

log("加载模型 (Qwen2-1.5B FP16)...")
model, tokenizer = load_model()
log("模型设备: %s", next(model.parameters()).device)

log("加载预处理数据...")
train_ds, val_ds, test_ds = load_processed_data()
log(train_ds.summary())

# 输出目录
out_dir = PROJECT_ROOT / "experiments" / "results" / "baseline"
out_dir.mkdir(parents=True, exist_ok=True)

# ---- P2.1-P2.2: PPL 方法 -------------------------------------------
log("=" * 50)
log("P2.1-P2.2: PPL 方法")
log("=" * 50)

from src.methods.probability import evaluate_ppl_method

ppl_results = evaluate_ppl_method(
    model=model, tokenizer=tokenizer,
    train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
    batch_size=8, max_length=128, threshold_metric="f1",
)
ppl_summary = {k: v for k, v in ppl_results.items()
               if k in ("method", "threshold", "train", "val", "test")}
with open(out_dir / "ppl_results.json", "w", encoding="utf-8") as f:
    json.dump(ppl_summary, f, indent=2, ensure_ascii=False, default=float)
log("PPL 结果已保存至 %s", out_dir / "ppl_results.json")
log("测试集: Acc=%.4f, Macro-F1=%.4f, AUROC=%.4f",
     ppl_results["test"]["accuracy"],
     ppl_results["test"]["macro_f1"],
     ppl_results["test"]["auroc"])

# ---- P2.3-P2.5: SAPLMA 方法 ----------------------------------------
log("=" * 50)
log("P2.3-P2.5: SAPLMA 方法")
log("=" * 50)

from src.methods.saplma import run_saplma_full

saplma_results = run_saplma_full(
    model=model, tokenizer=tokenizer,
    train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
    layer_idx=-1, pooling="last", batch_size=8, max_length=128,
)
for clf_name, clf_result in saplma_results.items():
    summary = {
        "method": clf_result["method"],
        "layer_idx": clf_result["layer_idx"],
        "pooling": clf_result["pooling"],
        "num_seeds": clf_result["num_seeds"],
        "test_summary": clf_result["test_summary"],
    }
    fname = f"saplma_{clf_name}_results.json"
    with open(out_dir / fname, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    log("SAPLMA (%s) 结果已保存至 %s", clf_name, out_dir / fname)
    ts = clf_result["test_summary"]
    log("  Accuracy:  %.4f ± %.4f", ts["accuracy"]["mean"], ts["accuracy"]["std"])
    log("  Macro-F1:  %.4f ± %.4f", ts["macro_f1"]["mean"], ts["macro_f1"]["std"])
    log("  AUROC:     %.4f ± %.4f", ts["auroc"]["mean"], ts["auroc"]["std"])

elapsed = time.time() - t_start
log("=" * 60)
log("Phase 2 全部完成! 耗时: %.0fs (%.1f min)", elapsed, elapsed / 60)
log("=" * 60)
