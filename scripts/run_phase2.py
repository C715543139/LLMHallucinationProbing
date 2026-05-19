"""
Phase 2 主运行脚本：执行 PPL 基线与 SAPLMA 基线，并将结果写入 experiments/results/baseline。

用法:
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llm_hallucination
    source ./.venv/bin/activate
    python -s scripts/run_phase2.py
    python -s scripts/run_phase2.py --ppl-only
    python -s scripts/run_phase2.py --saplma-only
"""
import sys
import os
import time
import json
import logging
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

# 配置日志：同时输出到控制台和文件
log_dir = PROJECT_ROOT / "experiments" / "results" / "baseline"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "phase2_run.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger("phase2")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="运行 Phase 2 基线实验并保存结果。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ppl-only", action="store_true", help="仅运行 PPL 基线")
    group.add_argument("--saplma-only", action="store_true", help="仅运行 SAPLMA 基线")
    args = parser.parse_args()

    run_ppl = not args.saplma_only
    run_saplma = not args.ppl_only

    t_start = time.time()

    print("=" * 60, flush=True)
    print("  Phase 2: 基础方法实现与评估", flush=True)
    print("=" * 60, flush=True)

    # ---- 加载模型和数据 -------------------------------------------------
    from src.models.loader import load_model, print_device_info
    from src.config import config
    from src.data.preprocessing import load_processed_data
    from src.utils.reproducibility import collect_runtime_info

    print_device_info()
    sys.stdout.flush()

    logger.info("加载模型 (Qwen2-1.5B %s)...", config.models.primary_dtype)
    model, tokenizer = load_model()
    logger.info("模型设备: %s", next(model.parameters()).device)

    logger.info("加载预处理数据...")
    train_ds, val_ds, test_ds = load_processed_data()
    logger.info(train_ds.summary())
    runtime_info = collect_runtime_info(model)

    # ---- P2.1-P2.2: PPL 方法 -------------------------------------------
    if run_ppl:
        print("\n" + "=" * 50, flush=True)
        print("  P2.1-P2.2: PPL 方法", flush=True)
        print("=" * 50, flush=True)

        from src.methods.probability import evaluate_ppl_method

        ppl_results = evaluate_ppl_method(
            model=model, tokenizer=tokenizer,
            train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
            batch_size=8, max_length=128, threshold_metric="f1",
        )
        ppl_summary = {k: v for k, v in ppl_results.items()
                       if k in ("method", "threshold", "threshold_metric", "train", "val", "test")}
        ppl_summary["runtime"] = runtime_info
        with open(log_dir / "ppl_results.json", "w", encoding="utf-8") as f:
            json.dump(ppl_summary, f, indent=2, ensure_ascii=False, default=float)
        logger.info("PPL 结果已保存至 %s", log_dir / "ppl_results.json")
        logger.info("测试集: Acc=%.4f, Macro-F1=%.4f, AUROC=%.4f",
                     ppl_results["test"]["accuracy"],
                     ppl_results["test"]["macro_f1"],
                     ppl_results["test"]["auroc"])

    # ---- P2.3-P2.5: SAPLMA 方法 ----------------------------------------
    if run_saplma:
        print("\n" + "=" * 50, flush=True)
        print("  P2.3-P2.5: SAPLMA 方法", flush=True)
        print("=" * 50, flush=True)

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
                "seeds": clf_result.get("seeds", []),
                "test_summary": clf_result["test_summary"],
                "runtime": runtime_info,
            }
            fname = f"saplma_{clf_name}_results.json"
            with open(log_dir / fname, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
            logger.info("SAPLMA (%s) 结果已保存至 %s", clf_name, log_dir / fname)
            ts = clf_result["test_summary"]
            logger.info("  Accuracy:  %.4f ± %.4f", ts["accuracy"]["mean"], ts["accuracy"]["std"])
            logger.info("  Macro-F1:  %.4f ± %.4f", ts["macro_f1"]["mean"], ts["macro_f1"]["std"])
            logger.info("  AUROC:     %.4f ± %.4f", ts["auroc"]["mean"], ts["auroc"]["std"])

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Phase 2 全部完成! 总耗时: %.0fs (%.1f min)", elapsed, elapsed / 60)
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
