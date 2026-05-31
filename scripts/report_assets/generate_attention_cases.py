"""
Phase 5 报告资产: 生成 Attention 案例可视化。

对测试子集前 150 条样本，使用最强 head (L16-H05) 的 attention matrix,
绘制 last token attending to previous tokens 的 heatmap，
并用 anchor 标注 subject / relation / tail token 位置。

用法:
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llm_hallucination
    source ./.venv/bin/activate
    python -s scripts/report_assets/generate_attention_cases.py
"""

from __future__ import annotations

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
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("attention_cases")


def load_model_eager():
    """加载 bfloat16 + eager attention 模型。"""
    from src.config import config
    from src.utils.reproducibility import configure_deterministic_runtime
    from transformers import AutoModelForCausalLM, AutoTokenizer

    configure_deterministic_runtime(seed=42, deterministic=False)

    model_path = str(config.paths.project_root / config.models.primary_local)
    logger.info("加载模型 (bfloat16 + eager): %s", model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    )
    model.eval()
    logger.info("模型加载完成, dtype: %s, attn_implementation: %s",
                next(model.parameters()).dtype,
                getattr(model.config, "_attn_implementation", "unknown"))
    return model, tokenizer


def load_a6_predictions():
    """加载 A6 逐样本预测。"""
    import csv
    a6_path = Path("experiments/results/phase4/a6_case_analysis.csv")
    rows = []
    with open(a6_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["label"] = int(row["label"])
            row["hidden_prob"] = float(row["hidden_prob"])
            row["hidden_pred"] = int(row["hidden_pred"])
            row["a6_prob"] = float(row["a6_prob"])
            row["a6_pred"] = int(row["a6_pred"])
            rows.append(row)
    return rows


def get_attention_matrix(model, tokenizer, statement: str, layer_idx: int):
    """对单条语句做前向，返回指定层的 attention matrix (batch forward)。

    返回:
        attn_matrix: (seq_len, seq_len) attention weights, last token row 或均值。
        input_ids: token IDs
        tokens: decoded token strings
    """
    device = next(model.parameters()).device

    inputs = tokenizer(statement, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # attentions: tuple of (batch, num_heads, seq_len, seq_len)
    attentions = outputs.attentions
    if attentions is None:
        raise ValueError("模型未返回 attention weights，请确认 output_attentions=True")

    # 取指定层的所有 head 的 attention
    layer_attn = attentions[layer_idx]  # (1, num_heads, seq_len, seq_len)
    # 对所有 head 取均值，先转为 float32 以兼容 numpy
    avg_attn = layer_attn[0].mean(dim=0).to(torch.float32).cpu().numpy()  # (seq_len, seq_len)

    input_ids = inputs["input_ids"][0].cpu().tolist()
    tokens = [tokenizer.decode([tid]) for tid in input_ids]

    return avg_attn, input_ids, tokens


def plot_attention_case(
    statement: str,
    label: int,
    tokens: list[str],
    attn_matrix: np.ndarray,
    hidden_prob: float,
    hidden_pred: int,
    a6_prob: float | None,
    a6_pred: int | None,
    anchors: dict,
    layer_idx: int,
    head_idx: int | None,
    case_type: str,
    description: str,
    save_path: str,
):
    """绘制单个 attention case 的 last-token attention heatmap。"""
    seq_len = len(tokens)
    if seq_len > 80:
        # 截断显示
        tokens = tokens[:80]
        attn_matrix = attn_matrix[:80, :80]
        seq_len = 80

    # last token attending to all previous tokens
    last_attn = attn_matrix[-1, :]  # (seq_len,)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                              gridspec_kw={"height_ratios": [1, 3]})

    # ---- 上半: bar heatmap of last-token attention ----
    ax = axes[0]
    x = np.arange(seq_len)
    colors = []
    for i in range(seq_len):
        if i in anchors.get("subject_indices", []):
            colors.append("#1f77b4")  # blue for subject
        elif i in anchors.get("relation_indices", []):
            colors.append("#ff7f0e")  # orange for relation
        elif i in anchors.get("tail_indices", []):
            colors.append("#2ca02c")  # green for tail
        elif i == anchors.get("last_token_index", -1):
            colors.append("#d62728")  # red for last
        else:
            colors.append("#aaaaaa")  # gray for other
    ax.bar(x, last_attn, color=colors, alpha=0.85, width=0.8)
    ax.axvline(x=anchors.get("last_token_index", seq_len - 1), color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlim(-0.5, seq_len - 0.5)
    ax.set_ylabel("Attention Weight")
    ax.set_title(f"Last Token Attention (L{layer_idx}" + (f" H{head_idx}" if head_idx is not None else "") +
                 f") | {case_type}")
    ax.grid(axis="y", alpha=0.3)

    # legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1f77b4", label="Subject"),
        Patch(facecolor="#ff7f0e", label="Relation"),
        Patch(facecolor="#2ca02c", label="Tail"),
        Patch(facecolor="#d62728", label="Last Token"),
        Patch(facecolor="#aaaaaa", label="Other"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=7, ncol=5)

    # ---- 下半: full attention matrix heatmap ----
    ax = axes[1]
    im = ax.imshow(attn_matrix, aspect="auto", origin="upper", cmap="YlOrRd")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")

    # 标注 anchor 位置
    for name, indices, color in [
        ("Subject", anchors.get("subject_indices", []), "#1f77b4"),
        ("Relation", anchors.get("relation_indices", []), "#ff7f0e"),
        ("Tail", anchors.get("tail_indices", []), "#2ca02c"),
    ]:
        for idx in indices:
            if idx < seq_len:
                ax.axvline(x=idx - 0.5, color=color, linewidth=0.5, alpha=0.4)
                ax.axhline(y=idx - 0.5, color=color, linewidth=0.5, alpha=0.4)

    # last token line
    last_idx = min(anchors.get("last_token_index", seq_len - 1), seq_len - 1)
    ax.axhline(y=last_idx - 0.5, color="red", linewidth=1.5, linestyle="--", alpha=0.6)

    plt.colorbar(im, ax=ax, label="Attention")

    # ---- 全局标题和信息 ----
    label_str = "TRUE" if label == 1 else "FALSE"
    h_pred_str = "TRUE" if hidden_pred == 1 else "FALSE"
    a6_pred_str = "TRUE" if a6_pred == 1 else "FALSE" if a6_pred is not None else "N/A"
    a6_prob_str = f"{a6_prob:.4f}" if a6_prob is not None else "N/A"

    info_text = (
        f"Statement: {statement[:120]}{'...' if len(statement) > 120 else ''}\n"
        f"Gold: {label_str} | Hidden: {hidden_prob:.4f} ({h_pred_str}) | "
        f"A6: {a6_prob_str} ({a6_pred_str})\n"
        f"Description: {description}"
    )
    fig.suptitle(info_text, fontsize=9, y=1.02, va="bottom")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("已保存: %s", save_path)


def select_cases(a6_rows: list[dict]) -> dict:
    """从 A6 分析结果中选择典型案例。

    返回:
        {"true_correct": [...], "false_correct": [...], "hard_or_failure": [...], "improvement": [...]}
    """
    true_correct = []
    false_correct = []
    hard_or_failure = []
    improvement = []

    for row in a6_rows:
        ct = row["case_type"]
        if ct == "hidden_correct_a6_correct":
            if row["label"] == 1:
                true_correct.append(row)
            else:
                false_correct.append(row)
        elif ct == "hidden_wrong_a6_correct":
            improvement.append(row)
            hard_or_failure.append(row)
        elif ct == "hidden_wrong_a6_wrong":
            hard_or_failure.append(row)
        elif ct == "hidden_correct_a6_wrong":
            hard_or_failure.append(row)

    # 选择策略：优先选概率接近 0.5 的（更有说服力），其次选远离 0.5 的边界案例
    def select_best(candidates, n=3, prefer_edge=False):
        if not candidates:
            return []
        if prefer_edge:
            # 选最远离 0.5 的（极端置信案例）
            sorted_cases = sorted(candidates, key=lambda r: abs(r["hidden_prob"] - 0.5), reverse=True)
        else:
            # 选最接近 0.5 的（不确定案例）
            sorted_cases = sorted(candidates, key=lambda r: abs(r["hidden_prob"] - 0.5))
        return sorted_cases[:n]

    return {
        "true_correct": select_best(true_correct, n=3, prefer_edge=False),
        "false_correct": select_best(false_correct, n=3, prefer_edge=False),
        "hard_or_failure": select_best(hard_or_failure, n=5, prefer_edge=False),
        "improvement": improvement[:3] if improvement else [],
    }


def main():
    from src.features.anchor_extraction import extract_anchors
    from src.data.preprocessing import load_processed_data

    t_start = time.time()

    # ---- 1. 读取 head selection ----
    with open("experiments/results/phase4/attention_head_selection.json", "r") as f:
        head_sel = json.load(f)
    top_head = head_sel["selected_heads"][0]
    target_layer = top_head["layer"]
    target_head = top_head["head"]
    logger.info("使用最强 head: L%d-H%d", target_layer, target_head)

    # ---- 2. 加载 A6 预测 ----
    a6_rows = load_a6_predictions()
    logger.info("加载了 %d 条 A6 预测", len(a6_rows))

    # ---- 3. 选择案例 ----
    cases = select_cases(a6_rows)
    logger.info("选择了: true_correct=%d, false_correct=%d, hard_or_failure=%d, improvement=%d",
                len(cases["true_correct"]), len(cases["false_correct"]),
                len(cases["hard_or_failure"]), len(cases["improvement"]))

    # ---- 4. 加载模型 ----
    model, tokenizer = load_model_eager()

    # ---- 5. 加载数据获取完整 test 语句 ----
    _, _, test_ds = load_processed_data()
    test_sub_statements = test_ds.statements[:150]
    test_sub_labels = test_ds.labels[:150]

    # ---- 6. 生成案例 ----
    out_dir = Path("experiments/results/phase4/case_viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases_meta = []

    # 定义案例类型映射
    case_specs = [
        ("true_correct_case", cases["true_correct"], "True statement, both hidden and A6 correct"),
        ("false_correct_case", cases["false_correct"], "False statement, both hidden and A6 correct"),
        ("hard_or_failure_case", cases["hard_or_failure"], "Hard or failure case"),
    ]

    if cases["improvement"]:
        case_specs.append(
            ("improvement_case", cases["improvement"], "Hidden wrong, A6 correct (improvement)")
        )

    for case_type_prefix, case_list, case_desc in case_specs:
        for i, case_row in enumerate(case_list):
            stmt = case_row["statement"]
            gold_label = case_row["label"]

            # 获取 anchor
            anchors = extract_anchors(tokenizer, stmt)

            # 前向获取 attention
            try:
                attn_matrix, input_ids, tokens = get_attention_matrix(
                    model, tokenizer, stmt, target_layer,
                )
            except Exception as e:
                logger.warning("前向失败: %s ... 跳过: %s", stmt[:50], e)
                continue

            # 保存路径
            suffix = f"_l{target_layer}_h{target_head}"
            if len(case_list) > 1:
                suffix += f"_{i + 1}"
            png_path = out_dir / f"{case_type_prefix}{suffix}.png"

            # 描述
            if case_row["case_type"] == "hidden_wrong_a6_correct":
                desc = "Improvement: hidden wrong, A6 correct. A6 fusion successfully corrected the hidden-only error."
            elif case_row["case_type"] == "hidden_wrong_a6_wrong":
                desc = "Both wrong. Challenging case where both methods fail."
            elif case_row["case_type"] == "hidden_correct_a6_wrong":
                desc = "Hidden correct, A6 wrong. Attention features degraded performance."
            else:
                desc = f"Both correct. {case_desc}."

            # 绘图
            plot_attention_case(
                statement=stmt,
                label=gold_label,
                tokens=tokens,
                attn_matrix=attn_matrix,
                hidden_prob=case_row["hidden_prob"],
                hidden_pred=case_row["hidden_pred"],
                a6_prob=case_row["a6_prob"],
                a6_pred=case_row["a6_pred"],
                anchors={
                    "subject_indices": anchors.subject_token_indices,
                    "relation_indices": anchors.relation_token_indices,
                    "tail_indices": anchors.tail_token_indices,
                    "last_token_index": anchors.last_token_index,
                },
                layer_idx=target_layer,
                head_idx=target_head,
                case_type=case_type_prefix,
                description=desc,
                save_path=str(png_path),
            )

            # 记录元数据
            cases_meta.append({
                "case_type": case_type_prefix,
                "statement": stmt,
                "gold_label": gold_label,
                "layer": target_layer,
                "head": target_head,
                "hidden_prob": case_row["hidden_prob"],
                "hidden_pred": case_row["hidden_pred"],
                "a6_prob": case_row["a6_prob"],
                "a6_pred": case_row["a6_pred"],
                "subject_token_indices": anchors.subject_token_indices,
                "relation_token_indices": anchors.relation_token_indices,
                "tail_token_indices": anchors.tail_token_indices,
                "last_token_index": anchors.last_token_index,
                "rule_name": anchors.rule_name,
                "valid": anchors.valid,
                "description": desc,
                "image_path": str(png_path),
            })

    # ---- 7. 保存 cases.json ----
    json_path = out_dir / "cases.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cases_meta, f, indent=2, ensure_ascii=False)
    logger.info("案例元数据已保存至 %s (%d 个案例)", json_path, len(cases_meta))

    # ---- 8. 检查 improvement cases ----
    if cases["improvement"]:
        logger.info("找到 %d 个 improvement cases (hidden-wrong, A6-correct)", len(cases["improvement"]))
    else:
        logger.warning("未找到 A6 improvement cases (hidden-wrong, A6-correct)")

    elapsed = time.time() - t_start
    logger.info("Attention 案例生成完成! 耗时: %.0fs (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
