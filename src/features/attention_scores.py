"""
注意力分数特征提取模块。

对每条陈述句提取 layer/head 级别的注意力分数统计特征。
特征设计围绕 last token 对 anchor tokens 的注意力模式。

要求模型以 eager attention 加载，确保 output_attentions=True 可用。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.features.anchor_extraction import extract_anchors, AnchorSpans
from src.utils.feature_cache import save_npz_cache

logger = logging.getLogger(__name__)


def _get_attention_implementation(model) -> str | None:
    """读取模型当前 attention 实现。"""
    return getattr(
        model.config,
        "attn_implementation",
        getattr(model.config, "_attn_implementation", None),
    )


def _compute_attention_entropy(attn_vector: np.ndarray, eps: float = 1e-10) -> float:
    """计算注意力分布的熵。"""
    v = np.asarray(attn_vector, dtype=np.float64)
    v = np.clip(v, eps, None)
    v = v / (v.sum() + eps)
    return float(-np.sum(v * np.log(v)))


def _compute_mass(attn_vector: np.ndarray, indices: list[int]) -> float:
    """计算注意力向量在指定 token 索引上的质量总和。"""
    if not indices:
        return 0.0
    total = float(np.sum(attn_vector))
    if total == 0:
        return 0.0
    mass = float(sum(attn_vector[i] for i in indices if 0 <= i < len(attn_vector)))
    return mass / total


def _extract_attention_features_single_head(
    attn_matrix: np.ndarray,
    anchor: AnchorSpans,
    seq_len: int,
) -> dict[str, float]:
    """对单个 head 的注意力矩阵提取统计特征。

    参数:
        attn_matrix: (seq_len, seq_len) 注意力矩阵，行=query, 列=key。
        anchor: token 级锚点信息。
        seq_len: 实际序列长度。

    返回:
        特征名 -> 特征值的字典。
    """
    n = min(seq_len, attn_matrix.shape[0])
    last_attn = attn_matrix[n - 1, :n].copy()  # last token 对所有 token 的注意力
    last_attn_total = float(np.sum(last_attn))
    if last_attn_total > 0:
        last_attn_norm = last_attn / last_attn_total
    else:
        last_attn_norm = np.zeros_like(last_attn)

    subj_idx = [i for i in anchor.subject_token_indices if i < n]
    rel_idx = [i for i in anchor.relation_token_indices if i < n]
    tail_idx = [i for i in anchor.tail_token_indices if i < n]
    anchor_idx = list(set(subj_idx + rel_idx + tail_idx))
    non_anchor_idx = [i for i in range(n) if i not in anchor_idx]

    features: dict[str, float] = {}

    # last token 视角
    features["last_to_subject_mass"] = _compute_mass(last_attn, subj_idx)
    features["last_to_relation_mass"] = _compute_mass(last_attn, rel_idx)
    features["last_to_tail_mass"] = _compute_mass(last_attn, tail_idx)
    features["last_to_anchor_mass"] = _compute_mass(last_attn, anchor_idx)
    features["last_to_non_anchor_mass"] = _compute_mass(last_attn, non_anchor_idx)
    features["attention_entropy_last"] = _compute_attention_entropy(last_attn_norm)
    features["max_attention_last"] = float(np.max(last_attn_norm))

    # top-3 attention mass
    top3_idx = np.argsort(last_attn_norm)[-3:]
    features["top3_attention_mass_last"] = float(np.sum(last_attn_norm[top3_idx]))

    # attention sink: 前几个 token 的注意力质量
    sink_n = min(4, n)
    features["attention_sink_mass"] = float(np.sum(last_attn_norm[:sink_n]))

    # 归一化特征（除以 anchor token 数量）
    for name, idx_list in [
        ("last_to_subject_mass_norm", subj_idx),
        ("last_to_relation_mass_norm", rel_idx),
        ("last_to_tail_mass_norm", tail_idx),
    ]:
        base = features[name.replace("_norm", "")]
        count = max(len(idx_list), 1)
        features[name] = base / count

    # token-to-token 结构视角
    if subj_idx and rel_idx:
        subj_to_rel = 0.0
        for si in subj_idx:
            if si < n:
                subj_to_rel += _compute_mass(attn_matrix[si, :n], rel_idx)
        features["subject_to_relation_mass"] = subj_to_rel / max(len(subj_idx), 1)
    else:
        features["subject_to_relation_mass"] = 0.0

    if rel_idx and subj_idx:
        rel_to_subj = 0.0
        for ri in rel_idx:
            if ri < n:
                rel_to_subj += _compute_mass(attn_matrix[ri, :n], subj_idx)
        features["relation_to_subject_mass"] = rel_to_subj / max(len(rel_idx), 1)
    else:
        features["relation_to_subject_mass"] = 0.0

    if rel_idx and tail_idx:
        rel_to_tail = 0.0
        for ri in rel_idx:
            if ri < n:
                rel_to_tail += _compute_mass(attn_matrix[ri, :n], tail_idx)
        features["relation_to_tail_mass"] = rel_to_tail / max(len(rel_idx), 1)
    else:
        features["relation_to_tail_mass"] = 0.0

    if tail_idx and rel_idx:
        tail_to_rel = 0.0
        for ti in tail_idx:
            if ti < n:
                tail_to_rel += _compute_mass(attn_matrix[ti, :n], rel_idx)
        features["tail_to_relation_mass"] = tail_to_rel / max(len(tail_idx), 1)
    else:
        features["tail_to_relation_mass"] = 0.0

    return features


_BASE_FEATURE_NAMES = [
    "last_to_subject_mass",
    "last_to_relation_mass",
    "last_to_tail_mass",
    "last_to_anchor_mass",
    "last_to_non_anchor_mass",
    "attention_entropy_last",
    "max_attention_last",
    "top3_attention_mass_last",
    "attention_sink_mass",
    "last_to_subject_mass_norm",
    "last_to_relation_mass_norm",
    "last_to_tail_mass_norm",
    "subject_to_relation_mass",
    "relation_to_subject_mass",
    "relation_to_tail_mass",
    "tail_to_relation_mass",
]


def _build_feature_names(layers: list[int], num_heads: int) -> list[str]:
    """构建所有 layer/head 组合的特征名列表。"""
    names: list[str] = []
    for layer_idx in layers:
        for head_idx in range(num_heads):
            for base in _BASE_FEATURE_NAMES:
                names.append(f"L{layer_idx}_H{head_idx:02d}_{base}")
    return names


def extract_attention_score_features_single(
    model,
    tokenizer,
    statement: str,
    layers: list[int],
) -> tuple[np.ndarray, list[str], dict]:
    """对单条 statement 提取 layer/head 级 attention score 特征。

    参数:
        model: HuggingFace CausalLM 模型（eager attention）。
        tokenizer: 分词器。
        statement: 陈述句文本。
        layers: 要提取的层索引列表。

    返回:
        (features: np.ndarray shape (D,), feature_names: list[str], metadata: dict)
    """
    device = next(model.parameters()).device

    # 抽取 anchor
    anchor = extract_anchors(tokenizer, statement)

    # 编码
    inputs = tokenizer(statement, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    seq_len = inputs["input_ids"].shape[1]

    num_heads = model.config.num_attention_heads

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, use_cache=False)

    # outputs.attentions: tuple of (batch, num_heads, seq_len, seq_len) per layer
    attentions = outputs.attentions
    if attentions is None:
        attn_impl = _get_attention_implementation(model)
        raise RuntimeError(
            "output_attentions=True 未返回 attention 权重。"
            f"当前 attn_implementation={attn_impl!r}，"
            "请先调用 model.set_attn_implementation('eager')。"
        )

    # 确保长度匹配
    if len(attentions) != model.config.num_hidden_layers:
        logger.warning(
            "attentions 长度 (%d) != num_hidden_layers (%d)",
            len(attentions),
            model.config.num_hidden_layers,
        )

    all_features: list[float] = []
    all_names: list[str] = []

    for layer_idx in layers:
        if layer_idx >= len(attentions):
            # 超出范围则填零
            for head_idx in range(num_heads):
                for base in _BASE_FEATURE_NAMES:
                    all_features.append(0.0)
                    all_names.append(f"L{layer_idx}_H{head_idx:02d}_{base}")
            continue

        layer_attn = attentions[layer_idx][0]  # (num_heads, seq_len, seq_len)
        layer_attn_np = layer_attn.to(dtype=torch.float32).cpu().numpy().astype(np.float64)

        for head_idx in range(num_heads):
            head_attn = layer_attn_np[head_idx]
            feats = _extract_attention_features_single_head(
                head_attn, anchor, seq_len
            )
            for base in _BASE_FEATURE_NAMES:
                all_features.append(feats.get(base, 0.0))
                all_names.append(f"L{layer_idx}_H{head_idx:02d}_{base}")

    metadata = {
        "statement": statement,
        "anchor_rule": anchor.rule_name,
        "fallback_reason": anchor.fallback_reason,
        "anchor_valid": anchor.valid,
        "seq_len": seq_len,
        "subject_token_count": len(anchor.subject_token_indices),
        "relation_token_count": len(anchor.relation_token_indices),
        "tail_token_count": len(anchor.tail_token_indices),
        "anchor_token_count": (
            len(anchor.subject_token_indices)
            + len(anchor.relation_token_indices)
            + len(anchor.tail_token_indices)
        ),
    }

    return np.array(all_features, dtype=np.float64), all_names, metadata


def extract_attention_score_features_dataset(
    model,
    tokenizer,
    dataset,
    layers: list[int],
    batch_size: int = 1,
    output_path: str | None = None,
) -> dict:
    """对数据集批量提取 attention score 特征。

    参数:
        model: 模型。
        tokenizer: 分词器。
        dataset: TrueFalseDataset 实例。
        layers: 要提取的层列表。
        batch_size: 批大小（注意：内存受限，建议=1）。
        output_path: 若提供，保存为 npz 缓存。

    返回:
        {"features": np.ndarray, "labels": np.ndarray,
         "feature_names": list[str], "metadata": dict}
    """
    num_heads = model.config.num_attention_heads
    feature_names = _build_feature_names(layers, num_heads)

    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    fallback_count = 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    for batch in tqdm(dataloader, desc="提取注意力分数特征"):
        for i in range(len(batch["statement"])):
            statement = batch["statement"][i]
            label = batch["label"][i].item() if isinstance(batch["label"][i], torch.Tensor) else int(batch["label"][i])

            feats, names, meta = extract_attention_score_features_single(
                model, tokenizer, statement, layers
            )
            all_features.append(feats)
            all_labels.append(label)

            if meta.get("fallback_reason"):
                fallback_count += 1

    features = np.stack(all_features, axis=0)
    labels_arr = np.array(all_labels, dtype=np.int64)

    logger.info("提取完成: %d 样本, %d 特征", features.shape[0], features.shape[1])
    logger.info("Fallback 比例: %d / %d", fallback_count, len(all_labels))

    metadata: dict[str, Any] = {
        "feature_version": "attention_scores_rule_v1",
        "layers": layers,
        "num_heads": num_heads,
        "num_features": features.shape[1],
        "num_samples": features.shape[0],
        "fallback_count": fallback_count,
        "anchor_version": "rule_v1",
        "include_length_features": False,
        "note": "layer/head-level attention score statistics",
    }

    if output_path is not None:
        save_npz_cache(output_path, features, labels_arr, feature_names, metadata)

    return {
        "features": features,
        "labels": labels_arr,
        "feature_names": feature_names,
        "metadata": metadata,
    }


def _build_length_feature_names(layers: list[int], num_heads: int) -> list[str]:
    """构建辅助 metadata 特征名（用于分析和去偏）。"""
    return [
        "sequence_length",
        "subject_token_count",
        "relation_token_count",
        "tail_token_count",
        "anchor_token_count",
        "fallback_flag",
    ]


def extract_length_metadata(
    model,
    tokenizer,
    dataset,
    layers: list[int],
    batch_size: int = 1,
) -> np.ndarray:
    """提取每条样本的长度相关 metadata（用于残差化去偏）。

    返回:
        np.ndarray shape (N, 6): seq_len, subj_count, rel_count, tail_count,
                                 anchor_count, fallback_flag
    """
    all_meta: list[list[float]] = []

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch in tqdm(dataloader, desc="提取长度元数据"):
        for i in range(len(batch["statement"])):
            statement = batch["statement"][i]
            _, _, meta = extract_attention_score_features_single(
                model, tokenizer, statement, layers
            )
            all_meta.append([
                float(meta["seq_len"]),
                float(meta["subject_token_count"]),
                float(meta["relation_token_count"]),
                float(meta["tail_token_count"]),
                float(meta["anchor_token_count"]),
                1.0 if meta["fallback_reason"] else 0.0,
            ])

    return np.array(all_meta, dtype=np.float64)
