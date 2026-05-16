"""
注意力特征提取模块。

Phase 4（方向 A）核心能力：
    1. 提取模型各层注意力矩阵
    2. 基于陈述句的 subject / relation / tail 锚点构造统计特征
    3. 为后续 attention-only 与 hidden+attention 融合分类提供特征输入

说明：
    - 优先使用 tokenizer 的 offset mapping 将文本跨度对齐到 token；
      若当前 tokenizer 不支持 offset mapping，则退化为基于位置的简单锚点。
    - 锚点抽取遵循项目计划中的“优先规则、失败时退化”的原则：
      这里默认实现轻量规则版本，不额外要求 spaCy 模型下载。
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional, Sequence, Union

import numpy as np
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from src.data.dataset import TrueFalseDataset

logger = logging.getLogger(__name__)

_WORD_PATTERN = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?")
_VERB_CUES = {
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had",
    "do", "does", "did",
    "can", "could", "may", "might", "must",
    "shall", "should", "will", "would",
    "contains", "contain", "located", "invented", "founded",
    "discovered", "created", "born", "capital", "belongs",
    "becomes", "became", "means", "refers", "uses", "use",
}
_RELATION_CONTINUATION_CUES = {
    "is", "are", "was", "were", "has", "have", "had",
    "in", "on", "at", "of", "for", "from", "to", "by", "with",
    "into", "over", "under", "between", "within",
}

ATTENTION_FEATURE_NAMES: tuple[str, ...] = (
    "attn_entropy_mean",
    "attn_entropy_std",
    "attn_entropy_last",
    "subject_attn_ratio",
    "relation_attn_ratio",
    "tail_attn_ratio",
    "subject_attn_last_layer",
    "relation_attn_last_layer",
    "tail_attn_last_layer",
    "last_to_subject",
    "last_to_relation",
    "last_to_tail",
    "subject_to_relation",
    "subject_to_tail",
    "relation_to_tail",
    "cross_head_agreement_mean",
    "cross_head_agreement_std",
    "subject_token_count",
    "relation_token_count",
    "sequence_length",
)


def _infer_model_device(model) -> torch.device:
    """兼容真实 HuggingFace 模型与测试中的轻量 dummy model。"""
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0


def _safe_entropy(probabilities: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    probs = np.clip(probabilities, eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=axis)


def _mean_pairwise_cosine(vectors: np.ndarray, eps: float = 1e-12) -> float:
    """计算多向量之间的平均两两余弦相似度。"""
    if vectors.ndim != 2:
        raise ValueError(f"vectors 期望二维，实际为 {vectors.shape}")
    if vectors.shape[0] < 2:
        return 1.0

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.clip(norms, eps, None)
    cosine = normalized @ normalized.T
    upper = cosine[np.triu_indices_from(cosine, k=1)]
    return float(np.mean(upper)) if upper.size > 0 else 1.0


def _tokenize_words_with_spans(statement: str) -> list[dict[str, str | int]]:
    words: list[dict[str, str | int]] = []
    for match in _WORD_PATTERN.finditer(statement):
        words.append(
            {
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
        )
    return words


def _looks_like_relation_word(word: str) -> bool:
    lower = word.lower()
    return (
        lower in _VERB_CUES
        or lower in _RELATION_CONTINUATION_CUES
        or lower.endswith("ed")
        or lower.endswith("ing")
    )


def extract_subject_relation_spans(statement: str) -> dict[str, object]:
    """基于轻量规则抽取 subject / relation 字符级跨度。"""
    words = _tokenize_words_with_spans(statement)
    if not words:
        return {
            "subject_span": None,
            "relation_span": None,
            "subject_text": "",
            "relation_text": "",
        }

    pivot = None
    for idx, word in enumerate(words):
        if _looks_like_relation_word(str(word["text"])):
            pivot = idx
            break

    if pivot is None:
        subject_end_idx = min(1, len(words) - 1)
        relation_start_idx = min(subject_end_idx + 1, len(words) - 1)
    else:
        subject_end_idx = max(0, pivot - 1)
        relation_start_idx = pivot

    subject_span = (
        int(words[0]["start"]),
        int(words[subject_end_idx]["end"]),
    )

    relation_end_idx = relation_start_idx
    max_extra_tokens = 1
    while (
        relation_end_idx + 1 < len(words)
        and max_extra_tokens > 0
        and _looks_like_relation_word(str(words[relation_end_idx + 1]["text"]))
    ):
        relation_end_idx += 1
        max_extra_tokens -= 1

    relation_span = (
        int(words[relation_start_idx]["start"]),
        int(words[relation_end_idx]["end"]),
    )

    return {
        "subject_span": subject_span,
        "relation_span": relation_span,
        "subject_text": statement[subject_span[0]:subject_span[1]].strip(),
        "relation_text": statement[relation_span[0]:relation_span[1]].strip(),
    }


def map_char_span_to_token_indices(
    offset_mapping: Optional[Union[np.ndarray, Sequence[Sequence[int]]]],
    char_span: Optional[tuple[int, int]],
) -> list[int]:
    """将字符跨度映射到 token 索引列表。"""
    if offset_mapping is None or char_span is None:
        return []

    start, end = int(char_span[0]), int(char_span[1])
    token_indices: list[int] = []
    for idx, pair in enumerate(offset_mapping):
        token_start = int(pair[0])
        token_end = int(pair[1])
        if token_end <= token_start:
            continue
        overlaps = token_end > start and token_start < end
        if overlaps:
            token_indices.append(idx)
    return token_indices


def identify_attention_anchor_tokens(
    statement: str,
    offset_mapping: Optional[Union[np.ndarray, Sequence[Sequence[int]]]],
    valid_length: int,
) -> dict[str, object]:
    """识别 subject / relation / tail 三类锚点 token。"""
    span_info = extract_subject_relation_spans(statement)

    valid_offsets = None
    if offset_mapping is not None:
        valid_offsets = np.asarray(offset_mapping)[:valid_length]

    content_token_indices: list[int] = []
    if valid_offsets is not None and valid_offsets.size > 0:
        for idx, pair in enumerate(valid_offsets):
            if int(pair[1]) > int(pair[0]):
                content_token_indices.append(idx)

    if not content_token_indices:
        content_token_indices = list(range(valid_length))

    subject_token_indices = map_char_span_to_token_indices(valid_offsets, span_info["subject_span"])
    relation_token_indices = map_char_span_to_token_indices(valid_offsets, span_info["relation_span"])

    if not subject_token_indices and content_token_indices:
        subject_token_indices = [content_token_indices[0]]
    if not relation_token_indices and content_token_indices:
        relation_token_indices = [content_token_indices[min(1, len(content_token_indices) - 1)]]

    tail_token_indices = [content_token_indices[-1]] if content_token_indices else [max(valid_length - 1, 0)]

    return {
        "subject_token_indices": subject_token_indices,
        "relation_token_indices": relation_token_indices,
        "tail_token_indices": tail_token_indices,
        "subject_text": span_info["subject_text"],
        "relation_text": span_info["relation_text"],
        "subject_span": span_info["subject_span"],
        "relation_span": span_info["relation_span"],
    }


def _normalize_attention(layer_attention: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = layer_attention.sum(axis=-1, keepdims=True)
    return np.divide(layer_attention, np.clip(denom, eps, None), out=np.zeros_like(layer_attention), where=denom > 0)


def _target_attention_ratio(layer_attention: np.ndarray, target_indices: Sequence[int]) -> float:
    if not target_indices:
        return 0.0
    return float(layer_attention[:, :, target_indices].sum(axis=-1).mean())


def _query_to_target_mass(
    layer_attention: np.ndarray,
    query_indices: Sequence[int],
    target_indices: Sequence[int],
) -> float:
    if not query_indices or not target_indices:
        return 0.0
    query_slice = layer_attention[:, list(query_indices), :]
    return float(query_slice[:, :, list(target_indices)].sum(axis=-1).mean())


def compute_attention_feature_dict(
    attentions: Sequence[np.ndarray | torch.Tensor],
    attention_mask: Union[np.ndarray, Sequence[int], torch.Tensor],
    anchor_tokens: dict[str, object],
) -> dict[str, float]:
    """基于单条样本的多层注意力矩阵计算统计特征。"""
    mask_array = np.asarray(attention_mask, dtype=np.int64).reshape(-1)
    valid_length = int(mask_array.sum()) if mask_array.size > 0 else 0
    if valid_length <= 0:
        raise ValueError("attention_mask 未包含任何有效 token")

    subject_indices = [int(i) for i in anchor_tokens.get("subject_token_indices", []) if 0 <= int(i) < valid_length]
    relation_indices = [int(i) for i in anchor_tokens.get("relation_token_indices", []) if 0 <= int(i) < valid_length]
    tail_indices = [int(i) for i in anchor_tokens.get("tail_token_indices", []) if 0 <= int(i) < valid_length]

    if not subject_indices:
        subject_indices = [0]
    if not relation_indices:
        relation_indices = [min(1, valid_length - 1)]
    if not tail_indices:
        tail_indices = [valid_length - 1]

    layer_entropies: list[float] = []
    subject_ratios: list[float] = []
    relation_ratios: list[float] = []
    tail_ratios: list[float] = []
    head_agreements: list[float] = []
    last_to_subject: list[float] = []
    last_to_relation: list[float] = []
    last_to_tail: list[float] = []
    subject_to_relation: list[float] = []
    subject_to_tail: list[float] = []
    relation_to_tail: list[float] = []

    for layer_attention in attentions:
        layer = layer_attention.detach().cpu().numpy() if hasattr(layer_attention, "detach") else np.asarray(layer_attention)
        layer = np.asarray(layer, dtype=np.float64)
        if layer.ndim == 4:
            layer = layer[0]
        if layer.ndim != 3:
            raise ValueError(f"单层注意力应为 (heads, seq, seq)，实际为 {layer.shape}")

        cropped = layer[:, :valid_length, :valid_length]
        normalized = _normalize_attention(cropped)

        entropy = _safe_entropy(normalized, axis=-1)
        layer_entropies.append(float(np.mean(entropy)))
        head_agreements.append(_mean_pairwise_cosine(normalized.reshape(normalized.shape[0], -1)))

        subject_ratios.append(_target_attention_ratio(normalized, subject_indices))
        relation_ratios.append(_target_attention_ratio(normalized, relation_indices))
        tail_ratios.append(_target_attention_ratio(normalized, tail_indices))

        last_query = [valid_length - 1]
        last_to_subject.append(_query_to_target_mass(normalized, last_query, subject_indices))
        last_to_relation.append(_query_to_target_mass(normalized, last_query, relation_indices))
        last_to_tail.append(_query_to_target_mass(normalized, last_query, tail_indices))
        subject_to_relation.append(_query_to_target_mass(normalized, subject_indices, relation_indices))
        subject_to_tail.append(_query_to_target_mass(normalized, subject_indices, tail_indices))
        relation_to_tail.append(_query_to_target_mass(normalized, relation_indices, tail_indices))

    return {
        "attn_entropy_mean": _safe_mean(layer_entropies),
        "attn_entropy_std": _safe_std(layer_entropies),
        "attn_entropy_last": float(layer_entropies[-1]) if layer_entropies else 0.0,
        "subject_attn_ratio": _safe_mean(subject_ratios),
        "relation_attn_ratio": _safe_mean(relation_ratios),
        "tail_attn_ratio": _safe_mean(tail_ratios),
        "subject_attn_last_layer": float(subject_ratios[-1]) if subject_ratios else 0.0,
        "relation_attn_last_layer": float(relation_ratios[-1]) if relation_ratios else 0.0,
        "tail_attn_last_layer": float(tail_ratios[-1]) if tail_ratios else 0.0,
        "last_to_subject": _safe_mean(last_to_subject),
        "last_to_relation": _safe_mean(last_to_relation),
        "last_to_tail": _safe_mean(last_to_tail),
        "subject_to_relation": _safe_mean(subject_to_relation),
        "subject_to_tail": _safe_mean(subject_to_tail),
        "relation_to_tail": _safe_mean(relation_to_tail),
        "cross_head_agreement_mean": _safe_mean(head_agreements),
        "cross_head_agreement_std": _safe_std(head_agreements),
        "subject_token_count": float(len(subject_indices)),
        "relation_token_count": float(len(relation_indices)),
        "sequence_length": float(valid_length),
    }


def _forward_with_attentions(model, model_inputs: dict[str, torch.Tensor]):
    """执行带注意力输出的前向传播，必要时回退到 eager attention。"""
    previous_impl = getattr(model.config, "_attn_implementation", None)
    switched_impl = False
    supported_with_outputs = {None, "eager", "eager_paged", "flex_attention"}

    if previous_impl not in supported_with_outputs:
        try:
            if hasattr(model, "set_attn_implementation"):
                model.set_attn_implementation("eager")
            else:
                model.config._attn_implementation = "eager"
            switched_impl = True
        except Exception:
            logger.warning("切换 eager attention 失败，将直接请求 output_attentions", exc_info=True)

    try:
        with torch.no_grad():
            outputs = model(**model_inputs, output_attentions=True, use_cache=False)
    finally:
        if switched_impl and previous_impl is not None:
            try:
                if hasattr(model, "set_attn_implementation"):
                    model.set_attn_implementation(previous_impl)
                else:
                    model.config._attn_implementation = previous_impl
            except Exception:
                logger.warning("恢复 attention implementation=%s 失败", previous_impl, exc_info=True)

    attentions = getattr(outputs, "attentions", None)
    if not attentions or any(attn is None for attn in attentions):
        raise RuntimeError("模型未返回 attention weights，无法执行 Phase 4 注意力分析")
    return outputs


def _convert_offsets(offset_mapping) -> Optional[np.ndarray]:
    if offset_mapping is None:
        return None
    if hasattr(offset_mapping, "detach"):
        return offset_mapping.detach().cpu().numpy()
    return np.asarray(offset_mapping)


def extract_attention_features(
    model,
    tokenizer,
    statements: Union[Sequence[str], str],
    batch_size: int = 4,
    max_length: int = 128,
) -> tuple[np.ndarray, list[str], list[dict[str, object]]]:
    """批量提取注意力统计特征。"""
    if isinstance(statements, str):
        statements = [statements]
    statements = list(statements)

    if batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")

    if not statements:
        empty = np.empty((0, len(ATTENTION_FEATURE_NAMES)), dtype=np.float64)
        return empty, list(ATTENTION_FEATURE_NAMES), []

    device = _infer_model_device(model)
    feature_rows: list[list[float]] = []
    metadata: list[dict[str, object]] = []

    for start in tqdm(range(0, len(statements), batch_size), desc="提取注意力特征"):
        batch_texts = statements[start:start + batch_size]
        token_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": max_length,
        }
        try:
            tokenized = tokenizer(list(batch_texts), return_offsets_mapping=True, **token_kwargs)
            offset_mapping = tokenized.pop("offset_mapping", None)
        except TypeError:
            tokenized = tokenizer(list(batch_texts), **token_kwargs)
            offset_mapping = None

        offsets_np = _convert_offsets(offset_mapping)
        model_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in tokenized.items()
        }
        outputs = _forward_with_attentions(model, model_inputs)
        attentions = outputs.attentions
        attention_mask = model_inputs.get("attention_mask")

        for local_idx, statement in enumerate(batch_texts):
            if attention_mask is not None:
                sample_mask = attention_mask[local_idx].detach().cpu().numpy()
                valid_length = int(sample_mask.sum())
            else:
                first_layer = attentions[0]
                valid_length = int(first_layer.shape[-1])
                sample_mask = np.ones(valid_length, dtype=np.int64)

            sample_offsets = offsets_np[local_idx][:valid_length] if offsets_np is not None else None
            anchors = identify_attention_anchor_tokens(
                statement=statement,
                offset_mapping=sample_offsets,
                valid_length=valid_length,
            )
            per_example_attentions = [layer[local_idx] for layer in attentions]
            feature_dict = compute_attention_feature_dict(
                per_example_attentions,
                attention_mask=sample_mask[:valid_length],
                anchor_tokens=anchors,
            )
            feature_rows.append([feature_dict[name] for name in ATTENTION_FEATURE_NAMES])
            metadata.append({
                "statement": statement,
                **anchors,
            })

    return np.asarray(feature_rows, dtype=np.float64), list(ATTENTION_FEATURE_NAMES), metadata


def extract_attention_features_dataset(
    model,
    tokenizer,
    dataset: TrueFalseDataset,
    batch_size: int = 4,
    max_length: int = 128,
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, object]]]:
    """从 TrueFalseDataset 提取注意力特征。"""
    features, feature_names, metadata = extract_attention_features(
        model=model,
        tokenizer=tokenizer,
        statements=dataset.statements,
        batch_size=batch_size,
        max_length=max_length,
    )
    labels = np.asarray(dataset.labels, dtype=np.int64)
    return features, labels, feature_names, metadata


def summarize_attention_feature_differences(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    top_k: int = 8,
) -> dict[str, object]:
    """汇总真/假陈述在注意力特征上的均值差异。"""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    feature_names = list(feature_names)

    if features.ndim != 2:
        raise ValueError(f"features 必须为二维矩阵，实际为 {features.shape}")
    if features.shape[1] != len(feature_names):
        raise ValueError("feature_names 数量与特征维度不一致")
    if labels.shape[0] != features.shape[0]:
        raise ValueError("labels 长度与特征行数不一致")

    true_mask = labels == 1
    false_mask = labels == 0

    per_feature: list[dict[str, float | str]] = []
    for idx, name in enumerate(feature_names):
        true_values = features[true_mask, idx]
        false_values = features[false_mask, idx]

        true_mean = float(np.mean(true_values)) if true_values.size else 0.0
        false_mean = float(np.mean(false_values)) if false_values.size else 0.0
        true_std = float(np.std(true_values, ddof=1)) if true_values.size > 1 else 0.0
        false_std = float(np.std(false_values, ddof=1)) if false_values.size > 1 else 0.0
        delta = true_mean - false_mean

        per_feature.append(
            {
                "name": name,
                "true_mean": true_mean,
                "true_std": true_std,
                "false_mean": false_mean,
                "false_std": false_std,
                "delta": delta,
                "abs_delta": abs(delta),
            }
        )

    top_features = sorted(per_feature, key=lambda item: float(item["abs_delta"]), reverse=True)[:top_k]

    return {
        "num_samples": int(features.shape[0]),
        "num_true": int(true_mask.sum()),
        "num_false": int(false_mask.sum()),
        "feature_names": feature_names,
        "per_feature": per_feature,
        "top_features": top_features,
    }


__all__ = [
    "ATTENTION_FEATURE_NAMES",
    "compute_attention_feature_dict",
    "extract_attention_features",
    "extract_attention_features_dataset",
    "extract_subject_relation_spans",
    "identify_attention_anchor_tokens",
    "map_char_span_to_token_indices",
    "summarize_attention_feature_differences",
]

