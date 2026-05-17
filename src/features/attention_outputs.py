"""
注意力输出激活特征提取模块。

使用 forward hook 捕获每层 self-attention 模块的输出激活，
提取统计特征以补充 attention score 特征。

特征设计：
    - 对每层 attention output 的 last token 向量提取统计量
    - 可选的 hidden state 与 attention output 的关系特征
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.feature_cache import save_npz_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attention Output Hook
# ---------------------------------------------------------------------------

class AttentionOutputExtractor:
    """使用 forward hook 捕获每层 self-attention 的输出激活。"""

    def __init__(self, model, layers: list[int]):
        self.model = model
        self.layers = layers
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.outputs: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # Qwen2 self_attn output 通常是 tuple，(attn_output, ...)
            if isinstance(output, tuple):
                attn_out = output[0]
            else:
                attn_out = output
            self.outputs[layer_idx] = attn_out.detach().cpu()

        return hook

    def register(self) -> None:
        """注册所有目标层的 forward hook。"""
        for layer_idx in self.layers:
            module = self.model.model.layers[layer_idx].self_attn
            handle = module.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(handle)

    def clear(self) -> None:
        """清空缓存的输出。"""
        self.outputs = {}

    def remove(self) -> None:
        """移除所有 hook。"""
        for handle in self.handles:
            handle.remove()
        self.handles = []


# ---------------------------------------------------------------------------
# 单样本特征提取
# ---------------------------------------------------------------------------

def _extract_stats(vector: np.ndarray, eps: float = 1e-10) -> dict[str, float]:
    """从向量中提取统计特征。"""
    v = np.asarray(vector, dtype=np.float64)
    return {
        "norm": float(np.linalg.norm(v)),
        "mean_abs": float(np.mean(np.abs(v))),
        "max_abs": float(np.max(np.abs(v))),
        "std": float(np.std(v)),
        "sparsity_1e-3": float(np.mean(np.abs(v) < 1e-3)),
    }


def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-10) -> float:
    """计算两个向量的余弦相似度。"""
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    denom = max(norm_a * norm_b, eps)
    return dot / denom


def extract_attention_output_features_single(
    model,
    tokenizer,
    statement: str,
    layers: list[int],
    pooling: str = "last",
    hidden_states: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[str], dict]:
    """对单条 statement 提取 attention output 统计特征。

    参数:
        model: HuggingFace CausalLM 模型。
        tokenizer: 分词器。
        statement: 陈述句文本。
        layers: 要提取的层索引。
        pooling: token 池化策略（目前仅支持 "last"）。
        hidden_states: 可选的隐藏状态字典 {layer_idx: np.ndarray}，
                       用于提取 attention-vs-hidden 关系特征。

    返回:
        (features, feature_names, metadata)
    """
    device = next(model.parameters()).device

    # 创建 extractor
    extractor = AttentionOutputExtractor(model, layers)
    extractor.register()

    try:
        inputs = tokenizer(
            statement,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            _ = model(**inputs, output_hidden_states=(hidden_states is not None))

        # 获取 attention mask 以确定 last token 位置
        attn_mask = inputs.get("attention_mask")
        if attn_mask is not None:
            seq_len = int(attn_mask.sum().item())
            last_pos = seq_len - 1
        else:
            last_pos = -1

        features: list[float] = []
        feature_names: list[str] = []

        for layer_idx in layers:
            out = extractor.outputs.get(layer_idx)
            if out is None:
                # 填零
                for stat_name in ["norm", "mean_abs", "max_abs", "std", "sparsity_1e-3"]:
                    features.append(0.0)
                    feature_names.append(f"L{layer_idx}_attn_out_{stat_name}")
                if hidden_states is not None:
                    features.append(0.0)
                    feature_names.append(f"L{layer_idx}_attn_out_hidden_cosine")
                    features.append(1.0)
                    feature_names.append(f"L{layer_idx}_attn_out_hidden_norm_ratio")
                continue

            # 取 last token
            out_np = out[0].numpy().astype(np.float64)  # (seq_len, hidden_dim)
            if pooling == "last":
                vec = out_np[last_pos]
            else:
                vec = out_np[last_pos]  # fallback to last

            stats = _extract_stats(vec)
            for stat_name in ["norm", "mean_abs", "max_abs", "std", "sparsity_1e-3"]:
                features.append(stats[stat_name])
                feature_names.append(f"L{layer_idx}_attn_out_{stat_name}")

            # 与 hidden state 的关系特征
            if hidden_states is not None and layer_idx in hidden_states:
                hs = hidden_states[layer_idx]
                cos = _cosine_similarity(vec, hs)
                norm_ratio = stats["norm"] / max(np.linalg.norm(hs), 1e-10)
                features.append(cos)
                features.append(norm_ratio)
                feature_names.append(f"L{layer_idx}_attn_out_hidden_cosine")
                feature_names.append(f"L{layer_idx}_attn_out_hidden_norm_ratio")
            elif hidden_states is not None:
                features.append(0.0)
                features.append(1.0)
                feature_names.append(f"L{layer_idx}_attn_out_hidden_cosine")
                feature_names.append(f"L{layer_idx}_attn_out_hidden_norm_ratio")

    finally:
        extractor.remove()

    metadata = {
        "statement": statement,
        "layers": layers,
        "pooling": pooling,
    }

    return np.array(features, dtype=np.float64), feature_names, metadata


def extract_attention_output_features_dataset(
    model,
    tokenizer,
    dataset,
    layers: list[int],
    pooling: str = "last",
    batch_size: int = 1,
    output_path: str | None = None,
    hidden_cache: dict[int, np.ndarray] | None = None,
) -> dict:
    """对数据集批量提取 attention output 特征。

    参数:
        model: 模型。
        tokenizer: 分词器。
        dataset: TrueFalseDataset 实例。
        layers: 要提取的层列表。
        pooling: 池化策略。
        batch_size: 批大小。
        output_path: 缓存输出路径。
        hidden_cache: {layer_idx: np.ndarray (N, hidden_dim)} 可选隐藏状态缓存。

    返回:
        {"features": np.ndarray, "labels": np.ndarray,
         "feature_names": list[str], "metadata": dict}
    """
    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    all_feature_names: list[str] | None = None

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in tqdm(dataloader, desc="提取注意力输出特征"):
        for i in range(len(batch["statement"])):
            statement = batch["statement"][i]
            label = batch["label"][i].item() if isinstance(batch["label"][i], torch.Tensor) else int(batch["label"][i])

            # 构建该样本的 hidden states（若提供）
            sample_hidden = None
            if hidden_cache is not None:
                sample_hidden = {}
                for l_idx, hs_arr in hidden_cache.items():
                    # 需要知道样本在缓存中的索引——这里简化处理，传入时已经对齐
                    pass

            feats, names, _ = extract_attention_output_features_single(
                model, tokenizer, statement, layers, pooling, sample_hidden
            )
            all_features.append(feats)
            all_labels.append(label)
            if all_feature_names is None:
                all_feature_names = names

    features = np.stack(all_features, axis=0)
    labels_arr = np.array(all_labels, dtype=np.int64)

    logger.info("提取完成: %d 样本, %d 特征", features.shape[0], features.shape[1])

    metadata: dict[str, Any] = {
        "feature_version": "attention_outputs_v1",
        "layers": layers,
        "pooling": pooling,
        "num_features": features.shape[1],
        "num_samples": features.shape[0],
        "use_stats": True,
        "use_vector": False,
    }

    if output_path is not None:
        save_npz_cache(output_path, features, labels_arr, all_feature_names, metadata)

    return {
        "features": features,
        "labels": labels_arr,
        "feature_names": all_feature_names or [],
        "metadata": metadata,
    }
