"""
隐藏状态特征提取模块。

从模型各 Transformer 层提取隐藏状态，支持多种 token 池化策略：
    - last:   最后一个 token 的表示
    - first:  第一个 token 的表示
    - mean:   所有 token 的均值池化
    - subject: 主语实体的 token 表示（可选，需 spaCy）

层索引约定：
    - 仅统计 Transformer block 输出，不包含 embedding output
    - hidden_states[0] = embedding output, hidden_states[1:] = Transformer blocks
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.config import config

if TYPE_CHECKING:
    from src.data.dataset import TrueFalseDataset

logger = logging.getLogger(__name__)


def _infer_model_device(model) -> torch.device:
    """兼容真实 HuggingFace 模型与测试中的轻量 dummy model。"""
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def extract_last_token_hidden(
    model,
    tokenizer,
    statement: str,
    layer_idx: int = -1,
    max_length: int = 128,
) -> np.ndarray:
    """提取指定 Transformer block 的最后一个 token 隐藏状态。"""
    device = _infer_model_device(model)

    try:
        inputs = tokenizer(
            statement,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
    except TypeError:
        inputs = tokenizer(statement, return_tensors="pt")

    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    if len(hidden_states) == model.config.num_hidden_layers + 1:
        block_hidden_states = hidden_states[1:]
    else:
        block_hidden_states = hidden_states

    num_layers = len(block_hidden_states)
    resolved_layer_idx = num_layers + layer_idx if layer_idx < 0 else layer_idx
    if resolved_layer_idx < 0 or resolved_layer_idx >= num_layers:
        raise ValueError(f"层索引 {layer_idx} 超出范围 [0, {num_layers - 1}]")

    target_hidden = block_hidden_states[resolved_layer_idx]
    if "attention_mask" in inputs:
        attention_mask = inputs["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(target_hidden.size(0), device=target_hidden.device)
        pooled = target_hidden[batch_indices, seq_lengths]
    else:
        pooled = target_hidden[:, -1, :]

    array = pooled.detach().to(dtype=torch.float32).cpu().numpy()
    return array[0] if array.shape[0] == 1 else array


def collate_statements(batch: list[dict]) -> list[str]:
    """DataLoader 的简单 collate 函数：仅收集 statement 文本。"""
    return [item["statement"] for item in batch]


def extract_hidden_states(
    model,
    tokenizer,
    statements: list[str],
    layers=None,
    pooling: str = "last",
    batch_size: int = 8,
    max_length: int = 128,
) -> np.ndarray:
    """批量提取指定层、指定池化策略的隐藏状态。

    参数:
        model: HuggingFace CausalLM 模型 (eval mode).
        tokenizer: 对应的分词器.
        statements: 陈述句文本列表.
        layers: 要提取的层索引列表（0-based，0 = 第一个 Transformer block）。
                None 表示只提取最后一层。层号不包含 embedding output。
        pooling: 池化策略 — "last", "first", "mean".
        batch_size: 批大小.
        max_length: 最大 token 长度.

    返回:
        numpy 数组，shape 取决于 layers:
        - 单层: (N, hidden_dim)
        - 多层: (num_layers, N, hidden_dim)
    """
    if layers is None:
        layers = [config.features.default_layer_idx]  # -1 → 最后一层

    device = _infer_model_device(model)
    num_hidden_layers = model.config.num_hidden_layers

    # 将负索引转为正索引
    resolved_layers: list[int] = []
    for l in layers:
        if l < 0:
            resolved_layers.append(num_hidden_layers + l)
        else:
            resolved_layers.append(l)
    resolved_layers.sort()

    # 验证层索引范围
    for l in resolved_layers:
        if l < 0 or l >= num_hidden_layers:
            raise ValueError(
                f"层索引 {l} 超出范围 [0, {num_hidden_layers - 1}]"
            )

    # 确定 output_hidden_states=True 时返回的 hidden_states 数量
    # hidden_states[0] = embedding, hidden_states[1..L] = transformer blocks
    max_layer_needed = max(resolved_layers)
    # 需要所有层的 hidden states
    need_all = True

    all_features: list[list[np.ndarray]] = [[] for _ in resolved_layers]

    dataloader = DataLoader(
        statements,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )

    for batch_texts in tqdm(dataloader, desc=f"提取隐藏状态 ({pooling})"):
        inputs = tokenizer(
            list(batch_texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (embedding_output, block_0, block_1, ..., block_{L-1})
        hidden_states = outputs.hidden_states
        # 剥离 embedding output: hidden_states[0] 是 embedding
        block_hidden = hidden_states[1:]  # 长度 = num_hidden_layers

        for i, layer_idx in enumerate(resolved_layers):
            hs = block_hidden[layer_idx]  # (batch, seq_len, hidden_dim)

            if pooling == "last":
                # 考虑 padding，取每个序列最后一个有效 token
                attention_mask = inputs["attention_mask"]  # (batch, seq_len)
                seq_lengths = attention_mask.sum(dim=1) - 1  # 最后一个有效位置
                batch_indices = torch.arange(hs.size(0), device=device)
                pooled = hs[batch_indices, seq_lengths]  # (batch, hidden_dim)
            elif pooling == "first":
                pooled = hs[:, 0, :]  # 第一个 token
            elif pooling == "mean":
                # 对有效 token 做均值池化
                attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hs * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
            else:
                raise ValueError(f"不支持的池化策略: {pooling}")

            all_features[i].append(pooled.to(dtype=torch.float32).cpu().numpy())

    # 拼接所有 batch
    result = []
    for feat_list in all_features:
        result.append(np.concatenate(feat_list, axis=0))

    if len(result) == 1:
        return result[0]  # (N, hidden_dim)
    else:
        return np.stack(result, axis=0)  # (num_layers, N, hidden_dim)


def extract_all_layer_hidden_states(
    model,
    tokenizer,
    statements: list[str],
    pooling: str = "last",
    batch_size: int = 8,
    max_length: int = 128,
) -> np.ndarray:
    """提取所有 Transformer 层的隐藏状态。

    返回:
        numpy 数组, shape (num_layers, N, hidden_dim)
    """
    num_layers = model.config.num_hidden_layers
    all_layers = list(range(num_layers))
    return extract_hidden_states(
        model=model,
        tokenizer=tokenizer,
        statements=statements,
        layers=all_layers,
        pooling=pooling,
        batch_size=batch_size,
        max_length=max_length,
    )


def extract_hidden_states_dataset(
    model,
    tokenizer,
    dataset: TrueFalseDataset,
    pooling: str = "last",
    layers=None,
    batch_size: int = 8,
    max_length: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """从 TrueFalseDataset 提取隐藏状态并返回特征和标签。

    参数:
        model, tokenizer: 模型与分词器.
        dataset: TrueFalseDataset 实例.
        pooling: 池化策略.
        layers: 层索引列表.
        batch_size: 批大小.
        max_length: 最大 token 长度.

    返回:
        (features, labels)
        features: (N, hidden_dim) 或 (num_layers, N, hidden_dim)
        labels: (N,) int 数组
    """
    statements = dataset.statements
    labels = np.array(dataset.labels, dtype=np.int64)

    features = extract_hidden_states(
        model=model,
        tokenizer=tokenizer,
        statements=statements,
        layers=layers,
        pooling=pooling,
        batch_size=batch_size,
        max_length=max_length,
    )

    return features, labels
