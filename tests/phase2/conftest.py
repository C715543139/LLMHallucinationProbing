"""
Phase 2 共享 Fixture 与辅助函数。

说明:
    - 仅在需要真实模型时才加载 Qwen2-1.5B
    - 复用 Phase 1 已验证的 `src.models.loader.load_model_fp16` 作为模型入口
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pytest


@pytest.fixture(scope="session")
def phase2_loaded_model(models_cache_dir: Path):
    """为 Phase 2 的真实模型测试加载一次模型。"""
    import torch
    from src.models.loader import load_model_fp16

    if not torch.cuda.is_available():
        pytest.skip("无 GPU，跳过 Phase 2 真实模型测试")

    qwen_dir = models_cache_dir / "Qwen2-1.5B"
    if not (qwen_dir / "config.json").exists():
        pytest.skip("本地 Qwen2-1.5B 权重不存在，跳过 Phase 2 真实模型测试")

    model, tokenizer = load_model_fp16(model_path=str(qwen_dir))
    yield model, tokenizer

    del model
    torch.cuda.empty_cache()


def pick_balanced_examples(dataset, n_per_label: int = 3) -> Tuple[List[str], List[int]]:
    """
    从数据集中抽取近似平衡的真/假样本。

    依赖 Phase 1 中已约定的数据项结构:
        dataset[i] -> {"statement": str, "label": int, ...}
    """
    positives: List[str] = []
    negatives: List[str] = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        statement = item["statement"]
        label = int(item["label"])

        if label == 1 and len(positives) < n_per_label:
            positives.append(statement)
        elif label == 0 and len(negatives) < n_per_label:
            negatives.append(statement)

        if len(positives) >= n_per_label and len(negatives) >= n_per_label:
            break

    statements = positives + negatives
    labels = [1] * len(positives) + [0] * len(negatives)

    if len(positives) == 0 or len(negatives) == 0:
        raise ValueError("无法从数据集中抽取同时包含真/假标签的样本")

    return statements, labels


def ensure_2d_feature_matrix(rows: Sequence) -> "object":
    """将若干单样本特征堆叠为二维矩阵。"""
    import numpy as np

    normalized = []
    for row in rows:
        arr = np.asarray(row)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        normalized.append(arr)

    return np.vstack(normalized)

