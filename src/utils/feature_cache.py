"""
统一特征缓存接口。

提供 npz 格式的特征保存/加载/检查功能，
供 Phase 4 各子阶段复用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_npz_cache(
    path: str | Path,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """将特征、标签、特征名和元数据保存为压缩 npz 文件。

    参数:
        path: 输出文件路径。
        features: 特征矩阵 (N, D)。
        labels: 标签数组 (N,)。
        feature_names: 特征名称列表，长度应为 D。
        metadata: 任意可 JSON 序列化的元数据字典。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        labels=labels,
        feature_names=np.array(feature_names or [], dtype=object),
        metadata=json.dumps(metadata or {}, ensure_ascii=False),
    )


def load_npz_cache(path: str | Path) -> dict[str, Any]:
    """加载 npz 缓存文件。

    返回:
        {"features": np.ndarray, "labels": np.ndarray,
         "feature_names": list[str], "metadata": dict}
    """
    data = np.load(path, allow_pickle=True)
    return {
        "features": data["features"],
        "labels": data["labels"],
        "feature_names": list(data["feature_names"]),
        "metadata": json.loads(str(data["metadata"])),
    }


def cache_exists(path: str | Path) -> bool:
    """检查缓存文件是否存在。"""
    return Path(path).exists()
