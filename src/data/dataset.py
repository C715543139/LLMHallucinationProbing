"""
True-False Dataset 数据加载模块。

支持从 data/raw/ 目录加载 CSV 格式的 True-False 陈述数据，
并提供 PyTorch Dataset 封装，供 DataLoader 使用。
"""

from __future__ import annotations

import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import pandas as pd


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _read_single_csv(filepath: Path) -> pd.DataFrame:
    """读取单个 CSV 文件并添加来源域名字段。

    参数:
        filepath: CSV 文件路径。

    返回:
        DataFrame，包含 statement, label, domain 三列。
    """
    df = pd.read_csv(filepath)
    # 标准化列名（兼容大小写和空格）
    df.columns = df.columns.str.strip().str.lower()
    if "statement" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"文件 {filepath} 必须包含 'statement' 和 'label' 两列，"
            f"实际列名为: {list(df.columns)}"
        )

    # 从文件名中提取领域名（e.g. cities_true_false.csv → cities）
    stem = filepath.stem  # e.g. "cities_true_false"
    domain = stem.replace("_true_false", "")
    df["domain"] = domain

    return df[["statement", "label", "domain"]]


def load_all_raw_data(raw_dir: Path) -> pd.DataFrame:
    """加载 raw/ 目录下全部 True-False CSV 文件并合并为单个 DataFrame。

    参数:
        raw_dir: data/raw/ 目录路径。

    返回:
        合并后的 DataFrame，包含 statement, label, domain 三列。
    """
    csv_files = sorted(raw_dir.glob("*_true_false.csv"))
    if not csv_files:
        raise FileNotFoundError(f"在 {raw_dir} 中未找到任何 *_true_false.csv 文件")

    frames = []
    for fp in csv_files:
        df = _read_single_csv(fp)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # 去除重复语句（保留首次出现）
    combined = combined.drop_duplicates(subset="statement", keep="first")
    combined = combined.reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class TrueFalseDataset(torch.utils.data.Dataset):
    """True-False 陈述数据集。

    每条样本包含:
        - statement: str  – 陈述句文本
        - label: int      – 0 (假) / 1 (真)
        - domain: str     – 所属领域名

    Args:
        statements: 陈述句列表。
        labels: 标签列表 (0/1)。
        domains: 领域名列表。
    """

    def __init__(
        self,
        statements: List[str],
        labels: List[int],
        domains: Optional[List[str]] = None,
    ) -> None:
        if len(statements) != len(labels):
            raise ValueError(
                f"statements 与 labels 长度不匹配: "
                f"{len(statements)} vs {len(labels)}"
            )
        self._statements = list(statements)
        self._labels = [int(l) for l in labels]
        self._domains = list(domains) if domains is not None else [""] * len(statements)

    # ---- 基本信息 ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._statements)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, int]]:
        return {
            "statement": self._statements[idx],
            "label": self._labels[idx],
            "domain": self._domains[idx],
        }

    # ---- 属性 --------------------------------------------------------------

    @property
    def statements(self) -> List[str]:
        return self._statements

    @property
    def labels(self) -> List[int]:
        return self._labels

    @property
    def domains(self) -> List[str]:
        return self._domains

    @property
    def n_true(self) -> int:
        return sum(1 for l in self._labels if l == 1)

    @property
    def n_false(self) -> int:
        return sum(1 for l in self._labels if l == 0)

    @property
    def class_balance(self) -> float:
        """真样本占比。"""
        n = len(self)
        return self.n_true / n if n > 0 else 0.0

    # ---- 统计 --------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"TrueFalseDataset: {len(self)} samples",
            f"  True : {self.n_true}",
            f"  False: {self.n_false}",
            f"  Balance (true ratio): {self.class_balance:.3f}",
        ]
        if self._domains:
            from collections import Counter
            domain_counts = Counter(self._domains)
            lines.append("  Domains:")
            for domain, count in sorted(domain_counts.items()):
                lines.append(f"    {domain}: {count}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 保存/加载预处理后的数据
# ---------------------------------------------------------------------------

def save_dataset(dataset: TrueFalseDataset, filepath: Path) -> None:
    """将数据集保存为 .pt 文件（内部以字典形式存储）。"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "statements": dataset._statements,
        "labels": dataset._labels,
        "domains": dataset._domains,
    }
    torch.save(data, str(filepath))


def load_dataset(filepath: Path) -> TrueFalseDataset:
    """从 .pt 文件加载数据集。"""
    data = torch.load(str(filepath), weights_only=False)
    return TrueFalseDataset(
        statements=data["statements"],
        labels=data["labels"],
        domains=data.get("domains", None),
    )
