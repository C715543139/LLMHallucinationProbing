"""
数据预处理与划分模块。

功能:
    1. 加载全部原始 CSV 数据
    2. 按领域分层随机划分训练集/验证集/测试集 (8:1:1)
    3. 保存处理后的 .pt 文件到 data/processed/
    4. 提供加载预处理数据的便捷接口
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import config
from src.data.dataset import (
    TrueFalseDataset,
    load_all_raw_data,
    save_dataset,
    load_dataset,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 核心: 数据划分
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
    stratify_by_domain: bool = True,
) -> Tuple[TrueFalseDataset, TrueFalseDataset, TrueFalseDataset]:
    """按领域分层划分数据集为训练集、验证集和测试集。

    参数:
        df: 包含 statement, label, domain 列的 DataFrame。
        train_ratio, val_ratio, test_ratio: 划分比例（应和为 1.0）。
        random_seed: 随机种子。
        stratify_by_domain: 是否按领域分层。

    返回:
        (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio 必须等于 1.0"

    statements = df["statement"].tolist()
    labels = df["label"].tolist()
    domains = df["domain"].tolist()

    # 分层键: 领域 + 标签 组合，确保每层在划分中都有代表
    if stratify_by_domain:
        stratify_key = [f"{d}_{l}" for d, l in zip(domains, labels)]
    else:
        stratify_key = labels

    # Step 1: 分出 test（test_ratio）
    X_temp, X_test, y_temp, y_test, d_temp, d_test, s_temp, s_test = train_test_split(
        statements, labels, domains, stratify_key,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=stratify_key,
    )

    # Step 2: 从剩余中分出 val（val_ratio / (train_ratio + val_ratio)）
    val_relative = val_ratio / (train_ratio + val_ratio)
    if stratify_by_domain:
        stratify_temp = [f"{d}_{l}" for d, l in zip(d_temp, y_temp)]
    else:
        stratify_temp = y_temp

    X_train, X_val, y_train, y_val, d_train, d_val, s_train, s_val = train_test_split(
        X_temp, y_temp, d_temp, stratify_temp,
        test_size=val_relative,
        random_state=random_seed,
        stratify=stratify_temp,
    )

    train_ds = TrueFalseDataset(X_train, y_train, d_train)
    val_ds = TrueFalseDataset(X_val, y_val, d_val)
    test_ds = TrueFalseDataset(X_test, y_test, d_test)

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# 主流程: 一键预处理
# ---------------------------------------------------------------------------

def run_preprocessing(
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    random_seed: Optional[int] = None,
    force: bool = False,
) -> Tuple[TrueFalseDataset, TrueFalseDataset, TrueFalseDataset]:
    """执行完整的数据预处理流水线。

    1. 加载原始 CSV 数据
    2. 划分 train/val/test
    3. 保存为 .pt 文件

    参数:
        raw_dir: 原始数据目录，默认使用 config.paths.data_raw。
        output_dir: 输出目录，默认使用 config.paths.data_processed。
        random_seed: 随机种子，默认使用 config.data.split_seed。
        force: 是否强制重新处理（即使已存在 .pt 文件）。

    返回:
        (train_dataset, val_dataset, test_dataset)
    """
    if raw_dir is None:
        raw_dir = config.paths.data_raw
    if output_dir is None:
        output_dir = config.paths.data_processed
    if random_seed is None:
        random_seed = config.data.split_seed

    train_path = output_dir / config.paths.train_file
    val_path = output_dir / config.paths.val_file
    test_path = output_dir / config.paths.test_file

    # 若已存在且不强制覆盖，直接加载
    if not force and train_path.exists() and val_path.exists() and test_path.exists():
        logger.info("预处理数据已存在，直接从 %s 加载", output_dir)
        return (
            load_dataset(train_path),
            load_dataset(val_path),
            load_dataset(test_path),
        )

    # ---- 加载原始数据 -------------------------------------------------------
    logger.info("从 %s 加载原始数据...", raw_dir)
    df = load_all_raw_data(raw_dir)
    logger.info("共加载 %d 条样本，领域数: %d", len(df), df["domain"].nunique())

    # ---- 划分 ----------------------------------------------------------------
    logger.info(
        "按 %.0f:%.0f:%.0f 比例分层划分 (seed=%d)...",
        config.data.train_ratio * 100,
        config.data.val_ratio * 100,
        config.data.test_ratio * 100,
        random_seed,
    )
    train_ds, val_ds, test_ds = split_data(
        df,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        random_seed=random_seed,
        stratify_by_domain=config.data.stratify_by_domain,
    )

    # ---- 保存 ----------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("保存训练集 (%d) → %s", len(train_ds), train_path)
    save_dataset(train_ds, train_path)
    logger.info("保存验证集 (%d) → %s", len(val_ds), val_path)
    save_dataset(val_ds, val_path)
    logger.info("保存测试集 (%d) → %s", len(test_ds), test_path)
    save_dataset(test_ds, test_path)

    # ---- 统计摘要 -----------------------------------------------------------
    logger.info("=== 数据划分完成 ===")
    logger.info("\n" + train_ds.summary())
    logger.info("\n" + val_ds.summary())
    logger.info("\n" + test_ds.summary())

    return train_ds, val_ds, test_ds


def load_processed_data(
    processed_dir: Optional[Path] = None,
) -> Tuple[TrueFalseDataset, TrueFalseDataset, TrueFalseDataset]:
    """加载已预处理的数据集（train/val/test）。

    参数:
        processed_dir: 处理后数据目录，默认使用 config.paths.data_processed。

    返回:
        (train_dataset, val_dataset, test_dataset)

    异常:
        FileNotFoundError: 若预处理数据不存在，提示先运行 run_preprocessing()。
    """
    if processed_dir is None:
        processed_dir = config.paths.data_processed

    train_path = processed_dir / config.paths.train_file
    val_path = processed_dir / config.paths.val_file
    test_path = processed_dir / config.paths.test_file

    missing = []
    for p in (train_path, val_path, test_path):
        if not p.exists():
            missing.append(str(p))

    if missing:
        raise FileNotFoundError(
            f"预处理数据文件不存在:\n" + "\n".join(f"  - {m}" for m in missing) +
            "\n请先运行 run_preprocessing() 生成数据。"
        )

    return (
        load_dataset(train_path),
        load_dataset(val_path),
        load_dataset(test_path),
    )


# ---------------------------------------------------------------------------
# CLI 入口（便于直接在终端运行预处理）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    run_preprocessing()
