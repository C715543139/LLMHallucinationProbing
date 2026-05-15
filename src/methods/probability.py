"""
基于生成概率 / 困惑度（PPL）的幻觉检测方法。

思路：
    模型对"真实"陈述的接受度更高 → PPL 更低。
    通过计算语句在模型下的序列困惑度，利用阈值做真伪二分类。

参考文献:
    Azaria & Mitchell (2023). The Internal State of an LLM Knows When It's Lying.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import (
    find_best_threshold,
    evaluate_with_threshold,
)

if TYPE_CHECKING:
    from src.data.dataset import TrueFalseDataset

logger = logging.getLogger(__name__)


def _infer_model_device(model) -> torch.device:
    """兼容真实 HuggingFace 模型与测试中的轻量 dummy model。"""
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration, TypeError):
        return torch.device("cpu")


def compute_statement_ppl(
    model,
    tokenizer,
    statement: str,
    max_length: int = 128,
) -> float:
    """计算单条陈述句的长度归一化困惑度。"""
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
        outputs = model(**inputs, labels=inputs["input_ids"])

    return float(torch.exp(outputs.loss).item())


def tune_ppl_threshold(
    ppl_scores,
    labels,
    metric: str = "f1",
) -> tuple[float, float]:
    """在验证集上调优 PPL 阈值。

    约定: PPL 越低，越可能为真。因此内部先取负号转成“分数越高越倾向正类”。
    """
    ppl_arr = np.asarray(ppl_scores, dtype=np.float64)
    label_arr = np.asarray(labels, dtype=np.int64)
    best_threshold_on_neg_scores, best_metric = find_best_threshold(
        label_arr,
        -ppl_arr,
        metric=metric,
    )
    return float(-best_threshold_on_neg_scores), float(best_metric)


find_best_ppl_threshold = tune_ppl_threshold
optimize_ppl_threshold = tune_ppl_threshold


def compute_ppl_scores(
    model,
    tokenizer,
    statements: list[str],
    batch_size: int = 8,  # 保留参数兼容性，当前逐条计算忽略此参数
    max_length: int = 128,
) -> np.ndarray:
    """逐条计算陈述句的困惑度 (Perplexity)。

    逐条计算以避免 padding 干扰，并复用模型内置 loss 快速计算，
    避免手动构造大词表 CrossEntropyLoss（对 Qwen2 151k 词表内存开销极大）。

    参数:
        model: HuggingFace CausalLM (eval mode).
        tokenizer: 分词器.
        statements: 陈述句文本列表.
        batch_size: 保留参数（当前逐条计算，忽略此值）.
        max_length: 最大 token 长度.

    返回:
        ppl_scores: shape (N,), 每条语句的困惑度。
                    数值越低 → 模型越认可该陈述。
    """
    device = _infer_model_device(model)
    all_ppls: list[float] = []

    dataloader = DataLoader(
        statements,
        batch_size=1,  # 逐条计算以获得每条语句的独立 PPL
        shuffle=False,
        collate_fn=lambda batch: batch,
    )

    for batch_texts in tqdm(dataloader, desc="计算 PPL"):
        # 逐条处理，避免 padding 干扰
        for text in batch_texts:
            try:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                )
            except TypeError:
                inputs = tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            all_ppls.append(compute_statement_ppl(model, tokenizer, text, max_length=max_length))

    return np.array(all_ppls, dtype=np.float64)


def evaluate_ppl_method(
    model,
    tokenizer,
    train_dataset: TrueFalseDataset,
    val_dataset: TrueFalseDataset,
    test_dataset: TrueFalseDataset,
    batch_size: int = 8,
    max_length: int = 128,
    threshold_metric: str = "f1",
) -> dict:
    """完整的 PPL 方法评估流水线。

    1. 在训练集上计算 PPL 分数分布（用于参考）
    2. 在验证集上搜索最优阈值
    3. 在测试集上汇报最终指标

    参数:
        model, tokenizer: 模型与分词器.
        train_dataset: 训练集.
        val_dataset: 验证集.
        test_dataset: 测试集.
        batch_size: 批大小.
        max_length: 最大 token 长度.
        threshold_metric: 阈值优化指标 ("f1" | "accuracy").

    返回:
        results 字典，包含所有指标。
    """
    logger.info("=" * 50)
    logger.info("PPL 方法评估")
    logger.info("=" * 50)

    # ---- Val: 计算 PPL & 搜索阈值 -----------------------------------------
    logger.info("验证集: 计算 PPL (%d 样本)...", len(val_dataset))
    val_ppls = compute_ppl_scores(
        model, tokenizer, val_dataset.statements,
        batch_size=batch_size, max_length=max_length,
    )
    val_labels = np.array(val_dataset.labels)

    best_threshold, best_val_metric = tune_ppl_threshold(
        val_ppls, val_labels, metric=threshold_metric,
    )
    # PPL: 越低越倾向正类 → higher_is_positive=False
    val_metrics = evaluate_with_threshold(
        val_labels, val_ppls, best_threshold, higher_is_positive=False,
    )
    logger.info("验证集结果 (阈值=%.4f):", best_threshold)
    for k, v in val_metrics.items():
        logger.info("  %s: %.4f", k, v)

    # ---- Test: 计算 PPL & 评估 --------------------------------------------
    logger.info("测试集: 计算 PPL (%d 样本)...", len(test_dataset))
    test_ppls = compute_ppl_scores(
        model, tokenizer, test_dataset.statements,
        batch_size=batch_size, max_length=max_length,
    )
    test_labels = np.array(test_dataset.labels)

    test_metrics = evaluate_with_threshold(
        test_labels, test_ppls, best_threshold, higher_is_positive=False,
    )
    logger.info("测试集结果:")
    for k, v in test_metrics.items():
        logger.info("  %s: %.4f", k, v)

    # ---- Train: 参考分数 -------------------------------------------------
    logger.info("训练集: 计算 PPL (%d 样本)...", len(train_dataset))
    train_ppls = compute_ppl_scores(
        model, tokenizer, train_dataset.statements,
        batch_size=batch_size, max_length=max_length,
    )
    train_labels = np.array(train_dataset.labels)
    train_metrics = evaluate_with_threshold(
        train_labels, train_ppls, best_threshold, higher_is_positive=False,
    )

    return {
        "method": "PPL",
        "threshold": float(best_threshold),
        "threshold_metric": threshold_metric,
        "val": val_metrics,
        "test": test_metrics,
        "train": train_metrics,
        "val_ppls": val_ppls,
        "test_ppls": test_ppls,
        "train_ppls": train_ppls,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "train_labels": train_labels,
    }
