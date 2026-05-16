"""
SAPLMA：基于隐藏状态分类的幻觉检测方法。

参考论文:
    Azaria & Mitchell (2023). The Internal State of an LLM Knows When It's Lying.

方法:
    1. 提取模型指定层的最后 token 隐藏状态作为特征
    2. 训练分类器（逻辑回归 / MLP）判断陈述真伪
    3. 支持多随机种子重复实验，汇报均值 ± 标准差
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import config
from src.features.hidden_states import extract_hidden_states_dataset
from src.utils.reproducibility import set_global_seed
from src.utils.metrics import (
    compute_metrics,
    compute_metrics_multi_seed,
)

if TYPE_CHECKING:
    from src.data.dataset import TrueFalseDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 分类器工厂
# ---------------------------------------------------------------------------

def _create_classifier(classifier_type: str = "logistic") -> object:
    """创建分类器实例。

    参数:
        classifier_type: "logistic" | "mlp"

    返回:
        sklearn 分类器实例（未训练）。
    """
    if classifier_type == "logistic":
        return LogisticRegression(
            C=config.training.logistic_C,
            max_iter=config.training.logistic_max_iter,
            penalty=config.training.logistic_penalty,
            random_state=config.training.random_seeds[0],
            n_jobs=config.training.n_jobs,
        )
    elif classifier_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=config.training.mlp_hidden_sizes,
            activation=config.training.mlp_activation,
            alpha=config.training.mlp_alpha,
            max_iter=config.training.mlp_max_iter,
            random_state=config.training.random_seeds[0],
        )
    else:
        raise ValueError(f"不支持的分类器类型: {classifier_type}")


def train_saplma_classifier(
    X: np.ndarray,
    y: np.ndarray,
    classifier_type: str = "logistic",
    random_state: int = 42,
    max_iter: Optional[int] = None,
):
    """训练一个可直接用于 predict / predict_proba 的 SAPLMA 分类器。"""
    set_global_seed(random_state)

    if classifier_type == "logistic":
        classifier = LogisticRegression(
            C=config.training.logistic_C,
            max_iter=max_iter or config.training.logistic_max_iter,
            penalty=config.training.logistic_penalty,
            random_state=random_state,
            n_jobs=config.training.n_jobs,
        )
    elif classifier_type == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=config.training.mlp_hidden_sizes,
            activation=config.training.mlp_activation,
            alpha=config.training.mlp_alpha,
            max_iter=max_iter or config.training.mlp_max_iter,
            random_state=random_state,
        )
    else:
        raise ValueError(f"不支持的分类器类型: {classifier_type}")

    pipeline = make_pipeline(StandardScaler(), classifier)
    pipeline.fit(X, y)
    return pipeline


fit_saplma_classifier = train_saplma_classifier
train_hidden_state_classifier = train_saplma_classifier


def predict_with_classifier(classifier, X: np.ndarray) -> np.ndarray:
    """使用已训练分类器做标签预测。"""
    return np.asarray(classifier.predict(X), dtype=np.int64)


predict_saplma = predict_with_classifier
predict_labels = predict_with_classifier


def predict_proba_with_classifier(classifier, X: np.ndarray) -> np.ndarray:
    """输出正类概率；若无 `predict_proba`，退化为 0/1 分数。"""
    if hasattr(classifier, "predict_proba"):
        return np.asarray(classifier.predict_proba(X), dtype=np.float64)

    preds = np.asarray(classifier.predict(X), dtype=np.float64)
    return np.stack([1.0 - preds, preds], axis=1)


predict_saplma_proba = predict_proba_with_classifier
predict_probabilities = predict_proba_with_classifier


# ---------------------------------------------------------------------------
# 单次实验
# ---------------------------------------------------------------------------

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    classifier_type: str = "logistic",
    random_seed: int = 42,
) -> Dict:
    """训练分类器并在验证集和测试集上评估。

    参数:
        X_train, y_train: 训练特征与标签.
        X_val, y_val: 验证集.
        X_test, y_test: 测试集.
        classifier_type: 分类器类型.
        random_seed: 随机种子.

    返回:
        包含 train/val/test 指标的字典。
    """
    set_global_seed(random_seed)

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 创建并训练分类器
    if classifier_type == "logistic":
        clf = LogisticRegression(
            C=config.training.logistic_C,
            max_iter=config.training.logistic_max_iter,
            penalty=config.training.logistic_penalty,
            random_state=random_seed,
            n_jobs=config.training.n_jobs,
        )
    elif classifier_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=config.training.mlp_hidden_sizes,
            activation=config.training.mlp_activation,
            alpha=config.training.mlp_alpha,
            max_iter=config.training.mlp_max_iter,
            random_state=random_seed,
        )
    else:
        raise ValueError(f"不支持的分类器类型: {classifier_type}")

    clf.fit(X_train_scaled, y_train)

    # 预测
    y_pred_train = clf.predict(X_train_scaled)
    y_pred_val = clf.predict(X_val_scaled)
    y_pred_test = clf.predict(X_test_scaled)

    # 获取概率分数（用于 AUROC）
    if hasattr(clf, "predict_proba"):
        y_score_train = clf.predict_proba(X_train_scaled)[:, 1]
        y_score_val = clf.predict_proba(X_val_scaled)[:, 1]
        y_score_test = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        y_score_train = y_pred_train.astype(float)
        y_score_val = y_pred_val.astype(float)
        y_score_test = y_pred_test.astype(float)

    return {
        "train": compute_metrics(y_train, y_pred_train, y_score_train),
        "val": compute_metrics(y_val, y_pred_val, y_score_val),
        "test": compute_metrics(y_test, y_pred_test, y_score_test),
        "classifier": clf,
        "scaler": scaler,
        "y_pred_test": y_pred_test,
        "y_score_test": y_score_test,
    }


# ---------------------------------------------------------------------------
# 多种子实验
# ---------------------------------------------------------------------------

def run_saplma_experiment(
    model,
    tokenizer,
    train_dataset: TrueFalseDataset,
    val_dataset: TrueFalseDataset,
    test_dataset: TrueFalseDataset,
    classifier_type: str = "logistic",
    layer_idx: int = -1,
    pooling: str = "last",
    batch_size: int = 8,
    max_length: int = 128,
    seeds=None,
) -> Dict:
    """完整 SAPLMA 实验流水线（多随机种子）。

    1. 提取指定层的隐藏状态
    2. 按多个随机种子分别训练分类器
    3. 汇总均值 ± 标准差

    参数:
        model, tokenizer: 模型与分词器.
        train_dataset: 训练集.
        val_dataset: 验证集.
        test_dataset: 测试集.
        classifier_type: "logistic" | "mlp".
        layer_idx: 层索引（-1 = 最后一层，0-based Transformer block）.
        pooling: 池化策略.
        batch_size: 批大小.
        max_length: 最大 token 长度.
        seeds: 随机种子元组，默认使用 config 中配置.

    返回:
        包含所有实验结果的字典。
    """
    if seeds is None:
        seeds = config.training.random_seeds

    num_layers = model.config.num_hidden_layers
    if layer_idx < 0:
        layer_idx = num_layers + layer_idx
    logger.info("=" * 50)
    logger.info("SAPLMA 实验: layer=%d/%d, pooling=%s, classifier=%s",
                layer_idx, num_layers, pooling, classifier_type)
    logger.info("=" * 50)

    # ---- 提取特征（一次性，不依赖随机种子）------------------------------
    logger.info("提取训练集隐藏状态...")
    X_train, y_train = extract_hidden_states_dataset(
        model, tokenizer, train_dataset,
        pooling=pooling, layers=[layer_idx], batch_size=batch_size, max_length=max_length,
    )
    # extract_hidden_states_dataset 返回 (N, hidden_dim) 当 layers 为单层

    logger.info("提取验证集隐藏状态...")
    X_val, y_val = extract_hidden_states_dataset(
        model, tokenizer, val_dataset,
        pooling=pooling, layers=[layer_idx], batch_size=batch_size, max_length=max_length,
    )

    logger.info("提取测试集隐藏状态...")
    X_test, y_test = extract_hidden_states_dataset(
        model, tokenizer, test_dataset,
        pooling=pooling, layers=[layer_idx], batch_size=batch_size, max_length=max_length,
    )

    logger.info("特征维度: %s", X_train.shape)

    # ---- 多种子训练 -------------------------------------------------------
    all_test_preds: List[np.ndarray] = []
    all_test_scores: List[np.ndarray] = []
    per_seed_results: List[Dict] = []

    for seed in seeds:
        logger.info("--- 随机种子 = %d ---", seed)
        result = train_and_evaluate(
            X_train, y_train, X_val, y_val, X_test, y_test,
            classifier_type=classifier_type, random_seed=seed,
        )
        per_seed_results.append(result)
        all_test_preds.append(result["y_pred_test"])
        all_test_scores.append(result["y_score_test"])
        # 记录指标
        logger.info(
            "  Test: Acc=%.4f, Macro-F1=%.4f, AUROC=%.4f",
            result["test"]["accuracy"],
            result["test"]["macro_f1"],
            result["test"]["auroc"],
        )

    # ---- 汇总 -------------------------------------------------------------
    # 使用多 seed 汇总
    test_summary = compute_metrics_multi_seed(
        [y_test] * len(seeds), all_test_preds, all_test_scores,
    )

    # 平均 test 指标
    logger.info("=== SAPLMA 最终结果 (%d seeds) ===", len(seeds))
    for metric_name, stats in test_summary.items():
        logger.info("  %s: %.4f ± %.4f", metric_name, stats["mean"], stats["std"])

    # 取第一个 seed 作为 representative
    best_seed_result = per_seed_results[0]

    return {
        "method": f"SAPLMA ({classifier_type})",
        "layer_idx": layer_idx,
        "pooling": pooling,
        "classifier_type": classifier_type,
        "num_seeds": len(seeds),
        "seeds": list(seeds),
        "test_summary": test_summary,
        "per_seed": per_seed_results,
        "best_seed_result": best_seed_result,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


# ---------------------------------------------------------------------------
# 便捷入口: 对比 LR 和 MLP
# ---------------------------------------------------------------------------

def run_saplma_full(
    model,
    tokenizer,
    train_dataset: TrueFalseDataset,
    val_dataset: TrueFalseDataset,
    test_dataset: TrueFalseDataset,
    layer_idx: int = -1,
    pooling: str = "last",
    batch_size: int = 8,
    max_length: int = 128,
) -> Dict[str, Dict]:
    """运行 SAPLMA 完整对比（LR + MLP）。

    返回:
        {"logistic": {...}, "mlp": {...}}
    """
    results = {}
    for clf_type in ("logistic", "mlp"):
        results[clf_type] = run_saplma_experiment(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            classifier_type=clf_type,
            layer_idx=layer_idx,
            pooling=pooling,
            batch_size=batch_size,
            max_length=max_length,
        )
    return results
