"""
P2.4 / P2.5 / P2.6 / M2 — 测试 SAPLMA 分类模块 `src/methods/saplma.py`。

覆盖:
    - 目标文件与模块可正常导入
    - 模块中存在训练与预测接口（兼容常见命名）
    - 逻辑回归分类器可在线性可分数据上完成训练与预测
    - MLP 分类器可在简单数据上完成训练与预测
    - 分类器能输出概率分数，以支持 AUROC 计算
    - 小规模端到端流程：真实模型提取隐藏状态 -> 训练分类器 -> 计算 Accuracy / Macro-F1 / AUROC
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest

from tests.phase2.conftest import ensure_2d_feature_matrix, pick_balanced_examples


TRAIN_FN_CANDIDATES = (
    "train_saplma_classifier",
    "train_hidden_state_classifier",
    "fit_saplma_classifier",
)
PREDICT_FN_CANDIDATES = (
    "predict_with_classifier",
    "predict_saplma",
    "predict_labels",
)
PREDICT_PROBA_FN_CANDIDATES = (
    "predict_proba_with_classifier",
    "predict_saplma_proba",
    "predict_probabilities",
)


def _get_train_fn(module: Any) -> Callable:
    for name in TRAIN_FN_CANDIDATES:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    pytest.fail(
        "`src.methods.saplma` 中缺少训练接口；"
        f"请至少实现以下名称之一: {TRAIN_FN_CANDIDATES}"
    )
    raise AssertionError("unreachable")


def _get_predict_fn(module: Any) -> Optional[Callable]:
    for name in PREDICT_FN_CANDIDATES:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def _get_predict_proba_fn(module: Any) -> Optional[Callable]:
    for name in PREDICT_PROBA_FN_CANDIDATES:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def _train_classifier(train_fn, X, y, classifier_type: str):
    signature = inspect.signature(train_fn)
    kwargs = {}

    if "classifier_type" in signature.parameters:
        kwargs["classifier_type"] = classifier_type
    elif "model_type" in signature.parameters:
        kwargs["model_type"] = classifier_type

    if "random_state" in signature.parameters:
        kwargs["random_state"] = 42
    if "max_iter" in signature.parameters:
        kwargs["max_iter"] = 500

    return train_fn(X, y, **kwargs)


def _predict_labels(module: Any, classifier: Any, X: Any):
    predict_fn = _get_predict_fn(module)
    if predict_fn is not None:
        return np.asarray(predict_fn(classifier, X))
    if hasattr(classifier, "predict"):
        return np.asarray(classifier.predict(X))
    pytest.fail("既没有预测函数，也没有 classifier.predict 方法")
    raise AssertionError("unreachable")


def _predict_scores(module: Any, classifier: Any, X: Any):
    predict_proba_fn = _get_predict_proba_fn(module)
    scores = None

    if predict_proba_fn is not None:
        scores = np.asarray(predict_proba_fn(classifier, X))
    elif hasattr(classifier, "predict_proba"):
        scores = np.asarray(classifier.predict_proba(X))
    else:
        pytest.fail("缺少概率输出接口，无法支持 AUROC 评估")
        raise AssertionError("unreachable")

    if scores.ndim == 2 and scores.shape[1] >= 2:
        return scores[:, 1]
    return scores.reshape(-1)


class TestSAPLMAImport:
    def test_saplma_file_exists(self, project_root: Path) -> None:
        fpath = project_root / "src" / "methods" / "saplma.py"
        assert fpath.exists(), f"缺少 Phase 2 文件: {fpath}"

    def test_module_importable(self) -> None:
        import src.methods.saplma  # noqa: F401

    def test_training_callable_present(self) -> None:
        import src.methods.saplma as saplma
        _ = _get_train_fn(saplma)


class TestSAPLMAClassifierTraining:
    @pytest.fixture
    def separable_data(self):
        X_train = np.array(
            [
                [-2.0, -1.5],
                [-1.8, -1.2],
                [-1.5, -1.0],
                [1.2, 1.4],
                [1.6, 1.7],
                [2.0, 1.9],
            ],
            dtype=float,
        )
        y_train = np.array([0, 0, 0, 1, 1, 1], dtype=int)

        X_test = np.array(
            [
                [-1.7, -1.1],
                [-1.3, -0.9],
                [1.4, 1.5],
                [1.8, 1.6],
            ],
            dtype=float,
        )
        y_test = np.array([0, 0, 1, 1], dtype=int)
        return X_train, y_train, X_test, y_test

    def test_logistic_classifier_can_fit_simple_data(self, separable_data) -> None:
        import src.methods.saplma as saplma
        from sklearn.metrics import accuracy_score

        X_train, y_train, X_test, y_test = separable_data
        train_fn = _get_train_fn(saplma)
        classifier = _train_classifier(train_fn, X_train, y_train, classifier_type="logistic")

        preds = _predict_labels(saplma, classifier, X_test)
        assert preds.shape[0] == X_test.shape[0]
        assert set(preds.tolist()) <= {0, 1}
        assert accuracy_score(y_test, preds) >= 0.75

    def test_mlp_classifier_can_fit_simple_data(self, separable_data) -> None:
        import src.methods.saplma as saplma
        from sklearn.metrics import accuracy_score

        X_train, y_train, X_test, y_test = separable_data
        train_fn = _get_train_fn(saplma)
        classifier = _train_classifier(train_fn, X_train, y_train, classifier_type="mlp")

        preds = _predict_labels(saplma, classifier, X_test)
        assert preds.shape[0] == X_test.shape[0]
        assert set(preds.tolist()) <= {0, 1}
        assert accuracy_score(y_test, preds) >= 0.50

    def test_probability_scores_available_for_auroc(self, separable_data) -> None:
        import src.methods.saplma as saplma

        X_train, y_train, X_test, _ = separable_data
        train_fn = _get_train_fn(saplma)
        classifier = _train_classifier(train_fn, X_train, y_train, classifier_type="logistic")

        scores = _predict_scores(saplma, classifier, X_test)
        assert scores.shape[0] == X_test.shape[0]
        assert np.isfinite(scores).all()

    def test_invalid_classifier_type_raises(self, separable_data) -> None:
        import src.methods.saplma as saplma

        X_train, y_train, _, _ = separable_data
        train_fn = _get_train_fn(saplma)

        with pytest.raises(ValueError):
            _train_classifier(train_fn, X_train, y_train, classifier_type="svm")

    def test_probability_helper_falls_back_to_binary_scores(self) -> None:
        import src.methods.saplma as saplma

        class PredictOnlyClassifier:
            def predict(self, X):
                return np.array([0, 1, 1], dtype=int)

        if not hasattr(saplma, "predict_proba_with_classifier"):
            pytest.skip("模块未暴露 predict_proba_with_classifier")

        scores = saplma.predict_proba_with_classifier(
            PredictOnlyClassifier(),
            np.zeros((3, 2), dtype=float),
        )

        assert scores.shape == (3, 2)
        assert np.allclose(scores.sum(axis=1), 1.0)


@pytest.mark.model
@pytest.mark.slow
class TestMilestoneM2:
    """Phase 2 小规模端到端流程检查。"""

    def test_saplma_end_to_end_pipeline_runs(self, phase2_loaded_model, data_processed_dir: Path) -> None:
        from src.data.dataset import load_dataset
        from src.features.hidden_states import extract_last_token_hidden
        import src.methods.saplma as saplma
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        train_pt = data_processed_dir / "train.pt"
        val_pt = data_processed_dir / "val.pt"
        test_pt = data_processed_dir / "test.pt"
        for fpath in (train_pt, val_pt, test_pt):
            if not fpath.exists():
                pytest.skip(f"缺少预处理数据文件: {fpath}")

        train_ds = load_dataset(train_pt)
        val_ds = load_dataset(val_pt)
        test_ds = load_dataset(test_pt)

        train_statements, train_labels = pick_balanced_examples(train_ds, n_per_label=3)
        _val_statements, _val_labels = pick_balanced_examples(val_ds, n_per_label=2)
        test_statements, test_labels = pick_balanced_examples(test_ds, n_per_label=2)

        model, tokenizer = phase2_loaded_model

        train_rows = [
            extract_last_token_hidden(model, tokenizer, stmt, layer_idx=-1)
            for stmt in train_statements
        ]
        test_rows = [
            extract_last_token_hidden(model, tokenizer, stmt, layer_idx=-1)
            for stmt in test_statements
        ]

        X_train = ensure_2d_feature_matrix(train_rows)
        X_test = ensure_2d_feature_matrix(test_rows)
        y_train = np.asarray(train_labels, dtype=int)
        y_test = np.asarray(test_labels, dtype=int)

        train_fn = _get_train_fn(saplma)
        classifier = _train_classifier(train_fn, X_train, y_train, classifier_type="logistic")

        preds = _predict_labels(saplma, classifier, X_test)
        scores = _predict_scores(saplma, classifier, X_test)

        accuracy = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        auroc = roc_auc_score(y_test, scores)

        for name, value in {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "auroc": auroc,
        }.items():
            assert math.isfinite(value), f"{name} 不是有限值"
            assert 0.0 <= value <= 1.0, f"{name} 超出 [0, 1] 区间: {value}"

