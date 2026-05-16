"""测试 Phase 3 分层分析模块。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.phase2.conftest import pick_balanced_examples


class DummyModel:
    class Config:
        num_hidden_layers = 3

    config = Config()


def test_layer_analysis_module_importable() -> None:
    import src.analysis.layer_analysis  # noqa: F401


def test_analyze_layer_performance_selects_best_layer(monkeypatch) -> None:
    import src.analysis.layer_analysis as layer_analysis

    def fake_extract_hidden_states_dataset(model, tokenizer, dataset, pooling, layers, batch_size, max_length):
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        values = {0: 0.20, 1: 0.85, 2: 0.50}
        feature_blocks = [np.full((4, 3), values[layer], dtype=np.float64) for layer in layers]
        return np.stack(feature_blocks, axis=0), labels

    def fake_train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, classifier_type, random_seed):
        score = float(X_train[0, 0])
        return {
            "val": {
                "accuracy": score,
                "macro_f1": score - 0.05,
                "auroc": score + 0.03,
            },
            "test": {
                "accuracy": score - 0.02,
                "macro_f1": score - 0.04,
                "auroc": score + 0.01,
            },
            "classifier": None,
            "scaler": None,
            "y_pred_test": np.array([0, 1, 0, 1], dtype=np.int64),
            "y_score_test": np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float64),
        }

    monkeypatch.setattr(layer_analysis, "extract_hidden_states_dataset", fake_extract_hidden_states_dataset)
    monkeypatch.setattr(layer_analysis, "train_and_evaluate", fake_train_and_evaluate)

    results = layer_analysis.analyze_layer_performance(
        model=DummyModel(),
        tokenizer=None,
        train_dataset=object(),
        val_dataset=object(),
        test_dataset=object(),
        classifier_type="logistic",
        pooling="last",
        layers=[0, 1, 2],
        seeds=(42, 123),
    )

    assert results["best_layer"]["layer_idx"] == 1
    assert len(results["per_layer"]) == 3

    curve = layer_analysis.extract_layer_metric_curve(results, split="test", metric="accuracy")
    assert curve["layer_indices"] == [0, 1, 2]
    assert len(curve["means"]) == 3


@pytest.mark.model
@pytest.mark.slow
def test_layer_analysis_end_to_end_small_pipeline(phase3_loaded_model, data_processed_dir: Path) -> None:
    from src.analysis.layer_analysis import analyze_layer_performance
    from src.data.dataset import load_dataset

    train_ds = load_dataset(data_processed_dir / "train.pt")
    val_ds = load_dataset(data_processed_dir / "val.pt")
    test_ds = load_dataset(data_processed_dir / "test.pt")

    train_statements, train_labels = pick_balanced_examples(train_ds, n_per_label=2)
    val_statements, val_labels = pick_balanced_examples(val_ds, n_per_label=2)
    test_statements, test_labels = pick_balanced_examples(test_ds, n_per_label=2)

    from src.data.dataset import TrueFalseDataset

    small_train = TrueFalseDataset(train_statements, train_labels, ["mini"] * len(train_labels))
    small_val = TrueFalseDataset(val_statements, val_labels, ["mini"] * len(val_labels))
    small_test = TrueFalseDataset(test_statements, test_labels, ["mini"] * len(test_labels))

    model, tokenizer = phase3_loaded_model
    results = analyze_layer_performance(
        model=model,
        tokenizer=tokenizer,
        train_dataset=small_train,
        val_dataset=small_val,
        test_dataset=small_test,
        classifier_type="logistic",
        pooling="last",
        layers=[-2, -1],
        batch_size=2,
        max_length=64,
        seeds=(42,),
    )

    assert len(results["per_layer"]) == 2
    assert results["best_layer"]["layer_idx"] in {26, 27}
