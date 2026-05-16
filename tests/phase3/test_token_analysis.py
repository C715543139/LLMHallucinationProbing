"""测试 Phase 3 token pooling 分析模块。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.phase2.conftest import pick_balanced_examples


class DummyModel:
    class Config:
        num_hidden_layers = 4

    config = Config()


def test_token_analysis_module_importable() -> None:
    import src.analysis.token_analysis  # noqa: F401


def test_analyze_token_pooling_selects_best_pooling(monkeypatch) -> None:
    import src.analysis.token_analysis as token_analysis

    def fake_extract_hidden_states_dataset(model, tokenizer, dataset, pooling, layers, batch_size, max_length):
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        values = {"first": 0.35, "mean": 0.60, "last": 0.78}
        features = np.full((4, 3), values[pooling], dtype=np.float64)
        return features, labels

    def fake_train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, classifier_type, random_seed):
        score = float(X_train[0, 0])
        return {
            "val": {
                "accuracy": score,
                "macro_f1": score - 0.03,
                "auroc": score + 0.02,
            },
            "test": {
                "accuracy": score - 0.01,
                "macro_f1": score - 0.02,
                "auroc": score + 0.01,
            },
            "classifier": None,
            "scaler": None,
            "y_pred_test": np.array([0, 1, 0, 1], dtype=np.int64),
            "y_score_test": np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float64),
        }

    monkeypatch.setattr(token_analysis, "extract_hidden_states_dataset", fake_extract_hidden_states_dataset)
    monkeypatch.setattr(token_analysis, "train_and_evaluate", fake_train_and_evaluate)

    results = token_analysis.analyze_token_pooling(
        model=DummyModel(),
        tokenizer=None,
        train_dataset=object(),
        val_dataset=object(),
        test_dataset=object(),
        classifier_type="logistic",
        layer_idx=-1,
        poolings=("first", "mean", "last"),
        seeds=(42, 123),
    )

    assert results["best_pooling"]["pooling"] == "last"

    bars = token_analysis.extract_token_metric_bars(results, split="test", metric="accuracy")
    assert bars["poolings"] == ["first", "mean", "last"]
    assert len(bars["means"]) == 3


@pytest.mark.model
@pytest.mark.slow
def test_token_analysis_end_to_end_small_pipeline(phase3_loaded_model, data_processed_dir: Path) -> None:
    from src.analysis.token_analysis import analyze_token_pooling
    from src.data.dataset import TrueFalseDataset, load_dataset

    train_ds = load_dataset(data_processed_dir / "train.pt")
    val_ds = load_dataset(data_processed_dir / "val.pt")
    test_ds = load_dataset(data_processed_dir / "test.pt")

    train_statements, train_labels = pick_balanced_examples(train_ds, n_per_label=2)
    val_statements, val_labels = pick_balanced_examples(val_ds, n_per_label=2)
    test_statements, test_labels = pick_balanced_examples(test_ds, n_per_label=2)

    small_train = TrueFalseDataset(train_statements, train_labels, ["mini"] * len(train_labels))
    small_val = TrueFalseDataset(val_statements, val_labels, ["mini"] * len(val_labels))
    small_test = TrueFalseDataset(test_statements, test_labels, ["mini"] * len(test_labels))

    model, tokenizer = phase3_loaded_model
    results = analyze_token_pooling(
        model=model,
        tokenizer=tokenizer,
        train_dataset=small_train,
        val_dataset=small_val,
        test_dataset=small_test,
        classifier_type="logistic",
        layer_idx=-1,
        poolings=("last", "first", "mean"),
        batch_size=2,
        max_length=64,
        seeds=(42,),
    )

    assert len(results["per_pooling"]) == 3
    assert results["best_pooling"]["pooling"] in {"last", "first", "mean"}
