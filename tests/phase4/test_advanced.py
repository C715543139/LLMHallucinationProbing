"""测试 Phase 4 注意力增强方法。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.phase2.conftest import pick_balanced_examples


class DummyModel:
    class Config:
        num_hidden_layers = 28

    config = Config()


def test_advanced_module_importable() -> None:
    import src.methods.advanced  # noqa: F401


def test_concatenate_feature_blocks_validates_rows() -> None:
    from src.methods.advanced import concatenate_feature_blocks

    left = np.ones((3, 2), dtype=np.float64)
    right = np.zeros((3, 1), dtype=np.float64)
    merged = concatenate_feature_blocks(left, right)
    assert merged.shape == (3, 3)

    with pytest.raises(ValueError):
        concatenate_feature_blocks(left, np.zeros((2, 1), dtype=np.float64))


def test_run_attention_ablation_study_selects_fusion(monkeypatch) -> None:
    import src.methods.advanced as advanced

    def fake_extract_hidden_states_dataset(model, tokenizer, dataset, pooling, layers, batch_size, max_length):
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        features = np.full((4, 3), 0.55, dtype=np.float64)
        return features, labels

    def fake_extract_attention_features_dataset(model, tokenizer, dataset, batch_size, max_length):
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        features = np.full((4, 2), 0.25, dtype=np.float64)
        return features, labels, ["a", "b"], [{"statement": "x"}] * 4

    def fake_train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, classifier_type, random_seed):
        feature_dim = X_train.shape[1]
        score = {2: 0.40, 3: 0.72, 5: 0.86}[feature_dim]
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
        }

    monkeypatch.setattr(advanced, "extract_hidden_states_dataset", fake_extract_hidden_states_dataset)
    monkeypatch.setattr(advanced, "extract_attention_features_dataset", fake_extract_attention_features_dataset)
    monkeypatch.setattr(advanced, "train_and_evaluate", fake_train_and_evaluate)

    results = advanced.run_attention_ablation_study(
        model=DummyModel(),
        tokenizer=None,
        train_dataset=object(),
        val_dataset=object(),
        test_dataset=object(),
        classifier_type="logistic",
        hidden_layer_idx=17,
        hidden_pooling="last",
        seeds=(42, 123),
    )

    assert results["best_variant"]["name"] == "hidden_plus_attention"
    assert set(results["variants"]) == {"attention_only", "hidden_only", "hidden_plus_attention"}
    assert results["variants"]["hidden_plus_attention"]["feature_dim"] == 5


def test_summarize_attention_feature_differences_returns_top_features() -> None:
    from src.features.attention import summarize_attention_feature_differences

    features = np.array(
        [
            [0.1, 0.8],
            [0.2, 0.9],
            [0.8, 0.2],
            [0.9, 0.1],
        ],
        dtype=np.float64,
    )
    labels = np.array([1, 1, 0, 0], dtype=np.int64)

    summary = summarize_attention_feature_differences(features, labels, ["f1", "f2"], top_k=1)

    assert summary["top_features"][0]["name"] in {"f1", "f2"}
    assert len(summary["per_feature"]) == 2


@pytest.mark.model
@pytest.mark.slow
def test_run_attention_ablation_end_to_end_small_pipeline(
    phase4_loaded_model,
    data_processed_dir: Path,
) -> None:
    from src.data.dataset import TrueFalseDataset, load_dataset
    from src.methods.advanced import run_attention_ablation_study

    train_ds = load_dataset(data_processed_dir / "train.pt")
    val_ds = load_dataset(data_processed_dir / "val.pt")
    test_ds = load_dataset(data_processed_dir / "test.pt")

    train_statements, train_labels = pick_balanced_examples(train_ds, n_per_label=2)
    val_statements, val_labels = pick_balanced_examples(val_ds, n_per_label=2)
    test_statements, test_labels = pick_balanced_examples(test_ds, n_per_label=2)

    small_train = TrueFalseDataset(train_statements, train_labels, ["mini"] * len(train_labels))
    small_val = TrueFalseDataset(val_statements, val_labels, ["mini"] * len(val_labels))
    small_test = TrueFalseDataset(test_statements, test_labels, ["mini"] * len(test_labels))

    model, tokenizer = phase4_loaded_model
    results = run_attention_ablation_study(
        model=model,
        tokenizer=tokenizer,
        train_dataset=small_train,
        val_dataset=small_val,
        test_dataset=small_test,
        classifier_type="logistic",
        hidden_layer_idx=17,
        hidden_pooling="last",
        batch_size=2,
        max_length=64,
        seeds=(42,),
    )

    assert set(results["variants"]) == {"attention_only", "hidden_only", "hidden_plus_attention"}
    assert results["best_variant"]["name"] in {"attention_only", "hidden_only", "hidden_plus_attention"}
    assert len(results["attention_feature_names"]) == results["num_attention_features"]

