"""测试 Phase 4 注意力特征提取模块。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.phase2.conftest import pick_balanced_examples


class DummyTokenizer:
    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=128, return_offsets_mapping=False):
        if isinstance(texts, str):
            texts = [texts]

        batch_tokens = []
        batch_offsets = []
        for text in texts:
            tokens = []
            offsets = []
            cursor = 0
            for word in text.split():
                start = text.index(word, cursor)
                end = start + len(word)
                cursor = end
                tokens.append(word)
                offsets.append((start, end))

            batch_tokens.append(tokens[:max_length])
            batch_offsets.append(offsets[:max_length])

        max_seq = max(len(tokens) for tokens in batch_tokens)
        input_ids = []
        attention_mask = []
        offsets = []
        for tokens, token_offsets in zip(batch_tokens, batch_offsets):
            seq_len = len(tokens)
            input_ids.append(list(range(1, seq_len + 1)) + [0] * (max_seq - seq_len))
            attention_mask.append([1] * seq_len + [0] * (max_seq - seq_len))
            offsets.append(token_offsets + [(0, 0)] * (max_seq - seq_len))

        encoding = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        if return_offsets_mapping:
            encoding["offset_mapping"] = torch.tensor(offsets, dtype=torch.long)
        return encoding


class DummyAttentionModel:
    class Config:
        num_hidden_layers = 2
        _attn_implementation = "eager"

    config = Config()

    def parameters(self):
        return iter(())

    def __call__(self, input_ids, attention_mask=None, output_attentions=False, use_cache=False):
        batch, seq_len = input_ids.shape
        assert output_attentions is True
        attentions = []
        for layer_idx in range(self.config.num_hidden_layers):
            layer = torch.zeros((batch, 2, seq_len, seq_len), dtype=torch.float32)
            for batch_idx in range(batch):
                valid_len = int(attention_mask[batch_idx].sum().item())
                for head_idx in range(2):
                    for query_idx in range(valid_len):
                        base = torch.zeros(valid_len, dtype=torch.float32)
                        base[0] = 0.20 + 0.05 * layer_idx
                        base[min(1, valid_len - 1)] = 0.25
                        base[valid_len - 1] = 0.55 - 0.05 * layer_idx
                        base = base / base.sum()
                        layer[batch_idx, head_idx, query_idx, :valid_len] = base
            attentions.append(layer)
        return SimpleNamespace(attentions=tuple(attentions))


def test_attention_module_importable() -> None:
    import src.features.attention  # noqa: F401


def test_extract_subject_relation_spans_uses_leading_phrase() -> None:
    from src.features.attention import extract_subject_relation_spans

    result = extract_subject_relation_spans("Paris is the capital of France.")

    assert result["subject_text"] == "Paris"
    assert result["relation_text"] == "is"


def test_compute_attention_feature_dict_prefers_tail_focus() -> None:
    from src.features.attention import compute_attention_feature_dict

    attention = np.array(
        [
            [
                [0.10, 0.20, 0.70],
                [0.10, 0.20, 0.70],
                [0.05, 0.15, 0.80],
            ],
            [
                [0.15, 0.20, 0.65],
                [0.10, 0.25, 0.65],
                [0.05, 0.10, 0.85],
            ],
        ],
        dtype=np.float64,
    )

    features = compute_attention_feature_dict(
        attentions=[attention],
        attention_mask=np.array([1, 1, 1], dtype=np.int64),
        anchor_tokens={
            "subject_token_indices": [0],
            "relation_token_indices": [1],
            "tail_token_indices": [2],
        },
    )

    assert features["tail_attn_ratio"] > features["subject_attn_ratio"]
    assert features["last_to_tail"] > features["last_to_subject"]
    assert features["sequence_length"] == 3.0


def test_extract_attention_features_dataset_with_dummy_components() -> None:
    from src.data.dataset import TrueFalseDataset
    from src.features.attention import extract_attention_features_dataset

    dataset = TrueFalseDataset(
        statements=["Paris is in France", "The moon is made of cheese"],
        labels=[1, 0],
        domains=["cities", "facts"],
    )

    features, labels, feature_names, metadata = extract_attention_features_dataset(
        model=DummyAttentionModel(),
        tokenizer=DummyTokenizer(),
        dataset=dataset,
        batch_size=2,
        max_length=32,
    )

    assert features.shape == (2, len(feature_names))
    assert labels.tolist() == [1, 0]
    assert len(metadata) == 2
    assert metadata[0]["subject_text"]


@pytest.mark.model
@pytest.mark.slow
def test_extract_attention_features_dataset_real_model_small_batch(
    phase4_loaded_model,
    data_processed_dir: Path,
) -> None:
    from src.data.dataset import TrueFalseDataset, load_dataset
    from src.features.attention import extract_attention_features_dataset

    train_ds = load_dataset(data_processed_dir / "train.pt")
    statements, labels = pick_balanced_examples(train_ds, n_per_label=2)
    small_ds = TrueFalseDataset(statements, labels, ["mini"] * len(labels))

    model, tokenizer = phase4_loaded_model
    features, y, feature_names, metadata = extract_attention_features_dataset(
        model=model,
        tokenizer=tokenizer,
        dataset=small_ds,
        batch_size=2,
        max_length=64,
    )

    assert features.shape[0] == len(y) == len(metadata)
    assert features.shape[1] == len(feature_names)
    assert len(feature_names) >= 10

