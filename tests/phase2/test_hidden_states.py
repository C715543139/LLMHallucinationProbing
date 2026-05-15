"""
P2.3 — 测试隐藏状态特征提取模块 `src/features/hidden_states.py`。

覆盖:
    - 目标文件与模块可正常导入
    - `extract_last_token_hidden` 可正常导入
    - 函数能显式剥离 embedding output，只按 Transformer block 编号
    - 默认 layer_idx=-1 时返回最后一个 block 的最后 token 表示
    - 指定 layer_idx 时能返回对应层表示
    - 返回结果维度与 hidden_size 一致，且无 NaN/Inf
    - 越界层索引应抛出异常
    - （可选集成）真实模型上可提取最后 token 隐藏状态
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


class DummyTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        input_ids = torch.tensor([[11, 12, 13, 14]])
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class DummyHiddenStateModel:
    def __init__(self, num_hidden_layers: int = 3, hidden_size: int = 5):
        self.config = SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
        )

    def __call__(self, **kwargs):
        assert kwargs.get("output_hidden_states") is True

        seq_len = kwargs["input_ids"].shape[1]
        emb = torch.zeros((1, seq_len, self.config.hidden_size), dtype=torch.float32)
        hidden_states = [emb]

        for block_idx in range(1, self.config.num_hidden_layers + 1):
            block = torch.full(
                (1, seq_len, self.config.hidden_size),
                fill_value=float(block_idx),
                dtype=torch.float32,
            )
            hidden_states.append(block)

        return SimpleNamespace(hidden_states=tuple(hidden_states))


class DummyHiddenStateModelNoEmbedding(DummyHiddenStateModel):
    def __call__(self, **kwargs):
        assert kwargs.get("output_hidden_states") is True

        seq_len = kwargs["input_ids"].shape[1]
        hidden_states = []
        for block_idx in range(1, self.config.num_hidden_layers + 1):
            block = torch.full(
                (1, seq_len, self.config.hidden_size),
                fill_value=float(block_idx),
                dtype=torch.float32,
            )
            hidden_states.append(block)

        return SimpleNamespace(hidden_states=tuple(hidden_states))


class TestHiddenStateImport:
    def test_hidden_states_file_exists(self, project_root: Path) -> None:
        fpath = project_root / "src" / "features" / "hidden_states.py"
        assert fpath.exists(), f"缺少 Phase 2 文件: {fpath}"

    def test_module_importable(self) -> None:
        import src.features.hidden_states  # noqa: F401

    def test_extract_last_token_hidden_importable(self) -> None:
        from src.features.hidden_states import extract_last_token_hidden  # noqa: F401


class TestExtractLastTokenHidden:
    TEST_STATEMENT = "Water is H2O."

    def test_default_uses_last_transformer_block(self) -> None:
        from src.features.hidden_states import extract_last_token_hidden

        model = DummyHiddenStateModel(num_hidden_layers=3, hidden_size=4)
        tokenizer = DummyTokenizer()
        hidden = extract_last_token_hidden(model, tokenizer, self.TEST_STATEMENT)

        arr = np.asarray(hidden)
        assert arr.shape[-1] == 4
        assert np.allclose(arr.reshape(-1), 3.0), (
            "默认 layer_idx=-1 时，应返回最后一个 Transformer block 的表示"
        )

    def test_selects_requested_block_index(self) -> None:
        from src.features.hidden_states import extract_last_token_hidden

        model = DummyHiddenStateModel(num_hidden_layers=3, hidden_size=6)
        tokenizer = DummyTokenizer()
        hidden = extract_last_token_hidden(model, tokenizer, self.TEST_STATEMENT, layer_idx=0)

        arr = np.asarray(hidden)
        assert arr.shape[-1] == 6
        assert np.allclose(arr.reshape(-1), 1.0), (
            "layer_idx=0 应指向第一个 Transformer block，而不是 embedding output"
        )

    def test_returns_finite_hidden_vector(self) -> None:
        from src.features.hidden_states import extract_last_token_hidden

        model = DummyHiddenStateModel(num_hidden_layers=2, hidden_size=7)
        tokenizer = DummyTokenizer()
        hidden = extract_last_token_hidden(model, tokenizer, self.TEST_STATEMENT)

        arr = np.asarray(hidden)
        assert np.isfinite(arr).all(), "隐藏状态中不应出现 NaN 或 Inf"

    def test_invalid_layer_index_raises(self) -> None:
        from src.features.hidden_states import extract_last_token_hidden

        model = DummyHiddenStateModel(num_hidden_layers=2, hidden_size=3)
        tokenizer = DummyTokenizer()

        with pytest.raises((IndexError, ValueError)):
            extract_last_token_hidden(model, tokenizer, self.TEST_STATEMENT, layer_idx=5)

    def test_negative_index_maps_to_correct_block(self) -> None:
        from src.features.hidden_states import extract_last_token_hidden

        model = DummyHiddenStateModel(num_hidden_layers=3, hidden_size=4)
        tokenizer = DummyTokenizer()
        hidden = extract_last_token_hidden(model, tokenizer, self.TEST_STATEMENT, layer_idx=-3)

        arr = np.asarray(hidden)
        assert np.allclose(arr.reshape(-1), 1.0), "layer_idx=-num_layers 应映射到第一个 block"

    def test_works_when_hidden_states_exclude_embedding_output(self) -> None:
        from src.features.hidden_states import extract_last_token_hidden

        model = DummyHiddenStateModelNoEmbedding(num_hidden_layers=3, hidden_size=5)
        tokenizer = DummyTokenizer()
        hidden = extract_last_token_hidden(model, tokenizer, self.TEST_STATEMENT, layer_idx=-1)

        arr = np.asarray(hidden)
        assert arr.shape[-1] == 5
        assert np.allclose(arr.reshape(-1), 3.0)


@pytest.mark.model
@pytest.mark.slow
class TestHiddenStateRealModel:
    def test_real_model_hidden_shape_matches_hidden_size(self, phase2_loaded_model) -> None:
        from src.features.hidden_states import extract_last_token_hidden

        model, tokenizer = phase2_loaded_model
        hidden = extract_last_token_hidden(model, tokenizer, "The sky is blue.", layer_idx=-1)

        arr = np.asarray(hidden)
        assert arr.shape[-1] == model.config.hidden_size
        assert np.isfinite(arr).all()

