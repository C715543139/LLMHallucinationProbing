"""测试 attention output 特征提取模块。"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.features.attention_outputs import (
    AttentionOutputExtractor,
    _extract_stats,
    _cosine_similarity,
)


class TestStatsFunctions:
    """测试统计函数。"""

    def test_extract_stats_basic(self):
        vec = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
        stats = _extract_stats(vec)
        assert "norm" in stats
        assert "mean_abs" in stats
        assert "max_abs" in stats
        assert "std" in stats
        assert "sparsity_1e-3" in stats
        assert stats["max_abs"] == 3.0
        assert 0.0 <= stats["sparsity_1e-3"] <= 1.0

    def test_extract_stats_all_zero(self):
        vec = np.zeros(10)
        stats = _extract_stats(vec)
        assert stats["norm"] == 0.0
        assert stats["mean_abs"] == 0.0
        assert stats["sparsity_1e-3"] == 1.0

    def test_cosine_similarity_same(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        cos = _cosine_similarity(a, b)
        assert abs(cos - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        cos = _cosine_similarity(a, b)
        assert abs(cos) < 1e-6

    def test_cosine_similarity_opposite(self):
        a = np.array([1.0, 2.0])
        b = np.array([-1.0, -2.0])
        cos = _cosine_similarity(a, b)
        assert abs(cos + 1.0) < 1e-6

    def test_cosine_similarity_zero_vector(self):
        a = np.array([1.0, 2.0])
        b = np.array([0.0, 0.0])
        cos = _cosine_similarity(a, b)
        assert not np.isnan(cos)


class TestAttentionOutputExtractor:
    """测试 AttentionOutputExtractor 类。"""

    def test_init(self):
        mock_model = MagicMock()
        extractor = AttentionOutputExtractor(mock_model, [0, 1])
        assert extractor.layers == [0, 1]
        assert len(extractor.handles) == 0
        assert len(extractor.outputs) == 0

    def test_clear(self):
        mock_model = MagicMock()
        extractor = AttentionOutputExtractor(mock_model, [0])
        extractor.outputs[0] = "test"
        extractor.clear()
        assert len(extractor.outputs) == 0

    def test_remove_handles(self):
        mock_model = MagicMock()
        mock_model.model.layers = [MagicMock() for _ in range(2)]
        for i in range(2):
            mock_model.model.layers[i].self_attn = MagicMock()
            mock_model.model.layers[i].self_attn.register_forward_hook = MagicMock(
                return_value=MagicMock()
            )

        extractor = AttentionOutputExtractor(mock_model, [0, 1])
        extractor.register()
        assert len(extractor.handles) == 2
        handles = list(extractor.handles)
        extractor.remove()
        for h in handles:
            h.remove.assert_called_once()
        assert extractor.handles == []

    @pytest.mark.model
    def test_hook_with_real_model(self, phase4_loaded_model):
        """使用真实模型测试 hook 是否能捕获输出。"""
        model, tokenizer = phase4_loaded_model
        import torch

        extractor = AttentionOutputExtractor(model, [0])
        extractor.register()

        try:
            inputs = tokenizer("Hello world", return_tensors="pt", truncation=True, max_length=128)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                _ = model(**inputs)

            assert 0 in extractor.outputs
            out = extractor.outputs[0]
            # 应该是 (batch=1, seq_len, hidden_dim)
            assert out.ndim == 3
            assert out.shape[0] == 1
        finally:
            extractor.remove()

    @pytest.mark.model
    def test_output_shape_consistent(self, phase4_loaded_model):
        """每层输出应有相同形状。"""
        model, tokenizer = phase4_loaded_model
        import torch

        layers = [0, 1, 2]
        extractor = AttentionOutputExtractor(model, layers)
        extractor.register()

        try:
            inputs = tokenizer("Test sentence for shape check.", return_tensors="pt", truncation=True, max_length=128)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                _ = model(**inputs)

            shapes = []
            for l in layers:
                if l in extractor.outputs:
                    shapes.append(extractor.outputs[l].shape)

            # 所有层输出应有相同形状
            if len(shapes) > 1:
                assert all(s == shapes[0] for s in shapes), f"形状不一致: {shapes}"
        finally:
            extractor.remove()

    @pytest.mark.model
    def test_extract_attention_output_features_no_nan(self, phase4_loaded_model):
        """attention output 特征不应有 NaN。"""
        model, tokenizer = phase4_loaded_model
        from src.features.attention_outputs import extract_attention_output_features_single

        features, names, metadata = extract_attention_output_features_single(
            model, tokenizer, "Paris is in France.", [0, 1], pooling="last",
        )
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
        assert len(features) == len(names)
        assert len(features) > 0
