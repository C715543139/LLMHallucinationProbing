"""测试 attention score 特征提取模块。"""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.features.attention_scores import (
    _compute_attention_entropy,
    _compute_mass,
    _build_feature_names,
    _BASE_FEATURE_NAMES,
)


class TestAttentionUtils:
    """测试基础注意力工具函数。"""

    def test_entropy_uniform(self):
        """均匀分布的熵应接近 log(N)。"""
        n = 10
        attn = np.ones(n) / n
        entropy = _compute_attention_entropy(attn)
        expected = np.log(n)
        assert abs(entropy - expected) < 0.01

    def test_entropy_one_hot(self):
        """one-hot 分布的熵应为 0（接近）。"""
        attn = np.zeros(10)
        attn[0] = 1.0
        entropy = _compute_attention_entropy(attn)
        assert entropy < 0.01

    def test_mass_simple(self):
        attn = np.array([0.1, 0.2, 0.3, 0.4])
        mass = _compute_mass(attn, [0, 1])
        assert abs(mass - 0.3) < 1e-6

    def test_mass_empty_indices(self):
        attn = np.array([0.1, 0.2, 0.3, 0.4])
        mass = _compute_mass(attn, [])
        assert mass == 0.0

    def test_mass_oob_indices(self):
        attn = np.array([0.5, 0.5])
        mass = _compute_mass(attn, [0, 5, 10])
        assert abs(mass - 0.5) < 1e-6


class TestFeatureNames:
    """测试特征名生成。"""

    def test_build_feature_names_count(self):
        layers = [13, 14]
        num_heads = 12
        names = _build_feature_names(layers, num_heads)
        expected_count = len(layers) * num_heads * len(_BASE_FEATURE_NAMES)
        assert len(names) == expected_count

    def test_build_feature_names_format(self):
        names = _build_feature_names([13], 1)
        for name in names:
            assert name.startswith("L13_H00_")

    def test_all_base_features_present(self):
        names = _build_feature_names([13], 1)
        for base in _BASE_FEATURE_NAMES:
            assert any(name.endswith(base) for name in names)


class TestAttentionScoreExtraction:
    """测试注意力分数特征提取（需要真实模型）。"""

    @pytest.mark.model
    def test_extract_single_feature_shape(self, phase4_loaded_model):
        """单条语句应返回正确形状的特征。"""
        model, tokenizer = phase4_loaded_model
        from src.features.attention_scores import extract_attention_score_features_single

        layers = [0, 1]  # 仅提取前两层
        features, names, metadata = extract_attention_score_features_single(
            model, tokenizer, "Paris is the capital of France.", layers
        )

        num_heads = model.config.num_attention_heads
        expected_len = len(layers) * num_heads * len(_BASE_FEATURE_NAMES)
        assert len(features) == expected_len
        assert len(names) == expected_len

    @pytest.mark.model
    def test_extract_no_nan(self, phase4_loaded_model):
        """特征中不应有 NaN。"""
        model, tokenizer = phase4_loaded_model
        from src.features.attention_scores import extract_attention_score_features_single

        features, names, metadata = extract_attention_score_features_single(
            model, tokenizer, "The sky is blue.", [0]
        )
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

    @pytest.mark.model
    def test_extract_attention_mass_valid(self, phase4_loaded_model):
        """注意力质量应在 [0, 1] 范围内。"""
        model, tokenizer = phase4_loaded_model
        from src.features.attention_scores import extract_attention_score_features_single

        features, names, metadata = extract_attention_score_features_single(
            model, tokenizer, "Water is a liquid.", [0]
        )

        for name, val in zip(names, features):
            if "mass" in name:
                assert 0.0 <= val <= 1.0, f"{name}={val} 超出 [0,1]"

    @pytest.mark.model
    def test_extract_entropy_non_negative(self, phase4_loaded_model):
        """注意力熵应非负。"""
        model, tokenizer = phase4_loaded_model
        from src.features.attention_scores import extract_attention_score_features_single

        features, names, metadata = extract_attention_score_features_single(
            model, tokenizer, "Gold is a precious metal.", [0]
        )

        for name, val in zip(names, features):
            if "entropy" in name:
                assert val >= 0.0, f"{name}={val} 为负值"

    @pytest.mark.model
    def test_metadata_contains_anchor_info(self, phase4_loaded_model):
        """metadata 应包含 anchor 信息。"""
        model, tokenizer = phase4_loaded_model
        from src.features.attention_scores import extract_attention_score_features_single

        _, _, metadata = extract_attention_score_features_single(
            model, tokenizer, "Paris is the capital of France.", [0]
        )

        assert "anchor_rule" in metadata
        assert "seq_len" in metadata
        assert "anchor_valid" in metadata

    @pytest.mark.model
    @pytest.mark.slow
    def test_extract_dataset_batch(self, phase4_loaded_model, sample_statements, sample_labels):
        """数据集的批量提取应保持顺序。"""
        model, tokenizer = phase4_loaded_model
        from src.data.dataset import TrueFalseDataset
        from src.features.attention_scores import extract_attention_score_features_dataset

        dataset = TrueFalseDataset(sample_statements[:4], sample_labels[:4])
        result = extract_attention_score_features_dataset(
            model, tokenizer, dataset, layers=[0], batch_size=2,
        )

        assert result["features"].shape[0] == 4
        assert len(result["labels"]) == 4
        assert len(result["feature_names"]) == result["features"].shape[1]
