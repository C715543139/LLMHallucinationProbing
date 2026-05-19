"""测试 head selection 模块。"""

from __future__ import annotations

import numpy as np

from src.methods.phase4_attention import (
    group_feature_indices_by_head,
    score_head_group,
    select_top_heads,
)


class TestGroupFeatures:
    """测试特征分组功能。"""

    def test_basic_grouping(self):
        names = [
            "L13_H00_last_to_subject_mass",
            "L13_H00_last_to_relation_mass",
            "L13_H01_last_to_subject_mass",
            "L14_H00_last_to_subject_mass",
        ]
        groups = group_feature_indices_by_head(names)
        assert len(groups) == 3  # (13,0), (13,1), (14,0)
        assert groups[(13, 0)] == [0, 1]
        assert groups[(13, 1)] == [2]
        assert groups[(14, 0)] == [3]

    def test_non_matching_names(self):
        names = ["some_random_feature", "not_matching_pattern"]
        groups = group_feature_indices_by_head(names)
        assert len(groups) == 0

    def test_mixed_names(self):
        names = [
            "L13_H00_attention_entropy_last",
            "global_bias",
            "L14_H02_max_attention_last",
        ]
        groups = group_feature_indices_by_head(names)
        assert len(groups) == 2
        assert (13, 0) in groups
        assert (14, 2) in groups


class TestScoreHeadGroup:
    """测试单个 head 评分功能。"""

    def test_score_basic(self):
        np.random.seed(42)
        n = 100
        X_train = np.random.randn(n, 10)
        X_val = np.random.randn(30, 10)
        y_train = np.random.randint(0, 2, n)
        y_val = np.random.randint(0, 2, 30)

        result = score_head_group(
            X_train, X_val, y_train, y_val,
            [0, 1, 2], metric="auroc",
        )
        assert "accuracy" in result
        assert "auroc" in result

    def test_score_returns_valid_range(self):
        np.random.seed(42)
        n = 100
        X_train = np.random.randn(n, 5)
        X_val = np.random.randn(30, 5)
        y_train = np.random.randint(0, 2, n)
        y_val = np.random.randint(0, 2, 30)

        result = score_head_group(
            X_train, X_val, y_train, y_val,
            list(range(5)), metric="accuracy",
        )
        assert 0.0 <= result["accuracy"] <= 1.0


class TestSelectTopHeads:
    """测试 top-head 选择功能。"""

    def test_select_basic(self):
        np.random.seed(42)
        n = 200
        n_features = 48  # e.g. 3 heads × 16 features
        feature_names = []
        for layer in [13, 14, 15]:
            for head in range(2):
                for base in ["last_to_subject_mass", "last_to_relation_mass",
                             "attention_entropy_last", "max_attention_last",
                             "last_to_anchor_mass", "last_to_non_anchor_mass",
                             "top3_attention_mass_last", "attention_sink_mass"]:
                    feature_names.append(f"L{layer}_H{head:02d}_{base}")

        X_train = np.random.randn(n, len(feature_names))
        X_val = np.random.randn(50, len(feature_names))
        y_train = np.random.randint(0, 2, n)
        y_val = np.random.randint(0, 2, 50)

        result = select_top_heads(
            X_train, X_val, y_train, y_val,
            feature_names, top_k_heads=3, metric="auroc",
        )

        assert len(result["selected_heads"]) == 3
        assert "selected_feature_indices" in result
        assert len(result["selected_feature_indices"]) > 0

    def test_select_top_k_exceeds_available(self):
        np.random.seed(42)
        n = 50
        feature_names = []
        for layer in [13]:
            for head in range(2):
                for base in ["last_to_subject_mass", "last_to_relation_mass"]:
                    feature_names.append(f"L{layer}_H{head:02d}_{base}")

        X_train = np.random.randn(n, len(feature_names))
        X_val = np.random.randn(20, len(feature_names))
        y_train = np.random.randint(0, 2, n)
        y_val = np.random.randint(0, 2, 20)

        result = select_top_heads(
            X_train, X_val, y_train, y_val,
            feature_names, top_k_heads=10, metric="auroc",
        )

        # 只有 2 个 heads，所以最多选 2 个
        assert len(result["selected_heads"]) <= 2

    def test_all_head_scores_included(self):
        np.random.seed(42)
        n = 50
        feature_names = []
        for layer in [13, 14]:
            for head in range(2):
                for base in ["last_to_subject_mass", "last_to_relation_mass"]:
                    feature_names.append(f"L{layer}_H{head:02d}_{base}")

        X_train = np.random.randn(n, len(feature_names))
        X_val = np.random.randn(20, len(feature_names))
        y_train = np.random.randint(0, 2, n)
        y_val = np.random.randint(0, 2, 20)

        result = select_top_heads(
            X_train, X_val, y_train, y_val,
            feature_names, top_k_heads=2, metric="auroc",
        )

        # all_head_scores 应包含所有 4 个 heads
        assert len(result["all_head_scores"]) == 4
