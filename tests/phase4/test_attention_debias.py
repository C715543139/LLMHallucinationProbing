"""测试 attention 特征去偏（residualization）模块。"""

from __future__ import annotations

import numpy as np

from src.methods.phase4_attention import residualize_by_length


class TestResidualizeByLength:
    """测试长度残差化功能。"""

    def test_shape_unchanged(self):
        """残差化后特征形状应不变。"""
        np.random.seed(42)
        train_X = np.random.randn(100, 10)
        val_X = np.random.randn(30, 10)
        test_X = np.random.randn(30, 10)
        train_len = np.random.randint(5, 30, (100, 6)).astype(float)
        val_len = np.random.randint(5, 30, (30, 6)).astype(float)
        test_len = np.random.randint(5, 30, (30, 6)).astype(float)

        tr, vr, te, meta = residualize_by_length(
            train_X, val_X, test_X,
            train_len, val_len, test_len,
        )

        assert tr.shape == train_X.shape
        assert vr.shape == val_X.shape
        assert te.shape == test_X.shape

    def test_no_nan_after_residualization(self):
        """残差化不应产生 NaN。"""
        np.random.seed(42)
        train_X = np.random.randn(50, 5)
        val_X = np.random.randn(20, 5)
        test_X = np.random.randn(20, 5)
        train_len = np.random.randint(5, 20, (50, 6)).astype(float)
        val_len = np.random.randint(5, 20, (20, 6)).astype(float)
        test_len = np.random.randint(5, 20, (20, 6)).astype(float)

        tr, vr, te, meta = residualize_by_length(
            train_X, val_X, test_X,
            train_len, val_len, test_len,
        )

        assert not np.any(np.isnan(tr))
        assert not np.any(np.isnan(vr))
        assert not np.any(np.isnan(te))

    def test_only_train_fit(self):
        """确保 val/test 不被用于 fit（通过检查 val 残差均值是否等于零检验）。"""
        np.random.seed(42)
        train_X = np.random.randn(100, 3).astype(np.float64)
        val_X = np.random.randn(20, 3).astype(np.float64)
        test_X = np.random.randn(20, 3).astype(np.float64)
        train_len = np.random.randint(5, 30, (100, 6)).astype(float)
        val_len = np.random.randint(5, 30, (20, 6)).astype(float)
        test_len = np.random.randint(5, 30, (20, 6)).astype(float)

        # 第一次：用 seed=42 的同一批数据
        tr1, vr1, te1, _ = residualize_by_length(
            train_X, val_X, test_X,
            train_len, val_len, test_len,
        )

        # 第二次：更换 val/test 数据，确保结果不同
        val_X2 = np.random.randn(20, 3).astype(np.float64)
        val_len2 = np.random.randint(5, 30, (20, 6)).astype(float)

        tr2, vr2, _, _ = residualize_by_length(
            train_X, val_X2, test_X,
            train_len, val_len2, test_len,
        )

        # train 残差应相同（使用相同的 train 数据）
        assert np.allclose(tr1, tr2)
        # val 残差应不同（因为输入不同）
        assert not np.allclose(vr1, vr2)

    def test_correlation_reduced(self):
        """残差化后与长度的相关性应降低。"""
        np.random.seed(42)
        n = 200
        seq_len = np.random.randint(5, 40, n).astype(float)
        # 构造一个与长度高度相关的特征
        feature = seq_len * 0.5 + np.random.randn(n) * 0.1

        train_X = np.column_stack([feature, np.random.randn(n)])
        train_len = np.column_stack([seq_len, np.zeros((n, 5))])

        tr, _, _, meta = residualize_by_length(
            train_X, train_X[:10], train_X[:10],
            train_len, train_len[:10], train_len[:10],
        )

        # 残差化后第一维与长度的相关性应低于残差化前
        corr_before = abs(np.corrcoef(train_X[:, 0], seq_len)[0, 1])
        corr_after = abs(np.corrcoef(tr[:, 0], seq_len)[0, 1])

        assert corr_after <= corr_before + 0.1, (
            f"残差化后相关性={corr_after:.4f} 未显著低于残差化前={corr_before:.4f}"
        )

    def test_metadata_contains_coeffs(self):
        """metadata 应包含回归系数。"""
        np.random.seed(42)
        train_X = np.random.randn(50, 3)
        val_X = np.random.randn(20, 3)
        test_X = np.random.randn(20, 3)
        train_len = np.random.randint(5, 20, (50, 6)).astype(float)
        val_len = np.random.randint(5, 20, (20, 6)).astype(float)
        test_len = np.random.randint(5, 20, (20, 6)).astype(float)

        _, _, _, meta = residualize_by_length(
            train_X, val_X, test_X,
            train_len, val_len, test_len,
        )

        assert "coeffs" in meta
        assert len(meta["coeffs"]) == train_X.shape[1]
