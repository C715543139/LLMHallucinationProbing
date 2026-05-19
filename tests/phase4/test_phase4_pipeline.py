"""Phase 4 小规模集成测试（使用极小样本验证流程）。"""

from __future__ import annotations

import numpy as np


class TestPhase4Pipeline:
    """测试 Phase 4 核心流水线是否可跑通。"""

    def test_train_eval_classifier_basic(self):
        """train_eval_classifier 应能在随机数据上运行。"""
        from src.methods.phase4_attention import train_eval_classifier

        np.random.seed(42)
        n = 50
        d = 20
        X_train = np.random.randn(n, d)
        X_val = np.random.randn(20, d)
        X_test = np.random.randn(20, d)
        y_train = np.random.randint(0, 2, n)
        y_val = np.random.randint(0, 2, 20)
        y_test = np.random.randint(0, 2, 20)

        result = train_eval_classifier(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            classifier_type="logistic",
            seeds=(42, 123),
        )

        assert "test_summary" in result
        assert "val_summary" in result
        for metric in ["accuracy", "macro_f1", "auroc"]:
            assert metric in result["test_summary"]
            assert "mean" in result["test_summary"][metric]

    def test_gated_fusion_probs(self):
        """gated fusion 概率计算。"""
        from src.methods.phase4_attention import gated_fusion_probs

        hidden = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        fusion = np.array([0.2, 0.8, 0.6, 0.5, 0.1])
        tau = 0.2

        result = gated_fusion_probs(hidden, fusion, tau)

        # |0.1 - 0.5| = 0.4 > 0.2 → keep hidden
        assert result[0] == 0.1
        # |0.3 - 0.5| = 0.2 <= 0.2 → use fusion
        assert result[1] == 0.8
        # |0.5 - 0.5| = 0.0 <= 0.2 → use fusion
        assert result[2] == 0.6
        # |0.7 - 0.5| = 0.2 <= 0.2 → use fusion
        assert result[3] == 0.5
        # |0.9 - 0.5| = 0.4 > 0.2 → keep hidden
        assert result[4] == 0.9

    def test_select_gated_fusion_tau(self):
        """验证 tau 选择功能。"""
        from src.methods.phase4_attention import select_gated_fusion_tau

        np.random.seed(42)
        n = 100
        hidden_probs = np.random.rand(n)
        fusion_probs = np.random.rand(n)
        y = np.random.randint(0, 2, n).astype(np.int64)

        result = select_gated_fusion_tau(
            hidden_probs, fusion_probs, y,
            tau_candidates=(0.05, 0.10, 0.15),
            metric="accuracy",
        )

        assert "best_tau" in result
        assert result["best_tau"] in (0.05, 0.10, 0.15)
        assert len(result["candidates"]) == 3

    def test_summarize_feature_differences(self):
        """特征差异分析功能。"""
        from src.methods.phase4_attention import summarize_feature_differences

        np.random.seed(42)
        n = 100
        d = 10
        X = np.random.randn(n, d)
        y = np.random.randint(0, 2, n).astype(np.int64)
        names = [f"feature_{i}" for i in range(d)]

        rows = summarize_feature_differences(X, y, names)
        assert len(rows) == d
        for row in rows:
            assert "feature_name" in row
            assert "delta" in row
            assert "single_feature_auroc" in row

    def test_build_error_analysis(self):
        """错误分析构建功能。"""
        from src.methods.phase4_attention import build_error_analysis

        statements = ["A", "B", "C", "D"]
        labels = np.array([1, 0, 1, 0])
        hidden_probs = np.array([0.9, 0.4, 0.6, 0.3])
        fusion_probs = np.array([0.8, 0.6, 0.7, 0.2])
        hidden_preds = np.array([1, 0, 1, 0])
        fusion_preds = np.array([1, 1, 1, 0])

        rows = build_error_analysis(
            statements, labels,
            hidden_probs, fusion_probs,
            hidden_preds, fusion_preds,
        )

        assert len(rows) == 4
        case_types = [r["case_type"] for r in rows]
        assert "hidden_correct_fusion_correct" in case_types
        assert "hidden_correct_fusion_wrong" in case_types

    def test_feature_cache_roundtrip(self, tmp_path):
        """特征缓存写入和读取。"""
        from src.utils.feature_cache import save_npz_cache, load_npz_cache, cache_exists

        features = np.random.randn(20, 50).astype(np.float64)
        labels = np.random.randint(0, 2, 20).astype(np.int64)
        names = [f"feat_{i}" for i in range(50)]
        meta = {"version": "test", "layers": [0, 1]}

        path = tmp_path / "test_cache.npz"
        save_npz_cache(path, features, labels, names, meta)

        assert cache_exists(path)

        loaded = load_npz_cache(path)
        assert loaded["features"].shape == features.shape
        assert loaded["labels"].shape == labels.shape
        assert loaded["feature_names"] == names
        assert loaded["metadata"]["version"] == "test"

    def test_full_ablation_minimal(self, tmp_path):
        """最小消融实验流程测试。"""
        from src.methods.phase4_attention import run_phase4_ablation

        np.random.seed(42)
        n = 30
        d_hidden = 64
        d_attn = 32
        d_output = 25

        X_h_train = np.random.randn(n, d_hidden)
        X_h_val = np.random.randn(10, d_hidden)
        X_h_test = np.random.randn(10, d_hidden)

        X_as_train = np.random.randn(n, d_attn)
        X_as_val = np.random.randn(10, d_attn)
        X_as_test = np.random.randn(10, d_attn)

        X_ao_train = np.random.randn(n, d_output)
        X_ao_val = np.random.randn(10, d_output)
        X_ao_test = np.random.randn(10, d_output)

        y_train = np.random.randint(0, 2, n).astype(np.int64)
        y_val = np.random.randint(0, 2, 10).astype(np.int64)
        y_test = np.random.randint(0, 2, 10).astype(np.int64)

        statements = [f"Statement {i}" for i in range(n)]
        test_statements = [f"Test statement {i}" for i in range(10)]

        top_indices = list(range(10))  # 取前 10 个 attention score 特征

        results = run_phase4_ablation(
            hidden_train=X_h_train,
            hidden_val=X_h_val,
            hidden_test=X_h_test,
            attn_score_train=X_as_train,
            attn_score_val=X_as_val,
            attn_score_test=X_as_test,
            attn_output_train=X_ao_train,
            attn_output_val=X_ao_val,
            attn_output_test=X_ao_test,
            top_head_indices=top_indices,
            train_labels=y_train,
            val_labels=y_val,
            test_labels=y_test,
            train_statements=statements,
            test_statements=test_statements,
            classifier_type="logistic",
            seeds=(42, 123),
            output_dir=tmp_path,
        )

        assert "A0_hidden_only" in results
        assert "A1_attention_score_raw" in results
        assert "A4_attention_output_only" in results
        assert "A6_hidden_plus_top_head_attention" in results
        assert "A7_hidden_plus_attention_output" in results
        assert "A8_hidden_plus_all_attention" in results
