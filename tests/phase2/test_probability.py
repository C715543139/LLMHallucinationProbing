"""
P2.1 / P2.2 — 测试基于序列概率（PPL）的方法 `src/methods/probability.py`。

覆盖:
    - 目标文件与模块可正常导入
    - 单条陈述的 PPL 可计算，且返回正且有限的标量
    - 更低的语言模型 loss 应对应更低的 PPL
    - 验证集阈值调优函数存在，且能在合成样本上找到合理阈值
    - （可选集成）真实 Qwen2-1.5B 上可对单条陈述完成 PPL 打分
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch


THRESHOLD_FN_CANDIDATES = (
    "tune_ppl_threshold",
    "find_best_ppl_threshold",
    "optimize_ppl_threshold",
)


def _get_threshold_fn(module: Any) -> Callable:
    for name in THRESHOLD_FN_CANDIDATES:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    pytest.fail(
        "`src.methods.probability` 中缺少阈值调优函数；"
        f"请至少实现以下名称之一: {THRESHOLD_FN_CANDIDATES}"
    )
    raise AssertionError("unreachable")


def _unwrap_threshold(result) -> float:
    if isinstance(result, (int, float)):
        return float(result)
    if isinstance(result, dict):
        for key in ("best_threshold", "threshold", "optimal_threshold"):
            if key in result:
                return float(result[key])
    if isinstance(result, tuple) and len(result) > 0:
        return _unwrap_threshold(result[0])
    pytest.fail(f"无法从返回值中提取阈值: {result!r}")


class DummyTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        token_count = max(2, len(text.split()))
        input_ids = torch.arange(1, token_count + 1).unsqueeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class DummyPPLModel:
    def __init__(self, loss_value: float):
        self.loss_value = float(loss_value)

    def __call__(self, **kwargs):
        assert "input_ids" in kwargs
        assert "labels" in kwargs
        return SimpleNamespace(loss=torch.tensor(self.loss_value, dtype=torch.float32))


class SequentialLossModel:
    def __init__(self, losses):
        self.losses = list(losses)
        self.index = 0

    def __call__(self, **kwargs):
        loss_value = self.losses[self.index]
        self.index += 1
        return SimpleNamespace(loss=torch.tensor(float(loss_value), dtype=torch.float32))


class TestProbabilityImport:
    """模块与文件存在性检查。"""

    def test_probability_file_exists(self, project_root: Path) -> None:
        fpath = project_root / "src" / "methods" / "probability.py"
        assert fpath.exists(), f"缺少 Phase 2 文件: {fpath}"

    def test_module_importable(self) -> None:
        import src.methods.probability  # noqa: F401

    def test_compute_statement_ppl_importable(self) -> None:
        from src.methods.probability import compute_statement_ppl  # noqa: F401

    def test_threshold_tuning_callable_present(self) -> None:
        import src.methods.probability as probability
        _ = _get_threshold_fn(probability)


class TestComputeStatementPPL:
    """基于文档伪代码约束 `compute_statement_ppl` 的最小行为。"""

    TEST_STATEMENT = "The sky is blue."

    def test_returns_positive_scalar(self) -> None:
        from src.methods.probability import compute_statement_ppl

        model = DummyPPLModel(loss_value=1.25)
        tokenizer = DummyTokenizer()
        ppl = compute_statement_ppl(model, tokenizer, self.TEST_STATEMENT)

        value = float(ppl)
        assert value > 0.0

    def test_returns_finite_scalar(self) -> None:
        from src.methods.probability import compute_statement_ppl

        model = DummyPPLModel(loss_value=0.5)
        tokenizer = DummyTokenizer()
        ppl = compute_statement_ppl(model, tokenizer, self.TEST_STATEMENT)

        value = float(ppl)
        assert math.isfinite(value), f"PPL 不是有限值: {value}"

    def test_lower_loss_means_lower_ppl(self) -> None:
        from src.methods.probability import compute_statement_ppl

        tokenizer = DummyTokenizer()
        lower = float(compute_statement_ppl(DummyPPLModel(0.2), tokenizer, self.TEST_STATEMENT))
        higher = float(compute_statement_ppl(DummyPPLModel(1.4), tokenizer, self.TEST_STATEMENT))

        assert lower < higher, "更低的 loss 应对应更低的 perplexity"

    def test_batch_ppl_scores_keep_input_order(self) -> None:
        from src.methods.probability import compute_ppl_scores

        tokenizer = DummyTokenizer()
        model = SequentialLossModel([0.1, 0.5, 1.0])
        scores = compute_ppl_scores(
            model,
            tokenizer,
            ["a", "b", "c"],
            batch_size=2,
        )

        assert scores.shape == (3,)
        assert scores[0] < scores[1] < scores[2], "PPL 分数应按输入顺序一一对应"


class TestPPLThresholdTuning:
    """P2.2: 验证集阈值调优。"""

    def test_threshold_is_between_true_and_false_clusters(self) -> None:
        import src.methods.probability as probability

        tune_threshold = _get_threshold_fn(probability)

        # 约定：PPL 越低越可能为真。
        ppl_scores = [1.05, 1.20, 1.35, 4.20, 4.50, 5.00]
        labels = [1, 1, 1, 0, 0, 0]

        threshold = _unwrap_threshold(tune_threshold(ppl_scores, labels))

        assert 1.35 <= threshold <= 4.50, (
            f"阈值 {threshold:.4f} 未落在真/假样本簇之间，"
            "不利于后续二分类"
        )

    def test_threshold_output_is_scalar_like(self) -> None:
        import src.methods.probability as probability

        tune_threshold = _get_threshold_fn(probability)
        threshold = _unwrap_threshold(tune_threshold([1.0, 2.0, 4.0, 5.0], [1, 1, 0, 0]))

        assert isinstance(threshold, float)
        assert math.isfinite(threshold)

    def test_constant_scores_still_return_finite_threshold(self) -> None:
        import src.methods.probability as probability

        tune_threshold = _get_threshold_fn(probability)
        threshold = _unwrap_threshold(tune_threshold([2.5, 2.5, 2.5, 2.5], [1, 0, 1, 0]))

        assert math.isfinite(threshold)
        assert threshold == pytest.approx(2.5)


@pytest.mark.model
@pytest.mark.slow
class TestProbabilityRealModel:
    """小规模真实模型集成检查。"""

    def test_real_model_statement_ppl_is_positive_and_finite(self, phase2_loaded_model) -> None:
        from src.methods.probability import compute_statement_ppl

        model, tokenizer = phase2_loaded_model
        ppl = compute_statement_ppl(model, tokenizer, "The sky is blue.")

        value = float(ppl)
        assert value > 0.0
        assert math.isfinite(value)

