"""Phase 4 共享 Fixture。"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def phase4_loaded_model(models_cache_dir: Path):
    """为 Phase 4 真实模型测试加载一次模型（eager attention）。"""
    import torch
    from src.models.loader import load_model_fp16

    if not torch.cuda.is_available():
        pytest.skip("无 GPU，跳过 Phase 4 真实模型测试")

    qwen_dir = models_cache_dir / "Qwen2-1.5B"
    if not (qwen_dir / "config.json").exists():
        pytest.skip("本地 Qwen2-1.5B 权重不存在，跳过 Phase 4 真实模型测试")

    model, tokenizer = load_model_fp16(model_path=str(qwen_dir))
    # 通过官方 setter 切换到 eager，确保 output_attentions=True 真正生效。
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    elif hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"

    yield model, tokenizer

    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def sample_statements():
    """测试用样本陈述句。"""
    return [
        "Paris is the capital of France.",
        "The sun rises in the west.",
        "Water is made of hydrogen and oxygen.",
        "Shakespeare wrote the Quran.",
        "Tokyo is located in Japan.",
        "Elephants can fly.",
        "Gold is a precious metal.",
        "The moon is made of cheese.",
    ]


@pytest.fixture(scope="session")
def sample_labels():
    """对应 sample_statements 的标签。"""
    return [1, 0, 1, 0, 1, 0, 1, 0]
