"""Phase 3 共享 Fixture。"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def phase3_loaded_model(models_cache_dir: Path):
    """为 Phase 3 真实模型测试加载一次模型。"""
    import torch
    from src.models.loader import load_model_fp16

    if not torch.cuda.is_available():
        pytest.skip("无 GPU，跳过 Phase 3 真实模型测试")

    qwen_dir = models_cache_dir / "Qwen2-1.5B"
    if not (qwen_dir / "config.json").exists():
        pytest.skip("本地 Qwen2-1.5B 权重不存在，跳过 Phase 3 真实模型测试")

    model, tokenizer = load_model_fp16(model_path=str(qwen_dir))
    yield model, tokenizer

    del model
    torch.cuda.empty_cache()