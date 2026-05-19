"""
GPU 与模型加载测试脚本：打印设备信息并执行一次最小模型前向传播。

用法:
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llm_hallucination
    source ./.venv/bin/activate
    python -s scripts/commands/test_gpu.py
"""

from __future__ import annotations

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)


def main() -> None:
    """快速测试 GPU、模型加载与隐藏状态输出。"""
    import torch

    from src.config import config
    from src.models.loader import load_model, print_device_info

    print_device_info()

    print(f"\n加载模型 (Qwen2-1.5B {config.models.primary_dtype})...")
    model, tokenizer = load_model()
    print(f"模型设备: {next(model.parameters()).device}")

    test_input = "The sky is blue."
    inputs = tokenizer(test_input, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states) - 1
    print(f"隐藏状态层数: {num_layers} (不含 embedding)")
    print(f"隐藏维度: {hidden_states[-1].shape[-1]}")
    print(f"序列长度: {hidden_states[-1].shape[1]}")
    print("✅ 模型前向传播测试通过!")


if __name__ == "__main__":
    main()