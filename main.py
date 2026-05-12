"""
LLM Hallucination Probing — 主入口。

用法:
    python -s main.py              # 显示项目状态
    python -s main.py preprocess   # 运行数据预处理
    python -s main.py test-gpu     # 测试 GPU 与模型加载

注意: 运行前必须依次激活环境:
    conda activate llm_hallucination
    .\.venv\Scripts\activate.ps1
"""

from __future__ import annotations

import sys
from pathlib import Path


def status() -> None:
    """打印项目状态：环境、数据、模型。"""
    print("=" * 60)
    print("  LLM Hallucination Probing — Phase 1 状态检查")
    print("=" * 60)

    # Python
    print(f"\nPython: {sys.version}")

    # PyTorch / CUDA
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"VRAM: {mem_total:.1f} GB")
    except ImportError:
        print("PyTorch: NOT INSTALLED")

    # Transformers
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: NOT INSTALLED")

    # 数据
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        pt_files = list(processed_dir.glob("*.pt"))
        print(f"\n预处理数据 ({len(pt_files)} 文件):")
        for f in sorted(pt_files):
            size_mb = f.stat().st_size / 1024**2
            print(f"  {f.name} ({size_mb:.1f} MB)")
    else:
        print("\n预处理数据: 尚未生成 (运行 preprocess)")

    # 模型缓存
    cache_dir = Path("models_cache")
    if cache_dir.exists():
        models = [d.name for d in cache_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        print(f"\n模型缓存: {models}")

    print("\n" + "=" * 60)
    print("  Phase 1 (P1.1-P1.8) 已完成 ✅")
    print("=" * 60)


def preprocess() -> None:
    """运行数据预处理流水线。"""
    from src.data.preprocessing import run_preprocessing
    run_preprocessing()


def test_gpu() -> None:
    """快速测试 GPU 与模型加载。"""
    from src.models.loader import print_device_info, load_model
    import torch

    print_device_info()

    print("\n加载模型 (Qwen2-1.5B FP16)...")
    model, tokenizer = load_model(use_4bit=False)
    print(f"模型设备: {next(model.parameters()).device}")

    # 简单前向传播测试
    test_input = "The sky is blue."
    inputs = tokenizer(test_input, return_tensors="pt")
    # 移至模型设备
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states
    num_layers = len(hidden) - 1  # 减去 embedding output
    print(f"隐藏状态层数: {num_layers} (不含 embedding)")
    print(f"隐藏维度: {hidden[-1].shape[-1]}")
    print(f"序列长度: {hidden[-1].shape[1]}")
    print("✅ 模型前向传播测试通过!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Hallucination Probing")
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=["status", "preprocess", "test-gpu"],
        help="要执行的命令 (默认: status)",
    )
    args = parser.parse_args()

    if args.command == "status":
        status()
    elif args.command == "preprocess":
        preprocess()
    elif args.command == "test-gpu":
        test_gpu()
