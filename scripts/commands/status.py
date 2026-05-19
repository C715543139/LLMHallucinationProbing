"""
项目状态脚本：打印 Python、PyTorch、CUDA、预处理数据与模型缓存的当前状态。

用法:
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llm_hallucination
    source ./.venv/bin/activate
    python -s scripts/commands/status.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """打印项目运行状态。"""
    print("=" * 60)
    print("  LLM Hallucination Probing — 项目状态检查")
    print("=" * 60)

    print(f"\nPython: {sys.version}")

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

    try:
        import transformers

        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        print("Transformers: NOT INSTALLED")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    if processed_dir.exists():
        pt_files = list(processed_dir.glob("*.pt"))
        print(f"\n预处理数据 ({len(pt_files)} 文件):")
        for path in sorted(pt_files):
            size_mb = path.stat().st_size / 1024**2
            print(f"  {path.name} ({size_mb:.1f} MB)")
    else:
        print("\n预处理数据: 尚未生成 (运行 preprocess)")

    cache_dir = PROJECT_ROOT / "models_cache"
    if cache_dir.exists():
        models = [path.name for path in cache_dir.iterdir() if path.is_dir() and not path.name.startswith(".")]
        print(f"\n模型缓存: {models}")

    print("\n" + "=" * 60)
    print("  状态检查完成")
    print("=" * 60)


if __name__ == "__main__":
    main()