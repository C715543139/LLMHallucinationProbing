"""
数据预处理脚本：执行原始数据读取、清洗、划分并生成 data/processed 下的缓存文件。

用法:
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llm_hallucination
    source ./.venv/bin/activate
    python -s scripts/commands/preprocess.py
"""

from __future__ import annotations

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)


def main() -> None:
    """运行预处理流水线。"""
    from src.data.preprocessing import run_preprocessing

    run_preprocessing()


if __name__ == "__main__":
    main()