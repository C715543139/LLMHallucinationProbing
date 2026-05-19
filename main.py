"""
LLM Hallucination Probing 命令分发入口。

职责:
    - 统一保留项目级 CLI 入口
    - 将具体实现转发到 scripts/commands 或 scripts/run 下的脚本
    - 原样透传命令后的附加参数，例如:
        python -s main.py phase4 --summary-only
        python -s main.py phase4 --use-cache --cache-dir experiments/results/phase4/1

用法:
    python -s main.py
    python -s main.py status
    python -s main.py preprocess
    python -s main.py test-gpu
    python -s main.py phase2
    python -s main.py phase2-ppl
    python -s main.py phase2-saplma
    python -s main.py phase3
    python -s main.py phase3-layer
    python -s main.py phase3-token
    python -s main.py phase4
    python -s main.py phase4-cache-hidden
    python -s main.py phase4-hidden-baseline
    python -s main.py phase4-extract-attention-scores
    python -s main.py phase4-extract-attention-outputs
    python -s main.py phase4-select-heads
    python -s main.py phase4-ablation
    python -s main.py phase4-visualize
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

COMMAND_MAP: dict[str, tuple[str, list[str]]] = {
    "status": ("scripts/commands/status.py", []),
    "preprocess": ("scripts/commands/preprocess.py", []),
    "test-gpu": ("scripts/commands/test_gpu.py", []),
    "phase2": ("scripts/run/phase2.py", []),
    "phase2-ppl": ("scripts/run/phase2.py", ["ppl"]),
    "phase2-saplma": ("scripts/run/phase2.py", ["saplma"]),
    "phase3": ("scripts/run/phase3.py", []),
    "phase3-layer": ("scripts/run/phase3.py", ["layer"]),
    "phase3-token": ("scripts/run/phase3.py", ["token"]),
    "phase4": ("scripts/run/phase4.py", []),
    "phase4-cache-hidden": ("scripts/run/phase4.py", ["cache-hidden"]),
    "phase4-hidden-baseline": ("scripts/run/phase4.py", ["hidden-baseline"]),
    "phase4-extract-attention-scores": ("scripts/run/phase4.py", ["extract-attention-scores"]),
    "phase4-extract-attention-outputs": ("scripts/run/phase4.py", ["extract-attention-outputs"]),
    "phase4-select-heads": ("scripts/run/phase4.py", ["select-heads"]),
    "phase4-ablation": ("scripts/run/phase4.py", ["ablation"]),
    "phase4-visualize": ("scripts/run/phase4.py", ["visualize"]),
}


def print_help() -> None:
    """打印顶层 CLI 帮助。"""
    print("LLM Hallucination Probing")
    print("")
    print("用法:")
    print("  python -s main.py [command] [command options]")
    print("")
    print("命令:")
    for command in COMMAND_MAP:
        print(f"  {command}")
    print("")
    print("说明:")
    print("  - 不带 command 时默认运行 status")
    print("  - command 后的参数会原样转发到对应脚本")
    print("  - 例如: python -s main.py phase4 --summary-only")


def resolve_command(argv: list[str]) -> tuple[str, list[str]]:
    """解析顶层命令和需要转发的附加参数。"""
    if not argv:
        return "status", []

    first = argv[0]
    if first in {"-h", "--help", "help"}:
        print_help()
        raise SystemExit(0)

    if first not in COMMAND_MAP:
        print(f"未知命令: {first}", file=sys.stderr)
        print("使用 python -s main.py --help 查看可用命令。", file=sys.stderr)
        raise SystemExit(2)

    return first, argv[1:]


def main() -> None:
    """将命令转发到 scripts 下的具体实现。"""
    command, extra_args = resolve_command(sys.argv[1:])
    script_relative_path, injected_args = COMMAND_MAP[command]
    script_path = PROJECT_ROOT / script_relative_path

    if not script_path.exists():
        print(f"目标脚本不存在: {script_path}", file=sys.stderr)
        raise SystemExit(1)

    os.execv(
        sys.executable,
        [sys.executable, "-s", str(script_path), *injected_args, *extra_args],
    )


if __name__ == "__main__":
    main()
