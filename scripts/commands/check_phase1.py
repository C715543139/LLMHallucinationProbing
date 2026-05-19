"""
Phase 1 环境检查脚本：在 Linux 环境下检查依赖、数据、模型与可选前向传播。

用法:
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llm_hallucination
    source ./.venv/bin/activate
    python -s scripts/commands/check_phase1.py
    python -s scripts/commands/check_phase1.py --include-model
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


@dataclass
class Summary:
    """检查结果计数。"""

    passed: int = 0
    failed: int = 0
    warned: int = 0


def write_section(title: str) -> None:
    """打印分节标题。"""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def write_ok(summary: Summary, message: str) -> None:
    """记录通过项。"""
    print(f"  [OK]   {message}")
    summary.passed += 1


def write_fail(summary: Summary, message: str) -> None:
    """记录失败项。"""
    print(f"  [FAIL] {message}")
    summary.failed += 1


def write_warn(summary: Summary, message: str) -> None:
    """记录警告项。"""
    print(f"  [WARN] {message}")
    summary.warned += 1


def write_info(message: str) -> None:
    """打印信息项。"""
    print(f"  [INFO] {message}")


def check(summary: Summary, condition: bool, ok_message: str, fail_message: str, is_warn: bool = False) -> None:
    """统一处理检查结果。"""
    if condition:
        write_ok(summary, ok_message)
        return

    if is_warn:
        write_warn(summary, fail_message)
        return

    write_fail(summary, fail_message)


def format_exception(exc: Exception) -> str:
    """压缩异常信息，便于命令行显示。"""
    return f"{type(exc).__name__}: {exc}"


def check_environment_files(summary: Summary) -> None:
    """检查 Python 环境与依赖文件。"""
    write_section("P1.1  环境与依赖文件")

    check(summary, (PROJECT_ROOT / "pyproject.toml").exists(), "pyproject.toml 存在", "pyproject.toml 缺失")
    check(summary, (PROJECT_ROOT / "uv.lock").exists(), "uv.lock 存在", "uv.lock 缺失（运行 uv sync 生成）", is_warn=True)
    check(summary, (PROJECT_ROOT / ".venv").exists(), ".venv 虚拟环境目录存在", ".venv 不存在（运行: uv sync）")

    version = f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    check(
        summary,
        sys.version_info.major == 3 and sys.version_info.minor in (10, 11),
        f"Python 版本符合要求: {version}",
        f"Python 版本不符合要求（需 3.10/3.11）: {version}",
        is_warn=True,
    )


def check_cuda_environment(summary: Summary) -> None:
    """检查 CUDA、GPU 与核心依赖。"""
    write_section("P1.2  CUDA / GPU 环境")

    try:
        import torch
    except Exception as exc:
        write_fail(summary, f"PyTorch 导入失败: {format_exception(exc)}")
        return

    check(summary, torch.cuda.is_available(), "CUDA 可用（torch.cuda.is_available() = True）", "CUDA 不可用，请检查驱动与 PyTorch CUDA 版本")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        write_ok(summary, f"GPU 设备: {device_name}")
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        check(summary, vram_gb >= 6.0, f"显存 {vram_gb} GB（≥6 GB 满足运行要求）", f"显存 {vram_gb} GB（建议 ≥6 GB）", is_warn=True)
    else:
        write_warn(summary, "GPU 信息无法获取（CUDA 不可用时跳过）")

    write_info(f"PyTorch: {torch.__version__}")

    try:
        import transformers

        write_info(f"Transformers: {transformers.__version__}")
    except Exception as exc:
        write_warn(summary, f"Transformers 导入失败: {format_exception(exc)}")


def check_model_files(summary: Summary) -> None:
    """检查模型权重文件。"""
    write_section("P1.3  模型权重 (Qwen2-1.5B)")

    model_dir = PROJECT_ROOT / "models_cache" / "Qwen2-1.5B"
    check(
        summary,
        model_dir.exists(),
        f"模型目录存在: {model_dir.relative_to(PROJECT_ROOT)}",
        "模型目录不存在: models_cache/Qwen2-1.5B（运行: hf download Qwen/Qwen2-1.5B --local-dir models_cache/Qwen2-1.5B）",
    )

    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
    for filename in required_files:
        check(summary, (model_dir / filename).exists(), f"模型文件存在: {filename}", f"模型文件缺失: {filename}")

    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        size_mb = round(safetensors_path.stat().st_size / 1024**2)
        check(summary, size_mb > 1024, f"model.safetensors 大小 {size_mb} MB（正常）", f"model.safetensors 大小 {size_mb} MB 异常（可能下载不完整）", is_warn=True)


def check_raw_data(summary: Summary) -> None:
    """检查原始数据集。"""
    write_section("P1.4  原始数据集 (data/raw/)")

    raw_dir = PROJECT_ROOT / "data" / "raw"
    check(summary, raw_dir.exists(), "data/raw/ 目录存在", "data/raw/ 目录不存在")

    raw_files = [
        "cities_true_false.csv",
        "inventions_true_false.csv",
        "elements_true_false.csv",
        "animals_true_false.csv",
        "companies_true_false.csv",
        "facts_true_false.csv",
    ]
    for filename in raw_files:
        check(summary, (raw_dir / filename).exists(), f"数据文件存在: {filename}", f"数据文件缺失: {filename}")


def check_dataset_module(summary: Summary) -> None:
    """检查数据加载模块。"""
    write_section("P1.5  数据加载模块 (src/data/dataset.py)")

    dataset_file = PROJECT_ROOT / "src" / "data" / "dataset.py"
    check(summary, dataset_file.exists(), "src/data/dataset.py 存在", "src/data/dataset.py 不存在")

    try:
        from src.data.dataset import TrueFalseDataset, load_all_raw_data  # noqa: F401

        write_ok(summary, "TrueFalseDataset 和 load_all_raw_data 可正常导入")
    except Exception as exc:
        write_fail(summary, f"导入失败: {format_exception(exc)}")
        return

    try:
        from src.data.dataset import TrueFalseDataset

        dataset = TrueFalseDataset(["A", "B", "C"], [1, 0, 1], ["d", "d", "d"])
        item = dataset[0]
        assert len(dataset) == 3
        assert dataset.n_true == 2
        assert dataset.n_false == 1
        assert "statement" in item and "label" in item and "domain" in item
        write_ok(summary, "TrueFalseDataset 基本功能正常")
    except Exception as exc:
        write_fail(summary, f"TrueFalseDataset 功能异常: {format_exception(exc)}")


def check_loader_module(summary: Summary) -> None:
    """检查模型加载模块。"""
    write_section("P1.6  模型加载模块 (src/models/loader.py)")

    loader_file = PROJECT_ROOT / "src" / "models" / "loader.py"
    check(summary, loader_file.exists(), "src/models/loader.py 存在", "src/models/loader.py 不存在")

    try:
        from src.models.loader import get_device_info, load_model, load_model_fp16  # noqa: F401

        write_ok(summary, "loader.py 函数可正常导入")
    except Exception as exc:
        write_fail(summary, f"导入失败: {format_exception(exc)}")
        return

    try:
        from src.models.loader import get_device_info

        info = get_device_info()
        assert "cuda_available" in info
        assert "device_count" in info
        write_ok(summary, "get_device_info() 返回结构正确")
    except Exception as exc:
        write_fail(summary, f"get_device_info() 异常: {format_exception(exc)}")


def check_processed_data(summary: Summary) -> None:
    """检查预处理产物与数据划分。"""
    write_section("P1.7  预处理数据 (data/processed/)")

    processed_dir = PROJECT_ROOT / "data" / "processed"
    for filename in ["train.pt", "val.pt", "test.pt"]:
        check(summary, (processed_dir / filename).exists(), f"预处理文件存在: {filename}", f"预处理文件缺失: {filename}（运行: python -s main.py preprocess）")

    required = [processed_dir / "train.pt", processed_dir / "val.pt", processed_dir / "test.pt"]
    if not all(path.exists() for path in required):
        write_warn(summary, "预处理文件不存在，跳过比例与重叠检查")
        return

    try:
        from src.data.dataset import load_dataset

        train = load_dataset(processed_dir / "train.pt")
        val = load_dataset(processed_dir / "val.pt")
        test = load_dataset(processed_dir / "test.pt")
        total = len(train) + len(val) + len(test)
        train_ratio = len(train) / total
        val_ratio = len(val) / total
        test_ratio = len(test) / total
        assert abs(train_ratio - 0.8) < 0.05
        assert abs(val_ratio - 0.1) < 0.05
        assert abs(test_ratio - 0.1) < 0.05

        train_set = set(train.statements)
        val_set = set(val.statements)
        test_set = set(test.statements)
        assert not (train_set & val_set)
        assert not (train_set & test_set)
        assert not (val_set & test_set)
        write_ok(summary, f"划分比例与无重叠检查通过: total={total} train={len(train)} val={len(val)} test={len(test)}")
    except Exception as exc:
        write_fail(summary, f"划分检查失败: {format_exception(exc)}")


def check_config_module(summary: Summary) -> None:
    """检查全局配置模块。"""
    write_section("P1.8  全局配置模块 (src/config.py)")

    config_file = PROJECT_ROOT / "src" / "config.py"
    check(summary, config_file.exists(), "src/config.py 存在", "src/config.py 不存在")

    try:
        from src.config import ExperimentConfig, config

        assert isinstance(config, ExperimentConfig)
        assert abs(config.data.train_ratio + config.data.val_ratio + config.data.test_ratio - 1.0) < 1e-6
        assert {42, 123, 2024}.issubset(set(config.training.random_seeds))
        assert config.paths.project_root.exists()
        write_ok(summary, "config 结构与默认值正确")
    except Exception as exc:
        write_fail(summary, f"config 检查失败: {format_exception(exc)}")


def check_model_forward(summary: Summary) -> None:
    """可选：执行一次真实模型前向传播。"""
    write_section("M1  模型前向传播（GPU + 本地权重）")
    write_info("加载 Qwen2-1.5B 进行前向传播测试，可能需要约 1-2 分钟...")

    model_dir = PROJECT_ROOT / "models_cache" / "Qwen2-1.5B"
    if not (model_dir / "config.json").exists():
        write_warn(summary, "本地 Qwen2-1.5B 权重不存在，跳过 M1 测试")
        return

    try:
        import torch
    except Exception as exc:
        write_fail(summary, f"PyTorch 导入失败，无法执行 M1: {format_exception(exc)}")
        return

    if not torch.cuda.is_available():
        write_warn(summary, "无 CUDA GPU，跳过 M1 测试")
        return

    try:
        from src.models.loader import load_model_fp16

        model, tokenizer = load_model_fp16(model_path=str(model_dir))
        statement = "The sky is blue."
        inputs = tokenizer(statement, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        num_layers = model.config.num_hidden_layers
        num_hidden_states = len(outputs.hidden_states)
        assert num_hidden_states == num_layers + 1
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        assert not torch.isnan(last_hidden).any()
        assert not torch.isinf(last_hidden).any()
        write_ok(summary, f"M1 前向传播通过: layers={num_layers} hidden_size={outputs.hidden_states[-1].shape[-1]}")
        del model
        torch.cuda.empty_cache()
    except Exception as exc:
        write_fail(summary, f"M1 前向传播失败: {format_exception(exc)}")


def print_summary(summary: Summary) -> None:
    """打印最终汇总与推荐命令。"""
    write_section("检验结果汇总")
    print("")
    print(f"  通过: {summary.passed}")
    print(f"  警告: {summary.warned}")
    print(f"  失败: {summary.failed}")
    print("")

    if summary.failed == 0 and summary.warned == 0:
        print("  Phase 1 全部检查通过！")
    elif summary.failed == 0:
        print(f"  Phase 1 核心检查通过，存在 {summary.warned} 项警告。")
    else:
        print(f"  Phase 1 存在 {summary.failed} 项失败，请逐一修复后重新运行。")

    print("\n  运行完整 pytest（快速，不含模型）:")
    print("    pytest tests/ -v -m 'not model'")
    print("\n  运行含 GPU 的 pytest（不含模型）:")
    print("    pytest tests/ -v -m 'gpu and not model'")
    print("\n  运行含模型加载的完整 pytest（慢）:")
    print("    pytest tests/ -v -m 'model'")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="检查 Phase 1 环境、数据与模型准备情况。")
    parser.add_argument("--include-model", action="store_true", help="额外执行一次真实模型前向传播测试")
    parser.add_argument("--verbose", action="store_true", help="保留兼容参数，当前脚本默认输出所有检查信息")
    return parser.parse_args()


def main() -> None:
    """运行 Phase 1 检查。"""
    args = parse_args()
    summary = Summary()

    check_environment_files(summary)
    check_cuda_environment(summary)
    check_model_files(summary)
    check_raw_data(summary)
    check_dataset_module(summary)
    check_loader_module(summary)
    check_processed_data(summary)
    check_config_module(summary)

    if args.include_model:
        check_model_forward(summary)

    print_summary(summary)
    raise SystemExit(1 if summary.failed else 0)


if __name__ == "__main__":
    main()