"""
pytest 公共配置与 Fixture。

标记说明:
    - gpu  : 需要真实 GPU 设备才能运行的测试
    - model: 需要本地 Qwen2-1.5B 模型权重才能运行的测试（同时隐含 gpu）
    - slow : 运行时间较长（>30 s）的测试

用法:
    pytest tests/                         # 仅运行快速单元测试
    pytest tests/ -m gpu                  # 运行 GPU 测试
    pytest tests/ -m model                # 运行需要模型的测试
    pytest tests/ -m "not model"          # 跳过模型测试
"""

from __future__ import annotations

import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# 标记注册
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: 需要 CUDA GPU 才能运行")
    config.addinivalue_line("markers", "model: 需要本地 Qwen2-1.5B 模型权重才能运行")
    config.addinivalue_line("markers", "slow: 运行时间较长（>30s）的测试")


# ---------------------------------------------------------------------------
# 公共路径 Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def project_root() -> Path:
    """项目根目录。"""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def data_raw_dir(project_root: Path) -> Path:
    return project_root / "data" / "raw"


@pytest.fixture(scope="session")
def data_processed_dir(project_root: Path) -> Path:
    return project_root / "data" / "processed"


@pytest.fixture(scope="session")
def models_cache_dir(project_root: Path) -> Path:
    return project_root / "models_cache"
