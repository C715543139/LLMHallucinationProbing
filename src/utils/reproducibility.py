"""可复现实验相关工具。"""

from __future__ import annotations

import os
import platform
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """设置 Python / NumPy / PyTorch 的全局随机种子。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_deterministic_runtime(
    seed: int,
    deterministic: bool = True,
    warn_only: bool = True,
) -> None:
    """配置 PyTorch 运行时的确定性选项。"""
    set_global_seed(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        except TypeError:
            torch.use_deterministic_algorithms(deterministic)


def collect_runtime_info(model: Optional[Any] = None) -> Dict[str, Any]:
    """收集与复现实验相关的运行环境信息。"""
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)

    try:
        import transformers

        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = None

    try:
        import sklearn

        info["scikit_learn"] = sklearn.__version__
    except Exception:
        info["scikit_learn"] = None

    if model is not None:
        try:
            first_param = next(model.parameters())
            info["model_device"] = str(first_param.device)
            info["model_dtype"] = str(first_param.dtype)
        except (StopIteration, TypeError):
            info["model_device"] = None
            info["model_dtype"] = None

        info["model_config_dtype"] = str(getattr(model.config, "torch_dtype", None))

    return info