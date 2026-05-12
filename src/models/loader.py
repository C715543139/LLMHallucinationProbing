"""
模型加载模块。

支持:
    - 主路径: Qwen2-1.5B FP16（通过 device_map="auto" 加载）
    - 可选路径: Llama-2-7B 4-bit 量化（通过 bitsandbytes）

同时提供便捷的 GPU 检测与显存查询工具。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from src.config import config

logger = logging.getLogger(__name__)

# 类型别名
ModelAndTokenizer = Tuple[PreTrainedModel, PreTrainedTokenizer | PreTrainedTokenizerFast]


# ---------------------------------------------------------------------------
# GPU / 环境工具
# ---------------------------------------------------------------------------

def get_device_info() -> Dict[str, Any]:
    """获取当前 GPU 设备信息。"""
    info: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info[f"device_{i}"] = {
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024**3), 1),
                "compute_capability": f"{props.major}.{props.minor}",
            }
    return info


def print_device_info() -> None:
    """打印 GPU 信息到控制台。"""
    info = get_device_info()
    print(f"CUDA available: {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"Device count: {info['device_count']}")
        for i in range(info["device_count"]):
            d = info[f"device_{i}"]
            print(f"  GPU {i}: {d['name']} ({d['total_memory_gb']:.1f} GB)")


# ---------------------------------------------------------------------------
# 模型加载（主路径: FP16 / auto）
# ---------------------------------------------------------------------------

def _resolve_model_path() -> Path:
    """解析模型路径：优先本地缓存，否则回退到 HuggingFace Hub 名称。

    返回:
        resolved: 本地路径（若存在）或原始 Hub ID。
        is_local: 是否为本地模型。
    """
    local_path = config.paths.models_cache / config.models.primary_local.split("/")[-1]
    # config.models.primary_local 例如 "models_cache/Qwen2-1.5B"
    candidate = config.paths.project_root / config.models.primary_local
    if candidate.exists() and (candidate / "config.json").exists():
        return candidate
    # 尝试 models_cache 下同名目录
    folder_name = Path(config.models.primary_local).name
    alt = config.paths.models_cache / folder_name
    if alt.exists() and (alt / "config.json").exists():
        return alt
    # 回退到 Hub ID
    logger.warning("本地模型 %s 不存在，将从 HuggingFace Hub 加载 %s",
                   candidate, config.models.primary_name)
    return Path(config.models.primary_name)


def load_model_fp16(
    model_path: Optional[str] = None,
    device_map: Optional[str] = None,
    torch_dtype: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
) -> ModelAndTokenizer:
    """以 FP16（或 auto）精度加载因果语言模型及其分词器。

    这是主实验路径（Qwen2-1.5B），也适用于其他 HF 兼容的 CausalLM 模型。

    参数:
        model_path: 模型路径或 HuggingFace Hub ID，默认使用 config 中的 primary。
        device_map: 设备映射策略，默认 "auto"。
        torch_dtype: 模型精度字符串，默认 "auto"。
        trust_remote_code: 是否允许执行远端代码。

    返回:
        (model, tokenizer)
    """
    if model_path is None:
        model_path = str(_resolve_model_path())
    if device_map is None:
        device_map = config.models.primary_device_map
    if torch_dtype is None:
        torch_dtype = config.models.primary_dtype
    if trust_remote_code is None:
        trust_remote_code = config.models.trust_remote_code

    logger.info("加载模型 (FP16/auto): %s", model_path)
    logger.info("  device_map=%s, torch_dtype=%s", device_map, torch_dtype)

    # 确定 torch_dtype 参数
    dtype_kwargs: Dict[str, Any] = {}
    if torch_dtype == "auto":
        # 让 transformers 从 config.json 中读取
        pass
    elif torch_dtype == "float16":
        dtype_kwargs["torch_dtype"] = torch.float16
    elif torch_dtype == "bfloat16":
        dtype_kwargs["torch_dtype"] = torch.bfloat16
    elif torch_dtype == "float32":
        dtype_kwargs["torch_dtype"] = torch.float32

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    # 设置 pad_token（若缺失则使用 eos_token）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("pad_token 未设置，已设为 eos_token: %s", tokenizer.eos_token)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **dtype_kwargs,
    )
    model.eval()

    _log_model_info(model)
    return model, tokenizer


# ---------------------------------------------------------------------------
# 模型加载（可选路径: 4-bit 量化）
# ---------------------------------------------------------------------------

def _check_bitsandbytes() -> bool:
    """检查 bitsandbytes 是否可用。"""
    try:
        import bitsandbytes as bnb  # noqa: F401
        return True
    except ImportError:
        return False


def load_model_4bit(
    model_path: Optional[str] = None,
    device_map: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
) -> ModelAndTokenizer:
    """以 4-bit 量化精度加载模型（用于 Llama-2-7B 等较大模型）。

    注意:
        - 需要安装 bitsandbytes（Windows 上需使用社区版 wheel）。
        - 若 bitsandbytes 不可用，将自动抛出 RuntimeError。

    参数:
        model_path: 模型路径，默认使用 config 中的 secondary_local。
        device_map: 设备映射。
        trust_remote_code: 是否允许执行远端代码。

    返回:
        (model, tokenizer)
    """
    if not _check_bitsandbytes():
        raise RuntimeError(
            "bitsandbytes 未安装或不可用。"
            "请使用 `uv add` 安装 Windows 兼容版 wheel，"
            "或改用 load_model_fp16() 加载较小模型。"
        )

    if model_path is None:
        # 先尝试本地路径
        local = config.paths.models_cache / Path(config.models.secondary_local).name
        if local.exists() and (local / "config.json").exists():
            model_path = str(local)
        else:
            alt = config.paths.project_root / config.models.secondary_local
            if alt.exists() and (alt / "config.json").exists():
                model_path = str(alt)
            else:
                model_path = config.models.secondary_name
                logger.warning("本地模型不存在，尝试从 Hub 下载: %s", model_path)

    if device_map is None:
        device_map = config.models.secondary_device_map
    if trust_remote_code is None:
        trust_remote_code = config.models.trust_remote_code

    logger.info("加载模型 (4-bit): %s", model_path)
    logger.info("  device_map=%s", device_map)

    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    _log_model_info(model)
    return model, tokenizer


# ---------------------------------------------------------------------------
# 统一加载入口
# ---------------------------------------------------------------------------

def load_model(
    use_4bit: bool = False,
    model_path: Optional[str] = None,
) -> ModelAndTokenizer:
    """统一模型加载入口。

    根据 use_4bit 参数选择加载路径:
        - False (默认): 使用 FP16/auto 加载（主路径，Qwen2-1.5B）
        - True: 使用 4-bit 量化加载（可选路径，Llama-2-7B）

    参数:
        use_4bit: 是否使用 4-bit 量化。
        model_path: 覆盖默认模型路径。

    返回:
        (model, tokenizer)
    """
    if use_4bit:
        return load_model_4bit(model_path=model_path)
    else:
        return load_model_fp16(model_path=model_path)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _log_model_info(model: PreTrainedModel) -> None:
    """记录模型基本信息。"""
    try:
        num_params = sum(p.numel() for p in model.parameters())
        num_params_b = num_params / 1e9
        logger.info("模型参数量: %.2f B", num_params_b)
    except Exception:
        pass

    try:
        cfg = model.config
        logger.info(
            "模型架构: %s, hidden_size=%s, num_layers=%s",
            cfg.model_type if hasattr(cfg, "model_type") else "unknown",
            getattr(cfg, "hidden_size", "?"),
            getattr(cfg, "num_hidden_layers", "?"),
        )
    except Exception:
        pass

    # 记录模型所在设备
    try:
        device = next(model.parameters()).device
        logger.info("模型设备: %s", device)
    except StopIteration:
        pass
