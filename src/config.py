"""
全局配置模块：统一管理项目路径、模型配置、超参数与随机种子。

使用方式:
    from src.config import config
    print(config.models.primary_name)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

# ---------------------------------------------------------------------------
# 项目根目录（src/config.py → src/ → 项目根）
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PathConfig:
    """项目内所有路径的统一入口（均为 pathlib.Path）。"""

    project_root: Path = _PROJECT_ROOT
    data_raw: Path = _PROJECT_ROOT / "data" / "raw"
    data_processed: Path = _PROJECT_ROOT / "data" / "processed"
    models_cache: Path = _PROJECT_ROOT / "models_cache"
    experiments_dir: Path = _PROJECT_ROOT / "experiments"
    results_dir: Path = _PROJECT_ROOT / "experiments" / "results"

    # 原始数据集文件名列表
    raw_files: Tuple[str, ...] = (
        "cities_true_false.csv",
        "inventions_true_false.csv",
        "elements_true_false.csv",
        "animals_true_false.csv",
        "companies_true_false.csv",
        "facts_true_false.csv",
        "generated_true_false.csv",
    )

    # 预处理后的数据文件
    train_file: str = "train.pt"
    val_file: str = "val.pt"
    test_file: str = "test.pt"


@dataclass
class ModelConfig:
    """模型加载配置。"""

    # 主实验模型（保底路径）
    primary_name: str = "Qwen/Qwen2-1.5B"
    primary_local: str = "models_cache/Qwen2-1.5B"
    primary_dtype: str = "bfloat16"        # Linux + RTX 3090 上更稳定，避免 eager attention NaN
    primary_device_map: str = "auto"

    # 通用
    trust_remote_code: bool = False
    torch_dtype_fallback: str = "bfloat16"


@dataclass
class DataConfig:
    """数据处理与划分配置。"""

    # 训练/验证/测试比例
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # 划分随机种子（一份划分全程复用）
    split_seed: int = 42

    # 是否按领域分层划分
    stratify_by_domain: bool = True

    # 配对保护（若存在真/假配对关系，同一对划入同一集合）
    pair_aware_split: bool = False


@dataclass
class TrainingConfig:
    """分类器训练超参数。"""

    # 分类器类型
    classifier_type: str = "logistic"       # "logistic" | "mlp"

    # 逻辑回归
    logistic_C: float = 1.0
    logistic_max_iter: int = 1000
    logistic_penalty: str = "l2"

    # MLP
    mlp_hidden_sizes: Tuple[int, ...] = (256, 128)
    mlp_activation: str = "relu"
    mlp_alpha: float = 0.0001
    mlp_max_iter: int = 500

    # 通用
    global_seed: int = 42
    random_seeds: Tuple[int, ...] = (42, 123, 2024)
    deterministic: bool = True
    deterministic_warn_only: bool = True
    n_jobs: int = -1


@dataclass
class FeatureConfig:
    """特征提取配置。"""

    # 默认提取的层索引（-1 表示最后一层）
    default_layer_idx: int = -1

    # Token 池化方式
    pooling_strategies: Tuple[str, ...] = ("last", "first", "mean")

    # 是否同时提取所有层的隐藏状态
    extract_all_layers: bool = False


@dataclass
class ExperimentConfig:
    """顶层聚合配置，统一入口。"""

    paths: PathConfig = field(default_factory=PathConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------
config = ExperimentConfig()

# 兼容 HuggingFace 镜像设置（若环境变量未设置则自动使用镜像）
if "HF_ENDPOINT" not in os.environ:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 将 models_cache 设为 HuggingFace 缓存目录（本 session 内有效）
os.environ.setdefault("HF_HOME", str(config.paths.models_cache.resolve()))
