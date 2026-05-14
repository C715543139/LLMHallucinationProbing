"""
P1.8 — 测试全局配置模块 (src/config.py)。

覆盖:
    - 配置对象可正常导入
    - 各子配置字段的默认值正确
    - 路径配置指向真实存在的目录
    - 数据划分比例之和为 1.0
    - 随机种子列表包含约定的三个值
"""

from __future__ import annotations

from pathlib import Path


class TestImport:
    """测试模块可正常导入。"""

    def test_config_importable(self) -> None:
        from src.config import config  # noqa: F401

    def test_experiment_config_class(self) -> None:
        from src.config import ExperimentConfig
        assert ExperimentConfig is not None

    def test_sub_config_classes(self) -> None:
        from src.config import (
            PathConfig,
            ModelConfig,
            DataConfig,
            TrainingConfig,
            FeatureConfig,
        )
        for cls in (PathConfig, ModelConfig, DataConfig, TrainingConfig, FeatureConfig):
            assert cls is not None


class TestConfigSingleton:
    """测试全局单例 config 的属性结构。"""

    def test_singleton_has_paths(self) -> None:
        from src.config import config
        assert hasattr(config, "paths")

    def test_singleton_has_models(self) -> None:
        from src.config import config
        assert hasattr(config, "models")

    def test_singleton_has_data(self) -> None:
        from src.config import config
        assert hasattr(config, "data")

    def test_singleton_has_training(self) -> None:
        from src.config import config
        assert hasattr(config, "training")

    def test_singleton_has_features(self) -> None:
        from src.config import config
        assert hasattr(config, "features")


class TestPathConfig:
    """测试路径配置。"""

    def test_project_root_is_path(self) -> None:
        from src.config import config
        assert isinstance(config.paths.project_root, Path)

    def test_project_root_exists(self) -> None:
        from src.config import config
        assert config.paths.project_root.exists()

    def test_project_root_contains_src(self) -> None:
        from src.config import config
        assert (config.paths.project_root / "src").exists()

    def test_data_raw_path_type(self) -> None:
        from src.config import config
        assert isinstance(config.paths.data_raw, Path)

    def test_data_processed_path_type(self) -> None:
        from src.config import config
        assert isinstance(config.paths.data_processed, Path)

    def test_models_cache_path_type(self) -> None:
        from src.config import config
        assert isinstance(config.paths.models_cache, Path)

    def test_raw_files_tuple_nonempty(self) -> None:
        from src.config import config
        assert len(config.paths.raw_files) > 0

    def test_raw_files_are_csv(self) -> None:
        from src.config import config
        for fname in config.paths.raw_files:
            assert fname.endswith(".csv"), f"{fname} 不是 CSV 文件"

    def test_processed_filenames(self) -> None:
        from src.config import config
        assert config.paths.train_file == "train.pt"
        assert config.paths.val_file == "val.pt"
        assert config.paths.test_file == "test.pt"


class TestModelConfig:
    """测试模型配置。"""

    def test_primary_name_is_qwen(self) -> None:
        from src.config import config
        assert "Qwen" in config.models.primary_name or "qwen" in config.models.primary_name.lower()

    def test_primary_local_set(self) -> None:
        from src.config import config
        assert config.models.primary_local != ""

    def test_device_map_set(self) -> None:
        from src.config import config
        assert config.models.primary_device_map in ("auto", "cpu", "cuda")


class TestDataConfig:
    """测试数据配置。"""

    def test_split_ratios_sum_to_one(self) -> None:
        from src.config import config
        total = config.data.train_ratio + config.data.val_ratio + config.data.test_ratio
        assert abs(total - 1.0) < 1e-6, f"划分比例之和为 {total}，应为 1.0"

    def test_split_ratios_are_positive(self) -> None:
        from src.config import config
        assert config.data.train_ratio > 0
        assert config.data.val_ratio > 0
        assert config.data.test_ratio > 0

    def test_train_ratio_dominant(self) -> None:
        """训练集比例应最大（计划要求 8:1:1）。"""
        from src.config import config
        assert config.data.train_ratio >= config.data.val_ratio
        assert config.data.train_ratio >= config.data.test_ratio

    def test_split_seed_is_int(self) -> None:
        from src.config import config
        assert isinstance(config.data.split_seed, int)


class TestTrainingConfig:
    """测试训练配置。"""

    def test_random_seeds_contain_required(self) -> None:
        """计划要求使用 42、123、2024 三个种子。"""
        from src.config import config
        seeds = set(config.training.random_seeds)
        for required in (42, 123, 2024):
            assert required in seeds, f"随机种子 {required} 未在配置中"

    def test_classifier_type_valid(self) -> None:
        from src.config import config
        assert config.training.classifier_type in ("logistic", "mlp")
