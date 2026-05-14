"""
P1.4 / P1.5 / P1.7 — 测试数据加载、Dataset 类与预处理输出。

覆盖:
    - data/raw/ 中存在所有必要的 CSV 文件 (P1.4)
    - TrueFalseDataset 的基本功能 (P1.5)
    - load_all_raw_data 能正确加载并合并数据 (P1.5)
    - 预处理后的 .pt 文件存在且结构正确 (P1.7)
    - train/val/test 比例约符合 8:1:1 (P1.7)
    - 三个集合之间无样本重叠（无信息泄漏）(P1.7)
"""

from __future__ import annotations

import pytest
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# P1.4：原始数据文件检查
# ---------------------------------------------------------------------------

EXPECTED_CSV_FILES = [
    "cities_true_false.csv",
    "inventions_true_false.csv",
    "elements_true_false.csv",
    "animals_true_false.csv",
    "companies_true_false.csv",
    "facts_true_false.csv",
]


class TestRawData:
    """检查 data/raw/ 目录中的原始数据文件 (P1.4)。"""

    def test_raw_dir_exists(self, data_raw_dir: Path) -> None:
        assert data_raw_dir.exists(), f"data/raw/ 目录不存在: {data_raw_dir}"

    @pytest.mark.parametrize("filename", EXPECTED_CSV_FILES)
    def test_required_csv_exists(self, data_raw_dir: Path, filename: str) -> None:
        fpath = data_raw_dir / filename
        assert fpath.exists(), f"缺少原始数据文件: {fpath}"

    def test_csv_files_nonempty(self, data_raw_dir: Path) -> None:
        for filename in EXPECTED_CSV_FILES:
            fpath = data_raw_dir / filename
            if fpath.exists():
                assert fpath.stat().st_size > 0, f"文件为空: {fpath}"

    def test_csv_has_required_columns(self, data_raw_dir: Path) -> None:
        """每个 CSV 必须包含 statement 和 label 列。"""
        import pandas as pd

        for filename in EXPECTED_CSV_FILES:
            fpath = data_raw_dir / filename
            if not fpath.exists():
                pytest.skip(f"文件不存在，跳过: {filename}")

            df = pd.read_csv(fpath)
            cols_lower = [c.strip().lower() for c in df.columns]
            assert "statement" in cols_lower, f"{filename} 缺少 'statement' 列"
            assert "label" in cols_lower, f"{filename} 缺少 'label' 列"

    def test_csv_labels_binary(self, data_raw_dir: Path) -> None:
        """标签列应只包含 0 和 1。"""
        import pandas as pd

        for filename in EXPECTED_CSV_FILES:
            fpath = data_raw_dir / filename
            if not fpath.exists():
                continue

            df = pd.read_csv(fpath)
            df.columns = df.columns.str.strip().str.lower()
            unique_labels = set(df["label"].unique())
            assert unique_labels <= {0, 1}, (
                f"{filename} 中存在非 0/1 标签: {unique_labels}"
            )


# ---------------------------------------------------------------------------
# P1.5：TrueFalseDataset 类测试
# ---------------------------------------------------------------------------

class TestTrueFalseDataset:
    """测试 TrueFalseDataset 类的基本功能 (P1.5)。"""

    @pytest.fixture
    def sample_data(self):
        statements = ["The sky is blue.", "The moon is made of cheese.", "Water is H2O."]
        labels = [1, 0, 1]
        domains = ["facts", "facts", "elements"]
        return statements, labels, domains

    def test_import(self) -> None:
        from src.data.dataset import TrueFalseDataset  # noqa: F401

    def test_instantiation(self, sample_data) -> None:
        from src.data.dataset import TrueFalseDataset
        stmts, labels, domains = sample_data
        ds = TrueFalseDataset(stmts, labels, domains)
        assert ds is not None

    def test_len(self, sample_data) -> None:
        from src.data.dataset import TrueFalseDataset
        stmts, labels, domains = sample_data
        ds = TrueFalseDataset(stmts, labels, domains)
        assert len(ds) == 3

    def test_getitem_keys(self, sample_data) -> None:
        from src.data.dataset import TrueFalseDataset
        stmts, labels, domains = sample_data
        ds = TrueFalseDataset(stmts, labels, domains)
        item = ds[0]
        assert "statement" in item
        assert "label" in item
        assert "domain" in item

    def test_getitem_values(self, sample_data) -> None:
        from src.data.dataset import TrueFalseDataset
        stmts, labels, domains = sample_data
        ds = TrueFalseDataset(stmts, labels, domains)
        item = ds[0]
        assert item["statement"] == stmts[0]
        assert item["label"] == labels[0]
        assert item["domain"] == domains[0]

    def test_label_coerced_to_int(self) -> None:
        from src.data.dataset import TrueFalseDataset
        ds = TrueFalseDataset(["test"], [True], ["d"])
        assert isinstance(ds[0]["label"], int)

    def test_n_true_n_false(self, sample_data) -> None:
        from src.data.dataset import TrueFalseDataset
        stmts, labels, domains = sample_data
        ds = TrueFalseDataset(stmts, labels, domains)
        assert ds.n_true == 2
        assert ds.n_false == 1
        assert ds.n_true + ds.n_false == len(ds)

    def test_class_balance(self, sample_data) -> None:
        from src.data.dataset import TrueFalseDataset
        stmts, labels, domains = sample_data
        ds = TrueFalseDataset(stmts, labels, domains)
        expected = 2 / 3
        assert abs(ds.class_balance - expected) < 1e-6

    def test_length_mismatch_raises(self) -> None:
        from src.data.dataset import TrueFalseDataset
        with pytest.raises((ValueError, AssertionError)):
            TrueFalseDataset(["a", "b"], [0])

    def test_without_domains(self) -> None:
        from src.data.dataset import TrueFalseDataset
        ds = TrueFalseDataset(["stmt"], [1])
        assert len(ds) == 1
        assert ds[0]["domain"] == ""

    def test_summary_returns_string(self, sample_data) -> None:
        from src.data.dataset import TrueFalseDataset
        stmts, labels, domains = sample_data
        ds = TrueFalseDataset(stmts, labels, domains)
        summary = ds.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_properties_readonly(self, sample_data) -> None:
        """statements/labels/domains 属性应返回列表。"""
        from src.data.dataset import TrueFalseDataset
        stmts, labels, domains = sample_data
        ds = TrueFalseDataset(stmts, labels, domains)
        assert isinstance(ds.statements, list)
        assert isinstance(ds.labels, list)
        assert isinstance(ds.domains, list)


class TestLoadAllRawData:
    """测试 load_all_raw_data 函数 (P1.5)。"""

    def test_import(self) -> None:
        from src.data.dataset import load_all_raw_data  # noqa: F401

    def test_loads_data(self, data_raw_dir: Path) -> None:
        from src.data.dataset import load_all_raw_data
        df = load_all_raw_data(data_raw_dir)
        assert len(df) > 0

    def test_output_columns(self, data_raw_dir: Path) -> None:
        from src.data.dataset import load_all_raw_data
        df = load_all_raw_data(data_raw_dir)
        assert "statement" in df.columns
        assert "label" in df.columns
        assert "domain" in df.columns

    def test_no_null_statements(self, data_raw_dir: Path) -> None:
        from src.data.dataset import load_all_raw_data
        df = load_all_raw_data(data_raw_dir)
        assert df["statement"].isna().sum() == 0

    def test_binary_labels_only(self, data_raw_dir: Path) -> None:
        from src.data.dataset import load_all_raw_data
        df = load_all_raw_data(data_raw_dir)
        unique = set(df["label"].unique())
        assert unique <= {0, 1}, f"存在非 0/1 标签: {unique}"

    def test_multiple_domains(self, data_raw_dir: Path) -> None:
        from src.data.dataset import load_all_raw_data
        df = load_all_raw_data(data_raw_dir)
        n_domains = df["domain"].nunique()
        assert n_domains >= 4, f"预期至少 4 个领域，实际: {n_domains}"

    def test_no_duplicate_statements(self, data_raw_dir: Path) -> None:
        from src.data.dataset import load_all_raw_data
        df = load_all_raw_data(data_raw_dir)
        assert df["statement"].duplicated().sum() == 0, "存在重复陈述句"

    def test_raises_on_missing_dir(self, tmp_path: Path) -> None:
        from src.data.dataset import load_all_raw_data
        with pytest.raises((FileNotFoundError, Exception)):
            load_all_raw_data(tmp_path / "nonexistent")


class TestSaveLoadDataset:
    """测试 save_dataset / load_dataset 序列化往返 (P1.5 / P1.7)。"""

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        import torch
        from src.data.dataset import TrueFalseDataset, save_dataset, load_dataset

        stmts = ["Hello world.", "AI is fascinating."]
        labels = [1, 0]
        domains = ["test", "test"]
        ds_orig = TrueFalseDataset(stmts, labels, domains)

        pt_path = tmp_path / "test_dataset.pt"
        save_dataset(ds_orig, pt_path)
        assert pt_path.exists()

        ds_loaded = load_dataset(pt_path)
        assert len(ds_loaded) == len(ds_orig)
        assert ds_loaded.statements == ds_orig.statements
        assert ds_loaded.labels == ds_orig.labels
        assert ds_loaded.domains == ds_orig.domains


# ---------------------------------------------------------------------------
# P1.7：预处理输出文件检查
# ---------------------------------------------------------------------------

class TestProcessedData:
    """检查 data/processed/ 中的 .pt 文件 (P1.7)。"""

    def test_processed_dir_exists(self, data_processed_dir: Path) -> None:
        assert data_processed_dir.exists(), (
            f"data/processed/ 目录不存在: {data_processed_dir}\n"
            "请先运行: python main.py preprocess"
        )

    @pytest.mark.parametrize("filename", ["train.pt", "val.pt", "test.pt"])
    def test_pt_file_exists(self, data_processed_dir: Path, filename: str) -> None:
        fpath = data_processed_dir / filename
        assert fpath.exists(), (
            f"缺少预处理文件: {fpath}\n请先运行: python main.py preprocess"
        )

    @pytest.mark.parametrize("filename", ["train.pt", "val.pt", "test.pt"])
    def test_pt_file_nonempty(self, data_processed_dir: Path, filename: str) -> None:
        fpath = data_processed_dir / filename
        if not fpath.exists():
            pytest.skip(f"文件不存在: {filename}")
        assert fpath.stat().st_size > 0

    def test_all_splits_loadable(self, data_processed_dir: Path) -> None:
        from src.data.dataset import load_dataset
        for fname in ("train.pt", "val.pt", "test.pt"):
            fpath = data_processed_dir / fname
            if not fpath.exists():
                pytest.skip(f"文件不存在: {fname}")
            ds = load_dataset(fpath)
            assert len(ds) > 0, f"{fname} 加载后为空"

    def test_split_sizes_approximate(self, data_processed_dir: Path) -> None:
        """验证 train:val:test ≈ 8:1:1（允许 ±5% 误差）。"""
        from src.data.dataset import load_dataset

        for fname in ("train.pt", "val.pt", "test.pt"):
            if not (data_processed_dir / fname).exists():
                pytest.skip("预处理文件缺失")

        train_ds = load_dataset(data_processed_dir / "train.pt")
        val_ds = load_dataset(data_processed_dir / "val.pt")
        test_ds = load_dataset(data_processed_dir / "test.pt")

        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total > 0

        train_ratio = len(train_ds) / total
        val_ratio = len(val_ds) / total
        test_ratio = len(test_ds) / total

        assert abs(train_ratio - 0.8) < 0.05, (
            f"训练集比例 {train_ratio:.3f} 偏离 0.8 超过 5%"
        )
        assert abs(val_ratio - 0.1) < 0.05, (
            f"验证集比例 {val_ratio:.3f} 偏离 0.1 超过 5%"
        )
        assert abs(test_ratio - 0.1) < 0.05, (
            f"测试集比例 {test_ratio:.3f} 偏离 0.1 超过 5%"
        )

    def test_no_overlap_between_splits(self, data_processed_dir: Path) -> None:
        """三个集合之间不应存在重叠样本（防止信息泄漏）。"""
        from src.data.dataset import load_dataset

        for fname in ("train.pt", "val.pt", "test.pt"):
            if not (data_processed_dir / fname).exists():
                pytest.skip("预处理文件缺失")

        train_ds = load_dataset(data_processed_dir / "train.pt")
        val_ds = load_dataset(data_processed_dir / "val.pt")
        test_ds = load_dataset(data_processed_dir / "test.pt")

        train_stmts = set(train_ds.statements)
        val_stmts = set(val_ds.statements)
        test_stmts = set(test_ds.statements)

        train_val_overlap = train_stmts & val_stmts
        assert len(train_val_overlap) == 0, (
            f"训练集与验证集有 {len(train_val_overlap)} 条重叠样本（信息泄漏！）"
        )

        train_test_overlap = train_stmts & test_stmts
        assert len(train_test_overlap) == 0, (
            f"训练集与测试集有 {len(train_test_overlap)} 条重叠样本（信息泄漏！）"
        )

        val_test_overlap = val_stmts & test_stmts
        assert len(val_test_overlap) == 0, (
            f"验证集与测试集有 {len(val_test_overlap)} 条重叠样本（信息泄漏！）"
        )

    def test_splits_contain_all_domains(self, data_processed_dir: Path) -> None:
        """训练集应包含所有领域（分层划分的效果）。"""
        from src.data.dataset import load_dataset

        train_pt = data_processed_dir / "train.pt"
        if not train_pt.exists():
            pytest.skip("train.pt 不存在")

        train_ds = load_dataset(train_pt)
        domains = set(train_ds.domains)
        assert len(domains) >= 4, (
            f"训练集只含 {len(domains)} 个领域，预期 ≥4"
        )

    def test_splits_have_both_labels(self, data_processed_dir: Path) -> None:
        """每个集合中都应有真/假两类标签。"""
        from src.data.dataset import load_dataset

        for fname in ("train.pt", "val.pt", "test.pt"):
            fpath = data_processed_dir / fname
            if not fpath.exists():
                pytest.skip(f"文件不存在: {fname}")
            ds = load_dataset(fpath)
            assert ds.n_true > 0, f"{fname} 中无真样本"
            assert ds.n_false > 0, f"{fname} 中无假样本"
