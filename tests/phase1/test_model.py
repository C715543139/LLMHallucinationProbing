"""
P1.2 / P1.3 / P1.6 / M1 — 测试模型加载模块与 GPU 环境 (src/models/loader.py)。

标记:
    - gpu  : 需要真实 CUDA GPU
    - model: 需要本地 Qwen2-1.5B 权重
    - slow : 运行时间较长

覆盖:
    - 模块可正常导入 (P1.6)
    - get_device_info 返回结构正确 (P1.2)
    - CUDA 可用性检测 (P1.2)
    - 本地模型权重目录存在且完整 (P1.3)
    - 模型加载返回 (model, tokenizer) (P1.6)
    - 模型前向传播并输出 hidden states (M1)
    - hidden states 数量与层数匹配 (M1)
"""

from __future__ import annotations

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# P1.6：模块导入检查
# ---------------------------------------------------------------------------

class TestLoaderImport:
    """模块可正常导入 (P1.6)。"""

    def test_module_importable(self) -> None:
        import src.models.loader  # noqa: F401

    def test_get_device_info_importable(self) -> None:
        from src.models.loader import get_device_info  # noqa: F401

    def test_load_model_fp16_importable(self) -> None:
        from src.models.loader import load_model_fp16  # noqa: F401

    def test_load_model_importable(self) -> None:
        from src.models.loader import load_model  # noqa: F401

    def test_print_device_info_importable(self) -> None:
        from src.models.loader import print_device_info  # noqa: F401


# ---------------------------------------------------------------------------
# P1.2：GPU / 设备信息检查
# ---------------------------------------------------------------------------

class TestDeviceInfo:
    """测试设备信息函数 (P1.2)。"""

    def test_get_device_info_returns_dict(self) -> None:
        from src.models.loader import get_device_info
        info = get_device_info()
        assert isinstance(info, dict)

    def test_device_info_has_cuda_available_key(self) -> None:
        from src.models.loader import get_device_info
        info = get_device_info()
        assert "cuda_available" in info

    def test_device_info_has_device_count_key(self) -> None:
        from src.models.loader import get_device_info
        info = get_device_info()
        assert "device_count" in info

    def test_cuda_available_is_bool(self) -> None:
        from src.models.loader import get_device_info
        info = get_device_info()
        assert isinstance(info["cuda_available"], bool)

    def test_device_count_is_int(self) -> None:
        from src.models.loader import get_device_info
        info = get_device_info()
        assert isinstance(info["device_count"], int)

    @pytest.mark.gpu
    def test_cuda_is_available(self) -> None:
        """验证 GPU 实际可用 (P1.2)。"""
        import torch
        assert torch.cuda.is_available(), (
            "CUDA 不可用！请检查 NVIDIA 驱动和 CUDA 工具包安装。"
        )

    @pytest.mark.gpu
    def test_gpu_memory_sufficient(self) -> None:
        """验证显存 ≥ 6 GB（Qwen2-1.5B FP16 约需 3 GB）。"""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("无 GPU")
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        assert total_gb >= 6.0, (
            f"显存 {total_gb:.1f} GB 不足，建议 ≥ 6 GB"
        )

    @pytest.mark.gpu
    def test_gpu_info_populated(self) -> None:
        """GPU 可用时，device_0 信息应被填充。"""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("无 GPU")
        from src.models.loader import get_device_info
        info = get_device_info()
        assert "device_0" in info
        d = info["device_0"]
        assert "name" in d
        assert "total_memory_gb" in d


# ---------------------------------------------------------------------------
# P1.3：本地模型权重检查
# ---------------------------------------------------------------------------

MODEL_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
]


class TestModelWeights:
    """验证 Qwen2-1.5B 模型权重已正确下载 (P1.3)。"""

    def test_models_cache_dir_exists(self, models_cache_dir: Path) -> None:
        assert models_cache_dir.exists(), (
            f"models_cache/ 目录不存在: {models_cache_dir}"
        )

    def test_qwen_model_dir_exists(self, models_cache_dir: Path) -> None:
        qwen_dir = models_cache_dir / "Qwen2-1.5B"
        assert qwen_dir.exists(), (
            f"模型目录不存在: {qwen_dir}\n"
            "请先运行: hf download Qwen/Qwen2-1.5B --local-dir models_cache/Qwen2-1.5B"
        )

    @pytest.mark.parametrize("filename", MODEL_FILES)
    def test_model_file_exists(self, models_cache_dir: Path, filename: str) -> None:
        fpath = models_cache_dir / "Qwen2-1.5B" / filename
        assert fpath.exists(), f"模型文件缺失: {fpath}"

    def test_model_safetensors_size(self, models_cache_dir: Path) -> None:
        """safetensors 文件应 > 1 GB（完整 FP16 权重）。"""
        st_path = models_cache_dir / "Qwen2-1.5B" / "model.safetensors"
        if not st_path.exists():
            pytest.skip("model.safetensors 不存在")
        size_gb = st_path.stat().st_size / 1024**3
        assert size_gb > 1.0, (
            f"model.safetensors 大小 {size_gb:.2f} GB 异常（可能下载不完整）"
        )

    def test_config_json_valid(self, models_cache_dir: Path) -> None:
        """config.json 应为合法 JSON 且含关键字段。"""
        import json
        config_path = models_cache_dir / "Qwen2-1.5B" / "config.json"
        if not config_path.exists():
            pytest.skip("config.json 不存在")
        with open(config_path) as f:
            cfg = json.load(f)
        assert "num_hidden_layers" in cfg, "config.json 缺少 num_hidden_layers"
        assert "hidden_size" in cfg, "config.json 缺少 hidden_size"


# ---------------------------------------------------------------------------
# P1.6 / M1：模型加载与前向传播（需要 GPU + 本地权重）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaded_model(models_cache_dir):
    """加载一次模型供本模块所有 model 标记测试复用。"""
    import torch
    from src.models.loader import load_model_fp16

    if not torch.cuda.is_available():
        pytest.skip("无 GPU，跳过模型加载测试")

    qwen_dir = models_cache_dir / "Qwen2-1.5B"
    if not (qwen_dir / "config.json").exists():
        pytest.skip("本地 Qwen2-1.5B 权重不存在，跳过模型加载测试")

    model, tokenizer = load_model_fp16(model_path=str(qwen_dir))
    yield model, tokenizer
    # 释放显存
    del model
    torch.cuda.empty_cache()


@pytest.mark.model
@pytest.mark.slow
class TestModelLoading:
    """测试模型加载功能 (P1.6)。"""

    def test_load_model_returns_tuple(self, loaded_model) -> None:
        model, tokenizer = loaded_model
        assert model is not None
        assert tokenizer is not None

    def test_model_in_eval_mode(self, loaded_model) -> None:
        model, _ = loaded_model
        assert not model.training, "模型应处于 eval 模式"

    def test_model_on_gpu(self, loaded_model) -> None:
        import torch
        model, _ = loaded_model
        device = next(model.parameters()).device
        assert device.type == "cuda", f"模型未在 GPU 上，当前设备: {device}"

    def test_tokenizer_has_pad_token(self, loaded_model) -> None:
        _, tokenizer = loaded_model
        assert tokenizer.pad_token is not None

    def test_model_config_has_num_layers(self, loaded_model) -> None:
        model, _ = loaded_model
        assert hasattr(model.config, "num_hidden_layers"), (
            "model.config 缺少 num_hidden_layers"
        )
        assert model.config.num_hidden_layers > 0

    def test_model_config_has_hidden_size(self, loaded_model) -> None:
        model, _ = loaded_model
        assert hasattr(model.config, "hidden_size"), (
            "model.config 缺少 hidden_size"
        )
        assert model.config.hidden_size > 0


@pytest.mark.model
@pytest.mark.slow
class TestMilestoneM1:
    """
    里程碑 M1：在 GPU 上完成一次完整的前向传播，输出 hidden states。

    验证:
        - 前向传播成功（无异常）
        - 返回 hidden_states 字段
        - hidden_states 数量 = num_hidden_layers + 1（含 embedding output）
        - 去掉 embedding output 后，block hidden states 数量 = num_hidden_layers
        - hidden_states[-1] 的形状符合 (batch, seq_len, hidden_size)
        - last token 的隐藏状态可正常提取，无 NaN/Inf
    """

    TEST_STATEMENT = "The sky is blue."

    def test_forward_pass_runs(self, loaded_model) -> None:
        import torch
        model, tokenizer = loaded_model
        inputs = tokenizer(self.TEST_STATEMENT, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        assert outputs is not None

    def test_hidden_states_present(self, loaded_model) -> None:
        import torch
        model, tokenizer = loaded_model
        inputs = tokenizer(self.TEST_STATEMENT, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        assert outputs.hidden_states is not None, "输出中缺少 hidden_states"
        assert len(outputs.hidden_states) > 0

    def test_hidden_states_count(self, loaded_model) -> None:
        """hidden_states 总数应为 num_hidden_layers + 1（含 embedding）。"""
        import torch
        model, tokenizer = loaded_model
        inputs = tokenizer(self.TEST_STATEMENT, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        expected_count = model.config.num_hidden_layers + 1
        actual_count = len(outputs.hidden_states)
        assert actual_count == expected_count, (
            f"hidden_states 数量 {actual_count} 应为 {expected_count} "
            f"(num_hidden_layers={model.config.num_hidden_layers} + 1 embedding)"
        )

    def test_block_hidden_states_count(self, loaded_model) -> None:
        """去掉 embedding output 后，block 输出数量应等于 num_hidden_layers。"""
        import torch
        model, tokenizer = loaded_model
        inputs = tokenizer(self.TEST_STATEMENT, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        block_hidden = outputs.hidden_states[1:]  # 剥离 embedding output
        assert len(block_hidden) == model.config.num_hidden_layers

    def test_hidden_state_shape(self, loaded_model) -> None:
        """最后一层 hidden state 形状应为 (1, seq_len, hidden_size)。"""
        import torch
        model, tokenizer = loaded_model
        inputs = tokenizer(self.TEST_STATEMENT, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        batch, seq_len, hidden_size = last_hidden.shape
        assert batch == 1
        assert seq_len > 0
        assert hidden_size == model.config.hidden_size

    def test_last_token_hidden_no_nan(self, loaded_model) -> None:
        """最后 token 的隐藏状态不应含 NaN 或 Inf。"""
        import torch
        model, tokenizer = loaded_model
        inputs = tokenizer(self.TEST_STATEMENT, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_token_hidden = outputs.hidden_states[-1][:, -1, :]
        assert not torch.isnan(last_token_hidden).any(), "隐藏状态含 NaN"
        assert not torch.isinf(last_token_hidden).any(), "隐藏状态含 Inf"

    def test_logits_present(self, loaded_model) -> None:
        """前向传播应同时输出 logits。"""
        import torch
        model, tokenizer = loaded_model
        inputs = tokenizer(self.TEST_STATEMENT, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        assert hasattr(outputs, "logits"), "输出缺少 logits"
        assert outputs.logits is not None
