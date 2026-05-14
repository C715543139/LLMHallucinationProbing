# Phase 1 完成情况检验脚本
# 用法: .\scripts\check_phase1.ps1
# 可选参数: -IncludeModel (测试模型加载与 M1 前向传播，需 GPU + 本地权重)

param(
    [switch]$IncludeModel,
    [switch]$Verbose
)

$ErrorActionPreference = "Continue"

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
function Write-Section($title) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  $title" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

function Write-Ok($msg)   { Write-Host "  [OK]  $msg" -ForegroundColor Green }
function Write-Fail($msg) { Write-Host "  [FAIL] $msg" -ForegroundColor Red }
function Write-Warn($msg) { Write-Host "  [WARN] $msg" -ForegroundColor Yellow }
function Write-Info($msg) { Write-Host "  [INFO] $msg" -ForegroundColor Gray }

$global:PassCount = 0
$global:FailCount = 0
$global:WarnCount = 0

function Check($condition, $okMsg, $failMsg, [switch]$IsWarn) {
    if ($condition) {
        Write-Ok $okMsg
        $global:PassCount++
    } elseif ($IsWarn) {
        Write-Warn $failMsg
        $global:WarnCount++
    } else {
        Write-Fail $failMsg
        $global:FailCount++
    }
}

# ---------------------------------------------------------------------------
# P1.1 — 环境与依赖文件
# ---------------------------------------------------------------------------
Write-Section "P1.1  环境与依赖文件"

Check (Test-Path "pyproject.toml") `
    "pyproject.toml 存在" `
    "pyproject.toml 缺失"

Check (Test-Path "uv.lock") `
    "uv.lock 存在" `
    "uv.lock 缺失（运行 uv sync 生成）" -IsWarn

Check (Test-Path ".venv") `
    ".venv 虚拟环境目录存在" `
    ".venv 不存在（运行: uv sync）"

# Python 版本检查
try {
    $pyver = python --version 2>&1
    Check ($pyver -match "3\.1[01]") `
        "Python 版本符合要求: $pyver" `
        "Python 版本不符合要求（需 3.10/3.11）: $pyver" -IsWarn
} catch {
    Write-Fail "无法检测 Python 版本"
    $global:FailCount++
}

# ---------------------------------------------------------------------------
# P1.2 — CUDA / GPU 环境
# ---------------------------------------------------------------------------
Write-Section "P1.2  CUDA / GPU 环境"

$cudaCheck = python -c "import torch; print(torch.cuda.is_available())" 2>&1
Check ($cudaCheck -eq "True") `
    "CUDA 可用（torch.cuda.is_available() = True）" `
    "CUDA 不可用，请检查驱动与 PyTorch CUDA 版本"

$gpuName = python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1
if ($gpuName -ne "N/A") {
    Write-Ok "GPU 设备: $gpuName"
    $global:PassCount++
    $vramGB = python -c "import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)) if torch.cuda.is_available() else print(0)" 2>&1
    Check ([float]$vramGB -ge 6.0) `
        "显存 ${vramGB} GB（≥6 GB 满足运行要求）" `
        "显存 ${vramGB} GB（建议 ≥6 GB）" -IsWarn
} else {
    Write-Warn "GPU 信息无法获取（CUDA 不可用时跳过）"
    $global:WarnCount++
}

# PyTorch 版本
$torchver = python -c "import torch; print(torch.__version__)" 2>&1
Write-Info "PyTorch: $torchver"

$transformersver = python -c "import transformers; print(transformers.__version__)" 2>&1
Write-Info "Transformers: $transformersver"

# ---------------------------------------------------------------------------
# P1.3 — 模型权重
# ---------------------------------------------------------------------------
Write-Section "P1.3  模型权重 (Qwen2-1.5B)"

$modelDir = "models_cache\Qwen2-1.5B"
Check (Test-Path $modelDir) `
    "模型目录存在: $modelDir" `
    "模型目录不存在: $modelDir（运行: hf download Qwen/Qwen2-1.5B --local-dir models_cache/Qwen2-1.5B）"

foreach ($f in @("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")) {
    $fp = Join-Path $modelDir $f
    Check (Test-Path $fp) `
        "模型文件存在: $f" `
        "模型文件缺失: $f"
}

# safetensors 大小检查
$stPath = Join-Path $modelDir "model.safetensors"
if (Test-Path $stPath) {
    $sizeMB = [math]::Round((Get-Item $stPath).Length / 1MB, 0)
    Check ($sizeMB -gt 1024) `
        "model.safetensors 大小 ${sizeMB} MB（正常）" `
        "model.safetensors 大小 ${sizeMB} MB 异常（可能下载不完整）" -IsWarn
}

# ---------------------------------------------------------------------------
# P1.4 — 原始数据集
# ---------------------------------------------------------------------------
Write-Section "P1.4  原始数据集 (data/raw/)"

$rawFiles = @(
    "cities_true_false.csv",
    "inventions_true_false.csv",
    "elements_true_false.csv",
    "animals_true_false.csv",
    "companies_true_false.csv",
    "facts_true_false.csv"
)

Check (Test-Path "data\raw") `
    "data/raw/ 目录存在" `
    "data/raw/ 目录不存在"

foreach ($f in $rawFiles) {
    $fp = "data\raw\$f"
    Check (Test-Path $fp) `
        "数据文件存在: $f" `
        "数据文件缺失: $f"
}

# ---------------------------------------------------------------------------
# P1.5 — 数据加载模块
# ---------------------------------------------------------------------------
Write-Section "P1.5  数据加载模块 (src/data/dataset.py)"

Check (Test-Path "src\data\dataset.py") `
    "src/data/dataset.py 存在" `
    "src/data/dataset.py 不存在"

$dataImport = python -c "from src.data.dataset import TrueFalseDataset, load_all_raw_data; print('ok')" 2>&1
Check ($dataImport -eq "ok") `
    "TrueFalseDataset 和 load_all_raw_data 可正常导入" `
    "导入失败: $dataImport"

# 快速功能验证
$dataTest = python -c @"
from src.data.dataset import TrueFalseDataset
ds = TrueFalseDataset(['A', 'B', 'C'], [1, 0, 1], ['d', 'd', 'd'])
assert len(ds) == 3
assert ds.n_true == 2
assert ds.n_false == 1
item = ds[0]
assert 'statement' in item and 'label' in item and 'domain' in item
print('ok')
"@ 2>&1
Check ($dataTest -eq "ok") `
    "TrueFalseDataset 基本功能正常" `
    "TrueFalseDataset 功能异常: $dataTest"

# ---------------------------------------------------------------------------
# P1.6 — 模型加载模块
# ---------------------------------------------------------------------------
Write-Section "P1.6  模型加载模块 (src/models/loader.py)"

Check (Test-Path "src\models\loader.py") `
    "src/models/loader.py 存在" `
    "src/models/loader.py 不存在"

$loaderImport = python -c "from src.models.loader import load_model_fp16, get_device_info, load_model; print('ok')" 2>&1
Check ($loaderImport -eq "ok") `
    "loader.py 函数可正常导入" `
    "导入失败: $loaderImport"

$deviceInfo = python -c @"
from src.models.loader import get_device_info
info = get_device_info()
assert 'cuda_available' in info
assert 'device_count' in info
print('ok')
"@ 2>&1
Check ($deviceInfo -eq "ok") `
    "get_device_info() 返回结构正确" `
    "get_device_info() 异常: $deviceInfo"

# ---------------------------------------------------------------------------
# P1.7 — 预处理数据
# ---------------------------------------------------------------------------
Write-Section "P1.7  预处理数据 (data/processed/)"

foreach ($f in @("train.pt", "val.pt", "test.pt")) {
    $fp = "data\processed\$f"
    Check (Test-Path $fp) `
        "预处理文件存在: $f" `
        "预处理文件缺失: $f（运行: python main.py preprocess）"
}

# 验证划分比例与无重叠
$splitCheck = python -c @"
from pathlib import Path
from src.data.dataset import load_dataset

proc = Path('data/processed')
required = [proc / 'train.pt', proc / 'val.pt', proc / 'test.pt']
if not all(p.exists() for p in required):
    print('SKIP')
else:
    train = load_dataset(proc / 'train.pt')
    val   = load_dataset(proc / 'val.pt')
    test  = load_dataset(proc / 'test.pt')
    total = len(train) + len(val) + len(test)
    tr = len(train)/total
    vr = len(val)/total
    te = len(test)/total
    assert abs(tr - 0.8) < 0.05, f'Train ratio {tr:.3f} out of range'
    assert abs(vr - 0.1) < 0.05, f'Val ratio {vr:.3f} out of range'
    assert abs(te - 0.1) < 0.05, f'Test ratio {te:.3f} out of range'
    # 无重叠检查
    ts = set(train.statements); vs = set(val.statements); tes = set(test.statements)
    assert len(ts & vs) == 0, 'train/val overlap'
    assert len(ts & tes) == 0, 'train/test overlap'
    assert len(vs & tes) == 0, 'val/test overlap'
    print(f'ok total={total} train={len(train)} val={len(val)} test={len(test)}')
"@ 2>&1

if ($splitCheck -eq "SKIP") {
    Write-Warn "预处理文件不存在，跳过比例与重叠检查"
    $global:WarnCount++
} elseif ($splitCheck -match "^ok") {
    Write-Ok "划分比例与无重叠检查通过: $splitCheck"
    $global:PassCount++
} else {
    Write-Fail "划分检查失败: $splitCheck"
    $global:FailCount++
}

# ---------------------------------------------------------------------------
# P1.8 — 全局配置模块
# ---------------------------------------------------------------------------
Write-Section "P1.8  全局配置模块 (src/config.py)"

Check (Test-Path "src\config.py") `
    "src/config.py 存在" `
    "src/config.py 不存在"

$configCheck = python -c @"
from src.config import config, ExperimentConfig
assert isinstance(config, ExperimentConfig)
assert abs(config.data.train_ratio + config.data.val_ratio + config.data.test_ratio - 1.0) < 1e-6
assert set([42, 123, 2024]).issubset(set(config.training.random_seeds))
assert config.paths.project_root.exists()
print('ok')
"@ 2>&1
Check ($configCheck -eq "ok") `
    "config 结构与默认值正确" `
    "config 检查失败: $configCheck"

# ---------------------------------------------------------------------------
# M1 — 模型前向传播（可选，需 GPU + 本地权重）
# ---------------------------------------------------------------------------
if ($IncludeModel) {
    Write-Section "M1  模型前向传播（GPU + 本地权重）"
    Write-Info "加载 Qwen2-1.5B 进行前向传播测试，可能需要约 1-2 分钟..."

    $m1Check = python -c @"
import torch
from pathlib import Path
from src.models.loader import load_model_fp16

model_dir = Path('models_cache/Qwen2-1.5B')
if not (model_dir / 'config.json').exists():
    print('SKIP:no_model')
elif not torch.cuda.is_available():
    print('SKIP:no_gpu')
else:
    model, tokenizer = load_model_fp16(model_path=str(model_dir))
    stmt = 'The sky is blue.'
    inputs = tokenizer(stmt, return_tensors='pt')
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    n_layers = model.config.num_hidden_layers
    n_hs = len(outputs.hidden_states)
    assert n_hs == n_layers + 1, f'hidden_states count {n_hs} != {n_layers+1}'
    block_hs = outputs.hidden_states[1:]
    assert len(block_hs) == n_layers
    last = outputs.hidden_states[-1][:, -1, :]
    assert not torch.isnan(last).any()
    assert not torch.isinf(last).any()
    del model; torch.cuda.empty_cache()
    print(f'ok layers={n_layers} hidden_size={outputs.hidden_states[-1].shape[-1]}')
"@ 2>&1

    if ($m1Check -match "^SKIP:no_model") {
        Write-Warn "本地 Qwen2-1.5B 权重不存在，跳过 M1 测试"
        $global:WarnCount++
    } elseif ($m1Check -match "^SKIP:no_gpu") {
        Write-Warn "无 CUDA GPU，跳过 M1 测试"
        $global:WarnCount++
    } elseif ($m1Check -match "^ok") {
        Write-Ok "M1 前向传播通过: $m1Check"
        $global:PassCount++
    } else {
        Write-Fail "M1 前向传播失败: $m1Check"
        $global:FailCount++
    }
}

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
Write-Section "检验结果汇总"

Write-Host ""
Write-Host "  通过: $($global:PassCount)" -ForegroundColor Green
Write-Host "  警告: $($global:WarnCount)" -ForegroundColor Yellow
Write-Host "  失败: $($global:FailCount)" -ForegroundColor Red
Write-Host ""

if ($global:FailCount -eq 0 -and $global:WarnCount -eq 0) {
    Write-Host "  Phase 1 全部检查通过！" -ForegroundColor Green
} elseif ($global:FailCount -eq 0) {
    Write-Host "  Phase 1 核心检查通过，存在 $($global:WarnCount) 项警告。" -ForegroundColor Yellow
} else {
    Write-Host "  Phase 1 存在 $($global:FailCount) 项失败，请逐一修复后重新运行。" -ForegroundColor Red
}

# 提示如何运行 pytest
Write-Host ""
Write-Host "  运行完整 pytest（快速，不含模型）:" -ForegroundColor Cyan
Write-Host "    pytest tests/ -v -m 'not model'" -ForegroundColor White
Write-Host ""
Write-Host "  运行含 GPU 的 pytest（不含模型）:" -ForegroundColor Cyan
Write-Host "    pytest tests/ -v -m 'gpu and not model'" -ForegroundColor White
Write-Host ""
Write-Host "  运行含模型加载的完整 pytest（慢）:" -ForegroundColor Cyan
Write-Host "    pytest tests/ -v -m 'model'" -ForegroundColor White
Write-Host ""

if ($global:FailCount -gt 0) { exit 1 } else { exit 0 }
