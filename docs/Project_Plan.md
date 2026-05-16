# 项目计划：利用大语言模型内部状态进行幻觉检测

---

## 一、项目概述

### 1.1 项目目标

本项目聚焦于利用大语言模型的内部状态判断给定陈述的真伪，并分析模型内部是否编码了与真实性相关的表征信号。实验数据以 True-False Dataset 中的陈述句为主，基础任务不以开放式问答生成作为主要评价对象。核心任务包括：

1. **简单任务**：实现并对比基于序列概率/困惑度的方法与基于隐藏状态分类的 SAPLMA 方法
2. **分析任务**：分析不同层深度、不同 token 表示方式对检测性能的影响
3. **进阶任务**：在以上基线之上探索注意力或 FFN 激活等内部特征，提出并评估改进方法

### 1.2 实验模型与数据

| 项       | 选择               | 说明                                                                                    |
| -------- | ------------------ | --------------------------------------------------------------------------------------- |
| 实验模型 | Qwen2-1.5B（FP16） | 在 Windows + 8GB 显存环境下稳定运行，可完整完成课程要求                                 |
| 数据集   | True-False Dataset | 6 个子领域：Cities, Inventions, Chemical Elements, Animals, Companies, Scientific Facts |

### 1.3 实验协议

为保证实验结果的可比性与可复现性，所有实验统一遵循以下协议：

- **数据划分**：按领域分层随机划分训练集/验证集/测试集，比例为 8:1:1；若数据中真/假陈述存在配对关系，则同一对样本必须划入同一集合，避免信息泄漏
- **集合职责**：训练集用于分类器训练；验证集仅用于阈值调优、模型选择和早停；测试集仅用于最终一次性能汇报与主结果图表生成，避免将调参结果误写为最终结果
- **随机种子**：所有分类实验固定随机种子（默认 42），建议在 3 个随机种子（42, 123, 2024）下重复实验，报告均值与标准差
- **种子约定**：数据划分使用固定 split seed（建议 42）一次生成并全程复用；42、123、2024 这 3 个随机种子仅用于分类器训练与阈值调优过程中的随机性控制，不重新划分数据集
- **运行时确定性**：最终基线与后续分析实验默认显式使用 float16 加载模型，不依赖 auto dtype；实验前固定 Python / NumPy / PyTorch 随机种子，关闭 cudnn benchmark，并启用 deterministic algorithms（warn_only 模式）
- **结果归档约定**：结果摘要文件需记录阈值优化指标、随机种子与关键运行环境信息（如 Python、PyTorch、Transformers、CUDA、GPU、模型 dtype），便于跨设备对账与复现
- **分析使用边界**：若层深度分析或 token 位置分析的结论将用于后续方法选择（如 Phase 3 指导 Phase 4 的特征选取），则该选择过程仅基于训练集与验证集完成；测试集只用于固定方案后的最终结果汇报，避免隐性数据泄漏
- **统一指标**：Accuracy、Macro-F1 与 AUROC，确保不同方法之间可比
- **层分析约定**：层分析统一以 Transformer block 输出为统计对象，不将 embedding output 计入层号；层索引通过 `model.config.num_hidden_layers` 动态获取，不硬编码具体数值
- **Subject token 提取**：若实验中需要定位主语实体，优先使用依存句法或 noun chunk 规则抽取句首主语短语；对于能够稳定识别的命名实体样本，可辅以 NER 工具（如 spaCy）做对齐检查。若自动解析不可靠，则退化为手工规则，并仅在报告中标注结果。Subject token 分析作为可选项，保底至少比较 First token、Last token 与 Mean pooling

### 1.4 当前实现同步（截至 2026-05-16）

结合当前工作区中的真实代码与目录结构，项目当前状态如下：

- **Phase 1 已完成**：环境、配置、数据加载/预处理、模型加载与 Phase 1 测试均已落地
- **Phase 2 已完成**：PPL 与 SAPLMA 两类基线方法已实现，并完成 `tests/phase2/` 自动化测试
- **Phase 2 复现性修正已完成**：当前代码已显式固定 float16、随机种子与确定性运行选项，并支持在结果摘要中记录 seeds / runtime 等关键信息
- **Phase 2 最终采用结果已收敛**：当前报告与后续分析默认以 `ppl_results.json`、`saplma_logistic_results.json` 与 `saplma_mlp_results_rerun_best.json` 作为已确认的 Phase 2 结果来源
- **Phase 3 已完成实现与收尾**：`src/analysis/` 中的层分析、token 分析与可视化模块已落地，`tests/phase3/` 已建立，完整结果与图像已生成到 `experiments/results/analysis/`，并已同步写入 `docs/Report.md`
- **Phase 4 已完成方向 A（注意力模式）**：`src/features/attention.py` 与 `src/methods/advanced.py` 已落地，已实现 attention-only / hidden-only / hidden+attention 三组消融，并生成 `experiments/results/advanced/attention_ablation_logistic_layer17_last.json`、`attention_ablation_accuracy.png` 与 `attention_feature_deltas.png`
- **Phase 4 当前观察结果已明确**：注意力特征单独使用时接近随机；以 `layer 17 + last` 隐藏状态为基线加入注意力特征后，测试集 Accuracy 从 0.8003 提升到 0.8067，Macro-F1 从 0.8001 提升到 0.8066，而 AUROC 基本持平（0.8879 vs 0.8878）
- **里程碑文档已整合**：M1 与 M2 的核心内容已同步合并到下方对应 Phase 段落中，原独立里程碑文档不再单独保留

---

## 二、硬件与软件环境

### 2.1 硬件配置

| 组件      | 规格                                 |
| --------- | ------------------------------------ |
| 操作系统  | Windows 10 (原生)                    |
| GPU       | NVIDIA RTX 4060 Laptop (8GB VRAM)    |
| CUDA 版本 | 12.4 (推荐)                          |
| 内存      | ≥16GB RAM                            |
| 磁盘      | ≥50GB 可用空间（模型下载约 15-20GB） |

### 2.2 显存预算分析

| 模型       | 精度          | 显存占用 | 是否可运行                            |
| ---------- | ------------- | -------- | ------------------------------------- |
| Qwen2-1.5B | FP16          | ~3 GB    | 完全可运行                            |
| Qwen2-1.5B | FP16 (含梯度) | ~6 GB    | 适合小 batch 的特征提取或参数高效实验 |

**结论**：Qwen2-1.5B FP16 在 8GB 显存下完全可运行，可直接支持小 batch 的特征提取与参数高效实验。

### 2.3 软件环境

| 工具              | 版本             | 用途                             |
| ----------------- | ---------------- | -------------------------------- |
| Python            | 3.10 / 3.11      | 主编程语言                       |
| Conda (Miniconda) | latest           | Python 版本管理                  |
| uv                | latest           | Python 包管理（替代 pip）        |
| PyTorch           | 2.4+ (CUDA 12.4) | 深度学习框架                     |
| Transformers      | 4.44+            | 模型加载与推理                   |
| PEFT / Accelerate | latest           | 高效推理                         |
| scikit-learn      | 1.5+             | 分类器训练                       |
| spaCy             | optional         | Subject token 自动抽取的可选依赖 |
| wandb             | latest           | 实验跟踪（可选）                 |
| pytest            | 8.0+             | 单元测试与集成测试框架           |
| pytest-cov        | 5.0+             | 测试覆盖率统计                   |

> **复现约定**：软件环境表中的版本范围用于说明依赖原则；实际提交与复现实验时，以 `pyproject.toml` 与 `uv.lock` 中维护的项目依赖配置为准。

---

## 三、项目目录结构（已按当前真实工作区同步）

### 3.1 当前真实目录结构

```text
LLMHallucinationProbing/
├── docs/
│   ├── Project_Plan.md
│   ├── Proposal.md
│   ├── Report.md
│   ├── 利用大语言模型内部状态进行幻觉检测.md
│   └── revision/
│       └── Project_Plan_review_v*.md
├── data/
│   ├── raw/
│   │   ├── animals_true_false.csv
│   │   ├── cities_true_false.csv
│   │   ├── companies_true_false.csv
│   │   ├── elements_true_false.csv
│   │   ├── facts_true_false.csv
│   │   ├── generated_true_false.csv
│   │   └── inventions_true_false.csv
│   └── processed/
│       ├── train.pt
│       ├── val.pt
│       └── test.pt
├── experiments/
│   └── results/
│       ├── advanced/
│       │   ├── attention_ablation_accuracy.png
│       │   ├── attention_ablation_logistic_layer17_last.json
│       │   └── attention_feature_deltas.png
│       ├── analysis/
│       │   ├── layer_accuracy_curve.png
│       │   ├── layer_analysis_logistic_last.json
│       │   ├── token_accuracy_comparison.png
│       │   └── token_analysis_logistic_last_layer.json
│       └── baseline/
│           ├── phase2_run.log
│           ├── ppl_results.json
│           ├── saplma_logistic_results.json
│           ├── saplma_mlp_results.json
│           └── saplma_mlp_results_rerun_best.json
├── models_cache/
│   └── Qwen2-1.5B/
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── model.safetensors
├── scripts/
│   ├── check_phase1.ps1
│   ├── run_phase2.py
│   └── run_phase2_simple.py
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/
│   │   └── loader.py
│   ├── methods/
│   │   ├── probability.py
│   │   ├── saplma.py
│   │   └── advanced.py
│   ├── features/
│   │   ├── hidden_states.py
│   │   └── attention.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── layer_analysis.py
│   │   ├── token_analysis.py
│   │   └── visualization.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       └── reproducibility.py
├── tests/
│   ├── conftest.py
│   ├── phase1/
│   │   ├── test_config.py
│   │   ├── test_data.py
│   │   └── test_model.py
│   ├── phase2/
│   │   ├── conftest.py
│   │   ├── test_probability.py
│   │   ├── test_hidden_states.py
│   │   └── test_saplma.py
│   ├── phase3/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_layer_analysis.py
│   │   ├── test_token_analysis.py
│   │   └── test_visualization.py
│   ├── phase4/
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_advanced.py
│   │   ├── test_attention.py
│   │   └── test_visualization.py
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

### 3.2 当前已实现模块与待实现模块

#### 已实现

- `src/config.py`：全局配置与路径/模型/训练超参数、确定性选项管理
- `src/data/dataset.py`：原始 CSV 加载、`TrueFalseDataset`、`.pt` 序列化
- `src/data/preprocessing.py`：分层划分与预处理流水线
- `src/models/loader.py`：Qwen2-1.5B 加载、GPU 信息查询与显式 float16 加载
- `src/methods/probability.py`：PPL 打分、阈值调优、PPL 方法评估
- `src/features/hidden_states.py`：最后 token 特征提取、批量/全层隐藏状态提取
- `src/methods/saplma.py`：逻辑回归 / MLP 分类器训练、预测与完整 SAPLMA 实验
- `src/features/attention.py`：注意力矩阵提取、subject / relation / tail 锚点识别与统计特征构造
- `src/methods/advanced.py`：attention-only、hidden-only 与 hidden+attention 融合实验
- `src/analysis/layer_analysis.py`：逐层隐藏状态分析与层性能曲线提取
- `src/analysis/token_analysis.py`：不同 token pooling 策略分析
- `src/analysis/visualization.py`：层曲线、token 对比图、方法对比图与 Phase 4 注意力消融图生成
- `src/utils/metrics.py`：Accuracy、Macro-F1、AUROC、阈值搜索等评估逻辑
- `src/utils/reproducibility.py`：随机种子、确定性运行与环境信息记录
- `main.py`：状态检查、预处理、Phase 2 / Phase 3 / Phase 4 运行入口

#### 计划中但尚未实现

- `src/features/ffn_activations.py`
- 计划中的 `experiments/configs/*.yaml` 与 `scripts/run_analysis.ps1`、`scripts/run_advanced.ps1`

---

## 四、环境搭建详细步骤

### 4.1 安装 Conda

```powershell
# 1. 安装 Miniconda（如未安装）
# 下载地址: https://docs.conda.io/en/latest/miniconda.html
# 安装完成后重启终端

# 验证安装
conda --version
```

### 4.2 创建 Conda 环境

```powershell
# 创建 Python 3.10 环境
conda create -n llm_hallucination python=3.10 -y
conda activate llm_hallucination
```

### 4.3 使用 uv 安装项目依赖

> **说明**：所有依赖已预先写入 `pyproject.toml`（含 CUDA 版 PyTorch 源），只需以下两步即可完成环境搭建。`uv add` 无需手动执行。

```powershell
# 在 Conda 环境中安装 uv
python -m pip install uv

# 一键安装全部依赖
uv sync

# 激活 uv 环境
.\.venv\Scripts\activate.ps1
```

> **依赖清单**（由 `pyproject.toml` 统一管理，无需手动安装）：

| 类别                | 包含的包                                                          |
| ------------------- | ----------------------------------------------------------------- |
| 深度学习框架        | `torch`, `torchvision`, `torchaudio`（CUDA 12.4 源）              |
| 模型与推理          | `transformers`, `accelerate`, `peft`, `sentencepiece`, `protobuf` |
| 数据处理            | `datasets`, `numpy`, `pandas`, `huggingface-hub`                  |
| 机器学习            | `scikit-learn`, `scipy`                                           |
| 可视化              | `matplotlib`, `seaborn`                                           |
| 工具                | `tqdm`, `pyyaml`                                                  |
| 可选：Subject token | `spacy`                                                           |
| 可选：实验跟踪      | `wandb`                                                           |
| 开发工具            | `ipykernel`, `jupyter`, `ipywidgets`, `black`, `ruff`, `mypy`     |
| 测试                | `pytest`, `pytest-cov`                                            |

> ⚠️ **重要**：后续所有命令执行前，必须依次激活两个环境：
>
> ```powershell
> conda activate llm_hallucination
> .\.venv\Scripts\activate.ps1
> ```
>
> 未激活环境将导致 `python` 指向错误的解释器，或缺少必要的依赖包。

### 4.4 下载模型

```powershell
# Qwen2-1.5B（FP16 直接可用，无需量化）
hf download Qwen/Qwen2-1.5B --local-dir models_cache/Qwen2-1.5B
```

> **当前实现同步**：当前工作区中已存在 `models_cache/Qwen2-1.5B/`，包括 `config.json`、`tokenizer.json`、`tokenizer_config.json` 与 `model.safetensors` 等核心文件。若仅复现实验，可优先检查本地缓存是否已齐全，再决定是否重新下载。

> **Windows 端注意事项**：
>
> - 模型缓存默认在 `C:\Users\<username>\.cache\huggingface\`，可能占用 C 盘空间
> - 建议使用以下 PowerShell 命令将缓存重定向到项目盘（当前终端会话生效）：
>
> ```powershell
> $env:HF_HOME = "F:\PythonCodes\LLMHallucinationProbing\models_cache"
> ```
>
> - 若无法访问 HuggingFace，可使用镜像加速：
>
> ```powershell
> $env:HF_ENDPOINT = "https://hf-mirror.com"
> ```
>
> - 如需持久化环境变量，请通过系统环境变量设置界面或 `setx` 命令完成

### 4.5 数据下载

从论文 [The Internal State of an LLM Knows When It's Lying](https://arxiv.org/pdf/2304.13734) 官方仓库获取 True-False Dataset：

```powershell
# 下载数据集到 data/raw/
# 具体下载地址为
http://azariaa.com/Content/Datasets/true-false-dataset.zip
```

> **当前实现同步**：当前工作区中的 `data/raw/` 已实际采用 CSV 组织形式，文件名为：
>
> - `animals_true_false.csv`
> - `cities_true_false.csv`
> - `companies_true_false.csv`
> - `elements_true_false.csv`
> - `facts_true_false.csv`
> - `generated_true_false.csv`
> - `inventions_true_false.csv`
>
> 其中前 6 个文件对应课程要求的主要领域数据；`generated_true_false.csv` 为当前配置中一并纳入的附加原始文件。

---

## 五、分阶段实施计划

> **时间标注说明**：各 Phase 标题中括号内的"X-Y 天"表示预估有效工作量，后侧日期表示自然日时间窗口，已包含缓冲与并行协作时间。

### Phase 1：环境搭建与数据准备（1-2 天）| 5.12 – 5.14 ｜**当前状态：已完成**

| 编号 | 任务                                    | 输出物                                      | 负责人建议 |
| ---- | --------------------------------------- | ------------------------------------------- | ---------- |
| P1.1 | 安装 Conda、uv，创建 Python 环境        | `pyproject.toml`, `uv.lock`                 | A          |
| P1.2 | 配置 CUDA 环境，验证 GPU 可用           | GPU 测试脚本                                | A          |
| P1.3 | 下载 Qwen2-1.5B 模型权重                | `models_cache/`                             | A          |
| P1.4 | 下载 True-False Dataset 6 个子集        | `data/raw/`                                 | B          |
| P1.5 | 撰写数据加载模块 `src/data/dataset.py`  | 可加载数据的 Dataset 类                     | B          |
| P1.6 | 实现模型加载模块 `src/models/loader.py` | 支持 FP16 的模型加载器                      | A          |
| P1.7 | 划分 train/val/test 并预处理            | `data/processed/`                           | B          |
| P1.8 | 搭建全局配置文件 `src/config.py`        | 统一配置入口                                | A          |
| P1.9 | 编写 Phase 1 验证测试，运行并报告结果   | `tests/phase1/`，`scripts/check_phase1.ps1` | A+B        |

**里程碑 M1**: 能够在 GPU 上成功加载目标模型，并对单条陈述语句完成一次完整的前向传播，输出 hidden states。

**当前同步结果**：

- `src/config.py`、`src/data/dataset.py`、`src/data/preprocessing.py`、`src/models/loader.py` 已实现
- `tests/phase1/` 与 `scripts/check_phase1.ps1` 已建立
- `data/processed/train.pt`、`val.pt`、`test.pt` 已生成
- Phase 1 的阶段性记录已同步整合到本节下方，不再单独保留里程碑文档

#### M1 里程碑整合

##### 里程碑定位

- **对应阶段**：Phase 1
- **目标**：完成基础环境、数据、模型与配置模块搭建，并满足“能够在 GPU 上成功加载目标模型，对单条陈述完成一次前向传播并输出 hidden states”的要求
- **对应任务范围**：P1.1–P1.9 与里程碑 M1

##### 已完成的具体任务

###### 1. 环境与依赖准备

- 已建立项目根目录与 `src/`、`tests/`、`scripts/` 等工程结构
- 已配置 Python 项目依赖管理文件：`pyproject.toml`、`uv.lock`
- 已建立统一测试入口与 Phase 1 验证脚本：`tests/phase1/`、`scripts/check_phase1.ps1`

###### 2. 数据准备与数据集封装

- 已完成 True-False Dataset 的本地准备，原始数据位于 `data/raw/`
- 已生成预处理后的训练集、验证集、测试集文件，位于 `data/processed/`
- 已实现原始 CSV 数据加载、合并逻辑与 `TrueFalseDataset` 数据集封装
- 已支持训练集 / 验证集 / 测试集的保存与加载

###### 3. 模型加载与前向传播基础能力

- 已建立本地模型缓存目录 `models_cache/Qwen2-1.5B/`
- 已实现模型加载模块 `src/models/loader.py`
- 已支持设备信息查询与 CUDA / GPU 可用性检测
- 已能在真实模型上完成单条陈述的前向传播
- 已能输出 `hidden_states` 与 `logits`

###### 4. 全局配置与项目约定

- 已建立统一配置入口：`src/config.py`
- 配置已覆盖路径、模型、数据划分比例、随机种子、分类器超参数与特征提取默认层设置

###### 5. Phase 1 自动化验证

- 已完成配置模块测试：`tests/phase1/test_config.py`
- 已完成数据模块测试：`tests/phase1/test_data.py`
- 已完成模型模块测试：`tests/phase1/test_model.py`
- 已建立公共 fixture：`tests/conftest.py`
- 测试已覆盖配置结构、原始数据存在性、数据集封装与序列化、模型权重完整性、模型加载与 hidden states 输出

##### 对应文件结构

以下结构是 M1 相关核心文件：

```text
LLMHallucinationProbing/
├── data/
│   ├── raw/
│   │   ├── animals_true_false.csv
│   │   ├── cities_true_false.csv
│   │   ├── companies_true_false.csv
│   │   ├── elements_true_false.csv
│   │   ├── facts_true_false.csv
│   │   ├── generated_true_false.csv
│   │   └── inventions_true_false.csv
│   └── processed/
│       ├── train.pt
│       ├── val.pt
│       └── test.pt
├── models_cache/
│   └── Qwen2-1.5B/
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── model.safetensors
├── scripts/
│   └── check_phase1.ps1
├── src/
│   ├── config.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   └── models/
│       └── loader.py
└── tests/
    ├── conftest.py
    └── phase1/
        ├── test_config.py
        ├── test_data.py
        └── test_model.py
```

##### 实践要点

- **路径管理统一使用 `pathlib.Path`**：在 Windows 环境下统一使用 `Path` 对象，避免硬编码路径导致的兼容性问题
- **数据划分强调可复现与防泄漏**：训练 / 验证 / 测试划分清晰，集合无重叠，并尽量保持真 / 假样本共存
- **模型侧优先验证最小闭环**：先验证模型可加载、tokenizer 可用、单句前向传播成功、`hidden_states` 数量与层数一致、最后 token 表示可稳定提取
- **测试先行为后续扩展保留接口契约**：`TrueFalseDataset` 字段结构、`.pt` 文件加载方式、模型输出 hidden states 的约定均已在 Phase 1 固定下来

##### M1 达成情况总结

- 数据、模型、配置与测试基础设施已经搭建完成
- Qwen2-1.5B 本地权重可被识别与加载
- 单条陈述可完成一次前向传播并稳定输出 hidden states

**结论**：项目已具备进入 Phase 2 基线方法实现与对比实验的条件。

---

### Phase 2：基础方法实现（3-4 天）| 5.15 – 5.21 ｜**当前状态：已完成**

| 编号 | 任务                                                      | 输出物                          | 负责人建议 |
| ---- | --------------------------------------------------------- | ------------------------------- | ---------- |
| P2.1 | **概率方法 (PPL)**：计算语句的序列困惑度                  | `src/methods/probability.py`    | A          |
| P2.2 | PPL 方法在验证集上调优判别阈值，在测试集上汇报最终结果    | baseline PPL 结果               | A          |
| P2.3 | **SAPLMA**：提取最后 token 的隐藏状态                     | `src/features/hidden_states.py` | B          |
| P2.4 | 实现隐藏状态分类器（逻辑回归 + MLP）                      | `src/methods/saplma.py`         | B          |
| P2.5 | SAPLMA 在验证集上完成模型选择，在测试集上汇报最终结果     | baseline SAPLMA 结果            | B          |
| P2.6 | 两种方法在测试集上的对比评估（Accuracy, Macro-F1, AUROC） | 对比表格与图表                  | A+B        |
| P2.7 | 整理 Phase 2 结果，写入报告初稿                           | 报告 3.1 节初稿                 | A+B        |

**关键技术细节**：

#### P2.1 概率方法实现要点

```python
# 伪代码：基于序列困惑度的幻觉检测
def compute_statement_ppl(model, tokenizer, statement):
    """
    计算陈述句的长度归一化困惑度（Perplexity），用作真伪判别分数。
    - 对语句进行 tokenize
    - 前向传播获取每个 token 的 logits
    - 计算序列的平均负对数似然 → perplexity = exp(loss)
    - PPL 越低表示模型越认可该陈述，根据阈值进行二分类
    """
    inputs = tokenizer(statement, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    perplexity = torch.exp(outputs.loss)
    return perplexity
```

#### P2.3 SAPLMA 方法实现要点

```python
# 伪代码：基于隐藏状态的幻觉检测
def extract_last_token_hidden(model, tokenizer, statement, layer_idx=-1):
    """
    提取指定 Transformer 层的最后 token 隐藏状态作为特征。
    - layer_idx 以 Transformer block 输出为编号，不包含 embedding output
    - 默认取最后一个 Transformer block 的输出
    - 也可配置提取所有层的状态用于后续分析
    """
    inputs = tokenizer(statement, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # 显式剥离 embedding output，仅保留 Transformer block 输出
    hidden_states = outputs.hidden_states
    if len(hidden_states) == model.config.num_hidden_layers + 1:
        block_hidden_states = hidden_states[1:]   # hidden_states[0] 是 embedding output
    else:
        block_hidden_states = hidden_states

    last_token_hidden = block_hidden_states[layer_idx][:, -1, :]
    return last_token_hidden.cpu().numpy()
```

**里程碑 M2**: 完成 PPL 与 SAPLMA 两类基线方法的实现、验证和对比；在验证集上调优后，以测试集结果作为最终汇报，输出 Accuracy、Macro-F1 与 AUROC。

**当前同步结果**：

- `src/methods/probability.py` 已实现 PPL 打分、阈值调优与完整评估流水线
- `src/features/hidden_states.py` 已实现最后 token 特征提取、批量提取与全层提取接口
- `src/methods/saplma.py` 已实现 LR / MLP 分类器训练、预测与多随机种子 SAPLMA 实验
- `src/config.py`、`src/models/loader.py` 与 `src/utils/reproducibility.py` 已补充显式 float16、全局随机种子与确定性运行设置
- `main.py` 已提供 `phase2`、`phase2-ppl`、`phase2-saplma` 运行入口
- `experiments/results/baseline/` 已存在 Phase 2 结果文件：`ppl_results.json`、`saplma_logistic_results.json`、`saplma_mlp_results.json`（历史本地结果）与 `saplma_mlp_results_rerun_best.json`（修改后方案重跑并在报告中采用的最终结果）
- 新的结果摘要写盘逻辑已支持记录 `threshold_metric`、`seeds` 与 `runtime` 等复现信息
- `tests/phase2/` 已建立并通过真实代码验证
- Phase 2 的阶段性记录已同步整合到本节下方，不再单独保留里程碑文档

#### M2 里程碑整合

##### 里程碑定位

- **对应阶段**：Phase 2
- **目标**：完成基于序列概率 / 困惑度的 PPL 方法、基于隐藏状态分类的 SAPLMA 方法，并建立对应的自动化测试与最小端到端验证流程
- **对应任务范围**：P2.1–P2.7 与里程碑 M2

##### 已完成的具体任务

###### 1. 概率方法（PPL）实现

- 已在 `src/methods/probability.py` 中实现单条陈述 PPL 计算：`compute_statement_ppl`
- 已支持多条陈述批量计算 PPL：`compute_ppl_scores`
- 已支持验证集阈值调优：`tune_ppl_threshold`、`find_best_ppl_threshold`、`optimize_ppl_threshold`
- 已支持按“PPL 越低越可能为真”的约定进行阈值搜索与评估
- 已支持完整 PPL 流水线评估：`evaluate_ppl_method`

###### 2. 隐藏状态特征提取实现

- 已在 `src/features/hidden_states.py` 中实现单条陈述指定层最后 token 表示提取：`extract_last_token_hidden`
- 已支持批量提取隐藏状态：`extract_hidden_states`
- 已支持提取所有层的隐藏状态：`extract_all_layer_hidden_states`
- 已支持从 `TrueFalseDataset` 直接提取特征与标签：`extract_hidden_states_dataset`

###### 3. SAPLMA 分类器实现

- 已支持训练 SAPLMA 分类器：`train_saplma_classifier`、`train_hidden_state_classifier`、`fit_saplma_classifier`
- 已支持预测标签：`predict_with_classifier`、`predict_saplma`、`predict_labels`
- 已支持输出概率分数：`predict_proba_with_classifier`、`predict_saplma_proba`、`predict_probabilities`
- 已支持完整 SAPLMA 实验流水线：`train_and_evaluate`、`run_saplma_experiment`、`run_saplma_full`

###### 4. Phase 2 测试与边界验证

- 已建立 `tests/phase2/` 测试套件，并在真实代码上完成验证
- 测试覆盖 PPL 方法接口存在性与基本行为
- 测试覆盖隐藏状态提取的层索引语义、返回维度与数值稳定性
- 测试覆盖 SAPLMA 训练、预测、概率输出与真实模型下的小规模端到端流程
- 测试覆盖多项边界情形，包括批量 PPL 顺序一致性、常数分数下的阈值稳定性、不含 embedding output 的兼容性、非法分类器类型异常处理与无 `predict_proba` 时的概率退化逻辑

###### 5. 真实代码验证结果

- `tests/phase2/` 全量通过
- 当前累计通过数为 **31 passed**
- 说明 PPL 与 SAPLMA 的最小闭环已经在真实代码与小规模真实模型验证下打通

##### 对应文件结构

以下结构是 M2 相关核心文件：

```text
LLMHallucinationProbing/
├── src/
│   ├── features/
│   │   └── hidden_states.py
│   ├── methods/
│   │   ├── probability.py
│   │   └── saplma.py
│   ├── data/
│   │   └── dataset.py
│   └── utils/
│       └── metrics.py
├── tests/
│   └── phase2/
│       ├── __init__.py
│       ├── conftest.py
│       ├── test_probability.py
│       ├── test_hidden_states.py
│       └── test_saplma.py
├── data/
│   └── processed/
│       ├── train.pt
│       ├── val.pt
│       └── test.pt
└── models_cache/
    └── Qwen2-1.5B/
```

##### 实践要点

- **明确两类基线方法的接口边界**：PPL 负责生成连续分数并通过阈值完成判别，SAPLMA 负责将隐藏状态映射为监督分类问题，因此需要独立设计分数函数、特征提取、训练、预测和统一评估入口
- **层索引语义必须与文档一致**：层号按 Transformer block 输出统计，不把 embedding output 计入层号，并兼容含 / 不含 embedding output 的返回结构
- **PPL 阈值方向不能反**：PPL 越低表示模型越认可该陈述，阈值搜索与评估实现必须遵守这一方向约定
- **测试不仅验证接口存在，还验证边界**：需要同时覆盖 dummy model、真实模型、批量与单样本逻辑、返回维度稳定性、异常路径与合理退化逻辑
- **延迟导入降低环境噪声影响**：按需导入可减少 Windows 环境下原生依赖初始化带来的额外干扰
- **复现实验要显式约束运行条件**：为减小跨设备漂移，Phase 2 最终代码路径已固定 float16、随机种子与确定性运行选项；后续 Phase 3 / Phase 4 建议沿用同一约定

##### M2 达成情况总结

- PPL 基线方法可运行，并支持阈值调优
- SAPLMA 基线方法可运行，并支持 LR / MLP 两类分类器
- 隐藏状态提取、训练、预测、评估链路已打通
- Phase 2 自动化测试与边界测试已建立并通过

**结论**：Phase 2 已为后续分析任务提供稳定基线，并已实际支撑当前工作区中完成的 Phase 3 层分析与 token 位置分析实验。

---

### Phase 3：分析实验（3-4 天）| 5.22 – 5.28 ｜**当前状态：已完成收尾**

| 编号 | 任务                                                     | 输出物                                                     | 负责人建议 |
| ---- | -------------------------------------------------------- | ---------------------------------------------------------- | ---------- |
| P3.1 | 提取模型全部 Transformer 层的隐藏状态                    | 分层特征矩阵（层数由 model.config.num_hidden_layers 确定） | B          |
| P3.2 | 逐层训练分类器，绘制层深度-准确率曲线                    | `src/analysis/layer_analysis.py`                           | B          |
| P3.3 | 分析不同 token 位置的检测效果（First/Last/Mean pooling） | `src/analysis/token_analysis.py`                           | A          |
| P3.4 | 对比 PPL vs SAPLMA 并撰写分析                            | 对比分析图表                                               | A+B        |
| P3.5 | 对分析结果做可视化                                       | `src/analysis/visualization.py`                            | B          |
| P3.6 | 整理 Phase 3 结果，写入报告                              | 报告 3.2 节初稿                                            | A+B        |

**分析维度详述**：

#### P3.2 层深度分析

- 遍历模型全部 Transformer 层（从第 1 个 Transformer block 到最后一层），逐层提取隐藏状态
- 每层分别训练逻辑回归分类器
- 绘制 "层深度 vs. 分类准确率" 曲线
- 当前观察：中后层编码了最强的真实性信息，浅层接近随机水平，最后层相对中后层存在回落
- 层分析统一以 Transformer block 输出为统计对象，不将 embedding output 计入层号

#### P3.3 Token 位置分析

比较以下 token 表示方式的检测效果（保底至少完成 First / Last / Mean pooling 三项）：

- **Last token**：最后一个 token 的表示（最常用，能看到完整上下文）
- **Mean pooling**：所有 token 的平均池化
- **First token**：句首 token 的表示
- **Subject token(s)**（可选）：主语实体的 token 表示。优先使用依存句法或 noun chunk 规则抽取句首主语短语，辅以 NER（如 spaCy）做对齐检查；如自动解析不可靠，退化为手工规则，并仅在报告中标注结果

分析不同位置的表示对分类结果的影响，并讨论自回归模型上下文化程度对表示的塑造作用。

#### P3.4 方法对比分析维度

| 对比维度   | PPL 方法                  | SAPLMA 方法                         |
| ---------- | ------------------------- | ----------------------------------- |
| 检测准确率 | Test Accuracy = 0.5293    | 最优观测配置 Test Accuracy = 0.7987 |
| 受句长影响 | 较大（长句 PPL 天然偏高） | 较小                                |
| 受词频影响 | 较大                      | 较小                                |
| 计算开销   | 低                        | 中（需存储隐藏状态）                |
| 可解释性   | 低                        | 中到高（可分析层与 token 位置贡献） |

**里程碑 M3**: 完成层深度和 token 位置的分析，明确最佳特征提取策略，为进阶实验提供方向。

**当前同步结果**：

- `src/analysis/layer_analysis.py`、`src/analysis/token_analysis.py` 与 `src/analysis/visualization.py` 已实现
- `main.py` 已提供 `phase3`、`phase3-layer` 与 `phase3-token` 运行入口
- `tests/phase3/` 已建立，对层分析、token 分析、可视化接口和小规模真实模型路径进行验收
- `experiments/results/analysis/` 已生成 `layer_analysis_logistic_last.json`、`token_analysis_logistic_last_layer.json`、`layer_accuracy_curve.png` 与 `token_accuracy_comparison.png`
- 当前已观察到的最佳层为 `layer 17`，在 `last` token + logistic 配置下测试集 Accuracy 为 0.7987、Macro-F1 为 0.7986、AUROC 为 0.8878
- 在最后层固定设置下，`last pooling` 的测试集 Accuracy 为 0.7433，高于 `mean pooling` 的 0.7021 与 `first pooling` 的 0.3867
- `docs/Report.md` 已同步写入 Phase 3 结果、讨论与结论，Phase 3 可视为已完成收尾

**结论**：Phase 3 已达到计划中的分析与收尾目标，可以正式作为 Phase 4 的起点。

---

### Phase 4：进阶方法探索（5-6 天）| 5.29 – 6.4 ｜**当前状态：已完成方向 A（注意力模式）**

从以下方向中选择 1-2 个深入研究：

#### 方向 A：基于注意力模式的幻觉检测

**动机**：真实陈述中，模型对关键实体的注意力更集中；虚假陈述中注意力可能分散。

> **数据适配说明**：True-False Dataset 的基础输入是陈述句，不天然具有 query-answer 结构。因此优先以陈述句内部的实体-关系注意力特征为主（主语 token、关系词 token、句尾 token 之间的注意力统计）。如需使用 query-answer attention 设计，将额外引入问答型数据集，并通过统一提示模板生成 answer 后再计算注意力特征。
>
> **关系词 token 提取规则**：在注意力实验中，优先使用依存句法解析得到与主语实体对应的核心谓词或系词短语作为关系词 span；若自动解析失败，则退化为主语后第一个核心动词（或系表结构）；若规则仍不稳定，则仅保留主语实体与句尾 token 的注意力特征，并在报告中说明退化比例。

| 编号  | 任务                                                               | 输出物             |
| ----- | ------------------------------------------------------------------ | ------------------ |
| P4.A1 | 提取各层注意力权重矩阵                                             | attention maps     |
| P4.A2 | 计算主语实体 token、关系词 token 与句尾 token 之间的注意力统计特征 | attention features |
| P4.A3 | 分析真实/虚假陈述的注意力模式差异                                  | 可视化热力图       |
| P4.A4 | 结合注意力特征 + 隐藏状态训练增强分类器                            | 改进对比结果       |
| P4.A5 | 消融实验：注意力特征独立 vs 组合的增益                             | 消融分析           |

**特征工程**：

```python
# 注意力相关特征（适配陈述句格式）
attention_features = {
    "attn_concentration": entropy(attention_weights),       # 注意力集中度（熵越低越集中）
    "entity_attn_ratio": attn_to_entity / attn_total,       # 对主语实体的关注比例
    "relation_attn_ratio": attn_to_relation / attn_total,   # 对关系词 token 的关注比例
    "tail_attn_ratio": attn_to_tail / attn_total,           # 对句尾 token 的关注比例
    "cross_head_agreement": mean_head_cosine_similarity,    # 多头注意力一致性
    "attn_entropy_per_layer": [h_i for h_i in entropies],   # 逐层注意力熵
}
```

#### 方向 B：基于 FFN 知识激活的幻觉检测

**动机**：FFN 存储事实知识；虚假陈述可能激活较少或不同的知识神经元。

| 编号  | 任务                                 | 输出物                 |
| ----- | ------------------------------------ | ---------------------- |
| P4.B1 | 提取各层 FFN 中间激活值              | FFN activation vectors |
| P4.B2 | 计算 Knowledge Entropy（知识熵）特征 | 知识熵曲线             |
| P4.B3 | 分析真实/虚假陈述的 FFN 激活模式差异 | 对比分析               |
| P4.B4 | 结合 FFN 特征训练增强分类器          | 改进对比结果           |

**特征工程**：

```python
# FFN 相关特征
ffn_features = {
    "knowledge_entropy": -sum(p * log(p) for p in activation_dist),  # 知识熵
    "sparsity": ratio_of_zero_activations,                            # 激活稀疏度
    "top_k_activation_mean": mean(top_k_activations),                 # Top-K 平均激活
    "layer_wise_entropy": [e_i for e_i in layer_entropies],          # 逐层知识熵
}
```

#### 方向 C：特定架构模型分析（MoE / DLLM）

如果时间和算力允许，可选择此方向：

- 下载 OLMoE-1B-7B 或 LLADA-8B-base
- 分析 MoE 路由模式与幻觉的关系
- 或分析扩散语言模型 (DLLM) 的解码轨迹

| 实验项        | 方法                                 | 对比基线    |
| ------------- | ------------------------------------ | ----------- |
| 基线 (SAPLMA) | 隐藏状态 + 逻辑回归                  | —           |
| 改进方法      | 隐藏状态 + 注意力特征 + 逻辑回归/MLP | vs 基线     |
| 改进方法      | 隐藏状态 + FFN 特征 + 逻辑回归/MLP   | vs 基线     |
| 最佳组合      | 多特征融合                           | vs 单一方法 |

**里程碑 M4**: 至少完成一种进阶特征的实现与对比实验（注意力或 FFN）；若性能未超过基线，也需分析失败原因、适用条件和观察到的内部模式差异。

**当前同步结果**：

- `src/features/attention.py` 已实现注意力矩阵读取、基于规则的 `subject / relation / tail` 锚点定位、offset mapping 对齐以及 20 维注意力统计特征构造
- `src/methods/advanced.py` 已实现 `attention_only`、`hidden_only` 与 `hidden_plus_attention` 三类变体，并支持多随机种子评估与特征差异汇总
- `src/analysis/visualization.py` 已补充 Phase 4 图表接口，可生成注意力消融对比图与注意力特征差异图
- `main.py` 已提供 `phase4` 与 `phase4-attention` 命令入口
- `tests/phase4/` 已建立，并完成轻量单元测试与小规模真实模型集成测试
- `experiments/results/advanced/attention_ablation_logistic_layer17_last.json` 已保存正式结果；对应图像 `attention_ablation_accuracy.png` 与 `attention_feature_deltas.png` 已生成

#### M4 阶段性结果（方向 A）

当前 Phase 4 默认沿用 Phase 3 的最优隐藏状态基线 `layer 17 + last + logistic`，并在其上比较三种设置：

| 变体 | 特征维度 | Test Accuracy | Test Macro-F1 | Test AUROC |
| ---- | -------- | ------------- | ------------- | ---------- |
| attention_only | 20 | 0.4976 | 0.4974 | 0.5031 |
| hidden_only | 1536 | 0.8003 | 0.8001 | 0.8879 |
| hidden_plus_attention | 1556 | **0.8067** | **0.8066** | 0.8878 |

以验证集 Accuracy 作为模型选择标准时，最佳变体为 **hidden_plus_attention**。这说明：

1. 仅使用当前构造的注意力统计特征几乎无法独立完成真假判别；
2. 注意力特征在与隐藏状态融合后，能够带来小幅但稳定的 Accuracy / Macro-F1 提升；
3. AUROC 与 hidden-only 基线几乎持平，说明当前注意力特征更多改善的是固定决策边界附近的分类效果，而不是整体排序能力。

进一步地，从结果文件中的注意力特征差异摘要可以看到，当前真 / 假样本差异最大的条目主要包括 `sequence_length`、`subject_attn_ratio`、`subject_attn_last_layer`、`attn_entropy_mean` 与 `cross_head_agreement_mean`。其中 `sequence_length` 的差异幅度最大，也提示后续仍需继续削弱表面统计特征对注意力实验的干扰。

**结论**：Phase 4 已达到“至少完成一种进阶特征并与基线对比”的课程要求；方向 A 已完成实现、测试和结果归档，可直接写入正式报告。

---

### Phase 5：报告撰写与答辩准备（5-6 天）| 6.5 – 6.11 ｜**当前状态：待实现**

| 编号 | 任务                          | 输出物       |
| ---- | ----------------------------- | ------------ |
| P5.1 | 整理所有实验结果表格和图表    | 高质量图表   |
| P5.2 | 撰写 3.1 节（基础方法）正式稿 | 报告 3.1     |
| P5.3 | 撰写 3.2 节（分析实验）正式稿 | 报告 3.2     |
| P5.4 | 撰写 3.3 节（进阶方法）正式稿 | 报告 3.3     |
| P5.5 | 撰写引言、相关工作、结论      | 报告其他章节 |
| P5.6 | 代码清理、注释、README 完善   | 可复现代码   |
| P5.7 | 制作答辩 PPT                  | 答辩材料     |
| P5.8 | 模拟答辩，内部预演            | 反馈改进     |

**报告结构建议**：

```
1. 引言
2. 相关工作
3. 方法
   3.1 基于生成概率的方法
   3.2 基于隐藏状态分类的方法 (SAPLMA)
   3.3 进阶改进方法
4. 实验设置
5. 实验结果与分析
   5.1 基础方法对比 (3.1)
   5.2 层深度与 Token 位置分析 (3.2)
   5.3 进阶方法评估 (3.3)
6. 讨论与局限
7. 结论
参考文献
```

### 最小可交付版本定义

为确保课程要求在任何环境风险下都能完成，定义以下保底交付内容：

- 使用 **Qwen2-1.5B** 完成 PPL、SAPLMA、层分析、token 位置分析的全部实验
- 至少完成一个进阶方向（注意力或 FFN）的小规模实验与对比
- 输出完整的代码、图表、实验记录和报告初稿

### 报告图表清单

课程文档要求 3.1 和 3.2 部分提供详尽的实验图表，提前规划以下图表清单：

| 编号 | 图表                            | 所属章节  | 说明                                 |
| ---- | ------------------------------- | --------- | ------------------------------------ |
| F1   | PPL vs SAPLMA 对比表            | 3.1 / 5.1 | Accuracy / Macro-F1 / AUROC 汇总     |
| F2   | 层深度 vs Accuracy 曲线         | 3.2 / 5.2 | 横轴为层索引，纵轴为分类准确率       |
| F3   | 不同 token 表示方式的柱状对比图 | 3.2 / 5.2 | First / Last / Mean pooling 效果对比 |
| F4   | PPL 分数分布直方图（真/假分别） | 3.1 / 5.1 | 展示两类陈述的 PPL 分布重叠程度      |
| F5   | 进阶方法 vs 基线对比表          | 3.3 / 5.3 | 多特征组合的消融实验结果             |
| F6   | 注意力热力图（真/假陈述对比）   | 3.3 / 5.3 | 选取典型案例可视化注意力差异         |
| F7   | 错误分析示例表                  | 5.1-5.3   | 列举各类方法的典型失败案例           |

---

## 六、关键风险与对策

| 风险                                  | 概率 | 影响 | 对策                                                    |
| ------------------------------------- | ---- | ---- | ------------------------------------------------------- |
| RTX 4060 8GB 显存不足                 | 低   | 高   | Qwen2-1.5B FP16 仅需 ~3GB，余量充足；在报告中说明限制   |
| HuggingFace 模型下载速度慢            | 高   | 中   | 使用 hf-mirror.com 镜像                                 |
| Windows 路径兼容性问题（`\\` vs `/`） | 中   | 低   | 统一使用 `pathlib.Path`；避免硬编码路径                 |
| 时间不足无法完成进阶方向              | 中   | 中   | 优先完成基础+分析任务（40分），进阶部分选择最简单的方向 |

---

## 七、核心参考文献

1. Azaria, A., & Mitchell, T. (2023). _The Internal State of an LLM Knows When It's Lying._ arXiv:2304.13734. → [https://arxiv.org/pdf/2304.13734](https://arxiv.org/pdf/2304.13734)
2. Li, K., et al. (2024). _Your Mixture-of-Experts LLM Is Secretly an Embedding Model For Free._ ICLR 2025.
3. Chen, Z., et al. (2025). _TRACEDET: Hallucination Detection from the Decoding Trace of Diffusion Large Language Models._ ICLR 2026.
4. Burns, C., et al. (2023). _Discovering Latent Knowledge in Language Models Without Supervision._ arXiv:2212.03827.
5. Li, J., et al. (2023). _HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models._ EMNLP 2023.
