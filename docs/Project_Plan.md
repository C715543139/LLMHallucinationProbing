# 项目计划：利用大语言模型内部状态进行幻觉检测

---

## 一、项目概述

### 1.1 项目目标

本项目聚焦于利用大语言模型的内部状态判断给定陈述的真伪，并分析模型内部是否编码了与真实性相关的表征信号。实验数据以 True-False Dataset 中的陈述句为主，基础任务不以开放式问答生成作为主要评价对象。核心任务包括：

1. **简单任务**：实现并对比基于序列概率/困惑度的方法与基于隐藏状态分类的 SAPLMA 方法
2. **分析任务**：分析不同层深度、不同 token 表示方式对检测性能的影响
3. **进阶任务**：在以上基线之上探索注意力或 FFN 激活等内部特征，提出并评估改进方法

### 1.2 实验模型与数据

| 项             | 选择                                                                  | 说明                                                                                    |
| -------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 优先尝试模型   | Llama-2-7B（4-bit 量化）                                              | 课程推荐模型；若环境稳定，优先用于主实验                                                |
| 保底主实验模型 | Qwen2-1.5B（FP16）                                                    | 在 Windows + 8GB 显存环境下更稳妥，可完整完成课程要求                                   |
| 使用策略       | 先完成 Qwen2-1.5B 端到端跑通，再尝试将核心实验迁移到 Llama-2-7B 4-bit | 降低环境风险                                                                            |
| 数据集         | True-False Dataset                                                    | 6 个子领域：Cities, Inventions, Chemical Elements, Animals, Companies, Scientific Facts |

### 1.3 实验协议

为保证实验结果的可比性与可复现性，所有实验统一遵循以下协议：

- **数据划分**：按领域分层随机划分训练集/验证集/测试集，比例为 8:1:1；若数据中真/假陈述存在配对关系，则同一对样本必须划入同一集合，避免信息泄漏
- **集合职责**：训练集用于分类器训练；验证集仅用于阈值调优、模型选择和早停；测试集仅用于最终一次性能汇报与主结果图表生成，避免将调参结果误写为最终结果
- **随机种子**：所有分类实验固定随机种子（默认 42），建议在 3 个随机种子（42, 123, 2024）下重复实验，报告均值与标准差
- **种子约定**：数据划分使用固定 split seed（建议 42）一次生成并全程复用；42、123、2024 这 3 个随机种子仅用于分类器训练与阈值调优过程中的随机性控制，不重新划分数据集
- **分析使用边界**：若层深度分析或 token 位置分析的结论将用于后续方法选择（如 Phase 3 指导 Phase 4 的特征选取），则该选择过程仅基于训练集与验证集完成；测试集只用于固定方案后的最终结果汇报，避免隐性数据泄漏
- **统一指标**：Accuracy、Macro-F1 与 AUROC，确保不同方法之间可比
- **层分析约定**：层分析统一以 Transformer block 输出为统计对象，不将 embedding output 计入层号；层索引通过 `model.config.num_hidden_layers` 动态获取，不硬编码具体数值
- **Subject token 提取**：若实验中需要定位主语实体，优先使用依存句法或 noun chunk 规则抽取句首主语短语；对于能够稳定识别的命名实体样本，可辅以 NER 工具（如 spaCy）做对齐检查。若自动解析不可靠，则退化为手工规则，并仅在报告中标注结果。Subject token 分析作为可选项，保底至少比较 First token、Last token 与 Mean pooling

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

| 模型       | 精度            | 显存占用    | 是否可运行                                            |
| ---------- | --------------- | ----------- | ----------------------------------------------------- |
| Llama-2-7B | FP16            | ~14 GB      | 超出 8GB                                              |
| Llama-2-7B | 8-bit 量化      | ~7-8 GB     | 临界（含推理开销可能不足）                            |
| Llama-2-7B | **4-bit (NF4)** | **~4-5 GB** | 可尝试，但在提取 hidden states / attention 时余量有限 |
| Qwen2-1.5B | FP16            | ~3 GB       | 完全可运行                                            |
| Qwen2-1.5B | FP16 (含梯度)   | ~6 GB       | 适合小 batch 的特征提取或参数高效实验                 |

**结论**：优先以 Qwen2-1.5B FP16 作为保底主实验模型，完成端到端实验流程；在 Windows + 8GB 环境下确认 bitsandbytes 兼容性后，可尝试将核心实验迁移到 Llama-2-7B 4-bit 进行补充验证（注意提取 hidden states / attention 时余量有限）。Qwen2-1.5B 可直接支持小 batch 的特征提取与参数高效实验。

### 2.3 软件环境

| 工具              | 版本             | 用途                             |
| ----------------- | ---------------- | -------------------------------- |
| Python            | 3.10 / 3.11      | 主编程语言                       |
| Conda (Miniconda) | latest           | Python 版本管理                  |
| uv                | latest           | Python 包管理（替代 pip）        |
| PyTorch           | 2.4+ (CUDA 12.4) | 深度学习框架                     |
| Transformers      | 4.44+            | 模型加载与推理                   |
| bitsandbytes      | optional         | Llama-2-7B 4-bit 量化的可选支持  |
| PEFT / Accelerate | latest           | 高效推理                         |
| scikit-learn      | 1.5+             | 分类器训练                       |
| spaCy             | optional         | Subject token 自动抽取的可选依赖 |
| wandb             | latest           | 实验跟踪（可选）                 |

> **复现约定**：软件环境表中的版本范围用于说明依赖原则；实际提交与复现实验时，以 `pyproject.toml` 与 `uv.lock` 中维护的项目依赖配置为准。

---

## 三、项目目录结构

```
LLMHallucinationProbing/
├── docs/                           # 文档
│   ├── 利用大语言模型内部状态进行幻觉检测.md   # 任务说明
│   ├── Proposal.md                 # 项目提案
│   └── Project_Plan.md             # 本计划文档
│
├── data/                           # 数据集
│   ├── raw/                        # 原始 True-False Dataset
│   │   ├── cities.json
│   │   ├── inventions.json
│   │   ├── chemical_elements.json
│   │   ├── animals.json
│   │   ├── companies.json
│   │   └── scientific_facts.json
│   └── processed/                  # 预处理后的数据（自动生成）
│       ├── train.pt
│       ├── val.pt
│       └── test.pt
│
├── src/                            # 源代码
│   ├── __init__.py
│   ├── config.py                   # 全局配置（路径、模型、超参数）
│   │
│   ├── data/                       # 数据加载与预处理
│   │   ├── __init__.py
│   │   ├── dataset.py              # True-False Dataset 加载
│   │   └── preprocessing.py        # 数据预处理与划分
│   │
│   ├── methods/                    # 幻觉检测方法
│   │   ├── __init__.py
│   │   ├── probability.py          # 3.1 基于生成概率(PPL)的方法
│   │   ├── saplma.py               # 3.1 基于隐藏状态分类(SAPLMA)
│   │   └── advanced.py             # 3.3 进阶方法（注意力/FFN）
│   │
│   ├── models/                     # 模型加载与工具
│   │   ├── __init__.py
│   │   └── loader.py               # 模型加载（FP16 主路径，兼容可选 4-bit 量化）
│   │
│   ├── features/                   # 特征提取
│   │   ├── __init__.py
│   │   ├── hidden_states.py        # 隐藏状态提取（各层/各token）
│   │   ├── attention.py            # 注意力模式提取
│   │   └── ffn_activations.py      # FFN 神经元激活提取
│   │
│   ├── analysis/                   # 分析与可视化
│   │   ├── __init__.py
│   │   ├── layer_analysis.py       # 3.2 不同层效果分析
│   │   ├── token_analysis.py       # 3.2 不同token位置分析
│   │   └── visualization.py        # 图表生成
│   │
│   └── utils/                      # 工具函数
│       ├── __init__.py
│       ├── metrics.py              # 评估指标
│       └── logging_utils.py        # 日志与实验管理
│
├── scripts/                        # 运行脚本（PowerShell）
│   ├── run_baseline.ps1            # Windows 下运行基础方法
│   ├── run_analysis.ps1            # Windows 下运行分析实验
│   └── run_advanced.ps1            # Windows 下运行进阶实验
│
├── experiments/                    # 实验配置与结果
│   ├── configs/                    # 各实验配置文件
│   │   ├── baseline.yaml
│   │   ├── layer_analysis.yaml
│   │   └── advanced.yaml
│   └── results/                    # 实验结果
│       ├── baseline/
│       ├── analysis/
│       └── advanced/
│
├── models_cache/                   # 预训练模型缓存
│   └── .gitkeep
│
├── pyproject.toml                  # uv 项目与依赖配置
├── uv.lock                         # uv 依赖锁定文件
├── .gitignore
└── README.md
```

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

> **说明**：所有依赖已预先写入 `pyproject.toml`（含 CUDA 版 PyTorch 源与 Windows bitsandbytes 兼容版），只需以下两步即可完成环境搭建。`uv add` 无需手动执行。

```powershell
# 在 Conda 环境中安装 uv
python -m pip install uv

# 一键安装全部依赖
uv sync

# 激活 uv 环境
.\.venv\Scripts\activate.ps1
```

> **依赖清单**（由 `pyproject.toml` 统一管理，无需手动安装）：

| 类别 | 包含的包 |
|------|---------|
| 深度学习框架 | `torch`, `torchvision`, `torchaudio`（CUDA 12.4 源） |
| 模型与推理 | `transformers`, `accelerate`, `peft`, `sentencepiece`, `protobuf` |
| 数据处理 | `datasets`, `numpy`, `pandas`, `huggingface-hub` |
| 机器学习 | `scikit-learn`, `scipy` |
| 可视化 | `matplotlib`, `seaborn` |
| 工具 | `tqdm`, `pyyaml` |
| 可选：4-bit 量化 | `bitsandbytes`（Windows 社区版 wheel） |
| 可选：Subject token | `spacy` |
| 可选：实验跟踪 | `wandb` |
| 开发工具 | `ipykernel`, `jupyter`, `ipywidgets`, `black`, `ruff`, `mypy` |

> ⚠️ **重要**：后续所有命令执行前，必须依次激活两个环境：
> ```powershell
> conda activate llm_hallucination
> .\.venv\Scripts\activate.ps1
> ```
> 未激活环境将导致 `python` 指向错误的解释器，或缺少必要的依赖包。


### 4.4 下载模型

```powershell
# 说明：此处下载的是原始模型权重；Llama-2 的 4-bit 量化在模型加载阶段完成（通过 bitsandbytes）
hf download meta-llama/Llama-2-7b-hf --local-dir models_cache/Llama-2-7b-hf

# Qwen2-1.5B（FP16 直接可用，无需量化；Llama-2 需要申请访问权限后方可下载）
hf download Qwen/Qwen2-1.5B --local-dir models_cache/Qwen2-1.5B
```

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

---

## 五、分阶段实施计划

> **时间标注说明**：各 Phase 标题中括号内的"X-Y 天"表示预估有效工作量，后侧日期表示自然日时间窗口，已包含缓冲与并行协作时间。

### Phase 1：环境搭建与数据准备（1-2 天）| 5.12 – 5.14

| 编号 | 任务                                       | 输出物                                    | 负责人建议 |
| ---- | ------------------------------------------ | ----------------------------------------- | ---------- |
| P1.1 | 安装 Conda、uv，创建 Python 环境           | `pyproject.toml`, `uv.lock`               | A          |
| P1.2 | 配置 CUDA 环境，验证 GPU 可用              | GPU 测试脚本                              | A          |
| P1.3 | 下载 Llama-2-7B 与 Qwen2-1.5B 原始模型权重 | `models_cache/`                           | A          |
| P1.4 | 下载 True-False Dataset 6 个子集           | `data/raw/`                               | B          |
| P1.5 | 撰写数据加载模块 `src/data/dataset.py`     | 可加载数据的 Dataset 类                   | B          |
| P1.6 | 实现模型加载模块 `src/models/loader.py`    | 支持 FP16 主路径与可选 4-bit 量化的加载器 | A          |
| P1.7 | 划分 train/val/test 并预处理               | `data/processed/`                         | B          |
| P1.8 | 搭建全局配置文件 `src/config.py`           | 统一配置入口                              | A          |

**里程碑 M1**: 能够在 GPU 上成功加载目标模型，并对单条陈述语句完成一次完整的前向传播，输出 hidden states。

---

### Phase 2：基础方法实现（3-4 天）| 5.15 – 5.21

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

---

### Phase 3：分析实验（3-4 天）| 5.22 – 5.28

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
- 预期发现：中层编码了最多的真实性信息，浅层与深层相对较弱
- 层分析统一以 Transformer block 输出为统计对象，不将 embedding output 计入层号

#### P3.3 Token 位置分析

比较以下 token 表示方式的检测效果（保底至少完成 First / Last / Mean pooling 三项）：

- **Last token**：最后一个 token 的表示（最常用，能看到完整上下文）
- **Mean pooling**：所有 token 的平均池化
- **First token**：句首 token 的表示
- **Subject token(s)**（可选）：主语实体的 token 表示。优先使用依存句法或 noun chunk 规则抽取句首主语短语，辅以 NER（如 spaCy）做对齐检查；如自动解析不可靠，退化为手工规则，并仅在报告中标注结果

分析不同位置的表示对分类结果的影响，并讨论自回归模型上下文化程度对表示的塑造作用。

#### P3.4 方法对比分析维度

| 对比维度   | PPL 方法                  | SAPLMA 方法              |
| ---------- | ------------------------- | ------------------------ |
| 检测准确率 | ?                         | ?                        |
| 受句长影响 | 较大（长句 PPL 天然偏高） | 较小                     |
| 受词频影响 | 较大                      | 较小                     |
| 计算开销   | 低                        | 中（需存储隐藏状态）     |
| 可解释性   | 低                        | 中（可分析哪些层贡献大） |

**里程碑 M3**: 完成层深度和 token 位置的分析，明确最佳特征提取策略，为进阶实验提供方向。

---

### Phase 4：进阶方法探索（5-6 天）| 5.29 – 6.4

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

---

### Phase 5：报告撰写与答辩准备（5-6 天）| 6.5 – 6.11

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

若环境条件允许，再使用 Llama-2-7B 4-bit 对核心实验进行补充验证，并在报告中对比两种模型的发现。

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

| 风险                                     | 概率 | 影响 | 对策                                                                    |
| ---------------------------------------- | ---- | ---- | ----------------------------------------------------------------------- |
| RTX 4060 8GB 显存不足，即使 4-bit 也 OOM | 低   | 高   | 使用 Qwen2-1.5B 完成实验；在报告中说明限制                              |
| Llama-2 访问权限未获批                   | 中   | 中   | 使用 Qwen2-1.5B / Qwen2-7B 等开源替代                                   |
| HuggingFace 模型下载速度慢               | 高   | 中   | 使用 hf-mirror.com 镜像                                                 |
| 4-bit 量化下隐藏状态精度下降，分类效果差 | 中   | 中   | 使用 8-bit 量化或缩小 batch size；使用 Qwen2-1.5B FP16 做对比           |
| Windows 路径兼容性问题（`\\` vs `/`）    | 中   | 低   | 统一使用 `pathlib.Path`；避免硬编码路径                                 |
| bitsandbytes Windows 兼容性问题          | 高   | 高   | 使用预编译的 bitsandbytes-windows 版本；或降级使用 8-bit via accelerate |
| 时间不足无法完成进阶方向                 | 中   | 中   | 优先完成基础+分析任务（40分），进阶部分选择最简单的方向                 |

### Windows + bitsandbytes 特别注意

Windows 上使用 bitsandbytes 4-bit 量化需要特殊处理：

```powershell
# 方案 1：使用社区维护的 Windows 兼容版
uv add https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.43.3-py3-none-win_amd64.whl

# 方案 2：如果方案 1 失败，使用 accelerate 的 device_map + 8-bit
# 在代码中设置：
# model = AutoModelForCausalLM.from_pretrained(..., load_in_8bit=True, device_map="auto")

# 方案 3：直接使用 Qwen2-1.5B FP16（最稳妥，3GB 显存足够）
```

---

## 七、核心参考文献

1. Azaria, A., & Mitchell, T. (2023). _The Internal State of an LLM Knows When It's Lying._ arXiv:2304.13734. → [https://arxiv.org/pdf/2304.13734](https://arxiv.org/pdf/2304.13734)
2. Li, K., et al. (2024). _Your Mixture-of-Experts LLM Is Secretly an Embedding Model For Free._ ICLR 2025.
3. Chen, Z., et al. (2025). _TRACEDET: Hallucination Detection from the Decoding Trace of Diffusion Large Language Models._ ICLR 2026.
4. Burns, C., et al. (2023). _Discovering Latent Knowledge in Language Models Without Supervision._ arXiv:2212.03827.
5. Li, J., et al. (2023). _HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models._ EMNLP 2023.
