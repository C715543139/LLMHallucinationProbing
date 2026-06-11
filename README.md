# LLM Hallucination Probing

利用大语言模型内部状态进行幻觉检测的实验项目。项目以 Qwen2-1.5B 和 True-False Dataset 为基础，比较序列概率、隐藏状态探测和注意力特征融合等方法，验证模型内部表征中是否存在可被读取的真假判别信号。

当前代码已完成 Phase 1 至 Phase 4 的主体实现、自动化测试与结果归档，后续工作主要集中在正式报告、图表整理、跨模型扩展和显著性分析。

## 项目概览

本项目围绕“给定一句陈述，判断其是否为真实陈述”展开，核心实验线如下：

- **PPL 基线**：使用生成概率/困惑度作为真假判别分数，并在验证集上选择阈值。
- **SAPLMA 基线**：提取模型隐藏状态，用 Logistic Regression 或 MLP 分类器判断真假。
- **层与 token 分析**：逐层比较隐藏状态可分性，并比较 `first` / `last` / `mean` pooling。
- **Phase 4 进阶方法**：基于陈述句 anchor 提取 attention score / attention output 特征，进行长度去偏、head selection 和多种融合消融。

Phase 4 围绕注意力信号做了系统探索：从 raw / debiased attention score、top-head selection、attention output activation，到 hidden 拼接、三路融合和 gated fusion 均纳入 A0-A9 消融。最终 A2 成为 A0-A9 attention-guided 消融中的最优方法，同时 A6/A8/A9 等复杂方案的收益边界也被实验证明，这部分负向结果是进阶探索的重要组成部分。

当前默认实验设置：

- 模型：`Qwen/Qwen2-1.5B`
- 本地模型路径：`models_cache/Qwen2-1.5B/`
- 数据：True-False Dataset 的 CSV 版本，位于 `data/raw/`
- 数据划分：按领域与标签分层的 `train/val/test = 8/1/1`
- 主要指标：Accuracy、Macro-F1、AUROC
- Phase 4 稳定路径：`eager attention + bfloat16`

## 结果摘要

以下结果来自当前仓库中的结果文件，可通过 `python -s scripts/show_results.py` 重新打印。

| 阶段    | 方法                          | Test Acc | Test Macro-F1 | Test AUROC | 说明                               |
| ------- | ----------------------------- | -------: | ------------: | ---------: | ---------------------------------- |
| Phase 2 | PPL                           |   0.5293 |        0.4180 |     0.6784 | 序列概率阈值基线                   |
| Phase 2 | SAPLMA LR                     |   0.7496 |        0.7496 |     0.8265 | 最后一层 `last` token hidden state |
| Phase 2 | SAPLMA MLP                    |   0.7771 |        0.7769 |     0.8770 | 非线性分类器基线                   |
| Phase 3 | Validation-selected L17       |   0.8003 |        0.8001 |     0.8876 | 中后层 hidden state 信号更强       |
| Phase 3 | Best pooling `last`           |   0.7496 |        0.7496 |     0.8265 | `last > mean >> first`             |
| Phase 4 | Hidden-only A0                |   0.8082 |        0.8081 |     0.8897 | 全量 hidden baseline               |
| Phase 4 | Debiased attn-score only A2   |   0.8193 |        0.8193 |     0.9010 | 全量 A0-A9 attention-guided 最优   |
| Phase 4 | Gated fusion A9               |   0.8130 |        0.8129 |     0.8903 | 全量轻微净修正，但非最优           |

主要结论：

- 单纯 PPL 只有较弱的真假排序信号，分类效果明显弱于内部状态方法。
- 隐藏状态中的真实性信号可被轻量分类器读取，SAPLMA 明显优于 PPL。
- 真实性信号在中后层更强，最后层不一定是最佳读出层。
- 对因果语言模型而言，最后一个有效 token 是更可靠的整句表示位置。
- Attention score 在数值稳定后可提供强判别信号；去长度偏置后的 attention-score only（A2）是当前 A0-A9 attention-guided 消融中的全量最优方法，复杂融合和 top-head 拼接未稳定超过该方法。
- 进阶探索覆盖了 score、output、head selection、feature fusion、gated routing、错误修正矩阵和 attention case 可视化；结论既包括 A2 的正向提升，也包括复杂融合策略的边界。

## 项目结构

```text
.
├── main.py                         # 顶层 CLI 分发入口
├── pyproject.toml                  # Python 版本、依赖与 CUDA PyTorch 源
├── src/
│   ├── config.py                   # 路径、模型、数据、训练与特征配置
│   ├── data/                       # 数据集封装与预处理
│   ├── models/                     # 模型加载
│   ├── features/                   # hidden states、anchor、attention 特征提取
│   ├── methods/                    # PPL、SAPLMA、Phase 4 方法实现
│   ├── analysis/                   # 层分析、token 分析、Phase 4 分析与可视化
│   └── utils/                      # 指标、复现性、特征缓存工具
├── scripts/
│   ├── commands/                   # 状态检查、预处理、GPU/Phase 1 检查
│   ├── run/                        # Phase 2/3/4 实验入口
│   └── show_results.py             # 汇总打印现有结果
├── tests/                          # Phase 1-4 自动化测试
├── experiments/results/            # 已归档实验结果
├── docs/                           # 项目计划、阶段报告与进阶方案说明
├── data/                           # 已跟踪的原始数据与预处理划分
└── models_cache/                   # 本地模型缓存，未纳入 Git
```

## 环境要求

推荐复现实验环境：

- Python `>=3.10,<3.12`
- CUDA 12.4 对应的 PyTorch 版本
- NVIDIA GPU；当前主实验结果基于 Linux + RTX 3090 24GB
- `uv` 用于安装 `pyproject.toml` 中锁定的依赖
- HuggingFace 模型下载能力，必要时使用 `HF_ENDPOINT=https://hf-mirror.com`

`models_cache/` 被 `.gitignore` 排除。克隆仓库后，如需完整复现实验，需要自行准备或下载 Qwen2-1.5B 模型文件；数据文件已随仓库跟踪。

## 配置引导

以下为 Linux 复现实验的最小路径，完整环境说明见 [docs/Project_Plan.md](docs/Project_Plan.md)。

```bash
conda create -n llm_hallucination python=3.10 -y
conda activate llm_hallucination

python -m pip install uv
uv sync
source ./.venv/bin/activate
```

准备模型：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$PWD/models_cache"
hf download Qwen/Qwen2-1.5B --local-dir models_cache/Qwen2-1.5B
```

准备数据后运行预处理：

```bash
python -s main.py preprocess
```

默认路径与关键配置集中在 [src/config.py](src/config.py)：

- 原始数据：`data/raw/`
- 预处理数据：`data/processed/`
- 模型缓存：`models_cache/`
- 结果目录：`experiments/results/`
- 默认模型精度：`bfloat16`

## 运行指南

查看项目状态：

```bash
python -s main.py status
```

查看当前已归档结果：

```bash
python -s scripts/show_results.py
python -s scripts/show_results.py --section phase4
```

运行各阶段实验：

```bash
python -s main.py phase2
python -s main.py phase2-ppl
python -s main.py phase2-saplma

python -s main.py phase3
python -s main.py phase3-layer
python -s main.py phase3-token

python -s main.py phase4 --summary-only
python -s main.py phase4 --use-cache
python -s main.py phase4
```

Phase 4 还支持更细粒度的子命令：

```bash
python -s main.py phase4-cache-hidden
python -s main.py phase4-hidden-baseline
python -s main.py phase4-extract-attention-scores
python -s main.py phase4-extract-attention-outputs
python -s main.py phase4-select-heads
python -s main.py phase4-ablation
python -s main.py phase4-visualize
```

运行测试：

```bash
pytest tests/phase1
pytest tests/phase2
pytest tests/phase3
pytest tests/phase4
```

部分测试或实验会加载模型并依赖 CUDA 环境；仅查看已有结果时，优先使用 `scripts/show_results.py`。

## 文档入口

目前只保留最常用的文档入口，暂不对 `docs/` 下全部文件做结构化索引：

- [docs/Project_Plan.md](docs/Project_Plan.md)：项目计划、环境搭建、阶段任务与运行说明。
- [docs/Report.md](docs/Report.md)：阶段性实验报告、方法解释、结果讨论与局限性。
- [docs/Report_ACL_zh.md](docs/Report_ACL_zh.md)：论文式中文对照稿，覆盖 Phase 4 全量消融与最终口径。
- [docs/outdated/Milestone.md](docs/outdated/Milestone.md)：历史中期里程碑归档，仅作提交记录参考。

## License

本项目采用 [MIT License](LICENSE)。
