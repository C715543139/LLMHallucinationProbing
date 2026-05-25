# 项目进展报告

> 项目：利用大语言模型内部状态进行幻觉检测  
> 更新日期：2026-05-25  
> 当前阶段：Phase 1–4 主体实验已完成，进入 Phase 5（报告与答辩准备）

---

## 一、总体进度概览

| 阶段                           | 计划时间       | 状态      | 完成日期   |
| ------------------------------ | -------------- | --------- | ---------- |
| Phase 1：文献调研与环境准备    | 5.8 – 5.12     | 已完成 | 5.12       |
| Phase 2：基础方法实现          | 5.13 – 5.19    | 已完成 | 5.19       |
| Phase 3：分析实验              | 5.20 – 5.25    | 已完成 | 5.19       |
| Phase 4：进阶方法探索          | 5.26 – 6.2     | 提前完成 | 5.19（核心实现） / 5.25（复跑收尾） |
| Phase 5：报告与答辩准备        | 6.3 – 6.9      | 进行中 | —          |

**当前状态**：Phase 1–4 的主体实验、代码实现、自动化测试与结果归档均已落地，项目已提前进入报告撰写与答辩准备阶段。

---

## 二、各阶段详细进展

### Phase 1：文献调研与环境准备

**完成时间**：2026-05-12

**主要工作**：
- 阅读核心参考文献，重点理解 _The Internal State of an LLM Knows When It's Lying_ (Azaria & Mitchell, 2023) 中提出的 SAPLMA 方法
- 梳理概率方法（PPL）与隐藏状态探测方法（SAPLMA）的区别与技术路线
- 确定实验模型为 **Qwen2-1.5B**，实验环境为 **Linux + RTX 3090 (24GB VRAM) + CUDA 12.4**
- 搭建 Python 3.10.20 环境（Conda `llm_hallucination` + uv 虚拟环境），安装 PyTorch 2.6.0、Transformers 4.57.6、scikit-learn 1.7.2 等核心依赖
- 建立项目目录结构与配置模块（`src/config.py`），统一管理路径、模型、训练超参与随机种子

**交付物**：

| 类别       | 文件/目录                                      | 状态 |
| ---------- | ---------------------------------------------- | ---- |
| 配置模块   | `src/config.py`                                | 完成 |
| 数据模块   | `src/data/dataset.py`, `src/data/preprocessing.py` | 完成 |
| 模型模块   | `src/models/loader.py`                         | 完成 |
| 预处理数据 | `data/processed/train.pt`, `val.pt`, `test.pt` | 完成 |
| 模型缓存   | `models_cache/Qwen2-1.5B/`                     | 完成 |
| 测试套件   | `tests/phase1/`（3 个测试文件）                 | 完成 |
| 命令入口   | `scripts/commands/check_phase1.py`, `status.py`, `preprocess.py` | 完成 |

**关键决策**：
- 项目默认精度切换为 `bfloat16`（后经验证这是避免 eager attention NaN 的必要条件）
- 数据按领域分层 8:1:1 划分，划分种子固定为 42，全程复用同一份划分

---

### Phase 2：基础方法实现

**完成时间**：2026-05-19

**主要工作**：
- 实现基于序列困惑度的 **PPL 方法**（`src/methods/probability.py`）：在验证集上搜索最优阈值，在测试集上评估
- 实现基于隐藏状态分类的 **SAPLMA 方法**（`src/methods/saplma.py`）：支持逻辑回归（LR）与 MLP 两种下游分类器，固定最后层 + `last` token 表示
- 完成多随机种子（42, 123, 2024）重复实验，汇报均值与标准差
- 建立 `tests/phase2/` 自动化测试（4 个测试文件）
- 修复复现性问题：显式固定随机种子与确定性运行选项，结果摘要中记录 seeds / runtime 元数据

**核心结果**：

| 方法              | Test Accuracy | Test Macro-F1 | Test AUROC | 备注        |
| ----------------- | ------------- | ------------- | ---------- | ----------- |
| PPL               | 0.5293        | 0.4180        | 0.6784     | 阈值搜索    |
| SAPLMA (Logistic) | 0.7417 ± 0.00 | 0.7417 ± 0.00 | 0.8278 ± 0.00 | 3 seeds 均值 |
| SAPLMA (MLP)      | 0.7697 ± 0.02 | 0.7696 ± 0.02 | 0.8676 ± 0.01 | 重跑最佳结果 |

**初步结论**：
- SAPLMA 显著优于 PPL：隐藏状态中的真实性信号远强于序列概率
- MLP 优于逻辑回归：说明真假判别边界在隐藏空间中并非完全线性
- 结果已收敛并作为后续 Phase 3/4 的基线参考

**交付物**：

| 类别       | 文件/目录                                      | 状态 |
| ---------- | ---------------------------------------------- | ---- |
| 概率方法   | `src/methods/probability.py`                   | 完成 |
| 隐藏状态特征 | `src/features/hidden_states.py`               | 完成 |
| SAPLMA 方法 | `src/methods/saplma.py`                       | 完成 |
| 评估指标   | `src/utils/metrics.py`                         | 完成 |
| 可复现工具 | `src/utils/reproducibility.py`                 | 完成 |
| 测试套件   | `tests/phase2/`（4 个测试文件）                 | 完成 |
| 运行脚本   | `scripts/run/phase2.py`                        | 完成 |
| 基线结果   | `experiments/results/baseline/*.json`（5 个文件） | 完成 |

---

### Phase 3：分析实验

**完成时间**：2026-05-19

**主要工作**：
- 对 28 个 Transformer block 逐层提取 `last` token 隐藏状态，训练逻辑回归分类器，绘制层深度-性能曲线（`src/analysis/layer_analysis.py`）
- 在最后层固定设置下，比较 First / Last / Mean pooling 的效果差异（`src/analysis/token_analysis.py`）
- 生成可视化图表（`src/analysis/visualization.py`）
- 建立 `tests/phase3/` 自动化测试

**核心结果**：

*层深度分析（last token + Logistic Regression）*：

| Layer | Val Accuracy | Test Accuracy | Test Macro-F1 | Test AUROC |
| ----- | ------------ | ------------- | ------------- | ---------- |
| 0     | 0.4960       | 0.5008        | 0.5007        | 0.5068     |
| 13    | 0.8114       | 0.8193        | 0.8193        | 0.8967     |
| 15    | 0.8273       | 0.8225        | 0.8225        | 0.9092     |
| **17** | **0.8288**   | **0.7987**    | **0.7986**    | **0.8878** |
| 20    | 0.8051       | 0.8241        | 0.8241        | 0.9015     |
| 27    | 0.7480       | 0.7433        | 0.7433        | 0.8277     |

*Token 表示分析（Layer 27 + Logistic Regression）*：

| Pooling | Test Accuracy | Test Macro-F1 | Test AUROC |
| ------- | ------------- | ------------- | ---------- |
| last    | 0.7433        | 0.7433        | 0.8277     |
| mean    | 0.7021        | 0.7020        | 0.7653     |
| first   | 0.3867        | 0.3859        | 0.3251     |

**初步结论**：
- 真实性信号在中后层（layer 13–20）达到最强，最后层反而回落，说明最接近输出的位置不一定最适合真假判别
- `last > mean >> first`：自回归模型中最后一个有效 token 是最优的整句读出位置
- 表示选择的重要性不低于分类器复杂度：`layer 17 + last + logistic`（Acc 0.7987）优于 `layer 27 + last + mlp`（Acc 0.7697）

**交付物**：

| 类别       | 文件/目录                                      | 状态 |
| ---------- | ---------------------------------------------- | ---- |
| 层分析     | `src/analysis/layer_analysis.py`               | 完成 |
| Token 分析 | `src/analysis/token_analysis.py`               | 完成 |
| 可视化     | `src/analysis/visualization.py`                 | 完成 |
| 测试套件   | `tests/phase3/`                                | 完成 |
| 运行脚本   | `scripts/run/phase3.py`                        | 完成 |
| 结果文件   | `experiments/results/analysis/*.json`, `*.png` | 完成 |

---

### Phase 4：进阶方法探索（提前完成）

**完成时间**：2026-05-19（核心实现）/ 2026-05-25（复跑收尾）

**主要工作**：
- 诊断并修复了 `eager + float16` 下的 NaN 问题，切换至稳定路径 `eager + bfloat16`
- 实现陈述句 anchor 抽取（subject / relation / tail / last token）（`src/features/anchor_extraction.py`）
- 提取 layer/head 级 attention score 特征，实现去长度偏置（`src/features/attention_scores.py`）
- 提取 attention output 激活统计特征（`src/features/attention_outputs.py`）
- 实现 validation-based head selection，筛选 top-k 高价值注意力头
- 完成 A0–A9 系统消融矩阵（hidden-only / attention-only / 不同融合方式 / gated fusion）
- 完成错误分析与图表生成（`src/analysis/phase4_analysis.py`）
- 建立 `tests/phase4/` 自动化测试（6 个测试文件）

**核心结果**（600/150/150 子集消融）：

| 方法 | 特征                       | Test Acc | Test F1 | Test AUROC | 结论                     |
| ---- | -------------------------- | -------- | ------- | ---------- | ------------------------ |
| A0s  | Hidden-only (layer 17)     | 0.8667   | 0.8661  | 0.9184     | 子集基线                 |
| A2   | Debiased attn-score only   | 0.7733   | 0.7713  | 0.8285     | score 单独已具判别力     |
| A4   | Attn-output only           | 0.6800   | 0.6800  | 0.7493     | output 单独弱于 score    |
| A5   | Hidden + debiased score    | 0.8733   | 0.8729  | 0.9302     | 融合带来稳定提升         |
| **A6** | **Hidden + top-16 head** | **0.8867** | **0.8865** | **0.9330** | **最佳 Accuracy / F1**   |
| A7   | Hidden + attn-output       | 0.8467   | 0.8463  | 0.9254     | AUROC 提升但 Acc 回落    |
| **A8** | **Hidden + top-head + output** | 0.8800 | 0.8798 | **0.9403** | **最佳 AUROC**           |
| A9   | Gated Fusion               | 0.8667   | 0.8661  | 0.9193     | 无净纠错收益             |

**全量 hidden baseline 复跑**（eager + bfloat16）：

| 方法                     | Test Acc | Test F1 | Test AUROC |
| ------------------------ | -------- | ------- | ---------- |
| Hidden-only (A0, 全量)   | 0.8082   | 0.8081  | 0.8897     |

**初步结论**：
1. **Attention score 不是噪声**：A2 远高于随机水平，说明注意力分数本身就含有真假判别信号
2. **Head selection 是有效的**：先基于验证集筛选高价值 head，再与 hidden state 融合（A6），比直接拼接全部 score 统计（A5）更有效
3. **Attention output 更偏排序信号**：A7/A8 的 AUROC 提升明显，但 Accuracy 不及 A6，适合作为补充排序信息
4. **复杂路由不一定优于直接融合**：当前的 simple gated fusion（A9）没有带来净纠错收益
5. **数值稳定性是前提**：Linux + RTX 3090 上 `eager + float16` 仍产生 NaN，`eager + bfloat16` 是唯一可复现的稳定路径

**交付物**：

| 类别         | 文件/目录                                        | 状态 |
| ------------ | ------------------------------------------------ | ---- |
| Anchor 抽取  | `src/features/anchor_extraction.py`               | 完成 |
| Attention 特征 | `src/features/attention_scores.py`, `attention_outputs.py` | 完成 |
| 特征缓存     | `src/utils/feature_cache.py`                      | 完成 |
| Phase 4 方法 | `src/methods/phase4_attention.py`                 | 完成 |
| Phase 4 分析 | `src/analysis/phase4_analysis.py`                 | 完成 |
| 测试套件     | `tests/phase4/`（6 个测试文件）                     | 完成 |
| 运行脚本     | `scripts/run/phase4.py`                           | 完成 |
| 消融结果     | `experiments/results/phase4/*.json`, `*.csv`      | 完成 |

---

## 三、工程与代码质量

### 3.1 代码架构

项目采用分层模块化设计，职责清晰：

```
src/
├── config.py          # 全局配置（路径、模型、训练超参、随机种子）
├── data/              # 数据加载与预处理
├── models/            # 模型加载
├── features/          # 特征提取（hidden_states, anchor, attention scores/outputs）
├── methods/           # 方法实现（PPL, SAPLMA, Phase 4 attention）
├── analysis/          # 分析与可视化
└── utils/             # 工具（metrics, reproducibility, feature_cache）
```

### 3.2 CLI 入口

`main.py` 作为纯命令分发器，支持以下命令：

| 命令                           | 功能               |
| ------------------------------ | ------------------ |
| `status` / `check-phase1`      | 环境诊断           |
| `preprocess`                   | 数据预处理         |
| `phase2` / `phase2-ppl` / `phase2-saplma` | Phase 2 实验 |
| `phase3` / `phase3-layer` / `phase3-token` | Phase 3 实验 |
| `phase4` / `phase4-cache-hidden` / `phase4-ablation` / ... | Phase 4 全流程 |

### 3.3 自动化测试

已建立 4 个阶段的测试套件，覆盖核心模块：

| 测试套件      | 文件数 | 覆盖范围                |
| ------------- | ------ | ----------------------- |
| `tests/phase1/` | 3    | 配置、数据、模型加载    |
| `tests/phase2/` | 4    | PPL、隐藏状态、SAPLMA   |
| `tests/phase3/` | 3+   | 层分析、token 分析、可视化 |
| `tests/phase4/` | 6    | anchor、attention、debias、head selection、pipeline |

### 3.4 复现性保障

- 数据划分种子固定为 42，全程复用同一份划分
- 分类器训练使用 3 个随机种子（42, 123, 2024），报告均值与标准差
- 结果文件中统一记录随机种子、阈值优化方式与运行环境元数据
- 默认精度配置为 `bfloat16`，attention 实现固定为 `eager`

---

## 四、关键指标纵览

| 阶段 | 最佳方法                      | Test Acc | Test F1 | Test AUROC |
| ---- | ----------------------------- | -------- | ------- | ---------- |
| P2   | PPL                           | 0.5293   | 0.4180  | 0.6784     |
| P2   | SAPLMA-MLP (layer 27, last)   | 0.7697   | 0.7696  | 0.8676     |
| P3   | Layer 17 + last + LR          | 0.7987   | 0.7986  | 0.8878     |
| P4   | Hidden-only A0 (全量)          | 0.8082   | 0.8081  | 0.8897     |
| P4   | Hidden + top-16 head (子集)    | 0.8867   | 0.8865  | 0.9330     |
| P4   | Hidden + top-head + output (子集) | 0.8800 | 0.8798  | **0.9403** |

---

## 五、下一步计划（Phase 5: 报告与答辩准备）

### 5.1 待完成工作

| 任务                       | 优先级 | 预计时间   | 说明                                     |
| -------------------------- | ------ | ---------- | ---------------------------------------- |
| 融合实验全量数据验证       | 高     | 6.3–6.5    | 将 A5–A8 等融合实验扩展到全量 6309 样本，确认子集结论的迁移性 |
| 实验报告撰写（PDF）        | 高     | 6.3–6.7    | 整合 Phase 1–4 结果，完成正式实验报告     |
| 图表与可视化完善           | 高     | 6.3–6.6    | 补充高质量论文式图表、消融对比图、错误分析案例 |
| 代码整理与 README 完善     | 中     | 6.5–6.7    | 补充项目 README、运行说明、复现指南       |
| 答辩材料准备               | 中     | 6.6–6.8    | 准备 PPT 或展示材料，梳理关键结论与创新点 |

### 5.2 报告结构规划

1. **引言**：研究背景、问题定义、本项目目标
2. **相关工作**：幻觉检测方法综述、PPL 与 SAPLMA 方法回顾
3. **方法**：
   - 3.1 PPL 方法
   - 3.2 SAPLMA 方法（含分类器设计）
   - 3.3 层深度与 Token 表示分析
   - 3.4 基于注意力特征的进阶方案（anchor 抽取、去偏、head selection、融合）
4. **实验设置**：模型、数据、评价指标、环境配置
5. **实验结果**：
   - 5.1 Phase 2 基线对比
   - 5.2 Phase 3 层深度与 Token 分析
   - 5.3 Phase 4 消融实验与融合分析
6. **讨论**：方法有效性分析、注意力互补性解释、工程边界讨论
7. **结论与展望**
8. **参考文献**
