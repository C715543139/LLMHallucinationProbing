# 利用大语言模型内部状态进行幻觉检测：实验报告初稿

> 版本：Draft v0.1  
> 更新时间：2026-05-15  
> 项目目录：`D:\LLMHallucinationProbing`

---

## 摘要

大语言模型在生成任务中展现出强大的语言理解与表达能力，但其“幻觉”问题仍然是影响可靠应用的关键瓶颈。与依赖外部知识库或检索系统的方法不同，基于模型内部状态的幻觉检测尝试直接利用模型自身对陈述真实性的内部表征进行判断。本项目围绕论文 *The Internal State of an LLM Knows When It's Lying* 的核心思想，基于 Qwen2-1.5B 模型和 True-False Dataset，对两类基础方法进行了实现与对比：一类是基于序列困惑度（Perplexity, PPL）的生成概率方法，另一类是基于隐藏状态分类的 SAPLMA 方法。

当前已完成 Phase 1 与 Phase 2。Phase 1 主要完成实验环境搭建、数据预处理、模型加载与测试体系构建；Phase 2 主要实现 PPL 基线与 SAPLMA 基线，并完成真实代码验证。现有结果表明，基于隐藏状态的 SAPLMA 方法明显优于基于 PPL 的方法：在当前实验设置下，PPL 在测试集上的 Accuracy 为 0.4992、Macro-F1 为 0.3358、AUROC 为 0.6784，而 SAPLMA-MLP 在测试集上的平均 Accuracy 达到 0.7670、Macro-F1 为 0.7666、AUROC 为 0.8683。该结果初步说明，相比于受句长和表面流畅性影响较大的概率分数，模型隐藏状态中编码了更适合真假判别的内部信号。

本报告初稿重点整理项目背景、方法设计、实验设置、Phase 1/2 已完成结果以及下一阶段的分析实验计划，为后续完善 Phase 3 与 Phase 4 的结果分析和图表撰写打下基础。

**关键词**：幻觉检测；大语言模型；隐藏状态；PPL；SAPLMA；Qwen2-1.5B

---

## 1. 引言

大语言模型（Large Language Models, LLMs）已经在问答、文本生成、推理和信息抽取等任务中取得显著进展。然而，这类模型经常会生成语法自然但事实错误的内容，即所谓“幻觉（hallucination）”。这一问题在知识密集型场景中尤为突出，例如教育、医疗、法律和科研辅助等领域。由于模型常以较高置信度输出错误内容，用户往往难以及时识别风险，因此如何检测幻觉已成为大模型可靠性研究的重要方向。

已有研究中，幻觉检测方法可以大致分为两类。第一类方法依赖外部知识库、检索系统或事实验证组件，通过将模型输出与外部证据进行比对来判定真实性。这类方法通常能够借助外部知识增强准确性，但会增加系统复杂度，并且依赖额外的知识来源。第二类方法则聚焦于模型自身，试图通过模型输出概率、隐藏状态、注意力分布或其他内部激活来判断模型是否“知道”一条陈述是真是假。相比外部校验，这种内部探测路线更具模型分析意义，也更符合“从模型自身认知状态中读取信号”的研究目标。

本项目采用第二类思路，以论文 *The Internal State of an LLM Knows When It's Lying* 为主要参考，围绕 True-False Dataset 中的陈述句开展实验。项目首先实现两类基线方法：其一是基于序列困惑度的 PPL 方法，其二是基于隐藏状态分类的 SAPLMA 方法。随后，项目计划进一步分析不同层深度、不同 token 位置表示对真实性检测的影响，并探索注意力或 FFN 激活等更丰富的内部特征。

本报告当前对应项目的“阶段性实验报告初稿”，主要反映已完成的 Phase 1 与 Phase 2。报告既总结当前实现结果，也为后续 Phase 3（层分析与 token 位置分析）和 Phase 4（进阶方法）预留结构和分析空间。

---

## 2. 相关工作

### 2.1 幻觉检测研究概述

大语言模型幻觉检测的研究通常从两个方向展开。一类工作强调外部知识增强，包括检索增强生成（RAG）、外部事实校验、多轮验证等；另一类工作则关注模型内部机制，希望不借助外部知识，直接从模型对文本的内部响应中检测其不确定性或知识缺失状态。

外部知识增强方法在应用层面较为实用，但其检测效果往往依赖检索质量、知识库完整性以及检索-生成协同机制。相比之下，内部状态方法更适合用来回答一个更基础的问题：模型自身的内部表示中，是否已经编码了足以区分真实与虚假陈述的信号？

### 2.2 基于概率的方法

最直观的内部方法是使用模型对一段文本的生成概率或序列困惑度作为真实性判断依据。直觉上，如果模型“更认可”某条陈述，那么它在生成这条陈述时的损失应更低、PPL 也应更低。然而，这类方法存在明显局限：PPL 不仅受到事实性影响，也会受到句长、词频、表面流畅度、措辞习惯等因素影响。因此，PPL 更多反映的是模型的语言接受度，而不一定等价于事实真实性。

### 2.3 基于隐藏状态的探测方法

论文 *The Internal State of an LLM Knows When It's Lying* 提出，模型在处理文本时的隐藏状态中可能已经编码了关于真实性的信息。SAPLMA 方法正是这一思路的代表：它提取某一层、某一 token 位置的隐藏状态作为特征，再训练轻量分类器判断真假。这种方法将真实性检测问题转化为一个表征探测问题，相对于 PPL 更少受到表面文本统计特征的干扰，也更有利于进行层深度和 token 位置分析。

### 2.4 本项目的工作定位

本项目聚焦于课程要求中的基础任务与分析任务，首先实现 PPL 与 SAPLMA 两类基线方法，并在统一实验设置下对比它们的表现。在此基础上，后续计划继续分析：

1. 哪些 Transformer 层编码了更多真实性相关信息；
2. 不同 token 位置（First / Last / Mean pooling）对检测效果有何影响；
3. 注意力或 FFN 激活是否能够进一步提升检测性能。

---

## 3. 方法

### 3.1 问题定义

给定一条陈述句 \(x\)，任务是预测其真假标签 \(y \in \{0,1\}\)，其中 1 表示真实陈述，0 表示虚假陈述。本项目使用 True-False Dataset 作为主要数据来源，输入以陈述句形式给出，而不是开放式问答输出。因此本任务可以视为一个标准的文本二分类问题，但分类特征并不是来自传统的词袋或句向量，而是来自语言模型的内部状态或语言建模分数。

### 3.2 基于生成概率的 PPL 方法

PPL 方法将一条陈述句送入因果语言模型，直接计算该句子的长度归一化损失，并以

\[
\text{PPL}(x) = \exp(\mathcal{L}(x))
\]

作为连续判别分数。PPL 越低，表示模型越“认可”该陈述。为了将连续分数转为真假分类，本项目在验证集上搜索最优阈值，再将该阈值固定到测试集进行最终评估。

当前实现中，PPL 方法的核心流程包括：

1. 对单条语句或批量语句进行 tokenize；
2. 调用语言模型前向传播并使用 `labels=input_ids` 获取平均 loss；
3. 将 loss 指数化得到 PPL；
4. 在验证集上搜索最优判别阈值；
5. 在测试集上汇报 Accuracy、Macro-F1 和 AUROC。

对应实现文件：`src/methods/probability.py`。

### 3.3 基于隐藏状态分类的 SAPLMA 方法

SAPLMA 方法将真假判别视为对模型内部表征的探测。对输入陈述句进行前向传播后，提取指定 Transformer block 的隐藏状态表示，作为下游分类器的输入特征。根据项目计划与当前实现，层编号统一按 Transformer block 输出编号，不将 embedding output 计入层号。

当前 Phase 2 的基础实现采用“最后一个 token 的隐藏状态”作为整句表示，即：

1. 对输入语句执行前向传播并开启 `output_hidden_states=True`；
2. 显式剥离 `hidden_states[0]` 对应的 embedding output；
3. 从指定层中取最后一个有效 token 的表示；
4. 将隐藏状态送入轻量分类器（逻辑回归或 MLP）；
5. 在测试集上汇报多随机种子下的均值与标准差。

对应实现文件：

- 隐藏状态提取：`src/features/hidden_states.py`
- SAPLMA 分类：`src/methods/saplma.py`

### 3.4 分类器设计

当前 SAPLMA 实现包含两类下游分类器：

- **Logistic Regression**：作为线性基线，适合衡量隐藏状态中特征是否已具有较好的线性可分性；
- **MLPClassifier**：引入非线性映射能力，用于探索隐藏状态中更复杂的真假判别边界。

在训练前，特征先通过 `StandardScaler` 进行标准化。当前配置中，随机种子固定为 `(42, 123, 2024)`，以减少单次训练带来的偶然波动，并报告测试集结果的均值与标准差。

### 3.5 后续分析方法规划

根据项目计划，后续报告正式版将继续补充以下内容：

- **层深度分析**：逐层提取隐藏状态并训练分类器，绘制层深度-性能曲线；
- **token 位置分析**：比较 Last / First / Mean pooling 等表示方式；
- **进阶特征**：探索注意力分布和 FFN 激活模式是否包含额外的真实性信号。

---

## 4. 实验设置

### 4.1 模型

本项目当前统一使用 **Qwen2-1.5B** 作为实验模型。选择该模型的主要原因如下：

1. 在当前 Windows + RTX 4060 Laptop 8GB 显存环境下可稳定运行；
2. 参数规模较适中，既能完成完整前向传播和隐藏状态提取，也便于多次实验复用；
3. 与课程要求中“若较大模型本地跑不动，可选 Qwen2-1.5B”这一条件一致。

当前模型缓存路径为：`models_cache/Qwen2-1.5B/`。

### 4.2 数据集

实验数据采用 True-False Dataset 的陈述句形式。当前工作区中的 `data/raw/` 已以 CSV 方式组织，包含如下原始文件：

- `animals_true_false.csv`
- `cities_true_false.csv`
- `companies_true_false.csv`
- `elements_true_false.csv`
- `facts_true_false.csv`
- `generated_true_false.csv`
- `inventions_true_false.csv`

其中前 6 个主要领域与课程说明中的数据集范围一致，`generated_true_false.csv` 为当前工程配置下同时纳入的附加原始文件。

### 4.3 数据划分与预处理

项目采用 8:1:1 的训练集、验证集、测试集划分。当前实现中，`src/data/preprocessing.py` 支持按“领域 + 标签”进行分层划分，以尽量保持各领域和真假标签在各划分中的代表性。预处理后的数据存储在：

- `data/processed/train.pt`
- `data/processed/val.pt`
- `data/processed/test.pt`

当前配置中，数据划分随机种子为 `42`，分类训练随机种子为 `(42, 123, 2024)`。

### 4.4 评价指标

本项目统一使用以下三个指标：

- **Accuracy**：总体分类正确率；
- **Macro-F1**：对真/假两类分别计算 F1 后取平均，更能反映类间平衡表现；
- **AUROC**：基于连续分数衡量模型排序能力，适合评估不同方法的判别质量。

这些指标由 `src/utils/metrics.py` 统一提供。

### 4.5 实验与测试环境

根据当前项目实现，Phase 1 和 Phase 2 已分别建立自动化测试：

- `tests/phase1/`：验证配置、数据、模型加载与前向传播；
- `tests/phase2/`：验证 PPL、隐藏状态提取、SAPLMA 分类及边界情况。

其中，`tests/phase2/` 已在真实代码上通过，共 **31 个测试通过**。这为当前报告中的 Phase 2 结果提供了较高的实现可信度。

---

## 5. 已完成实验结果

### 5.1 Phase 1：环境搭建与数据准备结果

Phase 1 已完成以下目标：

- 项目目录结构、配置模块、数据模块和模型模块已搭建完成；
- 原始数据与预处理数据已准备就绪；
- Qwen2-1.5B 模型已可被本地加载；
- 单条陈述句可完成一次完整前向传播，并输出 `hidden_states`；
- Phase 1 测试套件已建立并通过。

这一阶段的完成意味着后续所有内部状态实验所需的最小运行闭环已经形成。

### 5.2 Phase 2：PPL 与 SAPLMA 基线结果

#### 5.2.1 PPL 方法结果

根据 `experiments/results/baseline/ppl_results.json`，当前 PPL 基线在验证集上搜索得到的最佳阈值约为 **3.0028**。其结果如下：

| 数据集 | Accuracy | Macro-F1 | AUROC |
| ------ | -------- | -------- | ----- |
| Train  | 0.4991   | 0.3368   | 0.6695 |
| Val    | 0.4992   | 0.3358   | 0.6576 |
| Test   | 0.4992   | 0.3358   | 0.6784 |

从结果可以看出，PPL 方法在 Accuracy 与 Macro-F1 上接近随机水平，但 AUROC 略高于 0.5，说明其连续分数对真假样本仍有一定排序能力。这表明 PPL 中确实存在一定真实性相关信号，但该信号较弱，且在固定阈值下不足以支撑稳定的二分类决策。

#### 5.2.2 SAPLMA（Logistic Regression）结果

根据 `experiments/results/baseline/saplma_logistic_results.json`，当前 SAPLMA-LR 使用最后一层（`layer_idx = 27`）和 `last` token 表示，在 3 个随机种子下的测试集结果均值如下：

| 方法 | Layer | Pooling | Accuracy | Macro-F1 | AUROC |
| ---- | ----- | ------- | -------- | -------- | ----- |
| SAPLMA (logistic) | 27 | last | 0.7417 ± 0.0000 | 0.7417 ± 0.0000 | 0.8278 ± 0.0000 |

该结果明显优于 PPL，说明最后层隐藏状态中已经编码了较强的真假判别信息，并且这种信息在逻辑回归这一线性分类器下就可以被较好地利用。

#### 5.2.3 SAPLMA（MLP）结果

根据 `experiments/results/baseline/saplma_mlp_results.json`，当前 SAPLMA-MLP 在相同层与 token 设置下的测试集结果均值如下：

| 方法 | Layer | Pooling | Accuracy | Macro-F1 | AUROC |
| ---- | ----- | ------- | -------- | -------- | ----- |
| SAPLMA (mlp) | 27 | last | 0.7670 ± 0.0042 | 0.7666 ± 0.0049 | 0.8683 ± 0.0048 |

与逻辑回归相比，MLP 进一步提升了 Accuracy、Macro-F1 和 AUROC，说明隐藏状态中的真实性信号不完全是线性可分的，适当的非线性映射可以更充分地挖掘该信号。

### 5.3 基线方法对比分析

当前已完成的基线方法对比如下：

| 方法 | Accuracy | Macro-F1 | AUROC | 备注 |
| ---- | -------- | -------- | ----- | ---- |
| PPL | 0.4992 | 0.3358 | 0.6784 | 基于测试集固定阈值 |
| SAPLMA (logistic) | 0.7417 | 0.7417 | 0.8278 | 3 seeds 均值 |
| SAPLMA (mlp) | 0.7670 | 0.7666 | 0.8683 | 3 seeds 均值 |

从该表可以得到两个初步结论：

1. **SAPLMA 显著优于 PPL**。隐藏状态比单纯的序列概率更适合作为真假检测特征；
2. **MLP 优于逻辑回归**。说明真假信息在隐藏状态空间中的决策边界并非完全线性。

---

## 6. 结果讨论

### 6.1 为什么 PPL 表现较弱

PPL 的出发点是“模型越认可某句陈述，则其生成损失越低”，这一思路在概念上合理，但在实际中会受到多个因素干扰：

- 句长越长，累积损失模式更复杂；
- 高频词、常见句式会降低困惑度，即便陈述本身并不真实；
- 语言自然度与事实真实性并不完全一致；
- 某些虚假陈述在语言层面依然可能十分流畅，因此 PPL 无法稳定区分。

当前结果中，PPL 的 AUROC 略高于 0.5，而 Accuracy 接近 0.5，这与上述分析一致：PPL 可能携带部分真实性排序信号，但该信号不足以形成稳健的最终判别边界。

### 6.2 为什么 SAPLMA 更有效

隐藏状态是模型在逐层处理文本时形成的内部表示，它比单一的输出概率更接近模型的“中间认知状态”。如果模型内部已经存储了事实知识，那么在读取真实与虚假陈述时，其隐藏状态分布应存在系统性差异。当前实验中，SAPLMA 远优于 PPL，说明：

- 模型隐藏层确实包含与真实性相关的可分离信号；
- 这些信号在最后层已经较强；
- 通过轻量分类器即可将内部表征映射到真假标签。

这与参考论文的核心观点相一致，即：模型的内部状态往往“知道”一条陈述是否为真，即便其最终输出概率无法很好地直接体现这一点。

### 6.3 Logistic 与 MLP 的差异

逻辑回归已经能达到较好的结果，说明隐藏状态中存在明显的线性可分性；而 MLP 进一步提升表现，则说明这种可分性并非完全线性，仍存在一定的高阶交互模式。后续在层分析中，可以继续观察：

- 中间层是否更线性可分；
- 更深层是否需要更强的非线性分类器；
- 不同 token 表示方式下，分类边界的复杂度是否会发生变化。

---

## 7. 当前工程实现与复现说明

### 7.1 当前已实现模块

截至当前版本，已完成的核心模块包括：

- `src/config.py`
- `src/data/dataset.py`
- `src/data/preprocessing.py`
- `src/models/loader.py`
- `src/methods/probability.py`
- `src/features/hidden_states.py`
- `src/methods/saplma.py`
- `src/utils/metrics.py`
- `main.py`

同时已建立：

- `tests/phase1/`
- `tests/phase2/`
- `docs/milestone/M1.md`
- `docs/milestone/M2.md`

### 7.2 当前尚未完成的模块

以下模块已在项目计划中定义，但当前尚未落地：

- `src/analysis/layer_analysis.py`
- `src/analysis/token_analysis.py`
- `src/analysis/visualization.py`
- `src/features/attention.py`
- `src/features/ffn_activations.py`
- `src/methods/advanced.py`

因此，本报告当前主要是**Phase 1 + Phase 2 的阶段性报告初稿**，后续需要在 Phase 3 和 Phase 4 完成后继续补充实验图表、分析结果和结论。

### 7.3 当前运行入口

项目当前可通过 `main.py` 运行以下任务：

- `python -s main.py`：查看项目状态
- `python -s main.py preprocess`：运行数据预处理
- `python -s main.py phase2`：运行 Phase 2 全部实验
- `python -s main.py phase2-ppl`：只运行 PPL
- `python -s main.py phase2-saplma`：只运行 SAPLMA

---

## 8. 局限性与后续工作

### 8.1 当前局限性

1. **层分析尚未完成**：当前只验证了最后层 + last token 的 SAPLMA 结果，尚未系统比较不同层深度的特征质量；
2. **token 位置分析尚未完成**：目前还未比较 First / Last / Mean pooling 的效果差异；
3. **进阶特征尚未实现**：注意力模式与 FFN 激活分析仍停留在计划阶段；
4. **结果图表尚未系统生成**：目前已有结果 JSON，但尚未形成最终报告所需的柱状图、曲线图和热力图；
5. **报告仍为初稿**：当前文本已搭建完整结构，但部分章节仍需在后续实验完成后补入更细致的数据分析和图表解释。

### 8.2 后续工作计划

后续将按项目计划继续推进：

- 完成 Phase 3：
  - 逐层隐藏状态提取与层深度分析；
  - 不同 token 表示方式的对比分析；
  - PPL 与 SAPLMA 的机制层面对比；
- 完成 Phase 4：
  - 注意力特征或 FFN 特征的实现；
  - 与 SAPLMA 基线进行融合与消融实验；
- 完成 Phase 5：
  - 统一整理图表；
  - 完善实验分析、讨论与结论；
  - 输出正式报告版本。

---

## 9. 结论

本项目围绕“利用大语言模型内部状态进行幻觉检测”这一问题，基于 Qwen2-1.5B 与 True-False Dataset，完成了 Phase 1 和 Phase 2 的实现与验证。Phase 1 搭建了完整的实验基础设施，包括配置管理、数据预处理、模型加载和测试体系；Phase 2 实现了 PPL 与 SAPLMA 两类基线方法，并获得了初步实验结果。

当前结果表明，PPL 方法虽然具备一定排序能力，但在最终分类性能上明显不足；相比之下，基于隐藏状态的 SAPLMA 方法能够更有效地利用模型内部表征，取得显著更高的 Accuracy、Macro-F1 和 AUROC。这说明模型内部状态中确实编码了与真实性相关的表征信息，且这些信息可以被下游分类器成功读取。

从当前实验结果看，SAPLMA-MLP 是已实现方法中的最佳基线。这一结果为后续 Phase 3 的层深度分析与 token 位置分析，以及 Phase 4 的注意力 / FFN 特征探索，提供了坚实基础。

---

## 参考文献

1. Azaria, A., & Mitchell, T. (2023). *The Internal State of an LLM Knows When It's Lying.* arXiv:2304.13734.
2. Burns, C., et al. (2023). *Discovering Latent Knowledge in Language Models Without Supervision.* arXiv:2212.03827.
3. Li, J., et al. (2023). *HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models.* EMNLP 2023.
4. Li, K., et al. (2024). *Your Mixture-of-Experts LLM Is Secretly an Embedding Model For Free.* ICLR 2025.
5. Chen, Z., et al. (2025). *TRACEDET: Hallucination Detection from the Decoding Trace of Diffusion Large Language Models.* ICLR 2026.

---

## 附录 A：当前可直接引用的结果文件

- `experiments/results/baseline/ppl_results.json`
- `experiments/results/baseline/saplma_logistic_results.json`
- `experiments/results/baseline/saplma_mlp_results.json`
- `docs/milestone/M1.md`
- `docs/milestone/M2.md`

## 附录 B：后续正式版建议补充的图表

1. PPL vs SAPLMA 对比表（正式版排版）
2. 不同层深度 vs Accuracy 曲线
3. 不同 token 表示方式柱状图
4. PPL 分数分布直方图（真/假对比）
5. 进阶方法与基线方法的消融对比表

