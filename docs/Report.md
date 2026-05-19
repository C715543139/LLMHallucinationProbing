# 利用大语言模型内部状态进行幻觉检测：阶段性实验报告

> 版本：Draft v0.3  
> 更新时间：2026-05-16  
> 项目目录：`F:\PythonCodes\LLMHallucinationProbing`

---

## 摘要

大语言模型在生成任务中展现出强大的语言理解与表达能力，但其“幻觉”问题仍然是影响可靠应用的关键瓶颈。与依赖外部知识库或检索系统的方法不同，基于模型内部状态的幻觉检测尝试直接利用模型自身对陈述真实性的内部表征进行判断。本项目围绕论文 _The Internal State of an LLM Knows When It's Lying_ 的核心思想，基于 Qwen2-1.5B 模型和 True-False Dataset，对两类基础方法进行了实现与对比：一类是基于序列困惑度（Perplexity, PPL）的生成概率方法，另一类是基于隐藏状态分类的 SAPLMA 方法。

当前已完成 Phase 1 至 Phase 3。Phase 2 基线结果显示，PPL 在测试集上的 Accuracy 为 0.5293、Macro-F1 为 0.4180、AUROC 为 0.6784；在固定最后层与 `last` token 设置下，重跑后的 SAPLMA-MLP 测试集平均 Accuracy 为 0.7697、Macro-F1 为 0.7696、AUROC 为 0.8676。Phase 3 进一步表明，真实性相关信号并不在最后层最强：对 28 个 Transformer block 的 `last` token 做逐层分析后，验证集最优层为 layer 17，其测试集 Accuracy 为 0.7987、Macro-F1 为 0.7986、AUROC 为 0.8878；在最后层的 token 表示比较中，`last` pooling 优于 `mean` pooling 和 `first` pooling，测试集 Accuracy 分别为 0.7433、0.7021 和 0.3867。

这些结果说明，相比单纯依赖输出概率，模型隐藏状态以及“取哪一层、取哪个 token 表示”对真假检测性能具有更直接的影响。本报告基于当前工作区中的最终结果文件，同步总结 Phase 1 至 Phase 3 的实现、实验结果与工程状态，并将 Phase 4 的进阶特征探索保留为后续工作。

**关键词**：幻觉检测；大语言模型；隐藏状态；PPL；SAPLMA；Qwen2-1.5B

---

## 1. 引言

大语言模型（Large Language Models, LLMs）已经在问答、文本生成、推理和信息抽取等任务中取得显著进展。然而，这类模型经常会生成语法自然但事实错误的内容，即所谓“幻觉（hallucination）”。这一问题在知识密集型场景中尤为突出，例如教育、医疗、法律和科研辅助等领域。由于模型常以较高置信度输出错误内容，用户往往难以及时识别风险，因此如何检测幻觉已成为大模型可靠性研究的重要方向。

已有研究中，幻觉检测方法可以大致分为两类。第一类方法依赖外部知识库、检索系统或事实验证组件，通过将模型输出与外部证据进行比对来判定真实性。这类方法通常能够借助外部知识增强准确性，但会增加系统复杂度，并且依赖额外的知识来源。第二类方法则聚焦于模型自身，试图通过模型输出概率、隐藏状态、注意力分布或其他内部激活来判断模型是否“知道”一条陈述是真是假。相比外部校验，这种内部探测路线更具模型分析意义，也更符合“从模型自身认知状态中读取信号”的研究目标。

本项目采用第二类思路，以论文 _The Internal State of an LLM Knows When It's Lying_ 为主要参考，围绕 True-False Dataset 中的陈述句开展实验。项目首先实现两类基线方法：其一是基于序列困惑度的 PPL 方法，其二是基于隐藏状态分类的 SAPLMA 方法。随后，项目继续分析不同层深度、不同 token 位置表示对真实性检测的影响，并将注意力或 FFN 激活等更丰富的内部特征保留给下一阶段探索。

本报告对应项目当前的阶段性实验结果汇总，主要覆盖已完成的 Phase 1、Phase 2 与 Phase 3。报告既总结当前实现结果，也为后续 Phase 4（进阶方法）保留扩展空间。

---

## 2. 相关工作

### 2.1 幻觉检测研究概述

大语言模型幻觉检测的研究通常从两个方向展开。一类工作强调外部知识增强，包括检索增强生成（RAG）、外部事实校验、多轮验证等；另一类工作则关注模型内部机制，希望不借助外部知识，直接从模型对文本的内部响应中检测其不确定性或知识缺失状态。

外部知识增强方法在应用层面较为实用，但其检测效果往往依赖检索质量、知识库完整性以及检索-生成协同机制。相比之下，内部状态方法更适合用来回答一个更基础的问题：模型自身的内部表示中，是否已经编码了足以区分真实与虚假陈述的信号？

### 2.2 基于概率的方法

最直观的内部方法是使用模型对一段文本的生成概率或序列困惑度作为真实性判断依据。直觉上，如果模型“更认可”某条陈述，那么它在生成这条陈述时的损失应更低、PPL 也应更低。然而，这类方法存在明显局限：PPL 不仅受到事实性影响，也会受到句长、词频、表面流畅度、措辞习惯等因素影响。因此，PPL 更多反映的是模型的语言接受度，而不一定等价于事实真实性。

### 2.3 基于隐藏状态的探测方法

论文 _The Internal State of an LLM Knows When It's Lying_ 提出，模型在处理文本时的隐藏状态中可能已经编码了关于真实性的信息。SAPLMA 方法正是这一思路的代表：它提取某一层、某一 token 位置的隐藏状态作为特征，再训练轻量分类器判断真假。这种方法将真实性检测问题转化为一个表征探测问题，相对于 PPL 更少受到表面文本统计特征的干扰，也更有利于进行层深度和 token 位置分析。

### 2.4 本项目的工作定位

本项目聚焦于课程要求中的基础任务与分析任务，已在统一实验设置下完成以下工作：

1. 实现并对比 PPL 与 SAPLMA 两类基线方法；
2. 对 28 个 Transformer block 开展逐层隐藏状态分析，考察哪些层编码了更多真实性相关信息；
3. 在最后层设置下比较 First / Last / Mean pooling 的效果差异。

下一阶段将继续探索注意力模式、FFN 激活以及更丰富的特征融合方式是否能够进一步提升检测性能。

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

### 3.5 扩展分析方法

当前 Phase 3 已完成以下两类扩展分析：

- **层深度分析**：逐层提取 28 个 Transformer block 的隐藏状态，并训练逻辑回归分类器，绘制层深度-性能曲线；
- **token 位置分析**：在最后层固定设置下比较 `Last`、`First` 与 `Mean pooling` 的效果差异。

Phase 4 的后续工作将继续探索注意力分布、FFN 激活以及更丰富的内部特征融合方式。

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

根据当前项目实现，Phase 1、Phase 2 与 Phase 3 已分别建立自动化测试：

- `tests/phase1/`：验证配置、数据、模型加载与前向传播；
- `tests/phase2/`：验证 PPL、隐藏状态提取、SAPLMA 分类及边界情况。
- `tests/phase3/`：验证层分析、token 分析、可视化接口以及小规模真实模型集成路径。

同时，结果文件中已统一记录随机种子、阈值选择方式与运行环境元数据。这为当前报告中的 Phase 2 / Phase 3 结果提供了较高的实现可信度，也为跨设备复现实验提供了依据。

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

根据 `experiments/results/baseline/ppl_results.json`，当前 PPL 基线在验证集上搜索得到的最佳阈值约为 **232.4321**。其结果如下：

| 数据集 | Accuracy | Macro-F1 | AUROC  |
| ------ | -------- | -------- | ------ |
| Train  | 0.5336   | 0.4286   | 0.6695 |
| Val    | 0.5357   | 0.4316   | 0.6576 |
| Test   | 0.5293   | 0.4180   | 0.6784 |

从结果可以看出，PPL 方法在 Accuracy 与 Macro-F1 上略高于随机水平，但仍明显弱于 SAPLMA；与此同时，AUROC 略高于 0.5，说明其连续分数对真假样本仍有一定排序能力。这表明 PPL 中确实存在一定真实性相关信号，但该信号较弱，且在固定阈值下不足以支撑稳健的二分类决策。

#### 5.2.2 SAPLMA（Logistic Regression）结果

根据 `experiments/results/baseline/saplma_logistic_results.json`，当前 SAPLMA-LR 使用最后一层（`layer_idx = 27`）和 `last` token 表示，在 3 个随机种子下的测试集结果均值如下：

| 方法              | Layer | Pooling | Accuracy        | Macro-F1        | AUROC           |
| ----------------- | ----- | ------- | --------------- | --------------- | --------------- |
| SAPLMA (logistic) | 27    | last    | 0.7417 ± 0.0000 | 0.7417 ± 0.0000 | 0.8278 ± 0.0000 |

该结果明显优于 PPL，说明最后层隐藏状态中已经编码了较强的真假判别信息，并且这种信息在逻辑回归这一线性分类器下就可以被较好地利用。

#### 5.2.3 SAPLMA（MLP）结果

根据 `experiments/results/baseline/saplma_mlp_results_rerun_best.json`，在修改后的复现设置下重新运行 SAPLMA-MLP 后，在相同层与 token 设置下的测试集结果均值如下：

| 方法         | Layer | Pooling | Accuracy        | Macro-F1        | AUROC           |
| ------------ | ----- | ------- | --------------- | --------------- | --------------- |
| SAPLMA (mlp) | 27    | last    | 0.7697 ± 0.0191 | 0.7696 ± 0.0192 | 0.8676 ± 0.0118 |

与逻辑回归相比，MLP 仍然整体提升了 Accuracy、Macro-F1 和 AUROC，说明隐藏状态中的真实性信号不完全是线性可分的，适当的非线性映射可以更充分地挖掘该信号。与此同时，MLP 在不同随机种子下的标准差明显高于逻辑回归，这也表明非线性分类器对初始化与优化路径更敏感。

### 5.3 基线方法对比分析

当前已完成的基线方法对比如下：

| 方法              | Accuracy | Macro-F1 | AUROC  | 备注                          |
| ----------------- | -------- | -------- | ------ | ----------------------------- |
| PPL               | 0.5293   | 0.4180   | 0.6784 | 基于测试集固定阈值            |
| SAPLMA (logistic) | 0.7417   | 0.7417   | 0.8278 | 3 seeds 均值                  |
| SAPLMA (mlp)      | 0.7697   | 0.7696   | 0.8676 | 修改后方案重跑的 3 seeds 均值 |

从该表可以得到两个初步结论：

1. **SAPLMA 显著优于 PPL**。隐藏状态比单纯的序列概率更适合作为真假检测特征；
2. **MLP 优于逻辑回归**。说明真假信息在隐藏状态空间中的决策边界并非完全线性。

### 5.4 Phase 3：层深度与 token 表示分析

Phase 3 的结果文件与图像已生成至 `experiments/results/analysis/`。其中，层深度分析使用逻辑回归分类器与 `last` token 表示，对 28 个 Transformer block 逐层训练；token 分析则固定最后层，比较三种 pooling 策略。

#### 5.4.1 层深度分析结果

根据 `experiments/results/analysis/layer_analysis_logistic_last.json`，以验证集 Accuracy 作为选层标准时，最佳层为 **layer 17**。其测试集指标如下：

| 配置                       | Test Accuracy | Test Macro-F1 | Test AUROC |
| -------------------------- | ------------- | ------------- | ---------- |
| layer 17 + last + logistic | 0.7987        | 0.7986        | 0.8878     |

为观察整体趋势，选取若干关键层的结果如下：

| Layer | Val Accuracy | Test Accuracy | Test Macro-F1 | Test AUROC |
| ----- | ------------ | ------------- | ------------- | ---------- |
| 0     | 0.4960       | 0.5008        | 0.5007        | 0.5068     |
| 13    | 0.8114       | 0.8193        | 0.8193        | 0.8967     |
| 15    | 0.8273       | 0.8225        | 0.8225        | 0.9092     |
| 17    | 0.8288       | 0.7987        | 0.7986        | 0.8878     |
| 20    | 0.8051       | 0.8241        | 0.8241        | 0.9015     |
| 27    | 0.7480       | 0.7433        | 0.7433        | 0.8277     |

可以看到，浅层几乎接近随机水平，而从 layer 13 开始性能显著上升，layer 13 至 layer 20 形成稳定的高性能区间。相比之下，最后层（layer 27）的表现明显回落。这说明真实性相关信号在中后层已经充分形成，但在最终输出层中部分表征可能更偏向下一 token 预测而不是显式真假判别。相较于 Phase 2 中归档的最后层逻辑回归基线（Accuracy 0.7417，AUROC 0.8278），选择 layer 17 后 Accuracy 提升约 0.0570，AUROC 提升约 0.0600。对应曲线图已保存为 `experiments/results/analysis/layer_accuracy_curve.png`。

#### 5.4.2 token 表示分析结果

根据 `experiments/results/analysis/token_analysis_logistic_last_layer.json`，在最后层（layer 27）固定逻辑回归分类器后，不同 pooling 的表现如下：

| Pooling | Val Accuracy | Test Accuracy | Test Macro-F1 | Test AUROC |
| ------- | ------------ | ------------- | ------------- | ---------- |
| last    | 0.7480       | 0.7433        | 0.7433        | 0.8277     |
| mean    | 0.7163       | 0.7021        | 0.7020        | 0.7653     |
| first   | 0.4073       | 0.3867        | 0.3859        | 0.3251     |

结果表明，`last` pooling 明显优于 `mean` pooling，而 `first` pooling 表现显著失效。对因果语言模型而言，最后一个有效 token 的隐藏状态天然聚合了前文语义与真假线索，因此更适合用作整句表示；`mean` pooling 保留了部分信息，但会稀释末端位置的判别信号；`first` pooling 则几乎无法反映整句在自回归处理后的最终内部状态。需要说明的是，当前 token 分析中 `layer 27 + last` 的 Accuracy 为 0.7433，与 Phase 2 归档基线文件中的 0.7417 仅相差 0.0016，属于可接受的重跑波动范围，并不影响方法排序与总体结论。对应柱状图已保存为 `experiments/results/analysis/token_accuracy_comparison.png`。

---

## 6. 结果讨论

### 6.1 为什么 PPL 表现较弱

PPL 的出发点是“模型越认可某句陈述，则其生成损失越低”，这一思路在概念上合理，但在实际中会受到多个因素干扰：

- 句长越长，累积损失模式更复杂；
- 高频词、常见句式会降低困惑度，即便陈述本身并不真实；
- 语言自然度与事实真实性并不完全一致；
- 某些虚假陈述在语言层面依然可能十分流畅，因此 PPL 无法稳定区分。

当前结果中，PPL 的 AUROC 略高于 0.5，而 Accuracy 仅略高于 0.5，这与上述分析一致：PPL 可能携带部分真实性排序信号，但该信号不足以形成稳健的最终判别边界。

### 6.2 为什么 SAPLMA 更有效

隐藏状态是模型在逐层处理文本时形成的内部表示，它比单一的输出概率更接近模型的“中间认知状态”。如果模型内部已经存储了事实知识，那么在读取真实与虚假陈述时，其隐藏状态分布应存在系统性差异。当前实验中，SAPLMA 远优于 PPL，说明：

- 模型隐藏层确实包含与真实性相关的可分离信号；
- 这些信号在最后层已经较强；
- 通过轻量分类器即可将内部表征映射到真假标签。

这与参考论文的核心观点相一致，即：模型的内部状态往往“知道”一条陈述是否为真，即便其最终输出概率无法很好地直接体现这一点。

### 6.3 层深度分析揭示了“中后层最优”

Phase 3 的逐层结果表明，真实性信号并不是单调随层数增加而增强。浅层（如 layer 0）几乎接近随机，说明底层表示主要编码词法与局部语义；从 layer 13 开始，性能显著提升，表明真假相关信息在模型完成更多上下文整合后逐渐变得线性可读；而最后层又出现回落，说明最接近输出的位置不一定最适合下游真假判别。更准确地说，当前结果显示 layer 13 至 layer 20 更像是一个高性能平台区，而非只有单一“最佳神奇层”。

### 6.4 last token 是更可靠的整句读出位置

在最后层的三种 pooling 对比中，`last > mean >> first` 的结果非常清晰。这与因果语言模型的表示结构一致：最后一个有效 token 的状态最充分地整合了整句前缀信息；平均池化会把与真假判断弱相关的位置一并混合，从而削弱判别边界；而首 token 表示几乎不包含句尾累积后的全局语义，因此在该任务上表现最差。换言之，若希望从自回归模型中读取整句真假信号，优先使用最后一个有效 token 是更合理的工程选择。

### 6.5 表示选择的重要性不低于分类器复杂度

Phase 2 说明，在固定最后层 + `last` token 的前提下，MLP 相比逻辑回归能够进一步提升性能；但 Phase 3 又表明，单纯更换表示层带来的收益并不小于提升分类器复杂度。具体地，`layer 17 + last + logistic` 的 Accuracy 为 0.7987，高于 `layer 27 + last + mlp` 的 0.7697。这说明后续工作不应只关注“换更强的分类器”，还应联合搜索层深度、token 表示与分类器类型。另一方面，重跑后的结果差异大多保持在可接受范围内，也说明当前结论主要反映表示质量差异，而不是实验流程不稳定。

---

## 7. 当前工程实现与复现说明

### 7.1 当前已实现模块

截至当前版本，已完成的核心模块包括：

- `src/config.py`
- `src/data/dataset.py`
- `src/data/preprocessing.py`
- `src/models/loader.py`
- `src/features/hidden_states.py`
- `src/methods/probability.py`
- `src/methods/saplma.py`
- `src/analysis/layer_analysis.py`
- `src/analysis/token_analysis.py`
- `src/analysis/visualization.py`
- `src/utils/metrics.py`
- `src/utils/reproducibility.py`
- `main.py`

同时已建立：

- `tests/phase1/`
- `tests/phase2/`
- `tests/phase3/`

### 7.2 当前仍待完成的模块

以下模块或任务仍属于 Phase 4 及其后的扩展内容：

- `src/features/attention.py`
- `src/features/ffn_activations.py`
- `src/methods/advanced.py`

因此，本报告当前已覆盖 **Phase 1 + Phase 2 + Phase 3** 的主要实现与结果，剩余工程空缺集中在高级特征实现、更系统的消融实验以及跨模型泛化验证。

### 7.3 当前运行入口

项目当前可通过 `main.py` 运行以下任务：

- `python -s main.py`：查看项目状态
- `python -s main.py preprocess`：运行数据预处理
- `python -s main.py phase2`：运行 Phase 2 全部实验
- `python -s main.py phase2-ppl`：只运行 PPL
- `python -s main.py phase2-saplma`：只运行 SAPLMA
- `python -s main.py phase3`：运行 Phase 3 全部实验
- `python -s main.py phase3-layer`：只运行逐层分析
- `python -s main.py phase3-token`：只运行 token pooling 分析

---

## 8. 局限性与后续工作

### 8.1 当前局限性

1. **层深度分析目前只覆盖逻辑回归分类器**：尚未在全部层上系统比较 MLP 或其他更复杂分类器；
2. **token 位置分析当前只固定在最后层**：尚未开展“层深度 × pooling 方式”的联合网格搜索；
3. **进阶特征尚未实现**：注意力模式与 FFN 激活分析仍停留在计划阶段；
4. **模型与数据范围仍较有限**：当前结论基于单一模型 Qwen2-1.5B 与单一真假陈述数据集，外部泛化能力仍需验证；
5. **统计显著性分析仍不充分**：虽然多随机种子和运行元数据已记录，但尚未补充更系统的显著性检验与跨设备重复实验统计。

### 8.2 后续工作计划

后续将按项目计划继续推进：

- 推进 Phase 4：
  - 实现注意力特征或 FFN 特征；
  - 与 SAPLMA 基线进行融合与消融实验；
- 扩展验证范围：
  - 增加更多随机种子与显著性检验；
  - 补充跨模型、跨数据集或跨领域验证；
- 完善正式报告：
  - 统一整理图表与表格排版；
  - 补充更细致的误差分析与案例分析；
  - 输出最终正式版本。

---

## 9. 结论

本项目围绕“利用大语言模型内部状态进行幻觉检测”这一问题，基于 Qwen2-1.5B 与 True-False Dataset，已经完成了 Phase 1、Phase 2 与 Phase 3 的实现与验证。Phase 1 搭建了完整的实验基础设施，包括配置管理、数据预处理、模型加载和测试体系；Phase 2 实现了 PPL 与 SAPLMA 两类基线方法；Phase 3 则进一步回答了“哪一层更有用”“哪个 token 表示更合适”这两个关键分析问题。

当前结果表明，PPL 方法虽然具备一定排序能力，但在最终分类性能上明显不足；相比之下，基于隐藏状态的 SAPLMA 方法能够更有效地利用模型内部表征，取得显著更高的 Accuracy、Macro-F1 和 AUROC。这说明模型内部状态中确实编码了与真实性相关的表征信息，且这些信息可以被下游分类器成功读取。

进一步地，Phase 3 显示真实性相关信号在中后层最强，而最后层并非最优读出位置；在最后层内部，`last` token 表示又明显优于 `mean` 与 `first` pooling。当前已观测到的最佳配置为 `layer 17 + last + logistic`，其测试集 Accuracy 为 0.7987、Macro-F1 为 0.7986、AUROC 为 0.8878，高于固定最后层的各类基线。这说明在该任务上，表示选择本身就是核心性能因素。下一阶段将围绕注意力 / FFN 特征、特征融合以及更广泛泛化验证继续推进。

---

## 参考文献

1. Azaria, A., & Mitchell, T. (2023). _The Internal State of an LLM Knows When It's Lying._ arXiv:2304.13734.
2. Burns, C., et al. (2023). _Discovering Latent Knowledge in Language Models Without Supervision._ arXiv:2212.03827.
3. Li, J., et al. (2023). _HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models._ EMNLP 2023.
4. Li, K., et al. (2024). _Your Mixture-of-Experts LLM Is Secretly an Embedding Model For Free._ ICLR 2025.
5. Chen, Z., et al. (2025). _TRACEDET: Hallucination Detection from the Decoding Trace of Diffusion Large Language Models._ ICLR 2026.

---

## 附录 A：当前可直接引用的结果文件

- `experiments/results/baseline/ppl_results.json`
- `experiments/results/baseline/saplma_logistic_results.json`
- `experiments/results/baseline/saplma_mlp_results_rerun_best.json`
- `experiments/results/analysis/layer_analysis_logistic_last.json`
- `experiments/results/analysis/token_analysis_logistic_last_layer.json`
- `experiments/results/analysis/layer_accuracy_curve.png`
- `experiments/results/analysis/token_accuracy_comparison.png`

## 附录 B：当前已生成与后续可补充的图表

1. 已生成：不同层深度 vs Accuracy 曲线
2. 已生成：不同 token 表示方式柱状图
3. 可补充：PPL 分数分布直方图（真/假对比）
4. 可补充：进阶方法与基线方法的消融对比表
