# Probing Hallucination Signals in Large Language Model Internal States 中文对照稿

> 对应英文主文：`report/main.tex`
>
> 用途：中文审阅、分工与答辩准备。最终课程论文正文仍以英文 ACL LaTeX 版本为准。

## 摘要

幻觉检测通常被视为输出层面的判断问题，但语言模型的内部激活中可能已经编码了与事实正确性相关的证据。本文在 True-False 陈述句数据集上，使用 Qwen2-1.5B 研究二分类事实性检测问题。我们比较了困惑度（PPL）基线与 SAPLMA 风格的隐状态探测方法，分析了不同层和不同 token 位置中的事实性信号，并进一步评估了注意力引导特征扩展。实验表明，PPL 只能提供较弱但非随机的证据，在测试集上达到 52.93% accuracy 和 0.6784 AUROC。隐状态探测明显更强：最终层 logistic probe 达到 74.96% accuracy 和 0.8265 AUROC，MLP probe 达到 77.71% accuracy 和 0.8770 AUROC。逐层分析进一步表明，事实性信号并不是最终层最强；验证集选择的 layer 17 在测试集上达到 80.03% accuracy 和 0.8876 AUROC。在 Phase 4 子集实验中，加入 top-head attention 特征将 hidden-only 子集基线从 86.67% accuracy 提升到 88.67%，而 hidden + attention output 融合达到 0.9403 AUROC。这些结果支持一个结论：相比输出似然，模型内部状态是更可靠的幻觉检测信号，而经过筛选的注意力特征能提供有价值的互补证据。

## 1. 引言

大语言模型能够生成流畅但事实错误的回答，这一问题通常被称为 hallucination。它是 LLM 在教育、科研辅助、法律、医疗和决策支持等知识密集型场景中应用的主要风险之一。一个实用的检测器应能判断模型输出或待评估陈述是否可能为假，但检测信号的选择非常关键。输出似然和置信度容易获得，却主要衡量一段文本在模型看来是否自然，而不是它是否真实。

本项目采用另一种假设：事实性信息在模型内部可能比在最终输出概率中更清晰。已有研究认为，LLM 的内部激活中包含可被线性或非线性读取的 truthfulness 信息。因此，幻觉检测可以被转化为一个 probing 问题：冻结 LLM，从陈述句中提取内部特征，再训练轻量分类器进行真假判断。

我们在 True-False statement dataset 上使用 Qwen2-1.5B 评估这一假设。任务是二分类：给定一条陈述，判断它是真的还是假的。项目围绕三个问题展开：

1. 简单的 PPL 基线能做到什么程度？
2. 隐状态是否暴露出更强的事实性信号？哪些层和 token 位置最有用？
3. 注意力特征能否提供 hidden states 之外的互补信息？

本文贡献包括：

- 实现并比较 PPL 和 SAPLMA 风格 hidden-state probes；
- 分析层深度和 token pooling，证明中后层与 last-token 表示最有效；
- 设计基于 statement anchors、attention scores、head selection、attention outputs 和 feature fusion 的注意力扩展方法；
- 通过消融实验和样本级可视化展示 attention features 如何修正 hidden-only 错误。

## 2. 相关工作

### 幻觉检测

自然语言生成中的 hallucination 指生成内容虽然流畅，但缺乏事实支持、不一致或错误。已有 benchmark 如 HaluEval 从多个任务角度评估模型幻觉行为。许多应用系统使用检索、外部事实核验或后处理验证来减少幻觉。这些方法很实用，但依赖外部知识源，不能直接回答“模型自身计算过程中是否已经包含事实性信号”。

### 似然与不确定性信号

最直接的内部基线是 likelihood。如果模型给真实陈述更低 loss、给虚假陈述更高 loss，那么序列概率或 PPL 就可以作为检测信号。但 likelihood 会受到词频、风格、流畅度和句长影响。一个错误但常见的表述可能有低 PPL，而一个真实但罕见的事实可能有高 PPL。因此，PPL 更适合作为基线，而不是完整的事实性指标。

### 内部表示探测

内部状态方法关注模型激活中是否包含 truthfulness 信息。与直接读取 next-token probability 不同，probing 方法判断 hidden states 中是否存在可预测真假标签的表示。本项目采用这一思路，在冻结的 Qwen2-1.5B activations 上训练 SAPLMA 风格分类器。

### 注意力作为诊断信号

Transformer attention 不是完整的因果解释，但它能展示 token 间的信息路由关系。本项目不把 attention 当作独立解释，而是把它作为特征族：从 last token 到 statement anchors 的 attention mass、基于验证集筛选的 heads，以及 attention-output activations 都被用于测试是否能补充 hidden states。

## 3. 任务与实验设置

### 3.1 任务定义

每个样本是一条陈述句 \(x\)，标签为 \(y \in \{0,1\}\)。其中 1 表示 true statement，0 表示 false statement。检测器输出二分类预测以及连续分数。连续分数用于计算 AUROC，二分类预测用于计算 accuracy 和 macro-F1。

这个设置不同于开放式回答生成。我们不是先让模型生成回答再检测，而是直接探测模型在处理给定陈述时，内部计算是否区分真假。

### 3.2 数据集

处理后的数据集包含 6,309 条有效样本，来自 animals、cities、companies、chemical elements、scientific facts、inventions 和一个 generated subset。前六个领域对应课程任务说明，generated subset 是当前工程配置中额外纳入的部分。数据固定按 8:1:1 划分，并按 domain + label 分层。

| Split | Examples | True | False | 用途 |
| --- | ---: | ---: | ---: | --- |
| Train | 5,047 | 2,539 | 2,508 | 训练 probe |
| Validation | 631 | 317 | 314 | 阈值和模型选择 |
| Test | 631 | 317 | 314 | 最终评估 |

### 3.3 模型与特征提取

所有实验使用冻结的 Qwen2-1.5B。我们通过 Hugging Face Transformers 提取 hidden states 和 attention tensors。LLM 本身不进行微调，只训练轻量下游分类器。Phase 4 的稳定运行路径是 eager attention + bfloat16。此前 float16 在代表性 hidden / attention 切片中产生 NaN，因此高级实验结果均使用 bfloat16 重跑结果。

hidden-state 实验使用 28 个 Transformer blocks 的层编号，不把 embedding output 计入层号。token pooling 包括 first token、valid tokens mean pooling 和 final valid token。attention 实验为每条 statement 抽取 subject、relation、tail 和 last-token anchors，然后计算 attention-score features 和 attention-output features。

### 3.4 指标

统一报告 accuracy、macro-F1 和 AUROC。Accuracy 衡量固定阈值下的正确率；macro-F1 更关注类别平衡；AUROC 衡量连续分数的排序质量，在实际检测阈值可能变化时尤其重要。

## 4. 方法

### 4.1 PPL 基线

PPL 基线使用整条陈述的 causal language modeling loss。给定 statement \(x\)，计算平均 loss \(\mathcal{L}(x)\)，并定义：

\[
\mathrm{PPL}(x) = \exp(\mathcal{L}(x)).
\]

PPL 越低，说明模型认为该陈述越可能。我们在验证集上选择阈值，再固定该阈值到测试集。AUROC 使用连续 PPL-derived score 计算。该方法简单且属于模型内部信号，但无法区分事实性和表面自然度。

### 4.2 隐状态探测

SAPLMA 风格检测器从冻结 LLM 中提取 hidden representation，然后训练分类器。对 statement \(x\)，模型在每个 Transformer block 输出 hidden states。给定层 \(l\) 和 pooling 规则 \(p\)，特征向量为：

\[
h(x) = \mathrm{pool}_p(H_l(x)).
\]

我们评估 logistic regression 和 MLP。特征在分类前标准化。分类器随机种子为 42、123 和 2024。最终层基线使用 layer 27 和 last-token pooling；后续分析会改变层和 pooling。

### 4.3 层与 token pooling 分析

层分析对每个 Transformer block 训练一个 logistic probe，并使用 last-token pooling。通过 validation split 选择操作层，再在 test split 上评估，避免用 test 结果选层。token-pooling 分析固定最终层，比较 last-token、mean-token 和 first-token 表示。

### 4.4 注意力引导扩展

高级方法测试 attention 是否提供 hidden states 之外的信息。由于数据是 statement 而不是 question-answer pair，所以不使用 answer-to-question attention。我们为每条 statement 抽取 subject tokens、relation tokens、tail tokens 和 final valid token，并统计 last token 到这些 anchors 的 attention。

评估的特征包括：raw attention-score features、去长度偏置后的 debiased attention features、基于 validation AUROC 选择的 top-head features、attention-output features，以及与 hidden features 的拼接或 gated fusion。主要变体为 A0/A0s 到 A9，其中 A0 是全量 hidden-only baseline，A0s 是子集 hidden-only baseline，A6/A8 是最强的 attention-enhanced variants。

## 5. 结果与分析

### 5.1 PPL 与隐状态探测比较

| Method | Accuracy | Macro-F1 | AUROC |
| --- | ---: | ---: | ---: |
| PPL baseline | 52.93 | 41.80 | 67.84 |
| SAPLMA LR | 74.96 | 74.96 | 82.65 |
| SAPLMA MLP | 77.71 | 77.69 | 87.70 |

PPL 只达到 52.93% accuracy 和 0.6784 AUROC。它在排序上高于随机，但作为二分类器较弱。SAPLMA logistic probe 明显更强，达到 74.96% accuracy 和 0.8265 AUROC。MLP probe 进一步达到 77.71% accuracy 和 0.8770 AUROC。

PPL 分布图显示，false statements 的平均 PPL 更高，但 true 和 false distributions 有明显重叠，因此单一阈值难以稳定区分两类。这支持第一个核心结论：输出 likelihood 不足以可靠检测幻觉，而 hidden-state features 提供更清晰的事实性信号。

### 5.2 逐层分析

| Layer | Accuracy | Macro-F1 | AUROC |
| --- | ---: | ---: | ---: |
| 0 | 48.49 | 48.49 | 50.10 |
| 13 | 82.25 | 82.24 | 89.51 |
| 15 | 82.88 | 82.88 | 91.03 |
| 17 | 80.03 | 80.01 | 88.76 |
| 20 | 82.88 | 82.88 | 90.40 |
| 27 | 74.96 | 74.96 | 82.65 |

Layer 0 接近随机，而中后层表现很强。Layer 17 是 validation-selected layer，在测试集上达到 80.03% accuracy 和 0.8876 AUROC。由于 layer 15 和 layer 20 在 test AUROC 上更高，正文中不能把 layer 17 称为 test 上绝对最佳层。更稳妥的结论是：事实性信号主要集中在 middle-to-late layer band，而不是最后层。

这个结果也说明，表示选择的重要性不低于分类器复杂度。Validation-selected layer 17 的 logistic probe 在 accuracy 上超过了最终层 MLP baseline。

### 5.3 Token Pooling

| Pooling | Accuracy | Macro-F1 | AUROC |
| --- | ---: | ---: | ---: |
| Last token | 74.96 | 74.96 | 82.65 |
| Mean token | 70.52 | 70.52 | 76.36 |
| First token | 42.00 | 41.97 | 37.94 |

Last-token pooling 明显最强。Mean pooling 保留部分信号但会稀释局部证据；first-token pooling 表现很差。这符合自回归模型的结构：最后一个有效 token 已经整合前文信息，因此更适合作为整句 factuality readout。

### 5.4 Attention Feature Ablation

Phase 4 在 600/150/150 train/validation/test 子集上评估 attention features。必须强调：这些 attention-fusion 改进是子集结果，不是全量数据结果。

| ID | Feature family | Dim. | Accuracy | Macro-F1 | AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| A0 | Hidden-only, full data | 1536 | 80.82 | 80.81 | 88.97 |
| A0s | Hidden-only, subset | 1536 | 86.67 | 86.61 | 91.84 |
| A1 | Attention score only, raw | 1536 | 76.67 | 76.54 | 82.89 |
| A2 | Attention score only, debiased | 1536 | 77.33 | 77.13 | 82.85 |
| A3 | Attention score, top-16 heads | 256 | 76.00 | 75.65 | 79.34 |
| A4 | Attention output only | 40 | 68.00 | 68.00 | 74.93 |
| A5 | Hidden + debiased attention score | 3072 | 87.33 | 87.29 | 93.02 |
| A6 | Hidden + top-16 head attention | 1792 | 88.67 | 88.65 | 93.30 |
| A7 | Hidden + attention output | 1576 | 84.67 | 84.63 | 92.54 |
| A8 | Hidden + top-head + output | 1832 | 88.00 | 87.98 | 94.03 |
| A9 | Gated fusion, tau=0.05 | - | 86.67 | 86.61 | 91.93 |

Attention score alone 已经高于随机，但它最有效的使用方式是补充 hidden states。A5 相比 A0s 有提升；A6 使用 hidden features 加 validation-selected top-16 attention heads，达到最高 accuracy 和 macro-F1：88.67% 和 88.65%。A8 达到最高 AUROC：0.9403，说明 attention output 更像是排序补充信号。

最强单 head 是 L16-H05，validation AUROC 为 0.6776。单个 head 不如 hidden-state probe，但经过筛选后与 hidden states 融合可以提供互补信息。

### 5.5 样本级错误修正

|  | A6 correct | A6 wrong |
| --- | ---: | ---: |
| Hidden correct | 129 | 1 |
| Hidden wrong | 4 | 16 |

在 150 条测试子集样本上，A6 纠正了 4 个 hidden-only 错误，同时引入 1 个退化样本，净修正为 +3。被修正的例子包括 `The Earth has one moon.`。在该样本中，hidden-only classifier 将真实类概率预测为 0.386，因此判断为 false；A6 将真实类概率提高到 0.879，因此判断为 true。

注意力可视化可以帮助解释该样本，但不能单独作为因果证明。有效性仍应以 A0-A9 消融和 A6 correction matrix 为准。

## 6. 讨论

### 为什么 PPL 较弱

PPL 衡量的是模型预测一个序列的难度。它受到句长、词频、常见表达和流畅度影响，而这些因素与事实正确性并不完全一致。PPL 分布显示 false statements 有更长的高 PPL 尾部，但两类分布重叠严重，所以 PPL 有一定 AUROC，却难以形成稳健二分类器。

### 为什么 hidden states 有效

Hidden states 不只是最终 token probability。它们编码了语义组合、实体-关系信息和模型内部一致性信号。Logistic probes 的强表现说明相当一部分 factuality signal 是线性可读的；MLP 的进一步提升说明非线性边界还能挖掘额外信息。但逐层分析也表明，选择合适表示至少和换更复杂分类器一样重要。

### 为什么中后层最强

逐层实验说明，事实性信号不是在底层或最终层最强。浅层主要编码局部词汇和句法信息；中后层已经整合足够上下文，因此更容易暴露事实性信号。最终层可能更专门服务于 next-token prediction，因此未必是最适合下游真假判别的表示。

### 为什么 attention 有帮助

Attention features 保留了 last token 与 statement anchors 之间的结构化对齐信息。Hidden state 是压缩后的整句向量，而 attention scores 显式展示模型把信息路由到哪些 token。去偏和 validation-based head selection 之后，这些特征能与 hidden states 互补。不过，attention-only 仍弱于 hidden-state probes，而且 A6/A8 目前是子集结论，需要谨慎表述。

### 工程经验

Phase 4 复跑说明数值稳定性也是方法的一部分。在 RTX 3090 环境中，eager attention + float16 会产生 NaN，而 eager attention + bfloat16 对 hidden states、attention scores 和 attention outputs 均稳定。因此，复现实验时必须记录 dtype 和 attention implementation，而不只是记录模型和数据集。

## 7. 结论

本项目表明，LLM 内部状态提供的幻觉检测信号明显强于输出 PPL。全量测试集上，SAPLMA 风格 hidden-state probes 在 accuracy、macro-F1 和 AUROC 上都大幅优于 PPL。层分析和 token 分析进一步说明，最佳读出方式不是简单使用最终层：事实性信号在中后层更强，last-token pooling 明显优于 first-token 和 mean pooling。注意力引导特征则提供了可解释的扩展方向。在受控子集上，top-head attention fusion 提升了 hidden-only baseline，并修正了少量但具体的 hidden-only 错误。后续应在全量数据、更大模型和更多模型族上验证 attention-fusion 结果，并设计更稳健的融合或路由机制。

## Limitations

当前实验只使用单一基座模型 Qwen2-1.5B 和一个处理后的 True-False statement dataset。任务是二分类且为 statement-level，比真实开放式生成场景中的 hallucination detection 更简单。Phase 4 attention-fusion 结果基于 600/150/150 子集，不是全量结论。Anchor extraction 采用简单规则，面对复杂句法时可能不稳定。最后，attention visualizations 只是诊断性证据，不能证明因果机制。

## 附录 A：领域统计

| Domain | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Animals | 806 | 102 | 100 |
| Cities | 1152 | 144 | 144 |
| Companies | 960 | 120 | 120 |
| Elements | 744 | 92 | 94 |
| Facts | 489 | 61 | 61 |
| Generated | 194 | 25 | 25 |
| Inventions | 702 | 87 | 87 |

## 附录 B：图表位置

TeX 项目使用的图表已经复制到 `report/figures/`：

- `ppl_score_distribution.png`
- `layer_accuracy_curve.png`
- `token_accuracy_comparison.png`
- `method_accuracy_comparison.png`
- `method_auroc_comparison.png`
- `layer_head_auroc_heatmap.png`
- `improvement_case_l16_h5_1.png`

## 附录 C：复现说明

主结果来源为 `scripts/show_results.py`。关键归档文件包括：

- `experiments/results/baseline/ppl_results.json`
- `experiments/results/baseline/saplma_logistic_results.json`
- `experiments/results/baseline/saplma_mlp_results_rerun_best.json`
- `experiments/results/analysis/layer_analysis_logistic_last.json`
- `experiments/results/analysis/token_analysis_logistic_last_layer.json`
- `experiments/results/phase4/phase4_main_results.csv`
- `experiments/results/phase4/a6_correction_matrix.json`
