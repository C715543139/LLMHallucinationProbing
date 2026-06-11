# ACL 论文报告大纲（中文对照）

英文原文见 `docs/report_outline.md`。本文件用于写作讨论和分工对照，最终课程论文正文仍应统一使用英文。

建议题目：**Probing Hallucination Signals in Large Language Model Internal States**

中文含义：**探测大语言模型内部状态中的幻觉信号**

目标格式：ACL 模板；正文 8-9 页；正文统一英文。参考文献和附录不受页数限制。

## 核心论点

本项目研究：相比只依赖输出层面的不确定性指标，是否可以利用大语言模型内部状态来区分事实性回答与幻觉回答。论文应围绕三个结论展开：

1. 困惑度（perplexity, PPL）是一个有用但能力有限的幻觉检测基线。
2. 隐状态探测方法，尤其是在中后层表示上训练 SAPLMA 风格分类器，可以显著提升检测效果。
3. 注意力引导特征提供了可解释的互补信号，其中最强的子集实验相对 hidden-only baseline 同时提升了准确率并修正了一部分错误样本。

## 推荐页数分配

| 章节 | 建议长度 | 作用 |
| --- | ---: | --- |
| Abstract | 150-200 词 | 概括问题、方法、最强结果和核心发现。 |
| 1 Introduction | 0.8-1.0 页 | 引出幻觉检测问题，说明为什么要研究内部状态，并总结贡献。 |
| 2 Related Work | 0.7-0.9 页 | 简要定位幻觉检测、不确定性评分、表示探测和注意力分析。 |
| 3 Task and Experimental Setup | 1.0-1.2 页 | 定义任务、数据集、划分、模型、特征、指标和复现实验设置。 |
| 4 Methods | 1.5-1.8 页 | 说明 PPL、SAPLMA 隐状态探测、层/词元分析和注意力扩展方法。 |
| 5 Results and Analysis | 2.5-2.8 页 | 展示主要实验结果、图表、消融和案例分析。 |
| 6 Discussion | 0.7-0.9 页 | 解释为什么隐状态和注意力有效，并讨论失败模式。 |
| 7 Conclusion | 0.3-0.4 页 | 总结发现和实践经验。 |
| Limitations | 0.4-0.5 页 | ACL 风格的局限性章节，放在参考文献前。 |

预期总长度约为 8.0-9.0 页，不包括 references 和 appendices。

## 论文结构

### Abstract

建议写成一个紧凑段落，覆盖以下内容：

- 任务：面向 LLM 回答的二分类幻觉检测。
- 比较的信号：困惑度、隐状态、层选择、词元池化、注意力得分和注意力输出特征。
- 最强全量数据结果：SAPLMA MLP 达到 77.71% accuracy 和 0.8770 AUROC；验证集选择的第 17 层达到 80.03% accuracy 和 0.8876 AUROC。
- 最强注意力子集结果：hidden + top-16 head attention 达到 88.67% accuracy 和 0.9330 AUROC；hidden + top-head + output 达到 0.9403 AUROC。
- 结论：内部状态比输出困惑度更可靠，注意力特征能提供额外的可解释证据。

### 1 Introduction

建议叙述顺序：

1. 将 hallucination 定义为流畅生成与事实正确性之间的不一致。
2. 解释为什么输出层面的置信度或困惑度不足：错误陈述也可能很流畅、概率很高。
3. 引出假设：事实性会在模型内部表示中留下可测量的信号。
4. 明确项目范围：在带有 true/false 标签的数据上做二分类，使用 Qwen2-1.5B 的内部状态。
5. 列出贡献：
   - 将 PPL 与 SAPLMA 风格隐状态分类器进行比较。
   - 分析层深度和词元池化策略的影响。
   - 加入注意力引导特征并评估其增量价值。
   - 提供注意力修正错误的样本级可视化案例。

### 2 Related Work

这一节应短而集中，建议 2-3 段：

- 幻觉检测与事实性评估。
- 不确定性和似然检测方法，包括 PPL 类基线。
- 面向 truthfulness 或 factual consistency 的内部表示探测。
- 注意力分析与可解释性，同时谨慎说明：attention 不是完整解释，但可以作为有用的诊断特征。

不要写成宽泛综述，要给结果部分留出空间。

### 3 Task and Experimental Setup

#### 3.1 Task Definition

定义二分类任务：

- 输入：来自 true-false 数据集的 prompt/claim/response 样本。
- 标签：factual 或 hallucinated。
- 输出：二分类预测，以及用于计算 AUROC 的连续分数。

#### 3.2 Dataset

需要报告：

- 总样本数：6,309。
- 数据划分：5,047 train，631 validation，631 test。
- 如果最终统计中有标签平衡和领域组成，也应报告。
- 特征提取前使用的预处理步骤。

如果领域统计表较长，放到 Appendix A。

#### 3.3 Model and Feature Extraction

需要报告：

- 基座模型：Qwen2-1.5B。
- SAPLMA 风格探测使用的 hidden states。
- 分层提取和 token pooling 变体。
- Phase 4 使用的注意力特征。
- 如有必要，报告实际运行环境，例如 GPU、精度设置等。

#### 3.4 Metrics

使用 accuracy、macro-F1 和 AUROC。简要说明 AUROC 的意义：检测器输出连续分数，实际部署时阈值可能变化，因此排序能力也很重要。

### 4 Methods

#### 4.1 Perplexity Baseline

将 PPL 描述为输出似然基线。解释预期现象：幻觉文本平均上可能有更高 PPL，但似然本身并不直接衡量事实正确性。

#### 4.2 Hidden-State Probing

描述 SAPLMA 风格分类器：

- 从冻结的 LLM 中提取内部 hidden states。
- 训练轻量分类器，包括 logistic regression 和 MLP。
- 在保留的 validation/test 划分上评估。

需要明确：LLM 本身没有被微调。

#### 4.3 Layer and Token-Pooling Analysis

说明：

- 训练并评估每层 probe，用于定位对事实性敏感的层。
- 比较 last-token、mean-token 和 first-token 表示。
- 第 17 层必须写成 validation-selected layer 17，而不是根据 test performance 事后挑选的最佳层。

#### 4.4 Attention-Guided Extensions

紧凑总结 Phase 4 的变体：

- 来自选定 attention heads 的 attention score features。
- 如果使用了长度残差化或控制特征，也需要说明。
- 基于 validation AUROC 选择 top heads。
- Attention output features。
- A0-A9 融合变体。

如果篇幅允许，可以用一段文字加一个小表说明方法变体。实现细节和完整变体定义放到 Appendix C。

### 5 Results and Analysis

这一节应是论文主体。

#### 5.1 Main Baseline Comparison

主表：

| Method | Accuracy | Macro-F1 | AUROC |
| --- | ---: | ---: | ---: |
| PPL baseline | 52.93 | 41.80 | 67.84 |
| SAPLMA LR | 74.96 | 74.96 | 82.65 |
| SAPLMA MLP | 77.71 | 77.69 | 87.70 |

主要结论：hidden-state probes 在所有指标上都明显优于 PPL。

推荐图：PPL score distribution。用它说明 PPL 对 true/false 有一定区分趋势，但两个分布仍明显重叠。

#### 5.2 Layer-Wise Hidden-State Analysis

报告关键层结果：

| Layer | Accuracy | AUROC | Interpretation |
| --- | ---: | ---: | --- |
| 0 | 48.49 | 50.10 | 接近随机的词汇/输入层面信号。 |
| 13 | 82.25 | 89.51 | 强中层事实性信号。 |
| 15 | 82.88 | 91.03 | 当前报告中最高的 layer-wise AUROC。 |
| 17 | 80.03 | 88.76 | 验证集选择的最佳操作层。 |
| 20 | 82.88 | 90.40 | 强后中层表示。 |
| 27 | 74.96 | 82.65 | 最终层/默认 SAPLMA 设置。 |

推荐图：layer accuracy/AUROC curve。

重要措辞：不要说第 17 层是 test 上的绝对最佳层。应写成第 17 层是验证集选择的 operating point，并且在 test 上仍然保持较强表现。

#### 5.3 Token Pooling Analysis

报告：

| Pooling | Accuracy | AUROC |
| --- | ---: | ---: |
| Last token | 74.96 | 82.65 |
| Mean token | 70.52 | 76.36 |
| First token | 42.00 | 37.94 |

主要解释：

- Last-token 表示最强，因为它在评分前聚合了前文上下文。
- First-token 表示较弱，因为它几乎不包含回答相关的事实性信息。
- Mean pooling 可能稀释局部事实性线索。

#### 5.4 Attention Feature Ablation

对子集实验使用紧凑表格：

| Variant | Feature family | Accuracy | Macro-F1 | AUROC |
| --- | --- | ---: | ---: | ---: |
| A0s | Hidden only | 86.67 | 86.61 | 91.84 |
| A6 | Hidden + top-16 head attention | 88.67 | 88.65 | 93.30 |
| A8 | Hidden + top-head + attention output | 88.00 | 87.98 | 94.03 |
| A9 | Gated fusion | 简洁报告 | 简洁报告 | 简洁报告 |

主要解释：

- A6 是子集实验中 accuracy/F1 最强的配置。
- A8 达到最高 AUROC，说明即使阈值后 accuracy 略低于 A6，它的排序能力更强。
- 注意力应被表述为互补证据，而不是 hidden states 的替代方案。

推荐图：

- Method accuracy comparison。
- 如果篇幅允许，放 Method AUROC comparison。
- Layer-head AUROC heatmap 更适合放附录，除非正文需要突出可解释性叙事。

#### 5.5 Case-Level Error Correction and Visualization

使用 A6 case analysis：

| Category | Count |
| --- | ---: |
| Hidden correct, A6 correct | 129 |
| Hidden correct, A6 wrong | 1 |
| Hidden wrong, A6 correct | 4 |
| Hidden wrong, A6 wrong | 16 |

主要结论：相对 hidden-only subset baseline，A6 带来 +3 个样本的净修正。

推荐主图：

- A6 correction matrix 加一个代表性 improvement case。

其他 attention cases 放到 Appendix C 或 D。

### 6 Discussion

围绕三个解释组织：

1. **为什么 PPL 有局限。** 似然衡量的是流畅性和模型拟合程度，而不是事实正确性。
2. **为什么 hidden states 有效。** 中间层和后层表示比输出概率更直接地编码语义和事实一致性信号。
3. **为什么 attention 有帮助。** 选定的 heads 能捕捉回答词元与事实锚点之间的有用对齐，但 attention 仍然只是部分且有噪声的解释。

还应讨论实践经验：

- 基于 validation 的模型/层选择很重要。
- AUROC 和 accuracy 可能偏好不同方法变体。
- 子集注意力实验不能被过度表述为全量数据结论。

### 7 Conclusion

写一个短段落：

- 重申 hidden-state probing 相比 PPL 显著提升幻觉检测能力。
- 提到层分析和词元池化分析中的最强证据。
- 说明注意力引导特征在子集设置中提供了额外且可解释的增益。
- 以未来工作收尾：更大模型、更广数据集、更稳健的 attention/activation fusion。

### Limitations

ACL 论文通常在参考文献前包含 limitations section。建议包括：

- 只使用单一模型家族和单一模型规模。
- 二分类 true/false 设置可能简化了真实世界幻觉问题。
- 注意力实验基于子集，需要在更大样本上验证。
- 注意力可视化是诊断性证据，不是因果证明。
- 结果可能受数据集构造和特征提取策略影响。

## 图表安排

主文建议最多放 4-5 个图表块，以适配 8-9 页限制：

| 位置 | 类型 | 内容 | 建议位置 |
| --- | --- | --- | --- |
| Table 1 | Setup | 数据划分、模型、指标 | Section 3 |
| Table 2 | Main results | PPL vs SAPLMA LR/MLP | Section 5.1 |
| Figure 1 | Plot | PPL score distribution | Section 5.1 |
| Figure 2 | Plot | Layer-wise performance，可与 token pooling 合并 | Sections 5.2-5.3 |
| Table 3 | Ablation | A0s/A6/A8/A9 子集比较 | Section 5.4 |
| Figure 3 | Visualization | A6 correction matrix 和一个 improvement case | Section 5.5 |

如果篇幅紧张，把 Figure 1 或详细 attention visualization 移到附录，正文保留定量表格。

## 附录安排

Appendix A：数据集和预处理细节。

Appendix B：完整 Phase 2 和 Phase 3 结果表，包括所有测试层和 token pooling 变体。

Appendix C：完整 Phase 4 消融细节、attention head selection、layer-head heatmap 和 A0-A9 所有方法变体。

Appendix D：案例可视化，包括 true、false、hard 和 improvement examples。

Appendix E：复现说明、运行命令、环境和生成资产清单。

## 与课程要求的对应关系

| 课程要求 | 论文中对应位置 |
| --- | --- |
| 简单任务：使用 SAPLMA 并评估 layer-level hallucination signal | Sections 4.2, 4.3, 5.1, 5.2 |
| 分析 PPL 和 SAPLMA 是否有效，并解释原因 | Sections 5.1, 5.3, 6 |
| 高级任务：设计使用 attention/output 信息的改进方法 | Sections 4.4, 5.4, 5.5 |
| 提供实现和实验依据 | Sections 3, 4, 5 and Appendices B-E |
| 讨论局限性和未来工作 | Limitations, Section 7 |

## LaTeX 骨架

```latex
\begin{abstract}
...
\end{abstract}

\section{Introduction}
...

\section{Related Work}
...

\section{Task and Experimental Setup}
...

\section{Methods}
...

\section{Results and Analysis}
...

\section{Discussion}
...

\section{Conclusion}
...

\section*{Limitations}
...

\bibliography{custom}

\appendix
\section{Dataset and Preprocessing Details}
...
\section{Full Experimental Results}
...
\section{Attention Ablations and Case Visualizations}
...
\section{Reproducibility Notes}
...
```

## 写作注意事项

- 以 `scripts/show_results.py` 输出的精确指标作为结果来源。
- 所有 subset results 都必须明确标注为 subset results。
- 除非后续补充了全量注意力实验，否则不要把 A6/A8 表述为 full-dataset improvements。
- 提到第 17 层时，使用 "validation-selected layer 17"，不要笼统写成 "best layer"。
- 实现文件列表、命令日志和长表格放到附录。
- 使用简洁 ACL 风格表达：每个结论后面立刻跟对应表格、图片或指标证据。
