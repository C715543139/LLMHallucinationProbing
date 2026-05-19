# Advanced_Optimization：Phase 4 进阶方案分析、实现思路与结果总结

> 更新时间：2026-05-19  
> 项目：利用大语言模型内部状态进行幻觉检测  
> 对应阶段：Phase 4  
> 当前稳定运行配置：Qwen2-1.5B + eager attention + bfloat16 + Linux + RTX 3090

---

## 1. 文档定位

本文件不再承担“从 Phase 3 重新开始开发”的指导作用，而是用于总结当前项目中已经真正采用并验证过的进阶优化方案。它关注三个问题：

1. 为什么在 Phase 3 之后还需要继续做 Phase 4；
2. 当前进阶方案的核心设计与实现思路是什么；
3. 这些进阶特征最终带来了什么收益，又暴露了哪些边界。

因此，本文档强调的是方案分析、实现路径与结果解释，而不是逐步开发清单。

---

## 2. 为什么需要 Phase 4

Phase 2 与 Phase 3 已经得到两个关键结论：

1. 隐藏状态显著优于 PPL，说明真实性信号确实存在于模型内部表示中；
2. `layer 17 + last token` 比“最后层 + last token”更适合作为真假检测读出位置。

但这还没有回答另一个更具体的问题：在 hidden state 之外，注意力模块是否也编码了能够帮助真假判别的结构化信号？如果答案是肯定的，那么更合适的 Phase 4 目标就不是另起炉灶重新设计一个新分类器，而是围绕 Phase 3 已确定的最佳 hidden baseline，检查 attention score 和 attention output 能否形成稳定互补。

基于这一判断，当前 Phase 4 的研究问题被收敛为三条：

1. attention score 单独使用时是否已经具备判别力；
2. attention 特征与 hidden baseline 融合后是否优于 hidden-only；
3. 哪种 attention 特征真正有效，是全部统计、筛选后的高价值 head，还是 attention output activation。

---

## 3. 方案提出与关键设计

### 3.1 固定 hidden baseline，而不是重新搜索表示

Phase 4 直接沿用 Phase 3 已验证的最佳配置：`layer 17 + last token + logistic regression`。这样做的目的是把变量控制在“是否加入 attention 特征”这一条主线上，避免把进阶方法的收益与表示选择收益混在一起。

### 3.2 针对陈述句设计 anchor，而不是套用问答模板

True-False Dataset 的输入是完整陈述句，不是问答对，因此不能直接照搬“answer token attends to question token”的分析框架。当前实现选择为每条陈述抽取四类 anchor：

- subject
- relation
- tail
- last token

随后围绕这些 anchor 计算注意力分数统计，以便显式刻画模型是否把最终判别位置的注意力集中在陈述的关键实体关系上。

### 3.3 先处理长度偏置，再谈注意力可用性

原始 attention score 很容易混入句长、token 数量和位置分布带来的伪信号。当前方案没有直接把原始统计送入分类器，而是先构造三套版本：

1. raw attention score
2. debiased attention score
3. validation-based top-head attention score

其中 debias 的目标是去除长度相关成分，让模型更多依赖实体关系对齐而不是表面长度差异。

### 3.4 Head selection 比“全部拼接”更关键

Phase 4 的一个重要判断是：注意力 head 并不是越多越好。当前实现不追求保留所有候选层与候选 head 的统计，而是基于验证集 AUROC 做筛选，保留最具判别力的 top-k heads。这样既降低维度，也减少噪声特征对融合分类器的干扰。

### 3.5 Attention output 作为补充信号，而非主特征

attention output 与 attention score 不同，它代表的是注意力模块写回 residual stream 之后的激活统计。当前实现并不假设 output 一定比 score 更强，而是将其作为另一类内部状态补充信号，重点观察它更偏向提升 Accuracy 还是 AUROC。

### 3.6 用系统消融而不是单次对比来判断有效性

当前主线不是只比较一个“融合版”与一个“baseline”，而是保留 A0-A9 的系统消融矩阵：

- A0 / A0s：hidden-only baseline
- A1-A4：attention-only 变体
- A5-A8：不同融合方式
- A9：gated fusion

这种设计使得最终结论不依赖单个偶然结果，而是依赖多组方法之间的相对排序与稳定趋势。

---

## 4. 实现落地思路

### 4.1 当前代码落点

Phase 4 当前实际落地到以下模块：

- `src/utils/feature_cache.py`
- `src/features/anchor_extraction.py`
- `src/features/attention_scores.py`
- `src/features/attention_outputs.py`
- `src/methods/phase4_attention.py`
- `src/analysis/phase4_analysis.py`
- `tests/phase4/`

运行入口统一为：

- `python -s main.py phase4`
- `python -s scripts/run/phase4.py`

结果文件统一输出到：

- `experiments/results/phase4/`

### 4.2 当前实际流水线

当前采用的流水线可以概括为：

1. 固定 Phase 3 的 hidden-only baseline，并缓存 hidden features；
2. 对陈述句抽取 subject / relation / tail / last token anchors；
3. 提取 layer / head 级 attention score 统计，并做长度偏置处理；
4. 基于验证集选择高价值 head；
5. 提取 attention output activation 统计特征；
6. 对 hidden、score、output 的不同组合进行 A0-A9 消融、图表分析与错误分析。

### 4.3 当前真正稳定的运行条件

Phase 4 的实现过程表明，稳定运行条件本身就是方案的一部分。当前 Linux + RTX 3090 环境下，`eager + float16` 仍会在 hidden 与 attention 张量中产生 NaN，因此现行有效路径必须固定为：

- attention implementation: eager
- dtype: bfloat16

这不是可选优化，而是当前结果能否成立的前置条件。

---

## 5. 核心结果与分析

### 5.1 数值稳定性结论

Phase 4 复测首先得到的是一个工程结论而不是性能结论：Linux 平台本身并不会自动修复 NaN，真正有效的是 dtype 切换。当前最小诊断如下：

| 设置 | 检查对象 | NaN 情况 | 结论 |
| ---- | -------- | -------- | ---- |
| eager + float16 | 单样本 hidden[18] | 10752 / 10752 | 不可用 |
| eager + float16 | 单样本 attn[17] | 588 / 588 | 不可用 |
| eager + bfloat16 | hidden / attention 前向 | 0 NaN | 稳定 |
| eager + bfloat16 | attention score / output 缓存 | 0 / 1,382,400；0 / 36,000 | 稳定 |

因此，后续所有 attention 结果都建立在 `eager + bfloat16` 之上。

### 5.2 Full hidden baseline 与历史基线对比

在稳定路径下，先复跑全量 hidden-only baseline：

| 方法 | 数据范围 | Test Acc | Test Macro-F1 | Test AUROC |
| ---- | -------- | -------- | ------------- | ---------- |
| Phase 3: layer 17 + last + logistic | 全量 | 0.7987 | 0.7986 | 0.8878 |
| Phase 4: hidden-only (A0) | 全量 | 0.8082 | 0.8081 | 0.8897 |

这表明当前稳定配置既解决了 NaN，也保住了 hidden baseline，并在当前复跑中略优于历史 Phase 3 结果。

### 5.3 进阶方法核心结果

当前保留的关键消融如下：

| 方法 | 特征 | Test Acc | Test F1 | Test AUROC | 结论 |
| ---- | ---- | -------- | ------- | ---------- | ---- |
| A0s | Hidden-only | 0.8667 | 0.8661 | 0.9184 | 子集基线 |
| A2 | Debiased attn-score only | 0.7733 | 0.7713 | 0.8285 | score 单独已具判别力 |
| A4 | Attn-output only | 0.6800 | 0.6800 | 0.7493 | output 单独弱于 score |
| A5 | Hidden + debiased attn-score | 0.8733 | 0.8729 | 0.9302 | 融合带来稳定提升 |
| A6 | Hidden + top-16 head attn | 0.8867 | 0.8865 | 0.9330 | 最佳 Accuracy / F1 |
| A7 | Hidden + attn-output | 0.8467 | 0.8463 | 0.9254 | AUROC 提升但 Acc 回落 |
| A8 | Hidden + top-head + output | 0.8800 | 0.8798 | 0.9403 | 最佳 AUROC |
| A9 | Gated Fusion | 0.8667 | 0.8661 | 0.9193 | 无净纠错收益 |

### 5.4 如何理解这些结果

从当前结果看，Phase 4 最重要的结论有四条：

1. **attention score 不是噪声**：A2 远高于随机水平，说明在 NaN 消失后，attention score 本身就包含真假判别信号；
2. **head selection 是有效的**：A6 优于 A5，说明筛选高价值 head 比简单拼接全部 attention 统计更有效；
3. **attention output 更偏排序信号**：A7 与 A8 的 AUROC 提升明显，但 Accuracy 不如 A6，说明 output 更适合作为补充排序信息；
4. **复杂路由不一定优于直接融合**：A9 没有带来净纠错，说明当前简单 gated fusion 还不足以替代直接特征拼接。

换言之，当前 Phase 4 的最优实践不是“尽可能多加特征”，而是“在稳定数值路径上，保留高价值 top-head score，再与 hidden baseline 做受控融合”。

---

## 6. 当前结论与后续优化方向

### 6.1 当前已经可以确认的结论

1. `eager + bfloat16` 是当前 Linux + RTX 3090 上可复现的稳定 Phase 4 主路径；
2. hidden state 仍然是主判别特征，但 attention score 可以提供稳定互补；
3. top-head attention 融合给出当前最佳 Accuracy / F1；
4. attention output 更适合作为 AUROC 增强信号，而不是独立决策器；
5. gated fusion 在当前设置下没有形成额外净收益。

### 6.2 后续更值得推进的方向

如果继续扩展当前进阶方案，优先级更高的方向包括：

1. 把 A5-A8 扩展到全量数据集，确认子集上的收益能否稳定迁移；
2. 在更多模型上复现“中后层 hidden + top-head score”这一组合；
3. 为 anchor 抽取引入更稳健的句法或语义对齐机制；
4. 探索比当前 gated fusion 更强的校准式路由与错误修正策略；
5. 将数值稳定性诊断、错误案例与图表进一步整理为正式论文写作材料。

---

## 7. 一句话总结

当前 Phase 4 的真正有效路线不是“Linux 自动修好 attention”，而是“在 `eager + bfloat16` 的稳定前提下，用 top-head attention score 为 Phase 3 的 hidden baseline 提供可验证的互补信息”。