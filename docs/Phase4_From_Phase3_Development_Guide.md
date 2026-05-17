# Phase 4 开发指导文件：从 Phase 3 完成状态重新开始

> 项目：利用大语言模型内部状态进行幻觉检测  
> 起点：Phase 1、Phase 2、Phase 3 已完成；Phase 4 尚未开始  
> 本文件目标：在不复用任何旧 Phase 4 实现和旧实验结果的前提下，从 Phase 3 的最佳隐藏状态基线出发，重新设计、实现、评估 Phase 4 进阶方法。

---

## 0. 当前状态假设

本指导文件假设当前代码已经回退到 **Phase 3 刚完成** 的状态。

也就是说，项目中应当已经存在：

```text
src/
├── config.py
├── data/
│   ├── dataset.py
│   └── preprocessing.py
├── models/
│   └── loader.py
├── features/
│   └── hidden_states.py
├── methods/
│   ├── probability.py
│   └── saplma.py
├── analysis/
│   ├── layer_analysis.py
│   ├── token_analysis.py
│   └── visualization.py
└── utils/
    ├── metrics.py
    └── reproducibility.py
```

以及：

```text
experiments/results/
├── baseline/
│   ├── ppl_results.json
│   ├── saplma_logistic_results.json
│   └── saplma_mlp_results_rerun_best.json
└── analysis/
    ├── layer_analysis_logistic_last.json
    ├── token_analysis_logistic_last_layer.json
    ├── layer_accuracy_curve.png
    └── token_accuracy_comparison.png
```

本文件 **不要求** 项目中存在以下任何文件：

```text
src/features/attention.py
src/methods/advanced.py
tests/phase4/
experiments/results/advanced/
```

如果这些文件不存在，属于正常状态。

---

## 1. Phase 4 的重新定位

### 1.1 Phase 3 已经回答的问题

Phase 3 已经完成两个关键分析：

1. **层深度分析**：中后层比最后层更适合进行真实性检测；
2. **token 位置分析**：对因果语言模型而言，`last token` 是比 `first` 和 `mean pooling` 更可靠的整句表示。

因此，Phase 4 不应该重新做一个脱离 Phase 3 的新分类器，而应该以 Phase 3 的最佳隐藏状态读出方式作为主基线。

推荐固定基线：

```text
Hidden baseline:
- feature: layer 17 + last token hidden state
- classifier: Logistic Regression
- metrics: Accuracy, Macro-F1, AUROC
```

### 1.2 Phase 4 要解决的新问题

Phase 4 的目标是回答：

```text
除了 hidden state 之外，注意力模块的注意力分数或内部激活中，是否存在与真假判别互补的信号？
```

进一步拆分为四个问题：

1. 单独使用注意力分数是否能够检测真假陈述？
2. 注意力特征与 Phase 3 hidden state 融合后是否优于 hidden-only？
3. 哪些层、哪些 attention head 对真假检测更有用？
4. 注意力模块输出激活是否比纯 attention score 更适合作为增强特征？

### 1.3 方法名称

建议将 Phase 4 方法命名为：

```text
Attention-Guided SAPLMA
```

或：

```text
Attention-Augmented Internal State Probe
```

报告中可以写为：

> 在 Phase 3 确定的最佳隐藏状态读出层基础上，本文进一步引入注意力分数与注意力模块输出激活，构建 Attention-Guided SAPLMA，用于探索注意力模块是否能为隐藏状态幻觉检测器提供互补信号。

---

## 2. Phase 4 总体技术路线

Phase 4 分为 7 个子阶段：

```text
P4.0  固定 Phase 3 hidden-only 基线
P4.1  实现 statement anchor 抽取
P4.2  提取 layer/head 级 attention score 特征
P4.3  做 attention 特征去长度偏置
P4.4  做 validation-based layer/head selection
P4.5  提取 attention output activation
P4.6  训练融合分类器并做消融实验
P4.7  可视化、错误分析、报告整理
```

核心思想：

```text
hidden state 是主判别特征；
attention score 提供可解释结构信号；
attention output activation 提供注意力模块真正写回 residual stream 的内部激活信号；
最终通过系统消融判断 attention 是否提供稳定增益。
```

---

## 3. 新增目录结构

建议新增如下文件，不覆盖 Phase 1-3 代码。

```text
src/
├── features/
│   ├── anchor_extraction.py
│   ├── attention_scores.py
│   └── attention_outputs.py
├── methods/
│   └── phase4_attention.py
├── analysis/
│   └── phase4_analysis.py
└── utils/
    └── feature_cache.py
```

新增测试：

```text
tests/
└── phase4/
    ├── __init__.py
    ├── conftest.py
    ├── test_anchor_extraction.py
    ├── test_attention_scores.py
    ├── test_attention_debias.py
    ├── test_head_selection.py
    ├── test_attention_outputs.py
    └── test_phase4_pipeline.py
```

新增结果目录：

```text
experiments/results/phase4/
├── cache/
│   ├── hidden_layer17_last_train.npz
│   ├── hidden_layer17_last_val.npz
│   ├── hidden_layer17_last_test.npz
│   ├── attention_scores_train.npz
│   ├── attention_scores_val.npz
│   ├── attention_scores_test.npz
│   ├── attention_outputs_train.npz
│   ├── attention_outputs_val.npz
│   └── attention_outputs_test.npz
├── hidden_baseline.json
├── attention_score_feature_summary.csv
├── attention_head_selection.json
├── attention_output_feature_summary.csv
├── phase4_ablation_results.json
├── phase4_main_results.csv
├── phase4_error_analysis.csv
├── figures/
│   ├── layer_head_auroc_heatmap.png
│   ├── feature_delta_boxplot.png
│   ├── method_accuracy_comparison.png
│   ├── method_auroc_comparison.png
│   └── correction_matrix.png
└── phase4_summary.md
```

---

## 4. 新增配置文件

建议新增：

```text
experiments/configs/phase4_attention.yaml
```

内容建议如下：

```yaml
model:
  name: Qwen2-1.5B
  dtype: float16
  attn_implementation: eager

data:
  train_path: data/processed/train.pt
  val_path: data/processed/val.pt
  test_path: data/processed/test.pt

phase3_baseline:
  hidden_layer: 17
  pooling: last
  classifier: logistic

attention_scores:
  candidate_layers: [13, 14, 15, 16, 17, 18, 19, 20]
  anchor_version: rule_v1
  features:
    - last_to_subject_mass
    - last_to_relation_mass
    - last_to_tail_mass
    - last_to_anchor_mass
    - subject_to_relation_mass
    - relation_to_tail_mass
    - attention_entropy_last
    - max_attention_last
    - top3_attention_mass_last
    - attention_sink_mass
  keep_length_features_for_analysis: true
  remove_length_features_for_training: true
  residualize_by_length: true

head_selection:
  enabled: true
  top_k_heads_candidates: [8, 16, 32]
  metric: val_auroc

attention_outputs:
  enabled: true
  layers: [13, 14, 15, 16, 17, 18, 19, 20]
  pooling: last
  use_stats: true
  use_vector: false

classifier:
  seeds: [42, 123, 2024]
  types:
    - logistic
    - mlp

gated_fusion:
  enabled: true
  thresholds: [0.05, 0.10, 0.15, 0.20, 0.25]

metrics:
  - accuracy
  - macro_f1
  - auroc
```

注意：

```text
attn_implementation: eager
```

很重要。部分 Transformers 版本下，Qwen2 使用 SDPA / Flash Attention 时可能无法直接返回 attention weights。为确保 `output_attentions=True` 可用，建议在 Phase 4 模型加载时使用 eager attention。

---

## 5. P4.0：固定 Phase 3 hidden-only 基线

### 5.1 目标

在进入注意力特征前，先重新确认 Phase 3 的 hidden-only baseline。

基线配置：

```text
layer = 17
pooling = last
classifier = Logistic Regression
seeds = [42, 123, 2024]
```

### 5.2 新增文件：`src/utils/feature_cache.py`

实现统一特征缓存接口：

```python
from pathlib import Path
from typing import Any
import json
import numpy as np


def save_npz_cache(
    path: str | Path,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        labels=labels,
        feature_names=np.array(feature_names or [], dtype=object),
        metadata=json.dumps(metadata or {}, ensure_ascii=False),
    )


def load_npz_cache(path: str | Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {
        "features": data["features"],
        "labels": data["labels"],
        "feature_names": list(data["feature_names"]),
        "metadata": json.loads(str(data["metadata"])),
    }


def cache_exists(path: str | Path) -> bool:
    return Path(path).exists()
```

### 5.3 新增函数：缓存 hidden feature

在 `src/methods/phase4_attention.py` 中实现：

```python
def cache_phase3_hidden_features(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    test_dataset,
    output_dir,
    layer_idx: int = 17,
    pooling: str = "last",
):
    """
    使用 Phase 3 最优配置提取 train/val/test 的 hidden feature。
    输出：hidden_layer17_last_train.npz 等文件。
    """
```

如果 Phase 3 已经保存过 layer 17 hidden feature，也可以直接复用，但建议 Phase 4 独立缓存一次，保证实验可复现。

### 5.4 hidden-only 训练函数

```python
def run_hidden_baseline(
    train_hidden,
    val_hidden,
    test_hidden,
    train_labels,
    val_labels,
    test_labels,
    classifier_type: str = "logistic",
    seeds: tuple[int, ...] = (42, 123, 2024),
) -> dict:
    """
    训练 hidden-only 分类器，返回 mean/std metrics。
    """
```

应复用已有 `src/methods/saplma.py` 中的分类器训练逻辑，避免重复实现。

### 5.5 输出

```text
experiments/results/phase4/hidden_baseline.json
```

建议格式：

```json
{
  "method": "hidden_layer17_last_logistic",
  "feature": {
    "layer": 17,
    "pooling": "last",
    "dim": 1536
  },
  "classifier": "logistic",
  "seeds": [42, 123, 2024],
  "val": {
    "accuracy_mean": 0.0,
    "accuracy_std": 0.0,
    "macro_f1_mean": 0.0,
    "macro_f1_std": 0.0,
    "auroc_mean": 0.0,
    "auroc_std": 0.0
  },
  "test": {
    "accuracy_mean": 0.0,
    "accuracy_std": 0.0,
    "macro_f1_mean": 0.0,
    "macro_f1_std": 0.0,
    "auroc_mean": 0.0,
    "auroc_std": 0.0
  }
}
```

### 5.6 验收标准

该结果应接近 Phase 3 中的 layer 17 + last + logistic 结果。小幅波动可以接受，但如果明显下降，需要检查：

1. 数据划分是否变化；
2. layer index 是否仍然按 Transformer block 输出编号；
3. last token 是否排除了 padding；
4. StandardScaler 是否只在 train 上 fit；
5. 测试集是否没有参与模型选择。

---

## 6. P4.1：实现 statement anchor 抽取

### 6.1 为什么需要 anchor

True-False Dataset 的输入是陈述句，不天然包含 query-answer 结构。因此注意力分析不能直接照搬“answer token attends to query token”的设计。

对于陈述句，更合理的做法是抽取：

```text
subject tokens: 陈述主体
relation tokens: 关系 / 谓词 / 系词
object or tail tokens: 陈述尾部实体或属性
last token: 因果语言模型整句读出位置
```

然后分析：

```text
last token 对 subject / relation / tail 的注意力质量；
subject 与 relation 之间的注意力；
relation 与 tail 之间的注意力；
注意力是否集中于关键实体关系结构。
```

### 6.2 新增文件：`src/features/anchor_extraction.py`

建议定义数据结构：

```python
from dataclasses import dataclass


@dataclass
class AnchorSpans:
    statement: str
    subject_char_span: tuple[int, int] | None
    relation_char_span: tuple[int, int] | None
    tail_char_span: tuple[int, int] | None
    subject_token_indices: list[int]
    relation_token_indices: list[int]
    tail_token_indices: list[int]
    last_token_index: int
    rule_name: str
    valid: bool
    fallback_reason: str | None = None
```

### 6.3 字符级 span 抽取

实现：

```python
def extract_char_spans(statement: str) -> dict:
    """
    返回 subject/relation/tail 的字符级 span。
    优先使用简单稳定规则，不依赖复杂 parser。
    """
```

推荐规则顺序：

#### 规则 1：系表结构

匹配：

```text
X is Y.
X are Y.
X was Y.
X were Y.
X has Y.
X have Y.
```

例如：

```text
Paris is the capital of France.
```

抽取：

```text
subject = Paris
relation = is
 tail = the capital of France
```

#### 规则 2：常见关系短语

维护一个 relation keyword 列表：

```python
RELATION_PATTERNS = [
    "is located in",
    "are located in",
    "was founded by",
    "were founded by",
    "was invented by",
    "were invented by",
    "was discovered by",
    "were discovered by",
    "is made of",
    "are made of",
    "is part of",
    "are part of",
    "belongs to",
    "contain",
    "contains",
    "consists of",
    "is known for",
]
```

如果匹配到 relation phrase，则：

```text
subject = relation phrase 前面的文本
relation = relation phrase
 tail = relation phrase 后面的文本
```

#### 规则 3：fallback

如果没有匹配到明显关系词：

```text
subject = 前 1 到 3 个词
relation = 中间第一个动词或中间区域
 tail = 最后 1 到 3 个词
```

fallback 必须记录：

```text
rule_name = fallback
fallback_reason = no_relation_pattern_matched
```

### 6.4 token 对齐

实现：

```python
def align_char_span_to_token_indices(
    tokenizer,
    statement: str,
    char_span: tuple[int, int] | None,
) -> list[int]:
    """
    使用 tokenizer offset_mapping 将字符 span 对齐到 token indices。
    """
```

伪代码：

```python
encoded = tokenizer(
    statement,
    return_offsets_mapping=True,
    add_special_tokens=True,
)
offsets = encoded["offset_mapping"]

selected = []
for idx, (start, end) in enumerate(offsets):
    if start == end:
        continue
    if token_span_overlaps_char_span((start, end), char_span):
        selected.append(idx)
```

需要注意：

1. Qwen tokenizer 可能包含特殊 token；
2. padding token 不应进入 anchor；
3. offset `(0, 0)` 通常是特殊 token，应跳过；
4. 如果某个 anchor 没有 token，则回退到可用邻近 token 或标记为 invalid。

### 6.5 最终接口

```python
def extract_anchors(tokenizer, statement: str) -> AnchorSpans:
    """
    输入 statement，输出 token-level anchors。
    """
```

### 6.6 测试要求

在 `tests/phase4/test_anchor_extraction.py` 中测试：

1. 系表结构能正确识别；
2. relation keyword 能正确识别；
3. fallback 不报错；
4. token index 不越界；
5. last_token_index 对应最后一个非 padding token；
6. 特殊 token 不被错误选为 anchor；
7. 空字符串或极短句子能安全处理。

---

## 7. P4.2：提取 layer/head 级 attention score 特征

### 7.1 目标

不要直接对所有层和所有 head 做全局平均。Phase 4 应保留：

```text
layer × head × statistic
```

这样的粒度，后续才能分析哪些层、哪些 head 有用。

### 7.2 新增文件：`src/features/attention_scores.py`

建议实现核心函数：

```python
def extract_attention_score_features_single(
    model,
    tokenizer,
    statement: str,
    layers: list[int],
) -> tuple[np.ndarray, list[str], dict]:
    """
    对单条 statement 提取 layer/head 级 attention score 特征。
    返回：
    - features: shape [num_features]
    - feature_names: list[str]
    - metadata: anchor info, seq length, fallback rule 等
    """
```

批量接口：

```python
def extract_attention_score_features_dataset(
    model,
    tokenizer,
    dataset,
    layers: list[int],
    batch_size: int = 1,
    output_path: str | None = None,
) -> dict:
    """
    对 train/val/test dataset 批量提取 attention score features。
    """
```

### 7.3 模型前向传播设置

```python
with torch.no_grad():
    outputs = model(
        **inputs,
        output_attentions=True,
        output_hidden_states=False,
        use_cache=False,
    )
```

为了确保能拿到注意力矩阵，模型加载时建议：

```python
AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    attn_implementation="eager",
)
```

如果当前 `src/models/loader.py` 不支持 `attn_implementation`，可以在 loader 中新增可选参数。

### 7.4 attention tensor 格式

通常：

```python
outputs.attentions[layer_idx].shape
# [batch_size, num_heads, seq_len, seq_len]
```

注意：

```text
layer_idx 应继续使用 Transformer block 输出编号；
不要把 embedding output 当成 layer；
attention 层数量应等于 model.config.num_hidden_layers。
```

### 7.5 每个 head 的特征设计

对每个 layer `l`、head `h`，提取以下特征：

#### last token 视角

```text
L{l}_H{h}_last_to_subject_mass
L{l}_H{h}_last_to_relation_mass
L{l}_H{h}_last_to_tail_mass
L{l}_H{h}_last_to_anchor_mass
L{l}_H{h}_last_to_non_anchor_mass
L{l}_H{h}_attention_entropy_last
L{l}_H{h}_max_attention_last
L{l}_H{h}_top3_attention_mass_last
L{l}_H{h}_attention_sink_mass
```

解释：

- `last_to_subject_mass`：最后 token 对 subject tokens 的注意力总和；
- `last_to_relation_mass`：最后 token 对 relation tokens 的注意力总和；
- `last_to_tail_mass`：最后 token 对 tail tokens 的注意力总和；
- `last_to_anchor_mass`：最后 token 对 subject + relation + tail 的总注意力；
- `attention_entropy_last`：最后 token 的注意力分布熵；
- `attention_sink_mass`：最后 token 对前几个 token 的注意力质量，用于检测 attention sink。

#### token-to-token 结构视角

```text
L{l}_H{h}_subject_to_relation_mass
L{l}_H{h}_relation_to_subject_mass
L{l}_H{h}_relation_to_tail_mass
L{l}_H{h}_tail_to_relation_mass
```

解释：

这些特征用于观察实体关系结构是否被注意力连接起来。

#### 归一化特征

为避免 anchor token 数量不同造成偏差，增加：

```text
L{l}_H{h}_last_to_subject_mass_norm
L{l}_H{h}_last_to_relation_mass_norm
L{l}_H{h}_last_to_tail_mass_norm
```

定义：

```python
mass_norm = mass / max(num_anchor_tokens, 1)
```

### 7.6 全局辅助特征

除 layer/head 特征外，记录以下 metadata，但默认不作为训练特征：

```text
sequence_length
subject_token_count
relation_token_count
tail_token_count
anchor_token_count
anchor_rule_id
fallback_flag
```

这些 metadata 用于分析和去偏，不建议直接喂给最终分类器。

### 7.7 候选层范围

优先使用 Phase 3 发现的中后层高性能区间：

```python
candidate_layers = [13, 14, 15, 16, 17, 18, 19, 20]
```

理由：

```text
Phase 3 已说明真实性信号主要在中后层形成；
直接全层提取会增加计算和噪声；
先聚焦 layer 13-20 更高效，也更符合 Phase 3 结论。
```

如果时间允许，再补充全层版本：

```python
candidate_layers = list(range(model.config.num_hidden_layers))
```

### 7.8 输出缓存

```text
experiments/results/phase4/cache/attention_scores_train.npz
experiments/results/phase4/cache/attention_scores_val.npz
experiments/results/phase4/cache/attention_scores_test.npz
```

metadata 示例：

```json
{
  "feature_version": "attention_scores_rule_v1",
  "layers": [13, 14, 15, 16, 17, 18, 19, 20],
  "num_heads": 12,
  "anchor_version": "rule_v1",
  "include_length_features": false,
  "note": "features are layer/head-level attention score statistics"
}
```

### 7.9 测试要求

在 `tests/phase4/test_attention_scores.py` 中测试：

1. 单条 statement 返回固定长度 feature；
2. feature_names 与 feature 数量一致；
3. 无 NaN / inf；
4. attention mass 在合理范围；
5. 所有 layer/head 均生成特征；
6. 空 anchor 时不崩溃；
7. batch 提取后样本顺序与 labels 一致。

---

## 8. P4.3：attention 特征去长度偏置

### 8.1 动机

注意力统计很容易受到以下表面因素影响：

```text
sequence_length
subject_token_count
relation_token_count
tail_token_count
anchor_token_count
```

如果不控制这些因素，分类器可能学到“句长”或“anchor 数量”，而不是真正的注意力模式。

### 8.2 三套特征版本

Phase 4 应至少生成三套 attention score 特征：

```text
1. raw_attention
2. no_length_attention
3. residualized_attention
```

#### raw_attention

原始 attention 特征，用于分析，不作为最终主方法。

#### no_length_attention

删除以下字段：

```text
sequence_length
subject_token_count
relation_token_count
tail_token_count
anchor_token_count
fallback_flag
anchor_rule_id
```

如果这些字段原本没有进入 feature matrix，也要确保它们不会被拼接进训练特征。

#### residualized_attention

对每个 attention feature 做长度残差化。

### 8.3 残差化公式

对每个 attention feature \(\phi_j\)，在训练集上拟合：

\[
\phi_j = a_j \cdot \text{length} + b_j
\]

然后得到残差：

\[
\phi'_j = \phi_j - (a_j \cdot \text{length} + b_j)
\]

重要：

```text
只能在 train set 上 fit 残差化模型；
val/test 只能使用 train 上拟合出的 a_j、b_j 做 transform；
不能在 val/test 上重新 fit，否则数据泄漏。
```

### 8.4 实现函数

在 `src/methods/phase4_attention.py` 中实现：

```python
from sklearn.linear_model import LinearRegression


def residualize_by_length(
    train_X: np.ndarray,
    val_X: np.ndarray,
    test_X: np.ndarray,
    train_length: np.ndarray,
    val_length: np.ndarray,
    test_length: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    对每一维 feature 使用 train length 做线性残差化。
    """
```

### 8.5 验收标准

1. residualized feature shape 与原始 feature 相同；
2. train residualized features 与 length 的相关性应明显下降；
3. residualization 不应产生 NaN；
4. 只在 train 上 fit；
5. 输出 residualization metadata。

---

## 9. P4.4：validation-based layer/head selection

### 9.1 动机

不是所有 attention head 都有真假判别能力。直接拼接所有 head 的特征可能引入大量噪声。

因此要在验证集上选择最有用的 layer/head。

### 9.2 选择粒度

推荐按 head 分组，而不是单个 feature 分组。

例如以下特征都属于同一个 head：

```text
L17_H03_last_to_subject_mass
L17_H03_last_to_relation_mass
L17_H03_attention_entropy_last
L17_H03_max_attention_last
...
```

将它们归为：

```text
(layer=17, head=3)
```

### 9.3 实现函数

在 `src/methods/phase4_attention.py` 中实现：

```python
def group_feature_indices_by_head(feature_names: list[str]) -> dict[tuple[int, int], list[int]]:
    """
    将 L{layer}_H{head}_xxx 格式的特征名分组。
    """
```

```python
def score_head_group(
    train_X: np.ndarray,
    val_X: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    feature_indices: list[int],
    metric: str = "auroc",
) -> dict:
    """
    使用该 head 的全部特征训练一个小 logistic regression，
    在 val 上评估该 head 的单独判别能力。
    """
```

```python
def select_top_heads(
    train_X: np.ndarray,
    val_X: np.ndarray,
    train_y: np.ndarray,
    val_y: np.ndarray,
    feature_names: list[str],
    top_k_heads: int = 16,
    metric: str = "val_auroc",
) -> dict:
    """
    返回 selected_heads 和 selected_feature_indices。
    """
```

### 9.4 候选 top-k

在验证集上比较：

```text
top_k_heads = 8
top_k_heads = 16
top_k_heads = 32
```

最终选验证集表现最好的 top-k。

注意：

```text
test set 不能用于选择 top_k；
test set 只能用于最终汇报一次。
```

### 9.5 输出

```text
experiments/results/phase4/attention_head_selection.json
```

格式：

```json
{
  "selection_metric": "val_auroc",
  "top_k_heads": 16,
  "selected_heads": [
    {
      "layer": 15,
      "head": 3,
      "val_accuracy": 0.0,
      "val_macro_f1": 0.0,
      "val_auroc": 0.0
    }
  ],
  "selected_feature_names": []
}
```

### 9.6 可视化

在 `src/analysis/phase4_analysis.py` 中实现：

```python
def plot_layer_head_metric_heatmap(
    head_scores: list[dict],
    output_path: str,
    metric: str = "val_auroc",
):
    """
    绘制 layer × head heatmap。
    """
```

输出：

```text
experiments/results/phase4/figures/layer_head_auroc_heatmap.png
```

这张图是 Phase 4 报告中最重要的分析图之一。

---

## 10. P4.5：提取 attention output activation

### 10.1 动机

attention score 表示“看哪里”，但不表示“取回了什么信息”。

注意力模块真正写回 residual stream 的是 attention output：

\[
O^l = \text{Attention}(Q, K, V) W_O
\]

因此 attention output activation 可能比纯 attention score 更适合做真假检测。

### 10.2 新增文件：`src/features/attention_outputs.py`

使用 forward hook 获取每层 self-attention 输出。

Qwen2 的层结构通常类似：

```text
model.model.layers[i].self_attn
```

### 10.3 Hook 类

```python
class AttentionOutputExtractor:
    def __init__(self, model, layers: list[int]):
        self.model = model
        self.layers = layers
        self.handles = []
        self.outputs = {}

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            # Qwen2 self_attn output 通常是 tuple，output[0] 是 attention output
            if isinstance(output, tuple):
                attn_out = output[0]
            else:
                attn_out = output
            self.outputs[layer_idx] = attn_out.detach().cpu()
        return hook

    def register(self):
        for layer_idx in self.layers:
            module = self.model.model.layers[layer_idx].self_attn
            handle = module.register_forward_hook(self._make_hook(layer_idx))
            self.handles.append(handle)

    def clear(self):
        self.outputs = {}

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
```

### 10.4 单样本提取函数

```python
def extract_attention_output_features_single(
    model,
    tokenizer,
    statement: str,
    layers: list[int],
    pooling: str = "last",
) -> tuple[np.ndarray, list[str], dict]:
    """
    提取 attention module output activation 的统计特征。
    """
```

### 10.5 特征设计

对每层 attention output 的 last token 向量 \(o_l\)，提取：

```text
L{l}_attn_out_norm
L{l}_attn_out_mean_abs
L{l}_attn_out_max_abs
L{l}_attn_out_std
L{l}_attn_out_sparsity_1e-3
```

如果同时提取 hidden state，可加入：

```text
L{l}_attn_out_hidden_cosine
L{l}_attn_out_hidden_norm_ratio
```

其中：

```python
cosine = cosine_similarity(attn_out_last, hidden_last)
norm_ratio = norm(attn_out_last) / max(norm(hidden_last), eps)
```

### 10.6 两种使用方式

#### 方式 A：低维统计特征

优先实现。

```text
num_features = num_layers × num_stats
```

优点：

```text
显存小；
训练快；
报告容易解释；
适合当前项目时间限制。
```

#### 方式 B：高维向量特征

可选。

```text
使用 layer 17 attention output last vector，维度约为 hidden_size = 1536。
```

组合：

```text
attention_output_vector_only
hidden + attention_output_vector
```

如果时间紧，先不做方式 B。

### 10.7 输出缓存

```text
experiments/results/phase4/cache/attention_outputs_train.npz
experiments/results/phase4/cache/attention_outputs_val.npz
experiments/results/phase4/cache/attention_outputs_test.npz
```

### 10.8 测试要求

在 `tests/phase4/test_attention_outputs.py` 中测试：

1. hook 能成功注册和移除；
2. outputs 字典包含指定 layers；
3. 每层输出 shape 合理；
4. pooling 后 feature shape 固定；
5. 无 NaN / inf；
6. 多次调用后 outputs 不串样本；
7. remove 后不再继续记录输出。

---

## 11. P4.6：训练融合分类器与消融实验

### 11.1 必做方法列表

Phase 4 至少要比较以下方法：

| 编号 | 方法名 | 特征 | 目的 |
|---|---|---|---|
| A0 | hidden_only | layer17 last hidden | Phase 3 基线 |
| A1 | attention_score_only_raw | raw attention score | 检查注意力分数单独能力 |
| A2 | attention_score_only_debiased | 去长度偏置 score | 检查去偏后是否仍有信号 |
| A3 | attention_score_top_heads_only | selected heads score | 检查 head selection 是否有效 |
| A4 | attention_output_only | attention output stats | 检查内部激活单独能力 |
| A5 | hidden_plus_debiased_attention | hidden + 去偏 score | 检查 score 是否补充 hidden |
| A6 | hidden_plus_top_head_attention | hidden + selected heads | 检查筛选 heads 是否提升 |
| A7 | hidden_plus_attention_output | hidden + attention output | 检查内部激活是否补充 hidden |
| A8 | hidden_plus_all_attention | hidden + score + output | 检查多特征融合 |
| A9 | gated_fusion | hidden 与 fusion 的边界修正 | 检查是否修正不确定样本 |

### 11.2 训练函数

在 `src/methods/phase4_attention.py` 中实现：

```python
def train_eval_classifier(
    train_X,
    val_X,
    test_X,
    train_y,
    val_y,
    test_y,
    classifier_type: str = "logistic",
    seeds: tuple[int, ...] = (42, 123, 2024),
) -> dict:
    """
    对一个 feature setting 做多 seed 训练和评估。
    返回每个 seed 结果及 mean/std。
    """
```

分类器建议复用已有 `saplma.py` 逻辑：

```text
StandardScaler fit on train only；
LogisticRegression / MLPClassifier；
输出 label 和 probability；
指标使用 Accuracy、Macro-F1、AUROC。
```

### 11.3 特征融合方式

最简单的 concat：

```python
X_hidden_plus_attention = np.concatenate([X_hidden, X_attention], axis=1)
```

必须确保：

```text
train / val / test 使用同样 feature order；
StandardScaler 只在 train 上 fit；
如果 attention feature 做过 residualization，不能重新在 val/test fit。
```

### 11.4 Gated Fusion

Gated Fusion 不是重新训练一个复杂模型，而是将 hidden-only 和 fusion model 的概率做选择性组合。

公式：

\[
p(x)=
\begin{cases}
 p_{hidden}(x), & |p_{hidden}(x)-0.5| > \tau \\
 p_{fusion}(x), & |p_{hidden}(x)-0.5| \le \tau
\end{cases}
\]

含义：

```text
如果 hidden-only 很自信，则不改；
如果 hidden-only 接近 0.5，则使用 attention fusion 修正。
```

实现：

```python
def gated_fusion_probs(
    hidden_probs: np.ndarray,
    fusion_probs: np.ndarray,
    tau: float,
) -> np.ndarray:
    uncertain = np.abs(hidden_probs - 0.5) <= tau
    final_probs = hidden_probs.copy()
    final_probs[uncertain] = fusion_probs[uncertain]
    return final_probs
```

`tau` 只能在验证集上选择：

```python
candidate_tau = [0.05, 0.10, 0.15, 0.20, 0.25]
```

选择验证集 Macro-F1 或 Accuracy 最好的 `tau`，再固定到测试集。

### 11.5 输出文件

```text
experiments/results/phase4/phase4_ablation_results.json
experiments/results/phase4/phase4_main_results.csv
```

`phase4_main_results.csv` 格式：

```text
method,feature_dim,classifier,seed_count,val_accuracy_mean,val_macro_f1_mean,val_auroc_mean,test_accuracy_mean,test_macro_f1_mean,test_auroc_mean,test_accuracy_std,test_macro_f1_std,test_auroc_std
```

---

## 12. P4.7：特征分析、可视化与错误分析

### 12.1 特征差异分析

实现：

```python
def summarize_feature_differences(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    output_csv: str,
):
    """
    统计 true/false 样本在各特征上的均值、差值、标准差、单特征 AUROC。
    """
```

输出：

```text
attention_score_feature_summary.csv
attention_output_feature_summary.csv
```

字段：

```text
feature_name
true_mean
false_mean
delta
abs_delta
true_std
false_std
single_feature_auroc
```

### 12.2 layer-head heatmap

输出：

```text
figures/layer_head_auroc_heatmap.png
```

图含义：

```text
横轴：head index
纵轴：layer index
颜色：该 head 的 validation AUROC 或 effect size
```

### 12.3 方法对比图

输出：

```text
figures/method_accuracy_comparison.png
figures/method_auroc_comparison.png
```

展示方法：

```text
hidden_only
attention_score_only_debiased
attention_score_top_heads_only
attention_output_only
hidden_plus_top_head_attention
hidden_plus_attention_output
hidden_plus_all_attention
gated_fusion
```

### 12.4 错误修正分析

构建 hidden-only 与最佳 fusion 方法之间的 2×2 表：

| | Fusion correct | Fusion wrong |
|---|---:|---:|
| Hidden correct | n00 | n01 |
| Hidden wrong | n10 | n11 |

重点看：

```text
n10: hidden 错但 fusion 对
n01: hidden 对但 fusion 错
```

如果：

```text
n10 > n01
```

说明 attention fusion 带来了净修正。

### 12.5 错误案例表

输出：

```text
phase4_error_analysis.csv
```

字段：

```text
statement
label
hidden_prob
fusion_prob
hidden_pred
fusion_pred
case_type
key_attention_features
```

`case_type` 取值：

```text
hidden_correct_fusion_correct
hidden_correct_fusion_wrong
hidden_wrong_fusion_correct
hidden_wrong_fusion_wrong
```

报告中至少展示 4-8 条典型样本。

---

## 13. main.py 命令设计

建议在 `main.py` 中新增命令：

```bash
python -s main.py phase4
python -s main.py phase4-cache-hidden
python -s main.py phase4-extract-attention-scores
python -s main.py phase4-debias-attention
python -s main.py phase4-select-heads
python -s main.py phase4-extract-attention-outputs
python -s main.py phase4-ablation
python -s main.py phase4-visualize
```

### 13.1 一键运行

```bash
python -s main.py phase4
```

执行顺序：

```text
1. cache hidden features
2. run hidden baseline
3. extract attention score features
4. debias attention features
5. select top heads on validation set
6. extract attention output features
7. run ablation experiments
8. generate visualizations
9. write phase4_summary.md
```

### 13.2 分步运行

开发过程中建议分步运行，便于定位错误。

```bash
python -s main.py phase4-cache-hidden
python -s main.py phase4-extract-attention-scores
python -s main.py phase4-select-heads
python -s main.py phase4-extract-attention-outputs
python -s main.py phase4-ablation
python -s main.py phase4-visualize
```

---

## 14. 测试计划

### 14.1 单元测试

#### anchor extraction

```bash
pytest tests/phase4/test_anchor_extraction.py -q
```

覆盖：

```text
copula rule
relation phrase rule
fallback rule
offset mapping
invalid input
short input
```

#### attention score features

```bash
pytest tests/phase4/test_attention_scores.py -q
```

覆盖：

```text
feature shape
feature_names alignment
no NaN
valid attention mass
layer/head count
batch order
```

#### debias

```bash
pytest tests/phase4/test_attention_debias.py -q
```

覆盖：

```text
train-only fit
shape unchanged
correlation with length reduced
no NaN
```

#### head selection

```bash
pytest tests/phase4/test_head_selection.py -q
```

覆盖：

```text
feature_names grouping
top-k selection
no test leakage
selected indices valid
```

#### attention output

```bash
pytest tests/phase4/test_attention_outputs.py -q
```

覆盖：

```text
hook register/remove
output shape
clear between samples
feature shape
no NaN
```

### 14.2 小规模集成测试

```bash
pytest tests/phase4/test_phase4_pipeline.py -q
```

使用极小样本：

```text
train: 16 samples
val: 8 samples
test: 8 samples
```

只验证流程是否跑通，不要求指标有意义。

---

## 15. 实验记录要求

每个结果 JSON 必须记录：

```text
method name
feature version
feature dim
layers used
heads used
classifier type
seeds
split seed
metrics on train/val/test
runtime info
model dtype
transformers version
pytorch version
cuda device
```

禁止只保存最终数字，不保存配置。

---

## 16. 最终主结果表模板

报告中建议使用如下表格：

| Method | Feature | Classifier | Test Accuracy | Test Macro-F1 | Test AUROC |
|---|---|---|---:|---:|---:|
| PPL | sequence probability | threshold | 已有结果 | 已有结果 | 已有结果 |
| SAPLMA | layer27 last hidden | MLP | 已有结果 | 已有结果 | 已有结果 |
| Phase 3 hidden-only | layer17 last hidden | LR | 待复现 | 待复现 | 待复现 |
| Attention score only | debiased score | LR | 待实验 | 待实验 | 待实验 |
| Top-head attention only | selected heads | LR | 待实验 | 待实验 | 待实验 |
| Attention output only | attn output stats | LR | 待实验 | 待实验 | 待实验 |
| Hidden + top-head attention | hidden + selected score | LR | 待实验 | 待实验 | 待实验 |
| Hidden + attention output | hidden + output stats | LR | 待实验 | 待实验 | 待实验 |
| Hidden + all attention | hidden + score + output | LR/MLP | 待实验 | 待实验 | 待实验 |
| Gated fusion | boundary correction | rule | 待实验 | 待实验 | 待实验 |

---

## 17. 消融实验表模板

| Ablation | Accuracy | Macro-F1 | AUROC | Purpose |
|---|---:|---:|---:|---|
| hidden-only | | | | Phase 3 baseline |
| raw attention only | | | | attention score 单独能力 |
| debiased attention only | | | | 去长度偏置后是否仍有信号 |
| top-head attention only | | | | head selection 是否有效 |
| attention output only | | | | attention 内部激活是否有效 |
| hidden + debiased attention | | | | score 是否补充 hidden |
| hidden + top-head attention | | | | 选头后是否更好 |
| hidden + attention output | | | | output activation 是否补充 hidden |
| hidden + score + output | | | | 多特征融合 |
| gated fusion | | | | 是否修正边界样本 |

---

## 18. 报告 Phase 4 写作模板

### 18.1 Motivation

可以写：

```text
Phase 3 结果表明，模型中后层 hidden state 中已经包含较强的真实性判别信号。然而 hidden state 本身是高维向量，难以解释其具体来源。为了进一步分析模型在处理真实与虚假陈述时是否具有不同的信息聚合模式，本文在 Phase 4 中引入注意力模块特征。注意力分数可以刻画最后 token 对主语、关系词和尾部实体的关注程度；注意力输出激活则进一步反映注意力模块写回 residual stream 的内部信息。因此，Phase 4 尝试构建 Attention-Guided SAPLMA，在 Phase 3 hidden baseline 基础上评估注意力特征是否能提供互补信号。
```

### 18.2 Method

应分成三部分：

```text
1. Anchor extraction: subject / relation / tail token 定位
2. Attention score features: layer/head 级注意力统计
3. Attention output features: self-attention output activation
4. Fusion classifier: hidden-only, attention-only, hidden+attention, gated fusion
```

### 18.3 Experiment

说明：

```text
数据划分沿用 Phase 1-3；
hidden baseline 使用 Phase 3 选择的 layer 17 + last；
head selection 只使用 train/val；
test set 只用于最终汇报；
指标为 Accuracy、Macro-F1、AUROC；
分类实验使用 seeds = 42, 123, 2024 并报告 mean ± std。
```

### 18.4 Result

重点回答：

```text
attention-only 是否有效？
去长度偏置后 attention 是否仍有信号？
top-head selection 是否提升？
attention output 是否比 attention score 更有效？
hidden+attention 是否优于 hidden-only？
gated fusion 是否修正了 hidden-only 的错误样本？
```

### 18.5 Discussion

如果提升明显，可以写：

```text
结果说明注意力模块中确实存在与真实性判断互补的信号。相比直接使用所有注意力统计，经过长度去偏和验证集层/头选择后的注意力特征更稳定；attention output activation 进一步表明，注意力模块写回 residual stream 的内部激活比原始注意力分数包含更丰富的判别信息。
```

如果提升不明显，也可以写：

```text
结果表明，注意力分数单独不足以承担真假分类任务，但其层/头级差异和错误修正案例显示，注意力模块仍能提供一定解释性线索。该结果说明，在 True-False 陈述句任务中，hidden state 仍是主要判别来源，而 attention 更适合作为辅助分析工具和边界样本修正信号。
```

---

## 19. 成功标准

### 19.1 最低成功标准

满足以下即可认为 Phase 4 完成：

```text
1. 完成一种 attention score 或 attention output 特征；
2. 完成 attention-only、hidden-only、hidden+attention 三组消融；
3. 测试集报告 Accuracy、Macro-F1、AUROC；
4. 报告中解释 attention 方法的动机、实现细节和结果；
5. 若未超过 baseline，也要分析失败原因。
```

### 19.2 较好标准

```text
1. hidden + debiased attention >= hidden-only；
2. 去除长度偏置后仍有小幅增益；
3. layer/head heatmap 显示部分 head 有明显区分能力；
4. 错误分析显示 fusion 修正的样本多于破坏的样本。
```

### 19.3 优秀标准

```text
1. hidden + attention output 或 gated fusion 稳定超过 hidden-only；
2. Accuracy / Macro-F1 至少提升 0.5%-1.0%；
3. AUROC 不下降，最好略有提升；
4. 多 seed 结果稳定；
5. 报告中能结合 head selection、feature delta、case study 给出机制解释。
```

---

## 20. 推荐开发顺序清单

按以下顺序执行最稳：

```text
[ ] 1. 新建 experiments/results/phase4/ 目录
[ ] 2. 新建 src/utils/feature_cache.py
[ ] 3. 缓存 layer17 last hidden features
[ ] 4. 复现 hidden-only baseline
[ ] 5. 新建 src/features/anchor_extraction.py
[ ] 6. 完成 anchor extraction 单元测试
[ ] 7. 新建 src/features/attention_scores.py
[ ] 8. 提取 layer 13-20 的 layer/head attention score features
[ ] 9. 生成 attention score feature summary
[ ] 10. 实现 length debias / residualization
[ ] 11. 实现 validation-based head selection
[ ] 12. 生成 layer-head AUROC heatmap
[ ] 13. 新建 src/features/attention_outputs.py
[ ] 14. 提取 attention output stats
[ ] 15. 实现 phase4 ablation pipeline
[ ] 16. 跑 hidden-only / attention-only / hidden+attention
[ ] 17. 跑 attention output 和 gated fusion
[ ] 18. 生成主结果表
[ ] 19. 生成方法对比图
[ ] 20. 生成错误修正案例表
[ ] 21. 写 phase4_summary.md
[ ] 22. 更新 docs/Report.md 的 Phase 4 部分
```

---

## 21. 风险与解决方案

### 风险 1：Qwen2 不返回 attention weights

现象：

```text
outputs.attentions is None
```

解决：

```text
加载模型时设置 attn_implementation="eager"；
调用 forward 时设置 output_attentions=True；
设置 use_cache=False。
```

### 风险 2：显存不足

解决：

```text
batch_size=1 提取 attention；
只提取 layer 13-20；
提取后立即 detach.cpu()；
使用 npz 缓存；
不要一次保存所有 raw attention maps。
```

### 风险 3：anchor 抽取不稳定

解决：

```text
优先使用简单规则；
记录 fallback rate；
报告中说明 anchor 是启发式近似；
必要时只使用 subject + last token 相关特征。
```

### 风险 4：attention-only 结果很弱

这并不意味着 Phase 4 失败。报告中可以解释：

```text
attention score 是结构性辅助信号，不一定能单独替代 hidden state；
关键是 hidden+attention 是否提供互补增益，以及注意力分析是否揭示不同层/头的模式差异。
```

### 风险 5：hidden+attention 没有超过 hidden-only

解决：

```text
检查是否混入长度偏置；
检查 head selection；
尝试 attention output activation；
做 gated fusion；
补充错误案例和失败原因分析。
```

---

## 22. 最终提交物清单

Phase 4 完成后，应至少提交：

```text
代码：
- src/features/anchor_extraction.py
- src/features/attention_scores.py
- src/features/attention_outputs.py
- src/methods/phase4_attention.py
- src/analysis/phase4_analysis.py
- src/utils/feature_cache.py
- main.py 新增 phase4 命令

测试：
- tests/phase4/test_anchor_extraction.py
- tests/phase4/test_attention_scores.py
- tests/phase4/test_attention_debias.py
- tests/phase4/test_head_selection.py
- tests/phase4/test_attention_outputs.py
- tests/phase4/test_phase4_pipeline.py

结果：
- experiments/results/phase4/hidden_baseline.json
- experiments/results/phase4/attention_score_feature_summary.csv
- experiments/results/phase4/attention_head_selection.json
- experiments/results/phase4/attention_output_feature_summary.csv
- experiments/results/phase4/phase4_ablation_results.json
- experiments/results/phase4/phase4_main_results.csv
- experiments/results/phase4/phase4_error_analysis.csv

图表：
- layer_head_auroc_heatmap.png
- feature_delta_boxplot.png
- method_accuracy_comparison.png
- method_auroc_comparison.png
- correction_matrix.png

报告：
- docs/Report.md 中新增 Phase 4 方法、实验、结果、讨论
- phase4_summary.md
```

---

## 23. 一句话版本

Phase 4 从 Phase 3 开始的最稳路线是：

```text
固定 layer17 + last hidden state 作为主基线，
为陈述句抽取 subject / relation / tail anchors，
提取 layer/head 级 attention score 和 attention output activation，
通过长度去偏与 validation-based head selection 控制噪声，
最后比较 attention-only、hidden-only、hidden+attention 和 gated fusion，
用主结果表、layer-head heatmap 与错误修正案例证明注意力模块是否提供互补信号。
```
