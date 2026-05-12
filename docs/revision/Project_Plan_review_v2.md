# Project_Plan 审查与修改建议（v2）

## 1. 审查结论

相较于 v1 对应的问题清单，当前的 [docs/Project_Plan.md](docs/Project_Plan.md) 已经完成了大部分关键修正：

- 任务定义已经从“泛化的幻觉检测”收敛到“基于 True-False Dataset 的陈述真伪判别”；
- 主模型与保底模型的策略已经更合理，Qwen2-1.5B 被明确为可落地的保底主实验模型；
- 实验协议、PowerShell 脚本、最小可交付版本和图表清单都已补充；
- Phase 4 中注意力方向与陈述句数据之间的适配关系已经说明。

因此，这一版计划文档在“是否覆盖课程要求”这个层面上已经基本达标。当前剩余的问题，不再是方向偏题，而是执行层面的技术口径、环境依赖和实现表述仍有几处不够严谨。若作为正式执行版，建议再做一轮小修订。

## 2. 本轮审查重点

本轮 v2 不再重复 v1 已解决的问题，重点检查以下三类剩余风险：

1. 计划文本内部是否还存在前后口径不一致；
2. 伪代码和实验协议是否能直接指导实现；
3. 保底执行路径是否真的与高风险依赖解耦。

## 3. 剩余问题与修改建议

### P0. 层索引定义与伪代码实现仍然不一致

当前文档在“实验协议”和“P2.3 SAPLMA 方法实现要点”中已经明确提出：

- 层分析以 Transformer block 输出为统计对象；
- embedding output 不计入层号；
- layer_idx 从 Transformer block 输出开始计数。

但当前伪代码仍直接使用：

```python
last_token_hidden = outputs.hidden_states[layer_idx][:, -1, :]
```

同时注释又写明：

```python
# 注意：部分模型 hidden_states[0] 是 embedding output，分析时按需跳过
```

这两部分合在一起，意味着“文档定义的层号”和“代码实际取到的层”仍可能偏移一层。只要后续实现者按“第 k 个 Transformer block”去传 layer_idx，在不少 Hugging Face 模型上就会直接取错层。

影响：

- 层分析曲线可能整体错位；
- 不同模型之间的层结果不可比；
- 报告中关于“中层最有效”的分析可能建立在错误索引上。

建议：在伪代码中显式先剥离 embedding output，再对 Transformer block 输出编号。推荐把这一点写死在示例实现里，不要只留注释。

建议替换文案：

```python
def extract_last_token_hidden(model, tokenizer, statement, layer_idx=-1):
    """
    提取指定 Transformer 层的最后 token 隐藏状态作为特征。
    - layer_idx 以 Transformer block 输出为编号，不包含 embedding output
    - 默认取最后一个 Transformer block 的输出
    """
    inputs = tokenizer(statement, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    if len(hidden_states) == model.config.num_hidden_layers + 1:
        block_hidden_states = hidden_states[1:]
    else:
        block_hidden_states = hidden_states

    last_token_hidden = block_hidden_states[layer_idx][:, -1, :]
    return last_token_hidden.cpu().numpy()
```

### P1. 保底主实验路径仍未与 4-bit 依赖完全解耦

当前计划已经把 Qwen2-1.5B FP16 设为保底主实验模型，这是正确的；但文档中的若干地方仍然默认“4-bit 路径是标准路径”：

- 软件环境表仍把 bitsandbytes 列为常规依赖；
- Phase 1 的 P1.6 仍写成“支持 4-bit 量化的加载器”；
- 依赖安装命令默认直接安装 bitsandbytes。

这会削弱“先保底跑通”的策略。对于 Windows + PowerShell 环境，保底路径应当尽量避免一开始就绑定 bitsandbytes，否则环境准备阶段就会被高风险依赖卡住。

影响：

- 保底实验路径不够干净；
- Phase 1 的失败概率仍然偏高；
- 文档上的“Qwen2-1.5B 保底方案”与实际安装步骤不完全一致。

建议：把 bitsandbytes 明确改为“可选依赖”，并把加载器目标改成“支持 FP16 主路径，兼容可选 4-bit 量化”。

建议替换文案：

```markdown
| bitsandbytes | optional | Llama-2-7B 4-bit 量化的可选支持 |
```

```markdown
| P1.6 | 实现模型加载模块 `src/models/loader.py` | 支持 FP16 主路径与可选 4-bit 量化的加载器 | A |
```

依赖安装部分也建议改成分层安装：

```powershell
# 安装核心依赖（保底主路径）
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv add transformers accelerate
uv add peft sentencepiece protobuf
uv add scikit-learn numpy pandas matplotlib seaborn
uv add tqdm datasets huggingface_hub
uv add wandb pyyaml

# 仅在尝试 Llama-2-7B 4-bit 时再安装
uv add bitsandbytes
```

### P1. “下载 4-bit 模型”与“加载时量化”仍被混在一起

Phase 1 当前写的是“下载 Llama-2-7B (4-bit) 和 Qwen2-1.5B”，但从技术上说，文档后面的命令下载的是原始模型权重，而不是已经量化好的 4-bit 模型。4-bit 量化通常发生在加载阶段。

另外，注释里写了“直接从 ModelScope 下载更快”，但紧接着给出的仍是 Hugging Face CLI 命令，这会让读者误以为当前命令已经切到了 ModelScope。

影响：

- 新成员可能误解模型下载与量化的边界；
- 文档会显得技术表述不够严谨；
- 真正执行时容易出现“为什么下载完还没 4-bit”的困惑。

建议：

1. 把 P1.3 改成“下载原始模型权重”；
2. 在下载模型小节加一句“4-bit 量化在加载阶段完成”；
3. 如果不准备给出真实的 ModelScope 命令，就删除“直接从 ModelScope 下载更快”这句注释，避免口径不统一。

建议替换文案：

```markdown
| P1.3 | 下载 Llama-2-7B 与 Qwen2-1.5B 原始模型权重 | `models_cache/` | A |
```

```powershell
# 说明：此处下载的是原始模型权重；4-bit 量化在模型加载阶段完成
huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir models_cache/Llama-2-7b-hf
huggingface-cli download Qwen/Qwen2-1.5B --local-dir models_cache/Qwen2-1.5B
```

### P2. 时间估算和日期窗口的语义仍不够明确

当前每个 Phase 同时写了“1-2 天 / 3-4 天 / 5-6 天”以及具体日期区间。问题在于，日期跨度基本都长于前面的天数描述，因此容易让人误解为时间表前后矛盾。

这不算严重错误，但如果文档要作为团队执行计划，建议明确“前面的数字是预估工作量，后面的日期是自然日时间窗口”。

建议补一句统一说明，例如：

```markdown
注：各 Phase 前括号中的“X-Y 天”表示预估有效工作量，后侧日期表示自然日时间窗口，包含缓冲与并行协作时间。
```

### P2. Subject token 的可选实现需要补一个依赖说明

当前文档已经把 Subject token 分析降级为可选项，这是合理的；但既然仍提到了 spaCy 作为优先方案，就建议在软件环境或附注中明确它是“仅在开启 Subject token 分析时安装”的可选依赖。

否则，文档仍会给人一种“计划依赖 spaCy，但环境里没写”的不完整感。

建议补充文案：

```markdown
| spaCy | optional | Subject token 自动抽取的可选依赖 |
```

并在环境搭建部分补一句：

```powershell
# 仅在执行 Subject token 分析时安装
uv add spacy
python -m spacy download en_core_web_sm
```

## 4. 修改优先级建议

如果只打算再修一轮最小改动，建议按以下顺序处理：

1. 先修层索引伪代码，避免后续实现直接取错层；
2. 再把保底主路径与 bitsandbytes 解耦；
3. 接着修正模型下载与量化的表述；
4. 最后补时间窗口说明和 spaCy 可选依赖说明。

## 5. 一句话总结

这一版计划已经基本达到了“可以覆盖课程要求”的标准；v2 剩下的问题主要是技术口径和执行细节。如果把层索引、保底依赖和模型下载表述这三处再修稳，文档就可以作为较成熟的执行版计划使用。
