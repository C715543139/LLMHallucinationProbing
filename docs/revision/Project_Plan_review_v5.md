# Project_Plan 审查与修改建议（v5）

## 1. 审查结论

当前的 [docs/Project_Plan.md](docs/Project_Plan.md) 已经基本达到计划文档可执行版的标准。与前几轮相比，课程要求覆盖、实验协议、PowerShell 环境适配、风险控制和进阶方向定义都已经比较完整。

本轮 v5 不再处理方法设计或实验协议层面的核心问题，只记录少量仍可能在实际执行中造成困惑的细节。整体判断是：这些问题都属于低到中低优先级，但如果希望文档真正做到“团队成员照着就能执行”，仍建议补齐。

## 2. 本轮重点

本轮主要检查两类问题：

1. 文档中承诺的输出物是否与实际步骤一致；
2. 示例命令和特征定义是否与当前 PowerShell 环境和计划任务完全对齐。

## 3. 剩余问题与修改建议

### P2. P1.1 的输出物统一收敛为 pyproject.toml 与 uv.lock

当前这版计划更合理的口径应是：删除 `environment.yml`，保留 `pyproject.toml` 与 `uv.lock`。因此 P1.1、目录结构与复现约定都应统一围绕这两个文件展开。

```markdown
| P1.1 | 安装 Conda、uv，创建 Python 环境 | `pyproject.toml`, `uv.lock` | A |
```

但按照当前新版文档的维护思路，更合适的做法是以 `pyproject.toml` 作为依赖声明文件，以 `uv.lock` 作为锁定文件，同时删除 `environment.yml`。这样一来，P1.1 的输出物、目录结构和复现说明都应同步收敛，否则文档内部仍会存在口径分叉。

影响：

- 目录结构、复现约定和阶段输出物如果口径不统一，仍会让新成员无法判断最终应以哪个文件为准；
- 如果误删 `uv.lock`，复现性会弱于当前已经确定的 uv 工作流；
- 文档虽然整体正确，但在工程执行层面仍显得不够收口。

建议：既然决定删除 `environment.yml`，那就应在计划文档中同步保留 `pyproject.toml` 与 `uv.lock` 的相关表述，并把 P1.1 输出物与复现约定统一改成这两个文件。

建议替换文案：

```markdown
| P1.1 | 安装 Conda、uv，创建 Python 环境 | `pyproject.toml`, `uv.lock` | A |
```

并建议同步改为：

```markdown
**复现约定**：软件环境表中的版本范围用于说明依赖原则；实际提交与复现实验时，以 `pyproject.toml` 与 `uv.lock` 中维护的项目依赖配置为准。
```

### P2. 注意力特征示例与 P4.A2 的任务描述还差一小步对齐

当前 P4.A2 已经把任务写成：

```markdown
计算主语实体 token、关系词 token 与句尾 token 之间的注意力统计特征
```

但后面的特征工程示例仍主要体现：

- 对主语实体的关注比例；
- 注意力集中度；
- 多头一致性；
- 逐层注意力熵。

也就是说，示例代码里还没有明确体现“关系词 token”或“句尾 token”对应的特征。这样会让任务描述比示例实现更具体，后续实现者仍需要自己补定义。

建议：要么把 P4.A2 稍微放宽成“若干注意力统计特征”，要么把示例补全，至少增加一项关系词或句尾位置相关特征。后一种更好，因为和当前任务口径更一致。

建议替换文案：

```python
attention_features = {
    "attn_concentration": entropy(attention_weights),       # 注意力集中度（熵越低越集中）
    "entity_attn_ratio": attn_to_entity / attn_total,       # 对主语实体的关注比例
    "relation_attn_ratio": attn_to_relation / attn_total,   # 对关系词的关注比例
    "tail_attn_ratio": attn_to_tail / attn_total,           # 对句尾 token 的关注比例
    "cross_head_agreement": mean_head_cosine_similarity,    # 多头注意力一致性
    "attn_entropy_per_layer": [h_i for h_i in entropies],   # 逐层注意力熵
}
```

## 4. 修改优先级建议

如果只再做最后一轮文档修订，建议按以下顺序处理：

1. 先统一 P1.1 输出物、目录结构与复现约定的口径，删除 `environment.yml`，保留 `pyproject.toml` 与 `uv.lock`；
2. 然后让注意力特征示例和 P4.A2 完全对齐。

## 5. 一句话总结

当前计划文档已经非常接近收口版本；按目前的取舍，v5 剩下的重点主要是统一环境配置文件口径（删除 `environment.yml`，保留 `pyproject.toml` 与 `uv.lock`），以及让注意力特征示例与任务描述完全对齐。把这两处补平之后，计划文档层面的审查基本就可以结束了。
