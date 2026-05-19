# Project_Plan 变更日志（v15）

## 变更日期

2026-05-19

## 变更动机

在上一轮已经完成 Phase 4 结果复跑、文档去 Windows 化、计划文档与报告同步之后，当前文档层面仍存在一类新的结构性问题：

1. Phase 4 的核心方法、关键结论和结果分析同时分散在 `docs/Report.md`、`docs/对比结果.md` 与 `docs/Phase4_From_Phase3_Development_Guide.md` 中；
2. `docs/对比结果.md` 更像中间汇总文档，其中相当一部分内容已经应当被正式吸收入报告，而不应继续作为并行现行文档保留；
3. `docs/Phase4_From_Phase3_Development_Guide.md` 的原始定位是“从 Phase 3 重新开始开发”的指导文件，但当前仓库中 Phase 4 已经落地完成，该文档的名称与内容定位都已经落后于项目状态；
4. 如果继续维持这三份文档并行，后续极易再次出现“正式报告、计划文档与 Phase 4 说明彼此不同步”的问题。

因此，本次修订的目标是：

1. 将 `docs/对比结果.md` 中真正属于正式成果的核心内容写回 `docs/Report.md`；
2. 将旧的 Phase 4 Guide 改写为面向当前结果的进阶方案总结文档，并更名为 `docs/Advanced_Optimization.md`；
3. 删除已经被吸收的 `docs/对比结果.md`，避免继续出现并行版本；
4. 同步更新 `docs/Project_Plan.md` 中对现行文档结构的引用。

---

## 变更清单

### 1. `docs/Report.md` — 正式吸收 Phase 4 核心内容

| 位置 | 变更内容 |
| ---- | -------- |
| 摘要 | 将报告版本更新为 Draft v0.4，并补充 Phase 4 的稳定运行路径、hidden baseline、A6 / A8 核心结果与 attention 融合结论 |
| §2.4 本项目工作定位 | 将项目范围从 Phase 1-3 扩展为已覆盖 Phase 4 进阶优化任务 |
| §3.5 扩展分析方法 | 将 attention score / output、head selection 与 Phase 4 消融纳入正式方法说明 |
| §4.1 / §4.5 | 明确当前 Phase 4 的唯一有效运行口径为 `eager + bfloat16`，并补充 `tests/phase4/` 与 NaN 诊断背景 |
| §5 | 新增 `5.5 Phase 4：进阶优化结果`，正式写入数值稳定性结论、A0 全量 baseline、A2/A5/A6/A8/A9 等核心消融结果 |
| §6 | 新增 attention 融合收益与工程边界分析，不再把 Phase 4 视为未来工作 |
| §7-9 与附录 | 将当前实现状态、后续边界、结论与可引用结果文件更新到 Phase 4 已完成口径 |

### 2. Phase 4 Guide 改写并重命名

| 位置 | 变更内容 |
| ---- | -------- |
| `docs/Phase4_From_Phase3_Development_Guide.md` | 删除旧文件 |
| `docs/Advanced_Optimization.md` | 新增重写后的文档，改为“Phase 4 进阶方案分析、实现思路与结果总结” |

新的 `docs/Advanced_Optimization.md` 不再承担开发指导作用，而是重点保留以下内容：

1. 为什么在 Phase 3 之后仍需要做 Phase 4；
2. 当前真正采用的设计思路：固定 hidden baseline、rule-based anchor、debiased attention score、top-head selection、attention output 补充；
3. 当前实际实现流水线与稳定运行条件；
4. Phase 4 的核心结论、收益边界与后续优化方向。

### 3. 删除并行中间文档

| 位置 | 变更内容 |
| ---- | -------- |
| `docs/对比结果.md` | 删除旧文件，其核心结果与结论已吸收进 `docs/Report.md` |

这样处理后，Phase 4 结果不再同时依赖“正式报告 + 中间对比结果文档”两套现行说明，减少后续同步成本。

### 4. `docs/Project_Plan.md` — 同步现行文档结构

| 位置 | 变更内容 |
| ---- | -------- |
| §1.4 当前实现同步 | 将 Phase 4 的结果归档位置更新为 `docs/Report.md` 与 `docs/Advanced_Optimization.md` |
| §3.1 当前真实目录结构 | 删除 `docs/对比结果.md` 与旧 Guide 的现行引用，改为 `docs/Advanced_Optimization.md` |
| Phase 4 任务表 | 将 P4.7 的文档输出说明更新为 `docs/Report.md`、`docs/Advanced_Optimization.md` |

---

## 验证结果

| 验证项 | 结果 |
| ------ | ---- |
| `docs/Report.md` 文本搜索 | 已确认不再把 Phase 4 写为“下一阶段”或“待实现” |
| `docs/Advanced_Optimization.md` | 已创建并完成重写，内容定位与当前项目状态一致 |
| `docs/` 目录检查 | 当前仅保留 `Advanced_Optimization.md`、`Project_Plan.md`、`Proposal.md`、`Report.md` 与历史 `revision/` |
| 顶层 docs 搜索旧文件名 | 已确认现行顶层文档中不再引用 `docs/对比结果.md` 或 `docs/Phase4_From_Phase3_Development_Guide.md` |
| `get_errors` | `docs/Report.md`、`docs/Project_Plan.md`、`docs/Advanced_Optimization.md` 均无错误 |

---

## 影响评估

| 影响项 | 说明 |
| ------ | ---- |
| 正式结果归档 | Phase 4 核心结果已经进入 `docs/Report.md`，不再依赖单独的“对比结果”中间文档 |
| 文档角色清晰度 | `docs/Advanced_Optimization.md` 取代旧 Guide，改为面向当前成果的总结文档，而不是开发说明书 |
| 文档同步成本 | 现行 Phase 4 文档从三份并行收敛为“正式报告 + 进阶方案说明”两份主文档 |
| 计划文档一致性 | `docs/Project_Plan.md` 当前列出的现行文档结构已与仓库真实状态一致 |
| 历史记录完整性 | 旧变更过程没有被抹掉，而是通过本 revision 文档保留“为何合并、如何重命名、删除了什么”的解释 |

---

## 未修改的文件

- `docs/revision/Project_Plan_review_v13.md` / `v14.md`：保留上一轮修订上下文，不回写覆盖旧 revision 内容
- `docs/Proposal.md`：本轮未同步其任务完成度口径，只记录本次文档整合变更
- `src/`、`tests/`、`experiments/results/`：本轮不改代码与结果文件，只处理文档结构与写回关系

---

## 修订结论

本次修改属于 **Phase 4 文档体系收敛与正式归档修订**。完成后，Phase 4 的核心方法、重点结果与分析已经正式写回 `docs/Report.md`；旧的开发指导文档被改写并重命名为 `docs/Advanced_Optimization.md`；原 `docs/对比结果.md` 被吸收并删除。当前仓库中的现行文档结构因此更加清晰，也更符合“主体实验已完成、文档进入正式收口”这一项目状态。