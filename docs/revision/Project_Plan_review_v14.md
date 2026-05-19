# Project_Plan 变更日志（v14）

## 变更日期

2026-05-19

## 变更动机

在上一轮完成 `main.py` 分发器、Phase 4 主脚本与文档同步后，又出现了两类新的同步性问题：

1. 当前入口示例中仍残留错误的缓存目录示例 `experiments/results/phase4/1`，而真实缓存目录应为 `experiments/results/phase4/cache`；
2. Phase 1 的环境检查仍保留为 Windows PowerShell 脚本 `scripts/check_phase1.ps1`，与当前 Linux 主环境和 `scripts/commands/` 的命令组织方式不一致。

如果不修正，用户会继续参考错误的 cache 示例，或沿用已经不适配当前环境的旧检查脚本；同时，计划文档与结果文档也会再次落后于仓库真实状态。

因此，本次修订的目标是：

1. 修复 cache 示例路径，并检查当前仓库中是否还有同类错误引用；
2. 用 Linux 下可直接运行的命令脚本替换旧的 `check_phase1.ps1`；
3. 将新的环境检查入口接入 `main.py`；
4. 把这些变化同步写回现行文档与 revision 记录。

---

## 变更清单

### 1. 代码与脚本入口同步

| 位置 | 变更内容 |
| ---- | -------- |
| `scripts/commands/check_phase1.py` | 新增 Linux 版 Phase 1 环境检查脚本，覆盖依赖、CUDA、模型、数据、配置与可选 M1 前向传播检查 |
| `main.py` | 新增 `check-phase1` 命令入口，转发到 `scripts/commands/check_phase1.py` |
| `scripts/check_phase1.ps1` | 删除旧的 PowerShell 检查脚本，避免继续误用 |
| `main.py` / `scripts/run/phase4.py` | 将错误的 cache 示例 `experiments/results/phase4/1` 改为 `experiments/results/phase4/cache` |

### 2. 当前文档同步

| 位置 | 变更内容 |
| ---- | -------- |
| `docs/Project_Plan.md` | 将 `check_phase1.ps1` 的现行引用全部替换为 `scripts/commands/check_phase1.py`，并同步目录结构与 Phase 1 任务表 |
| `docs/对比结果.md` | 将旧的 `run_phase4_full.py` 入口改为当前 `scripts/run/phase4.py`，并补充正确的 cache 重跑参数说明 |
| `docs/Project_Plan.md` / `docs/对比结果.md` | 清理与当前仓库不一致的旧入口表述，保证用户看到的运行方式与仓库真实结构一致 |

### 3. 类似问题排查结果

本次对 `phase4/1`、`run_phase4_full` 与 `check_phase1.ps1` 做了全仓库搜索，结果如下：

1. 错误的 cache 示例只出现在现行代码入口说明中，已全部修复；
2. 现行文档中的 `run_phase4_full.py` 引用已全部替换为 `scripts/run/phase4.py`；
3. `check_phase1.ps1` 在现行文档中的引用已全部替换；
4. 剩余 `check_phase1.ps1` 仅出现在历史 revision 文档 `Project_Plan_review_v8.md` 中，这属于当时版本的真实记录，因此本次未改写历史记录。

---

## 验证结果

| 验证项 | 结果 |
| ------ | ---- |
| `python -s main.py check-phase1 --help` | 通过 |
| `python -s main.py phase4 --help` | 通过 |
| `python -s main.py check-phase1` | 通过，当前环境下返回 32 项通过、0 警告、0 失败 |
| `python -m py_compile main.py scripts/commands/check_phase1.py scripts/run/phase4.py` | 通过 |

---

## 影响评估

| 影响项 | 说明 |
| ------ | ---- |
| Cache 重跑可用性 | 用户不再会把 `experiments/results/phase4/1` 当成缓存目录示例 |
| 环境检查入口 | Phase 1 环境检查已统一迁移到 Linux 可执行命令脚本，并纳入 `main.py` 主入口 |
| 文档一致性 | `Project_Plan.md`、`对比结果.md` 与当前脚本结构重新对齐 |
| 历史记录完整性 | 历史 revision 文档保留原貌，仅修正现行文档和当前代码路径 |

---

## 未修改的文件

- `docs/revision/Project_Plan_review_v8.md`：保留历史记录，不因后续脚本重构而重写旧版本修订说明
- `docs/Report.md`：本轮未调整报告主体，只同步当前使用说明与计划文档
- `tests/phase1/`：本轮未改测试用例，环境检查脚本只是把现有检查逻辑迁移为 Linux 命令形式

---

## 修订结论

本次修改属于 **脚本入口与使用文档的一致性修补**。完成后，Phase 4 cache 示例、Phase 1 环境检查入口与现行文档说明都已经与当前 Linux 工作流保持一致；剩余旧引用仅存在于历史 revision 文档中，不影响当前代码与文档使用。