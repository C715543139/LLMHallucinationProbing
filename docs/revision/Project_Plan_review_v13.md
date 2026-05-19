# Project_Plan 变更日志（v13）

## 变更日期

2026-05-19

## 变更动机

在 Linux + RTX 3090 环境下完成 Phase 4 复跑、NaN 根因定位、默认 dtype 切换、脚本结构重构之后，`docs/Project_Plan.md` 中关于实验环境、目录结构、入口脚本和 Phase 4 状态的描述已经明显落后于当前仓库真实状态。

尤其是以下几处偏差已经不能继续保留：

1. 文档仍将主实验环境写为 Windows + RTX 4060 + FP16；
2. 文档仍将 Phase 4 写为“待实现”，而实际代码、测试和结果都已落地；
3. 文档仍沿用旧的脚本目录与入口描述，没有反映 `main.py` 分发器与 `scripts/commands` / `scripts/run` 结构；
4. `docs/Phase4_From_Phase3_Development_Guide.md` 中真正采用的 Phase 4 方法尚未回写到正式计划文档。

因此，本次修订的目标是：

1. 将计划文档同步到当前 Linux + bfloat16 + eager 的真实运行环境；
2. 将当前仓库目录结构、脚本入口和模块清单同步到计划文档；
3. 按 Phase 2 / Phase 3 的既有描述风格，把当前实际采用的 Phase 4 方法路线、关键实现和核心结果写回正式计划文档；
4. 明确当前项目状态已经从“等待实现 Phase 4”转为“主体实验基本完成，剩余以工程与文档收尾为主”。

---

## 变更清单

### 1. `docs/Project_Plan.md` — 同步实验环境与项目现状

| 位置 | 变更内容 |
| ---- | -------- |
| §1.2 实验模型与数据 | 将模型说明改为当前稳定路径 `Qwen2-1.5B + bfloat16 + eager`，补充 6309 个有效样本与固定划分口径 |
| §1.3 实验协议 | 将“默认 float16”更新为“显式指定 torch dtype，当前主路径为 bfloat16；Phase 4 需要 eager attention” |
| §1.4 当前实现同步 | 将“Phase 4 尚未开始”改为“Phase 4 已完成核心实现与复跑”，补充 NaN 修复、CLI 重构和项目整体完成度说明 |
| §2 硬件与软件环境 | 将 Windows + RTX 4060 更新为 Linux + RTX 3090，并补充当前 Python / PyTorch / Transformers / sklearn 版本 |

### 2. `docs/Project_Plan.md` — 同步目录结构与脚本入口

| 位置 | 变更内容 |
| ---- | -------- |
| §3.1 当前真实目录结构 | 增加 `docs/对比结果.md`、`experiments/results/phase4/`、`scripts/commands/`、`scripts/run/`、`src/features/*attention*`、`src/methods/phase4_attention.py`、`tests/phase4/` 等真实目录 |
| §3.2 当前已实现模块 | 将 Phase 4 相关模块全部纳入“已实现”，并把 `main.py` 描述更新为“纯命令分发器” |
| §3.2 待实现模块 | 删除已不适用的旧 attention 计划项，只保留 FFN / MoE / DLLM 等后续扩展方向 |
| Phase 2 / 3 当前同步结果 | 将旧的 `main.py` 单体入口描述改为 `scripts/run/*.py` 主脚本 + `main.py` 分发兼容入口 |

### 3. `docs/Project_Plan.md` — 重写 Phase 4 正式计划段落

| 位置 | 变更内容 |
| ---- | -------- |
| Phase 4 标题 | 将“当前状态：待实现”改为“当前状态：已完成核心实验与实现” |
| Phase 4 任务表 | 按当前真实落地情况改写为 P4.1-P4.7：hidden baseline、anchor 抽取、attention score、去偏、head selection、attention output、A0-A9 消融与错误分析 |
| Phase 4 核心技术细节 | 按现有 Phase 2 / 3 风格写回实际采用的方法：`layer 17 + last` 基线、`13-20` 候选层、rule-based anchor、train-fit 长度残差化、top-16 head selection、A0-A9 消融 |
| Phase 4 当前同步结果 | 补充当前主结果：A0 全量 baseline、A6 最佳 Accuracy / Macro-F1、A8 最佳 AUROC，以及 `bfloat16 + eager` 解决 NaN 的结论 |
| Phase 4 结论 | 明确当前主体主线为 attention-based Phase 4，FFN / MoE / DLLM 不再属于本项目主体实验阻塞项 |

### 4. `docs/Project_Plan.md` — 更新环境搭建命令口径

| 位置 | 变更内容 |
| ---- | -------- |
| §4 环境搭建详细步骤 | 将 PowerShell 命令统一改为当前 Linux 环境下的 bash 命令 |
| §4.3 环境激活 | 更新为 `conda activate llm_hallucination` + `source ./.venv/bin/activate` |
| §4.4 模型下载 | 增加 `HF_ENDPOINT=https://hf-mirror.com` 镜像说明 |
| §4.5 数据下载 | 将示例改为 Linux 下的 `wget` + `unzip` |

---

## 影响评估

| 影响项 | 说明 |
| ------ | ---- |
| Phase 4 状态判断 | 计划文档不再把已完成的 attention 路线误写为待实现 |
| 文档一致性 | `Project_Plan.md`、当前代码目录、Phase 4 结果文件与对比结果文档重新对齐 |
| 环境复现 | 文档明确了当前真正稳定的运行路径是 `bfloat16 + eager`，避免再按旧的 float16 方案复跑 attention 特征 |
| 工程入口理解 | 计划文档已反映 `main.py` 分发器、`scripts/commands` 和 `scripts/run` 的当前结构，减少后续误读 |
| 后续工作边界 | 项目主体实验已可视为基本完成，后续重点转向 README、计划文档、脚本与答辩材料收尾 |

---

## 未修改的文件

- `docs/Phase4_From_Phase3_Development_Guide.md`：本轮将其核心实现路线吸收进 `Project_Plan.md`，但未改写原指导文档本身
- `docs/Report.md`：本轮不重复调整报告主体，只同步计划文档与 revision 说明
- `src/` 与 `tests/`：本轮不改代码实现，仅记录当前真实状态
- `experiments/results/phase4/`：本轮不重跑实验，直接引用现有 Phase 4 结果

---

## 修订结论

本次修改属于 **Phase 4 落地之后必须进行的计划文档同步修订**。完成后，`docs/Project_Plan.md` 已与当前实验环境、代码结构、脚本入口、Phase 4 方法路线和核心结果保持一致，能够真实反映项目已经从“等待实现进阶方法”进入“主体实验完成、工程与文档收尾”的阶段状态。