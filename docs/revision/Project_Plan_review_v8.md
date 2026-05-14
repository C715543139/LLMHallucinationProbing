# Project_Plan 变更日志（v8）

## 变更日期

2026-05-14

## 变更动机

Phase 1 全部任务已完成并验证通过。为保证各阶段交付物的可核查性，补充了测试套件（pytest）与端到端检验脚本，并将相关依赖、目录结构和任务列表同步写入计划文档。同时修复了 `src/models/__init__.py` 中遗留的幽灵导入。

---

## 变更清单

### 1. `pyproject.toml` — 项目依赖

| 位置                | 变更内容                                                       |
| ------------------- | -------------------------------------------------------------- |
| `dependencies` 末尾 | 新增 `"pytest>=8.0,<9.0"` 和 `"pytest-cov>=5.0,<7.0"` 两条依赖 |

执行 `uv sync` 后实际安装版本：`pytest==8.4.2`、`pytest-cov==6.3.0`。

### 2. `tests/` — 测试套件（新建）

```
tests/
├── __init__.py
├── conftest.py              # pytest 公共 Fixture（路径、标记注册）
└── phase1/
    ├── __init__.py
    ├── test_config.py       # P1.8：全局配置模块（26 项）
    ├── test_data.py         # P1.4/P1.5/P1.7：数据加载、Dataset、预处理输出（55 项）
    └── test_model.py        # P1.2/P1.3/P1.6/M1：设备信息、权重完整性、模型加载与前向传播（22 项）
```

**测试标记体系**：

| 标记       | 含义                                 | 运行方式                       |
| ---------- | ------------------------------------ | ------------------------------ |
| _(无标记)_ | 快速单元测试，无 GPU 依赖            | `pytest tests/ -m "not model"` |
| `gpu`      | 需要 CUDA GPU                        | `pytest tests/ -m gpu`         |
| `model`    | 需要本地 Qwen2-1.5B 权重（隐含 gpu） | `pytest tests/ -m model`       |
| `slow`     | 运行时间 >30 s                       | 同 model                       |

**测试结果（Phase 1 验收）**：

| 运行模式                  | 通过    | 失败  | 跳过  |
| ------------------------- | ------- | ----- | ----- |
| `not model`（快速）       | 90      | 0     | 13    |
| `model`（含 M1 前向传播） | 13      | 0     | 0     |
| **合计**                  | **103** | **0** | **0** |

### 3. `scripts/check_phase1.ps1` — 端到端检验脚本（新建）

逐项检验 P1.1–P1.8 所有交付物，输出通过/警告/失败计数。

用法：

```powershell
# 快速检验（不加载模型）
.\scripts\check_phase1.ps1

# 含 M1 前向传播检验（需 GPU + 本地权重）
.\scripts\check_phase1.ps1 -IncludeModel
```

### 4. `src/models/__init__.py` — Bug 修复

移除对 `load_model_4bit` 的导入（该函数已在 v7 中随 4-bit 支持一并删除，但 `__init__.py` 未同步更新，导致模块导入时抛出 `ImportError`）。

### 5. `docs/Project_Plan.md` — 计划文档同步

| 位置            | 变更内容                                                                                 |
| --------------- | ---------------------------------------------------------------------------------------- |
| §2.3 软件环境表 | 新增 `pytest 8.0+` 和 `pytest-cov 5.0+` 两行                                             |
| §3 目录结构     | 新增 `tests/` 完整子树；`scripts/` 下新增 `check_phase1.ps1` 条目                        |
| §4.3 依赖清单表 | 新增"测试"类别行，列出 `pytest`、`pytest-cov`                                            |
| Phase 1 任务表  | 新增 P1.9：编写 Phase 1 验证测试，输出物为 `tests/phase1/` 与 `scripts/check_phase1.ps1` |

---

## 影响评估

| 影响项   | 说明                                                                                                      |
| -------- | --------------------------------------------------------------------------------------------------------- |
| 依赖体积 | 新增 pytest、pytest-cov、coverage、pluggy、iniconfig 共 5 个轻量包，不影响推理环境                        |
| 可复现性 | 任何人 clone 仓库后可用 `pytest tests/ -m "not model"` 在 ~9 s 内验证 Phase 1 基础交付物                  |
| 后续阶段 | Phase 2–4 只需在 `tests/phase2/`、`tests/phase3/`、`tests/phase4/` 下按相同模式新增测试，无需改动现有结构 |
| 修复影响 | `src/models/__init__.py` 的修复消除了 `import src.models` 时的隐性错误，不影响已有实验逻辑                |

---

## 未修改的文件

- `docs/revision/Project_Plan_review_v1.md` ~ `v7.md`：历史审查记录，保留原样
- `src/config.py`、`src/data/`、`src/models/loader.py`：生产代码未做任何改动
