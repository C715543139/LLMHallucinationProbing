# Project_Plan 变更日志（v7）

## 变更日期

2026-05-14

## 变更动机

由于无法获取 Llama-2-7B 模型访问权限，项目决定仅使用 **Qwen2-1.5B（FP16）** 作为唯一实验模型。删除所有与 Llama 模型、4-bit 量化及 bitsandbytes 相关的规划与代码。

---

## 变更清单

### 1. `docs/Project_Plan.md` — 计划文档

| 位置 | 变更内容 |
|------|---------|
| §1.2 实验模型与数据 | 精简表格，移除 Llama-2-7B 行及"优先/保底"双模型策略，仅保留 Qwen2-1.5B |
| §2.2 显存预算分析 | 移除 Llama 三行（FP16/8-bit/4-bit），仅保留 Qwen2-1.5B 数据；重写结论 |
| §2.3 软件环境 | 移除 `bitsandbytes` 行 |
| §3 目录结构 | `loader.py` 描述从"FP16 主路径，兼容可选 4-bit 量化"改为"FP16" |
| §4.3 依赖说明 | 移除 bitsandbytes 相关描述；从依赖清单中移除"可选：4-bit 量化"行 |
| §4.4 下载模型 | 移除 Llama-2-7B 下载命令及相关注释 |
| Phase 1 任务表 | P1.3 仅下载 Qwen2-1.5B；P1.6 简化为"支持 FP16 的模型加载器" |
| 最小可交付版本 | 移除"若环境条件允许，再使用 Llama-2-7B 4-bit 补充验证" |
| §6 关键风险与对策 | 移除 Llama 访问权限、4-bit 精度下降、bitsandbytes 兼容性三条风险 |
| §6 Windows + bitsandbytes | 整个小节删除 |

### 2. `src/config.py` — 全局配置

- 移除 `ModelConfig` 中的 `secondary_*` 字段（`secondary_name`、`secondary_local`、`secondary_load_in_4bit`、`secondary_dtype`、`secondary_device_map`）

### 3. `src/models/loader.py` — 模型加载模块

- 更新模块文档字符串，移除 Llama 及 bitsandbytes 相关描述
- 删除 `_check_bitsandbytes()` 函数
- 删除 `load_model_4bit()` 函数
- 简化 `load_model()` 统一入口：移除 `use_4bit` 参数，直接委托 `load_model_fp16()`

### 4. `pyproject.toml` — 项目依赖

- 从 `dependencies` 中移除 `"bitsandbytes"`（含注释行）
- 从 `[tool.uv.sources]` 中移除 `bitsandbytes` 的 Windows 社区版 wheel 源

### 5. 未修改的文件

- `docs/revision/Project_Plan_review_v1.md` ~ `v6.md`：历史审查记录，保留原样不做修改

---

## 影响评估

| 影响项 | 说明 |
|--------|------|
| 实验模型 | 仅 Qwen2-1.5B，无多模型对比 |
| 依赖数量 | 减少（移除 bitsandbytes），安装更简单 |
| 代码复杂度 | 降低（移除 4-bit 加载路径和条件分支） |
| 报告内容 | 无需讨论 Llama vs Qwen2 模型对比 |
