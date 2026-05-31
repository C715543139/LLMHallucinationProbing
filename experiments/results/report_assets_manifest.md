# Phase 5 报告/PPT 资产清单

> 生成日期: 2026-05-31
> 项目: LLM Hallucination Probing (利用大语言模型内部状态进行幻觉检测)
> 模型: Qwen2-1.5B (bfloat16 + eager attention)

---

## 一、新生成文件列表

### 1. Phase 4 标准统计图

| 文件 | 用途 | 建议放置位置 |
|------|------|-------------|
| `experiments/results/phase4/figures/layer_head_auroc_heatmap.png` | Layer x Head AUROC 热力图，展示各层各 head 的判别能力 | 报告 4.2 节 / PPT 方法对比页 |
| `experiments/results/phase4/figures/method_accuracy_comparison.png` | A0-A9 方法 Test Accuracy 对比柱状图 | 报告 4.3 节 / PPT 消融实验页 |
| `experiments/results/phase4/figures/method_auroc_comparison.png` | A0-A9 方法 Test AUROC 对比柱状图 | 报告 4.3 节 / PPT 消融实验页 |
| `experiments/results/phase4/figures/correction_matrix.png` | A9 Gated Fusion 修正矩阵 (Hidden vs Fusion) | 报告 4.4 节 / PPT 融合分析页 |

### 2. PPL 分布图

| 文件 | 用途 | 建议放置位置 |
|------|------|-------------|
| `experiments/results/baseline/ppl_score_distribution.csv` | 逐样本 PPL 分数 (train/val/test, 共 6309 条) | 数据附录 |
| `experiments/results/baseline/ppl_score_distribution.png` | Test 集 True/False 陈述 PPL 分布 (直方图 + KDE) | 报告 3.1 节 / PPT 基线对比页 |

### 3. Attention 案例可视化

| 文件 | 用途 | 建议放置位置 |
|------|------|-------------|
| `experiments/results/phase4/case_viz/true_correct_case_l16_h5_*.png` | True 陈述, hidden + A6 均正确 (3 张) | PPT 案例分析页 |
| `experiments/results/phase4/case_viz/false_correct_case_l16_h5_*.png` | False 陈述, hidden + A6 均正确 (3 张) | PPT 案例分析页 |
| `experiments/results/phase4/case_viz/hard_or_failure_case_l16_h5_*.png` | 困难/失败案例 (5 张, 含 hidden/A6 分歧) | PPT 案例分析页 |
| `experiments/results/phase4/case_viz/improvement_case_l16_h5_*.png` | A6 纠正 hidden-only 错误的案例 (3 张) | PPT 案例分析页 (重点展示) |
| `experiments/results/phase4/case_viz/cases.json` | 所有案例的结构化元数据 (14 条) | 数据附录 |

### 4. A6 对比分析

| 文件 | 用途 | 建议放置位置 |
|------|------|-------------|
| `experiments/results/phase4/a6_case_analysis.csv` | A6 vs Hidden-only 逐样本对比 (150 条 test subset) | 数据附录 |
| `experiments/results/phase4/a6_correction_matrix.json` | A6 修正矩阵及 net correction | 报告 4.4 节 |

### 5. 生成脚本

| 文件 | 用途 |
|------|------|
| `scripts/report_assets/generate_ppl_distribution.py` | 计算逐样本 PPL 分数并生成分布图 |
| `scripts/report_assets/generate_a6_analysis.py` | 基于 Phase 4 缓存生成 A6 对比分析 |
| `scripts/report_assets/generate_attention_cases.py` | 生成 attention 案例可视化 |

---

## 二、关键指标摘录

### A0-A9 消融结果 (Test Subset 150)

| Method | Accuracy | Macro-F1 | AUROC |
|--------|----------|----------|-------|
| A0 (Hidden-only L17, full) | 0.8082 | 0.8081 | 0.8897 |
| A0s (Hidden-only, subset) | 0.8667 | 0.8661 | 0.9184 |
| A1 (Attn-score raw) | 0.7667 | 0.7654 | 0.8289 |
| A2 (Attn-score debiased) | 0.7733 | 0.7713 | 0.8285 |
| A3 (Attn-score top-16) | 0.7600 | 0.7565 | 0.7934 |
| A4 (Attn-output only) | 0.6800 | 0.6800 | 0.7493 |
| A5 (Hidden + debiased attn) | 0.8733 | 0.8729 | 0.9302 |
| **A6 (Hidden + top-16 head)** | **0.8867** | **0.8865** | **0.9330** |
| A7 (Hidden + attn-output) | 0.8467 | 0.8463 | 0.9254 |
| A8 (Full fusion) | 0.8800 | 0.8798 | 0.9403 |
| A9 (Gated Fusion, tau=0.05) | 0.8667 | 0.8661 | 0.9193 |

### A6 vs Hidden-only 修正矩阵 (Test Subset 150)

|  | A6 Correct | A6 Wrong |
|--|-----------|----------|
| **Hidden Correct** | 129 | 1 |
| **Hidden Wrong** | 4 | 16 |

**Net Correction: +3** (4 个 hidden-wrong 被 A6 纠正, 1 个 hidden-correct 被 A6 误判)

### PPL 基线 (Test Set 631)

- Mean PPL (True): 77.23
- Mean PPL (False): 225.72
- AUROC: 0.6690
- Accuracy (threshold=232.43): 0.5261
- Macro-F1: 0.4160

### 最强 Attention Head

- **L16-H05** (val_auroc=0.6776, val_accuracy=0.6400)

---

## 三、注意事项

1. **子集限制**: Phase 4 融合结果 (A0s-A9) 基于 600/150/150 子集，不是全量结论。报告/PPT 中必须注明 "test subset n=150"。
2. **A9 修正矩阵为 0**: A9 Gated Fusion 在 150 样本子集上无实际修正 (net=0)，因为 hidden-only 在该子集上表现已足够好，tau=0.05 下无样本落入不确定区间。
3. **Attention 案例为定性展示**: 案例可视化用于直观理解 attention pattern，最终结论仍以 A0-A9 消融指标为准。
4. **PPL 分布重叠严重**: PPL 方法的 AUROC 仅 0.67，远低于 hidden-state 方法 (0.89-0.93)，说明序列概率不是可靠的幻觉检测信号。
5. **环境配置**: 所有生成使用 bfloat16 + eager attention，与 Phase 4 主实验一致。
6. **可复现性**: 所有生成脚本保存在 `scripts/report_assets/`，可重新运行以复现结果。
