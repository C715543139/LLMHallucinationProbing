# Phase 5 报告/PPT 资产清单

> 生成日期: 2026-05-31
> 项目: LLM Hallucination Probing (利用大语言模型内部状态进行幻觉检测)
> 模型: Qwen2-1.5B (bfloat16 + eager attention)

---

## 一、当前生成文件列表

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
| `experiments/results/phase4/case_viz/true_correct_case_l15_h6_*.png` | True 陈述代表案例 | PPT 案例分析页 |
| `experiments/results/phase4/case_viz/false_correct_case_l15_h6_*.png` | False 陈述代表案例 | PPT 案例分析页 |
| `experiments/results/phase4/case_viz/hard_or_failure_case_l15_h6_*.png` | 困难/失败案例 | PPT 案例分析页 |
| `experiments/results/phase4/case_viz/improvement_case_l15_h6_*.png` | attention 路径改善 hidden-only 错误的案例 | PPT 案例分析页 |
| `experiments/results/phase4/case_viz/cases.json` | 所有案例的结构化元数据 (14 条) | 数据附录 |

### 4. A9 对比分析

| 文件 | 用途 | 建议放置位置 |
|------|------|-------------|
| `experiments/results/phase4/phase4_ablation_results.json` | A0-A9 全量消融主结果，含 A9 correction matrix | 数据附录 |
| `experiments/results/phase4/a9_correction_matrix.json` | A9 修正矩阵及 net correction | 报告 5.5 节 |
| `experiments/results/phase4/phase4_error_analysis.csv` | A9 vs Hidden-only 逐样本错误分析 | 数据附录 |

### 5. 生成脚本

| 文件 | 用途 |
|------|------|
| `scripts/report_assets/generate_ppl_distribution.py` | 计算逐样本 PPL 分数并生成分布图 |
| `scripts/report_assets/generate_a6_analysis.py` | 历史 A6 诊断脚本；当前报告主结论不再引用其输出 |
| `scripts/report_assets/generate_attention_cases.py` | 生成 attention 案例可视化 |

---

## 二、关键指标摘录

### A0-A9 消融结果 (Full Test Set 631)

| Method | Accuracy | Macro-F1 | AUROC |
|--------|----------|----------|-------|
| A0 (Hidden-only L17, full) | 0.8082 | 0.8081 | 0.8897 |
| A0s (Hidden-only, attention-aligned) | 0.8082 | 0.8081 | 0.8897 |
| A1 (Attn-score raw) | 0.8146 | 0.8146 | 0.9003 |
| **A2 (Attn-score debiased)** | **0.8193** | **0.8193** | **0.9010** |
| A3 (Attn-score top-16) | 0.7100 | 0.7099 | 0.8072 |
| A4 (Attn-output only) | 0.6751 | 0.6751 | 0.7210 |
| A5 (Hidden + debiased attn) | 0.8051 | 0.8050 | 0.8963 |
| A6 (Hidden + top-16 head) | 0.8003 | 0.8003 | 0.8820 |
| A7 (Hidden + attn-output) | 0.8114 | 0.8113 | 0.8906 |
| A8 (Hidden + top-head + output) | 0.7971 | 0.7971 | 0.8843 |
| A9 (Gated Fusion, tau=0.15) | 0.8130 | 0.8129 | 0.8903 |

### A9 vs Hidden-only 修正矩阵 (Full Test Set 631)

|  | A9 Correct | A9 Wrong |
|--|-----------|----------|
| **Hidden Correct** | 509 | 1 |
| **Hidden Wrong** | 4 | 117 |

**Net Correction: +3** (4 个 hidden-wrong 被 A9 纠正, 1 个 hidden-correct 被 A9 误判)

### PPL 基线 (Test Set 631)

- Mean PPL (True): 77.23
- Mean PPL (False): 225.72
- AUROC: 0.6784
- Accuracy (threshold=232.43): 0.5293
- Macro-F1: 0.4180

### 最强 Attention Head

- **L15-H06** (val_auroc=0.6527)

---

## 三、注意事项

1. **全量口径**: Phase 4 主结果基于固定 5,047/631/631 train/validation/test split，A0-A9 均按 full test set 汇报。
2. **A2 结论范围**: A2 是 A0-A9 attention-guided 消融中的最优方法；Phase 3 中部分 hidden-state layer probe 的 accuracy 更高，应分开表述。
3. **Attention 案例为定性展示**: 案例可视化用于直观理解 attention pattern，最终结论仍以 A0-A9 消融指标为准。
4. **PPL 分布重叠严重**: PPL 方法的 AUROC 约 0.68，远低于 hidden-state 与 attention-guided 方法，说明序列概率不是可靠的幻觉检测信号。
5. **环境配置**: 所有生成使用 bfloat16 + eager attention，与 Phase 4 主实验一致。
6. **可复现性**: 所有生成脚本保存在 `scripts/report_assets/`，可重新运行以复现结果。
