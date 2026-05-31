# ACL Report Outline

Working title: **Probing Hallucination Signals in Large Language Model Internal States**

Target format: ACL template, English only, 8-9 pages for the main paper body. References and appendices are not counted by the stated course requirement.

## Core Message

This project studies whether hallucinated and factual answers can be separated using signals inside an LLM, rather than relying only on output-level uncertainty. The report should argue three points:

1. Perplexity is a useful but limited baseline for hallucination detection.
2. Hidden-state probing, especially SAPLMA-style classifiers on late/middle-layer representations, provides substantially stronger detection performance.
3. Attention-guided features add interpretable complementary signals, with the strongest subset experiment improving both accuracy and error correction over the hidden-state-only baseline.

## Recommended Page Budget

| Section | Target length | Purpose |
| --- | ---: | --- |
| Abstract | 150-200 words | State problem, method family, strongest metrics, and main finding. |
| 1 Introduction | 0.8-1.0 page | Motivate hallucination detection from internal states and state contributions. |
| 2 Related Work | 0.7-0.9 page | Briefly position hallucination detection, uncertainty scoring, probing, and attention analysis. |
| 3 Task and Experimental Setup | 1.0-1.2 pages | Define task, dataset, split, model, features, metrics, and reproducibility setup. |
| 4 Methods | 1.5-1.8 pages | Explain PPL, SAPLMA hidden-state probing, layer/token analysis, and attention-guided extensions. |
| 5 Results and Analysis | 2.5-2.8 pages | Present main quantitative results, figures, ablations, and case analysis. |
| 6 Discussion | 0.7-0.9 page | Interpret why hidden states and attention work; discuss failure modes. |
| 7 Conclusion | 0.3-0.4 page | Summarize findings and practical lessons. |
| Limitations | 0.4-0.5 page | ACL-style limitations section before references. |

Expected total: about 8.0-9.0 pages, excluding references and appendices.

## Paper Structure

### Abstract

Write one compact paragraph covering:

- The task: binary hallucination detection for LLM responses.
- The compared signals: perplexity, hidden states, token/layer choices, attention score/output features.
- The strongest full-data result: SAPLMA MLP reaches 77.71% accuracy and 0.8770 AUROC; the validation-selected layer 17 reaches 80.03% accuracy and 0.8876 AUROC.
- The strongest attention subset result: hidden + top-16 head attention reaches 88.67% accuracy and 0.9330 AUROC; hidden + top-head + output reaches 0.9403 AUROC.
- The conclusion: internal states are more reliable than output perplexity, while attention features provide additional interpretable evidence.

### 1 Introduction

Suggested flow:

1. Define hallucination as a mismatch between fluent generation and factual correctness.
2. Explain why output-level confidence or perplexity is insufficient: false statements can still be fluent and high-probability.
3. Introduce the hypothesis: factuality leaves measurable traces in internal representations.
4. State the project scope: binary classification over labeled true/false examples, using Qwen2-1.5B internal states.
5. List contributions:
   - Compare PPL against SAPLMA-style hidden-state classifiers.
   - Analyze layer depth and token pooling choices.
   - Add attention-guided features and evaluate their incremental value.
   - Provide case-level visualization for attention-based correction.

### 2 Related Work

Keep this section short and selective. Use 2-3 paragraphs:

- Hallucination detection and factuality evaluation.
- Uncertainty and likelihood-based detection, including perplexity-style baselines.
- Internal representation probing for truthfulness or factual consistency.
- Attention analysis and interpretability, with a careful note that attention is not a complete explanation but can still be a useful diagnostic feature.

Avoid turning this into a broad survey. Save space for results.

### 3 Task and Experimental Setup

#### 3.1 Task Definition

Define the binary task:

- Input: a prompt/claim/response example from the true-false dataset.
- Label: factual vs hallucinated.
- Output: a binary prediction and a continuous score for AUROC.

#### 3.2 Dataset

Report:

- Total examples: 6,309.
- Split: 5,047 train, 631 validation, 631 test.
- Label balance and domain composition if available in the final dataset statistics.
- Any preprocessing used before feature extraction.

If detailed domain statistics are long, place the full table in Appendix A.

#### 3.3 Model and Feature Extraction

Report:

- Base model: Qwen2-1.5B.
- Hidden states used by SAPLMA-style probes.
- Layer-wise extraction and token pooling variants.
- Attention features used in Phase 4.
- Practical execution environment, including GPU and precision, if relevant.

#### 3.4 Metrics

Use accuracy, macro-F1, and AUROC. Briefly justify AUROC as important because the detector produces continuous scores and threshold choice can vary by deployment setting.

### 4 Methods

#### 4.1 Perplexity Baseline

Describe PPL as an output likelihood baseline. Explain expected behavior: hallucinated text may have higher PPL on average, but likelihood alone does not directly measure factuality.

#### 4.2 Hidden-State Probing

Describe the SAPLMA-style classifier:

- Extract internal hidden states from the frozen LLM.
- Train lightweight classifiers, including logistic regression and MLP.
- Evaluate on held-out validation/test splits.

Make clear that the LLM itself is not fine-tuned.

#### 4.3 Layer and Token-Pooling Analysis

Explain:

- Per-layer probes are trained/evaluated to locate factuality-sensitive layers.
- Token pooling compares last-token, mean-token, and first-token representations.
- Validation-selected layer 17 must be described as validation-selected, not retrospectively selected from test performance.

#### 4.4 Attention-Guided Extensions

Summarize the Phase 4 variants compactly:

- Attention score features from selected heads.
- Length residualization or control features if used.
- Top-head selection by validation AUROC.
- Attention output features.
- Fusion variants A0-A9.

Use one paragraph plus a compact table of method variants if space allows. Put implementation details and full variant definitions in Appendix C.

### 5 Results and Analysis

This should be the central section of the paper.

#### 5.1 Main Baseline Comparison

Main table:

| Method | Accuracy | Macro-F1 | AUROC |
| --- | ---: | ---: | ---: |
| PPL baseline | 52.93 | 41.80 | 67.84 |
| SAPLMA LR | 74.96 | 74.96 | 82.65 |
| SAPLMA MLP | 77.71 | 77.69 | 87.70 |

Main claim: hidden-state probes clearly outperform PPL across all metrics.

Recommended figure: PPL score distribution. Use it to explain that PPL has a visible but overlapping separation between true and false examples.

#### 5.2 Layer-Wise Hidden-State Analysis

Report key layer results:

| Layer | Accuracy | AUROC | Interpretation |
| --- | ---: | ---: | --- |
| 0 | 48.49 | 50.10 | Near-random lexical/input-level signal. |
| 13 | 82.25 | 89.51 | Strong middle-layer factuality signal. |
| 15 | 82.88 | 91.03 | Highest reported layer-wise AUROC. |
| 17 | 80.03 | 88.76 | Validation-selected best layer. |
| 20 | 82.88 | 90.40 | Strong late-middle representation. |
| 27 | 74.96 | 82.65 | Final-layer/default SAPLMA setting. |

Recommended figure: layer accuracy/AUROC curve.

Important wording: do not claim layer 17 is the absolute best test layer. Claim that it is the validation-selected operating point and remains strong on test.

#### 5.3 Token Pooling Analysis

Report:

| Pooling | Accuracy | AUROC |
| --- | ---: | ---: |
| Last token | 74.96 | 82.65 |
| Mean token | 70.52 | 76.36 |
| First token | 42.00 | 37.94 |

Main interpretation:

- Last-token representations are strongest because they aggregate preceding context before answer scoring.
- First-token representations are weak because they contain little response-specific factuality information.
- Mean pooling can dilute localized factuality cues.

#### 5.4 Attention Feature Ablation

Use a compact table for the subset experiment:

| Variant | Feature family | Accuracy | Macro-F1 | AUROC |
| --- | --- | ---: | ---: | ---: |
| A0s | Hidden only | 86.67 | 86.61 | 91.84 |
| A6 | Hidden + top-16 head attention | 88.67 | 88.65 | 93.30 |
| A8 | Hidden + top-head + attention output | 88.00 | 87.98 | 94.03 |
| A9 | Gated fusion | Report compactly | Report compactly | Report compactly |

Main interpretation:

- A6 is the strongest accuracy/F1 configuration in the subset experiment.
- A8 reaches the strongest AUROC, suggesting better ranking even when its thresholded accuracy is slightly lower than A6.
- Attention should be presented as complementary evidence, not as a replacement for hidden states.

Recommended figures:

- Method accuracy comparison.
- Method AUROC comparison, if page space allows.
- Layer-head AUROC heatmap, likely in appendix unless the main text needs the interpretability story.

#### 5.5 Case-Level Error Correction and Visualization

Use A6 case analysis:

| Category | Count |
| --- | ---: |
| Hidden correct, A6 correct | 129 |
| Hidden correct, A6 wrong | 1 |
| Hidden wrong, A6 correct | 4 |
| Hidden wrong, A6 wrong | 16 |

Main claim: A6 gives a net correction of +3 examples over the hidden-only subset baseline.

Recommended main figure:

- A6 correction matrix plus one representative improvement case.

Move the remaining attention cases to Appendix C or D.

### 6 Discussion

Organize around three explanations:

1. **Why PPL is limited.** Likelihood measures fluency and model fit, not factual correctness.
2. **Why hidden states work.** Intermediate and late representations encode semantic and factual consistency signals more directly than output probability alone.
3. **Why attention helps.** Selected heads can capture useful alignment between response tokens and factual anchors, but attention remains a partial and noisy explanation.

Also discuss practical lessons:

- Validation-based model/layer selection is important.
- AUROC and accuracy can favor different variants.
- Subset attention experiments should not be overstated as full-data conclusions.

### 7 Conclusion

Write one short paragraph:

- Reiterate that hidden-state probing substantially improves hallucination detection over PPL.
- Mention the strongest evidence from layer/token analysis.
- State that attention-guided features provide additional interpretable gains in the subset setting.
- End with a future direction: larger models, broader datasets, and more robust attention/activation fusion.

### Limitations

ACL papers normally include a limitations section before references. Include:

- Single base model family and model size.
- Binary true/false setup may simplify real-world hallucination.
- Attention experiments are subset-based and should be validated on larger samples.
- Attention visualization is diagnostic, not causal proof.
- Results may depend on dataset construction and feature extraction choices.

## Figures and Tables Plan

Main paper should include at most 4-5 visual/table blocks to fit 8-9 pages:

| Slot | Type | Content | Suggested location |
| --- | --- | --- | --- |
| Table 1 | Setup | Dataset split, model, metrics | Section 3 |
| Table 2 | Main results | PPL vs SAPLMA LR/MLP | Section 5.1 |
| Figure 1 | Plot | PPL score distribution | Section 5.1 |
| Figure 2 | Plot | Layer-wise performance, optionally combined with token pooling | Sections 5.2-5.3 |
| Table 3 | Ablation | A0s/A6/A8/A9 subset comparison | Section 5.4 |
| Figure 3 | Visualization | A6 correction matrix and one improvement case | Section 5.5 |

If space is tight, move Figure 1 or detailed attention visualization to the appendix and keep the quantitative tables in the main paper.

## Appendix Plan

Appendix A: Dataset and preprocessing details.

Appendix B: Full Phase 2 and Phase 3 result tables, including all tested layers and token pooling variants.

Appendix C: Full Phase 4 ablation details, attention head selection, layer-head heatmap, and all method variants A0-A9.

Appendix D: Case visualizations, including true, false, hard, and improvement examples.

Appendix E: Reproducibility notes, commands, environment, and generated asset manifest.

## Mapping to Course Requirements

| Course requirement | Where to answer it |
| --- | --- |
| Simple task: use SAPLMA and evaluate layer-level hallucination signal | Sections 4.2, 4.3, 5.1, 5.2 |
| Analyze whether PPL and SAPLMA work, and explain why | Sections 5.1, 5.3, 6 |
| Advanced task: design an improved method using attention/output information | Sections 4.4, 5.4, 5.5 |
| Provide implementation and experiment evidence | Sections 3, 4, 5 and Appendices B-E |
| Discuss limitations and future work | Limitations, Section 7 |

## LaTeX Skeleton

```latex
\begin{abstract}
...
\end{abstract}

\section{Introduction}
...

\section{Related Work}
...

\section{Task and Experimental Setup}
...

\section{Methods}
...

\section{Results and Analysis}
...

\section{Discussion}
...

\section{Conclusion}
...

\section*{Limitations}
...

\bibliography{custom}

\appendix
\section{Dataset and Preprocessing Details}
...
\section{Full Experimental Results}
...
\section{Attention Ablations and Case Visualizations}
...
\section{Reproducibility Notes}
...
```

## Writing Notes

- Use exact metrics from `scripts/show_results.py` as the source of truth.
- Keep all subset results explicitly labeled as subset results.
- Do not describe A6/A8 as full-dataset improvements unless full-dataset attention experiments are later added.
- Use "validation-selected layer 17" instead of "best layer" when referring to the selected operating point.
- Keep implementation file lists, command logs, and long tables in the appendix.
- Use concise ACL-style prose: short claims followed immediately by the table, figure, or metric that supports them.
