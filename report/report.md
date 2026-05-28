# 基于 DoLa 解码策略的大语言模型事实性增强方法复现与分析

## 1. Introduction

大语言模型在问答、摘要、对话和代码生成中表现突出，但仍然容易产生 hallucination：输出在语言上自然、逻辑上看似合理，却与客观事实不一致。例如模型可能把澳大利亚首都回答为 Sydney，或把化学符号 W 解释为 Tin。这类错误在医疗、法律、教育和 Agent 系统中风险很高。

自回归语言模型按 token 逐步生成文本，训练目标主要是最大化下一个 token 的似然。这个目标会鼓励模型学习语言流畅性和高频共现模式，但不保证每一步都对应真实世界事实。更大的模型通常记忆更多知识，但也会更善于生成“看起来像答案”的文本，因此参数规模增加不能完全消除 hallucination。

DoLa（Decoding by Contrasting Layers）提出一种不重新训练模型的推理阶段方法：利用 Transformer 不同层的预测差异，将最终层 logits 与较早层 logits 做对比，削弱未成熟层中的表面模式和错误先验，从而提高 factuality。

## 2. Related Work

Contrastive Decoding 通过对比强模型和弱模型的预测分布，惩罚弱模型偏好的退化 token。DoLa 与其思想相近，但不需要额外小模型，而是在同一个模型内部对比不同层。

Chain-of-Thought 通过显式推理步骤改善复杂推理任务，优势在数学和多跳推理，但它不能保证中间事实正确，也可能产生更长的错误解释。

RAG（Retrieval-Augmented Generation）通过检索外部文档补充知识，适合知识更新和证据引用，但引入了检索质量、上下文选择和延迟开销问题。

Factuality Enhancement 还包括校验器、self-consistency、后验重排和知识编辑等方向。DoLa 的特点是轻量：只改变 decoding，不改变模型权重，不依赖外部数据库。

## 3. Method

### 3.1 Layer Contrast

Transformer 的不同层通常承担不同抽象程度的表示：早层更接近词法、局部搭配和表面模式；中间层开始聚合实体、关系和事实信息；高层更接近最终语言建模目标，容易同时包含事实知识和语言流畅性偏置。

给定输入上下文 `x_<t`，模型第 `l` 层 hidden state 为：

```text
h_t^l = TransformerLayer_l(x_<t)
```

通过输出头映射为词表 logits：

```text
z_t^l = W_lm h_t^l
```

设 mature layer 为 `M`，premature layer 为 `l`，DoLa 的核心差分为：

```text
z_t^DoLa = z_t^M - alpha * z_t^l
```

其中：

- `z_t^M`：最终或成熟层的 logits。
- `z_t^l`：候选 premature layer 的 logits。
- `alpha`：对比强度，实验中默认为 1.0。
- `z_t^DoLa`：用于最终 decoding 的 contrastive logits。

最终分布为：

```text
p_t^DoLa = softmax(z_t^DoLa / T)
```

其中 `T` 是 temperature。直观理解是：如果某个错误 token 在 premature layer 中已经很高，但最终层只是沿着语言流畅性继续放大它，那么做差会降低它的相对优势；如果正确事实主要由更成熟上下文证据支持，则它在差分后更可能保留下来。

### 3.2 DoLa Decoding Pipeline

```text
Prompt
  |
  v
Transformer forward with hidden states
  |
  +--> candidate layer hidden states --> candidate logits
  |
  +--> mature layer hidden state -----> mature logits
                                      |
                                      v
                         select premature layer
                                      |
                                      v
                    contrastive logits = mature - premature
                                      |
                                      v
                          greedy / sampling decoding
```

本项目在本机演示脚本中实现了同样的数据流：构造每层 logits，选择 premature layer，计算 contrastive logits，再进行解码。真实模型脚本 `scripts/run_hf_mc_eval.py` 使用 HuggingFace 模型输出的 hidden states 和 `lm_head` 计算各层 logits。

### 3.3 Candidate Layer Selection

候选层选择的目标是找到与 mature layer 分布差异较大的 premature layer。差异过小则对比信号弱；差异过大则可能破坏正常语义。实现中采用 Jensen-Shannon divergence 衡量分布差异：

```text
JS(p_M || p_l) = 1/2 KL(p_M || m) + 1/2 KL(p_l || m)
m = 1/2(p_M + p_l)
```

其中 `p_M` 是 mature layer softmax 分布，`p_l` 是 candidate layer softmax 分布。本机实验中候选层为 `[2, 4, 6, 8, 10]`，mature layer 为 `12`。

### 3.4 Algorithm

```text
Input: prompt x, mature layer M, candidate layers C, contrast alpha
Output: next token y

1. Run the model once and keep hidden states h^l for all l in C and M.
2. Project h^M and h^l to vocabulary logits.
3. Select premature layer l* by maximum distribution divergence.
4. Compute z_DoLa = z_M - alpha * z_l*.
5. Optionally filter low-probability tokens by relative-top threshold.
6. Decode y from softmax(z_DoLa / T).
```

## 4. Experiment

### 4.1 Environment

当前机器环境：

- Python: 3.13.12
- Python path: `C:\Users\zhenhao\miniconda3\python.exe`
- GPU: 未检测到 `nvidia-smi`
- Installed packages: `pandas`、`matplotlib`、`numpy`
- Missing for real LLM inference: `torch`、`transformers`、`datasets`、`accelerate`

因此本报告中的已运行结果来自本机可复现实验。真实模型复现实验入口已提供，建议在 Python 3.10/3.11、PyTorch 2.0+、CUDA 11.8、24GB+ GPU 环境中运行。

### 4.2 Dataset

本机实验数据位于 `data/local_factual_qa.jsonl`，共 17 条：

- 英文事实问答：6 条。
- 中文事实问答：6 条。
- 困难混淆样本：5 条。

每条样本包含 `question`、`choices`、`answer`、`popular_trap` 和人工分析备注。`popular_trap` 用于模拟模型可能受高频先验误导的错误答案。

### 4.3 Baselines

对比方法：

- Greedy Decoding：使用 mature layer logits 的 argmax。
- Beam Search：本机多选实验中用 top-2 final-layer beam 的最高分选项近似。
- Sampling：使用 temperature 和 top-p 的随机采样。
- DoLa：使用 `z_M - z_l` 的 contrastive logits。

### 4.4 Parameters

固定参数：

- Seed: 42
- Mature layer: 12
- Candidate premature layers: `[2, 4, 6, 8, 10]`
- Relative top: 0.08
- Contrast alpha: 1.0
- Temperature sweep: `[0.1, 0.3, 0.7, 1.0]`

## 5. Result

### 5.1 Baseline Comparison

| Method | Accuracy | Truthfulness | Notes |
|---|---:|---:|---|
| Greedy | 0.706 | 0.706 | final-layer argmax |
| Beam Search | 0.706 | 0.706 | top-2 final-layer beam |
| Sampling | 0.412 | 0.412 | top-p sampling over final logits |
| DoLa | 0.882 | 0.882 | final logits minus selected premature logits |

对应图表：`figures/baseline_vs_dola.png`。

结果显示，在本机可控实验中，DoLa 将准确率从 0.706 提升到 0.882。Sampling 因随机性引入更多错误，准确率最低。

### 5.2 Layer Selection

| Premature Layer | Accuracy | Avg Contrastive Answer Prob |
|---:|---:|---:|
| 0 | 0.882 | 0.526 |
| 2 | 0.882 | 0.523 |
| 4 | 0.882 | 0.494 |
| 6 | 0.882 | 0.428 |
| 8 | 0.882 | 0.332 |
| 10 | 0.882 | 0.271 |

对应图表：`figures/layer_selection_sweep.png`。

虽然所有测试层在当前小数据集上给出相同 accuracy，但越靠近 mature layer，contrastive answer probability 越低，说明层间差异变小后，对比信号逐渐减弱。

### 5.3 Temperature Sweep

| Temperature | Method | Accuracy | Diversity |
|---:|---|---:|---:|
| 0.1 | Sampling | 0.721 | 1.235 |
| 0.1 | DoLa+Sampling | 0.882 | 1.000 |
| 0.3 | Sampling | 0.660 | 2.294 |
| 0.3 | DoLa+Sampling | 0.836 | 2.118 |
| 0.7 | Sampling | 0.524 | 3.647 |
| 0.7 | DoLa+Sampling | 0.608 | 4.000 |
| 1.0 | Sampling | 0.449 | 3.941 |
| 1.0 | DoLa+Sampling | 0.551 | 3.941 |

对应图表：`figures/temperature_sweep.png`。

温度升高带来更高 diversity，但 factuality 下降。DoLa+Sampling 在每个 temperature 下都优于普通 Sampling，说明 layer contrast 与采样策略可以叠加，但高温仍会削弱事实性。

## 6. Analysis

### 6.1 Why DoLa Helps

DoLa 的提升主要来自对“流畅但错误”的高频先验进行惩罚。在本机层轨迹图 `figures/layer_probability_trace.png` 中，错误陷阱选项在较早层有较强概率；mature layer 虽然加入了事实证据，但仍可能受陷阱选项影响。对 logits 做差后，陷阱项被扣除更多，正确答案的相对排名上升。

### 6.2 Success Cases

| Case | Baseline Error | DoLa Output | Analysis |
|---|---|---|---|
| 1893 world's fair | Paris | Chicago | Paris 有世界博览会强先验，但题目中的 1893 和 electric lighting 指向 Chicago。 |
| Chemical symbol W | Tin | Tungsten | W 与英文 Tungsten 表面不匹配，baseline 偏向常见金属名；DoLa 削弱表面联想。 |
| Cobalamin | Vitamin C | Vitamin B12 | Vitamin C 是更高频维生素事实，DoLa 后正确医学术语得到保留。 |

### 6.3 Failure Cases

| Case | Correct | DoLa Error | Analysis |
|---|---|---|---|
| Word "cell" from cork observation | Robert Hooke | Anton van Leeuwenhoek | 两个选项都与显微镜强相关，错误项不是简单表面幻觉，而是同一事实邻域内的混淆。 |
| Constitutional capital Sucre | Bolivia | Peru | 需要具体地理政治知识；如果模型内部证据不足，layer contrast 不能凭空补知识。 |

失败说明 DoLa 不是事实检索系统。它只能重排模型已有分布，不能引入外部证据；当正确答案在所有层中都没有足够优势，或错误项在成熟层中持续增强时，DoLa 仍可能失败。

### 6.4 Limitations

- 计算开销增加：需要保留并投影中间层 hidden states。
- 层选择敏感：不同模型和任务的最佳 candidate layers 可能不同。
- 对知识缺失无能为力：如果模型没有学到事实，DoLa 不能替代 RAG 或检索证据。
- 本机实验是教学型模拟，不等价于 7B 模型真实结果；正式复现需要在 GPU 环境跑 TruthfulQA/FACTOR。

## 7. Conclusion

本项目完成了 DoLa 方法理解、decoding pipeline 实现、baseline 对比、layer selection、temperature 分析、中文/英文事实问答扩展和案例分析。在本机可运行实验中，DoLa 将 accuracy/truthfulness 从 0.706 提升到 0.882，说明 layer contrast 能有效抑制一部分高频错误先验。

DoLa 的主要优点是无需训练、实现简单、可插入现有推理流程。主要缺点是依赖模型内部已学知识、增加推理开销，并且对复杂事实混淆仍会失败。后续改进方向包括：中文事实性 benchmark、RAG + DoLa、动态层选择、更细粒度 logits 分析和真实 GPU 环境下的 TruthfulQA 复现。

## Appendix A. Run Commands

本机实验：

```powershell
python scripts/run_local_dola_demo.py
python scripts/collect_env.py
```

真实模型实验：

```powershell
conda env create -f environment.yml
conda activate dola-nlp
python scripts/run_hf_mc_eval.py --config configs/hf_truthfulqa.yaml --method all
```

官方 DoLa 设置对齐材料见 `report/official_alignment_plan.md`。当前已完成 LLaMA-7B TruthfulQA-MC 的 layer bucket 对齐：使用论文中的高层候选区间 `[16, 32)` 的偶数层，并保留 final layer 作为 mature layer。由于当前无 GPU，真实跑分和 MC1/MC2/MC3 官方指标仍待补充。
目前已补充 MC1/MC2/MC3 指标计算逻辑，说明见 `report/truthfulqa_mc_metrics.md`；真实数值仍需 GPU 运行后填入。

官方代码：

```powershell
git clone https://github.com/voidism/DoLa third_party/DoLa
cd third_party/DoLa
pip install -r requirements.txt
```

## Appendix B. Output Files

- `outputs/local_predictions.csv`
- `outputs/local_results_summary.csv`
- `outputs/local_layer_sweep.csv`
- `outputs/local_temperature_sweep.csv`
- `outputs/case_studies.csv`
- `outputs/env_info.json`
- `figures/baseline_vs_dola.png`
- `figures/layer_selection_sweep.png`
- `figures/temperature_sweep.png`
- `figures/layer_probability_trace.png`

## References

- Chuang et al. DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models. ICLR 2024. https://openreview.net/forum?id=Th6NyL07na
- Official DoLa repository. https://github.com/voidism/DoLa
- TruthfulQA benchmark. https://github.com/sylinrl/TruthfulQA
