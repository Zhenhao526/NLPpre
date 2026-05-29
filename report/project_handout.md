# DoLa 项目讲义：从论文理解到课堂汇报

适用场景：阅读本讲义后，能够直接完成 10-15 分钟课堂 pre，并能回答常见追问。  
对应材料：`report/dola_beamer.pdf`、`report/dola_beamer_speaker_notes.md`、`report/report.md`。

## 1. 项目一句话概括

本项目复现并分析 DoLa（Decoding by Contrasting Layers）思想：在不重新训练大语言模型、不引入外部检索知识的情况下，只在推理阶段对比 Transformer 不同层的 logits，用 mature layer 减去 premature layer 的预测倾向，从而抑制一部分“语言上流畅但事实错误”的输出。

课堂汇报时可以这样开场：

> 我的工作关注大语言模型 hallucination 问题。DoLa 的核心不是训练新模型，而是利用同一个 Transformer 内部不同层的预测差异，在 decoding 阶段提高事实性。本项目完成了方法理解、本机可复现实验、baseline 对比、参数分析、案例分析、Colab 补充实验，以及官方 LLaMA-7B TruthfulQA-MC 和 FACTOR 复现。

## 2. 研究背景：为什么需要 DoLa

### 2.1 Hallucination 是什么

Hallucination 指模型生成与客观事实不一致的内容。它的危险之处在于，错误答案往往不是语法错误，而是“看起来很像真的”。

例子：

- 问：澳大利亚首都是哪里？
- 错误但常见回答：Sydney。
- 正确答案：Canberra。

Sydney 是高频城市名，模型容易受曝光度和语言先验影响。但事实问答要求的是真实世界知识，而不是最常见或最流畅的补全。

### 2.2 为什么自回归 decoding 容易产生幻觉

自回归模型按如下方式生成：

```text
p(x_1, ..., x_T) = product_t p(x_t | x_<t)
```

每一步只选择或采样下一个 token。这个目标有三个问题：

1. 局部最优不等于事实正确  
   下一个 token 概率高，只说明它在训练语料中符合上下文分布，不保证它对应真实事实。

2. 错误会被后续上下文放大  
   一旦模型生成了错误实体，后续 token 会围绕这个实体继续补全，使错误越来越连贯。

3. 流畅性和事实性不是同一个目标  
   语言模型天然擅长生成流畅文本，但 factuality 需要外部世界约束或内部知识被正确调用。

### 2.3 为什么模型变大仍然会 hallucinate

更大的模型通常有更多知识，但并不自动保证更可靠：

- 它学到更多事实，也学到更多错误共现和偏见。
- 它更擅长生成有说服力的解释，因此错误更难被肉眼发现。
- 如果 decoding 目标仍然是最大化语言概率，模型仍可能优先输出高频、流畅但错误的答案。

所以 DoLa 的动机是：不改变模型参数，而是改变推理时如何使用模型内部信息。

## 3. DoLa 方法核心

### 3.1 基本观察

Transformer 的不同层并不是完全等价的。一般可以粗略理解为：

- 早层：更偏词法、局部搭配、表面模式。
- 中间层：开始整合实体、关系和事实信息。
- 高层：更接近最终语言模型输出，也更容易混入流畅性、模板化和延续性偏置。

DoLa 利用这个现象：如果某个错误 token 在较早层就被强烈偏好，它可能来自未成熟的表面先验；如果正确 token 更多由高层上下文整合后支持，层间做差可能提高正确 token 的相对排名。

### 3.2 公式

给定上下文 `x_<t`，第 `l` 层 hidden state 为：

```text
h_t^l
```

通过语言模型输出头映射到词表：

```text
z_t^l = W_lm h_t^l
```

其中：

- `h_t^l`：第 `t` 个位置、第 `l` 层的 hidden state。
- `W_lm`：语言模型输出头。
- `z_t^l`：第 `l` 层对应的 vocabulary logits。

设 mature layer 为 `M`，premature layer 为 `l`，DoLa 使用：

```text
z_t^DoLa = z_t^M - alpha * z_t^l
```

再得到最终分布：

```text
p_t^DoLa = softmax(z_t^DoLa / T)
```

其中：

- `z_t^M`：成熟层 logits。
- `z_t^l`：未成熟层 logits。
- `alpha`：对比强度。
- `T`：temperature。

直觉解释：

- 如果错误答案在 premature layer 中很高，它会在差分中被扣掉。
- 如果正确答案主要由 mature layer 支持，它的相对优势会保留。
- 所以 DoLa 不是创造新知识，而是重排模型内部已有预测分布。

### 3.3 与普通 contrastive decoding 的区别

普通 Contrastive Decoding 常常需要两个模型：

```text
强模型 logits - 弱模型 logits
```

DoLa 不需要额外弱模型，而是在同一个模型内部做：

```text
成熟层 logits - 未成熟层 logits
```

优势：

- 不需要训练新模型。
- 不需要额外小模型。
- 可以插入已有推理流程。

不足：

- 需要保留和投影中间层 hidden states。
- 对 layer selection 敏感。
- 不能补充模型本来不知道的知识。

## 4. DoLa 推理流程

完整 pipeline：

```text
输入 prompt
  |
  v
模型前向传播，保留 hidden states
  |
  v
对 mature layer 和 candidate premature layers 投影到 logits
  |
  v
选择 premature layer
  |
  v
计算 contrastive logits: z_M - alpha * z_l
  |
  v
使用 greedy / sampling / top-p 等策略生成下一个 token
```

课堂展示时强调两点：

1. DoLa 改的是 decoding，不是训练。
2. DoLa 用的是模型内部层差异，不是外部检索。

## 5. 本项目实现内容

当前项目分两层：

### 5.1 本机可复现实验

脚本：`scripts/run_local_dola_demo.py`

作用：

- 在无 GPU、无 Transformers 的当前机器上复现 DoLa 的核心机制。
- 生成 baseline 对比、layer selection、temperature 分析、案例分析和图表。

输出：

- `outputs/local_results_summary.csv`
- `outputs/local_layer_sweep.csv`
- `outputs/local_temperature_sweep.csv`
- `outputs/case_studies.csv`
- `figures/baseline_vs_dola.png`
- `figures/layer_selection_sweep.png`
- `figures/temperature_sweep.png`
- `figures/layer_probability_trace.png`

### 5.2 真实模型评测

脚本：

- 官方 TruthfulQA 主实验：`scripts/run_official_dola_truthfulqa.sh`
- 官方 FACTOR 主实验：`scripts/run_official_dola_factor.sh`
- 自写补充评测：`scripts/run_hf_mc_eval.py`

作用：

- 在 GPU 环境中加载 LLaMA-7B / HuggingFace causal LM。
- 使用 hidden states 和模型输出 embedding head 计算各层 logits。
- 支持 TruthfulQA 多选评测。
- 输出官方 MC1/MC2/MC3 指标。

运行条件：

- Python 3.10/3.11。
- PyTorch + CUDA。
- Transformers、datasets、accelerate。
- Pythia-1.4B 可在 Colab T4 16GB 上运行。
- LLaMA-7B 已在双 RTX 3090 服务器上完成官方 DoLa TruthfulQA-MC 和 FACTOR News/Wiki 复现。

官方实验设置对齐材料：

```text
report/official_alignment_plan.md
```

该文档已经把论文中的 LLaMA layer bucket、TruthfulQA/FACTOR/GSM8K 设置、APC 参数、结果记录模板和当前项目配置逐项对齐。当前已完成官方 DoLa LLaMA-7B TruthfulQA-MC 和 FACTOR News/Wiki 主实验复现。

## 6. 实验设置

### 6.1 数据

本机实验数据位于：

```text
data/local_factual_qa.jsonl
```

共 17 条：

- 英文事实问答：6 条。
- 中文事实问答：6 条。
- 困难混淆样本：5 条。

每条数据包含：

- `question`：问题。
- `choices`：多选候选。
- `answer`：正确答案。
- `popular_trap`：高频但错误的陷阱选项。
- `note`：人工分析备注。

### 6.2 Baseline

本项目对比四种方法：

1. Greedy Decoding  
   直接选择 mature layer logits 最大的答案。

2. Beam Search  
   在本机多选实验中，用 top-2 final-layer beam 的最高分选项近似。

3. Sampling  
   使用 temperature 和 top-p 采样。

4. DoLa  
   使用 `z_M - alpha * z_l` 的 contrastive logits。

### 6.3 固定参数

```text
seed = 42
mature layer = 12
candidate premature layers = [2, 4, 6, 8, 10]
relative_top = 0.08
contrast alpha = 1.0
temperature sweep = [0.1, 0.3, 0.7, 1.0]
```

## 7. 当前结果

### 7.1 Baseline 对比

| Method | Accuracy | Truthfulness | Notes |
|---|---:|---:|---|
| Greedy | 0.706 | 0.706 | final-layer argmax |
| Beam Search | 0.706 | 0.706 | top-2 final-layer beam |
| Sampling | 0.412 | 0.412 | top-p sampling over final logits |
| DoLa | 0.882 | 0.882 | final logits minus selected premature logits |

解读：

- DoLa 比 Greedy/Beam 高 17.6 个百分点。
- Sampling 最低，因为随机性带来更多错误选择。
- 当前结果说明 layer contrast 能够抑制部分高频错误先验。

讲述时必须补一句：

> 这里的结果来自本机可复现实验，重点验证机制和分析流程；论文主设置的证据来自官方 DoLa LLaMA-7B TruthfulQA-MC 和 FACTOR 复现实验。

### 7.2 Official DoLa LLaMA-7B TruthfulQA-MC

运行环境：

- 双 RTX 3090 服务器。
- 官方 DoLa 仓库 `tfqa_mc_eval.py`。
- 模型：本地下载的 `huggyllama/llama-7b`。
- DoLa 层设置：`16,18,20,22,24,26,28,30,32`。
- 其中 `32` 是 mature layer，前面的偶数层是 candidate premature layers。

结果：

| Method | MC1 | MC2 | MC3 | n |
|---|---:|---:|---:|---:|
| baseline | 0.2392 | 0.3925 | 0.1807 | 790 |
| DoLa high-layer | 0.3278 | 0.6540 | 0.3289 | 790 |

解读：

- DoLa 在 MC1/MC2/MC3 上均显著高于 baseline。
- 该结果方向与论文主结论一致，因此可作为本项目真实模型主复现结果。
- 自写 HuggingFace 版本中 DoLa 低于 vanilla，说明后续应核对 prompt、tokenization、relative-top filtering 和 early-exit scoring 细节。

### 7.3 Official DoLa LLaMA-7B FACTOR

运行环境：

- 双 RTX 3090 服务器。
- 官方 DoLa 仓库 `factor_eval.py`。
- 模型：本地下载的 `huggyllama/llama-7b`。
- DoLa 层设置：`0,2,4,6,8,10,12,14,32`。
- 其中 `32` 是 mature layer，前面的低层偶数层是 FACTOR 的 candidate premature layers。

结果：

| Dataset | Method | Accuracy | n |
|---|---|---:|---:|
| News | baseline | 0.5859 | 1036 |
| News | DoLa | 0.6149 | 1036 |
| Wiki | baseline | 0.5862 | 2994 |
| Wiki | DoLa | 0.6219 | 2994 |

解读：

- DoLa 在 News 和 Wiki 上均高于 baseline。
- FACTOR 是 TruthfulQA 之外的第二个官方事实性 benchmark。
- 该结果说明 DoLa 收益不只局限于短答案 TruthfulQA-MC，也能迁移到事实性判别任务。

### 7.4 Colab Pythia-1.4B TruthfulQA-MC

运行环境：

- Google Colab T4 16GB。
- `EleutherAI/pythia-1.4b`。
- TruthfulQA-MC validation 全量 817 条。
- 脚本：`scripts/run_hf_mc_eval.py`。
- Notebook：`notebooks/colab_pythia_truthfulqa.ipynb`。

结果：

| Model | Method | MC1 | MC2 | MC3 | n |
|---|---|---:|---:|---:|---:|
| Pythia-1.4B | vanilla | 0.2081 | 0.3609 | 0.1879 | 817 |
| Pythia-1.4B | DoLa | 0.1628 | 0.3672 | 0.1311 | 817 |

解读：

- 真实 HuggingFace 模型评测链路已经跑通。
- DoLa 在 MC2 上小幅提升，从 0.3609 到 0.3672。
- DoLa 在 MC1 和 MC3 上下降，说明 Pythia-1.4B 小模型上收益不稳定。
- 该实验应定位为低显存补充实验，不等价于论文 LLaMA-7B/13B/33B 主结果复现。

### 7.5 Layer Selection

当前 layer sweep 结果：

| Premature Layer | Accuracy | Avg Contrastive Answer Prob |
|---:|---:|---:|
| 0 | 0.882 | 0.526 |
| 2 | 0.882 | 0.523 |
| 4 | 0.882 | 0.494 |
| 6 | 0.882 | 0.428 |
| 8 | 0.882 | 0.332 |
| 10 | 0.882 | 0.271 |

解读：

- 小数据集上 accuracy 相同。
- 但越接近 mature layer，对比后的 answer probability 越低。
- 这说明 premature layer 与 mature layer 太相似时，contrast 信号变弱。

答辩时如果老师问“为什么所有 accuracy 一样”，可以答：

> 因为当前数据集规模很小，accuracy 是离散指标，不够敏感；但 answer probability 曲线仍显示层间差异。后续 GPU 实验中应扩大样本量，并同时报告 logits/probability 级别指标。

### 7.6 Temperature

| Temperature | Sampling Accuracy | DoLa+Sampling Accuracy |
|---:|---:|---:|
| 0.1 | 0.721 | 0.882 |
| 0.3 | 0.660 | 0.836 |
| 0.7 | 0.524 | 0.608 |
| 1.0 | 0.449 | 0.551 |

解读：

- temperature 越高，输出越随机，diversity 上升。
- 但 factuality 下降。
- DoLa+Sampling 在每个 temperature 下都优于普通 Sampling。
- 说明 DoLa 和采样可以叠加，但高温仍然会削弱事实性。

## 8. 案例分析

### 8.1 成功案例

#### 案例 1：1893 world's fair

- 问题：1893 年哪个城市举办了世界博览会并让很多参观者接触到电灯？
- 正确答案：Chicago。
- Greedy：Paris。
- DoLa：Chicago。

分析：

Paris 与世界博览会有很强历史关联，是高频先验；但题目中的 1893 和 electric lighting 指向 Chicago。DoLa 削弱了 Paris 这种高频陷阱。

#### 案例 2：Chemical symbol W

- 问题：化学符号 W 对应哪个元素？
- 正确答案：Tungsten。
- Greedy：Tin。
- DoLa：Tungsten。

分析：

W 和 Tungsten 的英文表面形式不一致，所以模型容易选择更直观的金属名 Tin。DoLa 减弱了表面形式联想。

#### 案例 3：Cobalamin

- 问题：cobalamin 是哪种维生素？
- 正确答案：Vitamin B12。
- Greedy：Vitamin C。
- DoLa：Vitamin B12。

分析：

Vitamin C 是更常见的维生素事实，容易被模型当作默认选项。DoLa 帮助正确术语浮现。

### 8.2 失败案例

#### 案例 1：cell 一词来源

- 问题：谁在观察软木时引入了 cell 这个词？
- 正确答案：Robert Hooke。
- DoLa 输出：Anton van Leeuwenhoek。

分析：

这不是简单的表面幻觉。Robert Hooke 和 Anton van Leeuwenhoek 都与显微镜史强相关，错误项和正确项处于同一事实邻域。DoLa 不能稳定解决这种细粒度实体混淆。

#### 案例 2：Sucre constitutional capital

- 问题：哪个国家以 Sucre 为 constitutional capital？
- 正确答案：Bolivia。
- DoLa 输出：Peru。

分析：

这需要具体地理政治知识。如果模型内部对 Bolivia 的证据不足，layer contrast 不能凭空补充外部事实。

### 8.3 从案例得到的边界

DoLa 有效的情况：

- 错误答案主要来自高频先验。
- 错误答案在 premature layer 中很强。
- mature layer 对正确答案仍有足够证据。

DoLa 容易失败的情况：

- 正确答案和错误答案属于同一知识簇。
- 模型本身没有掌握该事实。
- 问题需要外部证据或实时信息。
- mature layer 也强烈偏向错误答案。

## 9. 如何讲这场 pre

建议按以下节奏：

### 9.1 0-2 分钟：讲问题

重点：

- Hallucination 是事实错误，不是语言不通顺。
- 自回归 decoding 优化 token 概率，不直接优化事实性。
- DoLa 的价值是“不训练，只改 decoding”。

过渡句：

> 既然问题出在推理阶段的 token 选择，那么一个自然思路是：能不能在不改模型参数的情况下，重新利用模型内部信息来调整 token 分布？

### 9.2 2-5 分钟：讲方法

重点：

- 不同层表达不同信息。
- mature layer 和 premature layer 做 logits difference。
- 公式要讲清楚每个变量。

过渡句：

> 有了这个公式后，整个 decoding pipeline 就很直接：前向传播拿到各层 hidden states，投影成 logits，选 premature layer，做差，再解码。

### 9.3 5-8 分钟：讲实验

重点：

- 说明当前环境限制。
- 解释本机实验定位。
- 展示 baseline 表格。
- 展示官方 DoLa LLaMA-7B TruthfulQA-MC 主结果。
- 展示官方 DoLa LLaMA-7B FACTOR News/Wiki 结果。
- 展示 Colab Pythia-1.4B TruthfulQA-MC 补充结果。

必须说：

> 本机实验解释机制，官方 DoLa LLaMA-7B TruthfulQA-MC 和 FACTOR 结果验证论文主结论，Pythia-1.4B 和自写 HF 结果用于补充分析模型规模与实现细节影响。

### 9.4 8-11 分钟：讲分析

重点：

- Layer selection：越接近 mature layer，contrast 信号越弱。
- Temperature：随机性提高会降低 factuality。
- 成功案例：高频陷阱被削弱。
- 失败案例：知识缺失或同一事实簇混淆无法解决。

### 9.5 11-13 分钟：讲结论和后续

重点：

- DoLa 是轻量 decoding 方法。
- 能缓解一部分 hallucination。
- 不能替代 RAG 或外部验证。
- 下一步是分析自写 HF 实现与官方实现的差异，并扩展中文事实性数据集、RAG + DoLa 和效率评估。

## 10. 常见答辩问题与回答

### Q1：DoLa 为什么不用重新训练？

因为它只改变推理阶段的 logits 使用方式。模型参数不变，只是在生成下一个 token 时，不直接用最终层 logits，而是用最终层 logits 减去某个 premature layer 的 logits。

### Q2：DoLa 和 RAG 有什么区别？

RAG 引入外部检索文档，能补充模型不知道或过期的知识。DoLa 不引入外部知识，只重排模型内部已有分布。因此 DoLa 更轻量，但不能解决模型本身不知道的事实。

### Q3：为什么 layer contrast 能提高 factuality？

因为 premature layer 可能保留表面模式和高频先验，错误 token 在这些层中可能已经很强。做差相当于惩罚这类早期就很强的未成熟预测，让成熟层中更有事实证据的 token 相对突出。

### Q4：为什么 DoLa 仍会失败？

DoLa 不会创造新知识。如果正确答案在模型所有层中都没有足够证据，或者错误答案和正确答案属于同一事实邻域，layer contrast 可能无法区分。

### Q5：当前结果能否说明 DoLa 在真实 LLM 上有效？

可以。官方 DoLa LLaMA-7B TruthfulQA-MC 已经完成，DoLa 在 MC1、MC2、MC3 上均显著高于 baseline；官方 FACTOR News/Wiki 也已经完成，DoLa accuracy 均高于 baseline。Pythia-1.4B 补充实验说明小模型上收益不稳定；自写 HF 版本和官方版本的差异则提示后续需要核对 prompt、tokenization 和 scoring 细节。

### Q6：为什么 Sampling 比 Greedy 差？

Sampling 引入随机性，可能选择非最大概率选项。temperature 越高，分布越平，错误选项被采到的概率越高，所以 factuality 下降。

### Q7：Layer selection 应该怎么选？

论文和官方代码通常设置一组 candidate premature layers，并根据分布差异选择。真实实验中应比较不同层的 accuracy、JS divergence、answer logits 和 latency，不能只凭经验固定一层。

## 11. 这份项目的不足与改进

当前不足：

1. 自写 HuggingFace 评测与官方 DoLa 结果仍存在差异，需要进一步定位 prompt、tokenization 和 scoring 细节。
2. 本机数据规模小，accuracy 指标离散。
3. Beam Search 在多选任务中只是近似实现。
4. Pythia-1.4B 结果显示 DoLa 收益不稳定，需要进一步分析层选择和模型规模影响。
5. 尚未与 RAG、CoT、self-consistency 做真实对比。

最优先的改进：

1. 分析自写 HF TruthfulQA 实现与官方 DoLa 实现差异。
2. 复核官方 TruthfulQA/FACTOR 的 prompt、tokenization、relative-top filtering 和 early-exit scoring。
3. 扩大中文事实性数据集。
4. 做 logits 级案例可视化。
5. 记录 latency、显存和 tokens/s。

## 12. 最终总结

DoLa 的核心贡献是提出一种非常轻量的 factuality enhancement 方法：不用训练、不用检索，只利用模型内部不同层之间的预测差异。它适合缓解一部分由高频先验和表面模式导致的 hallucination，但不能解决知识缺失、事实过期和复杂实体混淆。

本项目目前完成了从方法理解到可运行实验、图表、报告、PPT、Colab 真实模型补充实验、官方 DoLa LLaMA-7B TruthfulQA-MC 和 FACTOR 主复现实验的闭环。后续补上实现差异分析、中文 benchmark 和效率分析后，就能从“课程复现实验”进一步提升为更完整的论文复现项目。
