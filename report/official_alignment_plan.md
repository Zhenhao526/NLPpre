# 官方 DoLa 设置对齐计划

目标：在暂时没有 GPU 的情况下，先把论文/官方代码的实验设置、参数命名、评测指标和本项目实现逐项对齐。等 GPU 可用后，只需要按本文档执行命令并填表。

## 1. 为什么现在可以做第二步

“官方结果对齐”不等于立刻跑完 7B 模型。它包括：

- 明确论文使用的数据集、模型、指标和 decoding 设置。
- 把论文中的 layer bucket 映射到本项目配置。
- 准备可直接运行的命令模板。
- 准备结果记录表和差异分析表。
- 明确哪些地方目前已经一致，哪些地方等 GPU 后补。

这些工作不需要 GPU，但能显著提高报告严谨性。

## 2. 论文关键实验设置摘录

来源：`DoLa_ICLR2024.pdf`。

### 2.1 模型

论文主实验使用 LLaMA family：

| Model | Transformer Layers | GPU 数量设置 |
|---|---:|---:|
| LLaMA-7B | 32 | 1 |
| LLaMA-13B | 40 | 2 |
| LLaMA-33B | 60 | 4 |
| LLaMA-65B | 80 | 8 |

论文实验环境：

- NVIDIA V100 GPUs。
- HuggingFace Transformers。
- 16-bit floating point。
- batch size = 1。
- 多 GPU 使用 HuggingFace Accelerate 做模型权重切分。

### 2.2 数据集和任务

| Task | 用途 | 论文设置 |
|---|---|---|
| TruthfulQA-MC | 短答案事实性，多选评估 | default QA prompt，two-fold validation，根据 MC3 选 bucket |
| FACTOR News/Wiki | 长段落事实性，多选评估 | News/Wiki 作 two-fold validation |
| TruthfulQA open-ended | 开放式事实性 | fine-tuned GPT-3 评估 Truth/Info |
| StrategyQA | CoT reasoning | greedy decode |
| GSM8K | 数学推理 | 随机抽 10% training subset 作 validation |
| Vicuna QA | 指令跟随/聊天质量 | GPT-4 pairwise evaluation |

### 2.3 Decoding 设置

论文附录 F 给出的设置：

- TruthfulQA、StrategyQA、GSM8K：greedy decode。
- Vicuna QA：random sampling，temperature = 0.7，max new tokens = 1024。
- latency/throughput：TruthfulQA 817 examples，default 6-shot prompt，强制生成 50 new tokens。
- adaptive plausibility constraint：`alpha = 0.1`。注意论文中的 APC alpha 不是本项目脚本里的 `contrast_alpha`。
- repetition penalty：`theta = 1.2`，但 TruthfulQA/FACTOR 多选似然评分不需要 repetition penalty。

## 3. Candidate Layer Bucket 对齐

论文将层划分为 bucket，并只用偶数层作为 candidate layers。

### 3.1 TruthfulQA

论文中 TruthfulQA 选择高层 bucket：

| Model | Selected Bucket | Layer Range | 本项目 candidate layers |
|---|---|---|---|
| LLaMA-7B | 2nd / 2 | `[16, 32)` | `[16,18,20,22,24,26,28,30]` |
| LLaMA-13B | 2nd / 2 | `[20, 40)` | `[20,22,...,38]` |
| LLaMA-33B | 3rd / 3 | `[40, 60)` | `[40,42,...,58]` |
| LLaMA-65B | 4th / 4 | `[60, 80)` | `[60,62,...,78]` |

当前已更新：

```text
configs/hf_truthfulqa.yaml
```

其中 LLaMA-7B TruthfulQA-MC 已设为：

```yaml
candidate_premature_layers: [16, 18, 20, 22, 24, 26, 28, 30]
mature_layer: -1
relative_top: 0.1
```

### 3.2 FACTOR / GSM8K / StrategyQA / Vicuna QA

论文中 FACTOR 和 GSM8K 选择低层 bucket。StrategyQA 和 Vicuna QA 复用 GSM8K/FACTOR 的 bucket。

| Model | Selected Bucket | Layer Range | 本项目 candidate layers |
|---|---|---|---|
| LLaMA-7B | 1st / 2 | `[0, 16)` | `[0,2,4,6,8,10,12,14]` |
| LLaMA-13B | 1st | `[0, 20)` | `[0,2,...,18]` |
| LLaMA-33B | 1st | `[0, 20)` | `[0,2,...,18]` |
| LLaMA-65B | 1st | `[2, 8)` | `[2,4,6]` |

当前已新增模板：

```text
configs/hf_factor_llama7b.yaml
```

该配置只完成 layer bucket 对齐；FACTOR 数据集 loader 还需要单独补 adapter。

## 4. 论文指标对齐

### 4.1 TruthfulQA-MC

论文报告：

- MC1
- MC2
- MC3

含义简述：

- MC1：winner-takes-all，要求最佳答案被选中。
- MC2/MC3：考虑多个 true/false answers 的综合指标，更稳定。

当前本项目 `scripts/run_hf_mc_eval.py` 已补 TruthfulQA 官方 MC1/MC2/MC3 计算逻辑。

当前阶段可在报告中写：

> 本项目的 HuggingFace 入口已按 DoLa layer bucket、contrastive likelihood 和 MC1/MC2/MC3 指标对齐 TruthfulQA-MC；真实数值结果仍需 GPU 环境运行 LLaMA-7B 后补充。

### 4.2 FACTOR

论文报告：

- News accuracy
- Wiki accuracy

当前项目尚未实现 FACTOR loader。可作为第三阶段扩展。

### 4.3 Open-ended TruthfulQA

论文使用 fine-tuned GPT-3 评估：

- %Truth
- %Info
- %Truth * Info
- %Reject

当前无 API/评估器，不建议优先做。可作为报告中的“未复现原因”。

## 5. 代码实现对齐状态

| 项目 | 论文/官方设置 | 本项目当前状态 | 是否对齐 |
|---|---|---|---|
| DoLa 核心 | mature logits 与 premature logits 对比 | 已实现 `final_logits - contrast_alpha * premature_logits` | 基本对齐 |
| Premature selection | candidate layers 中按 JSD 动态选择 | 已实现 JSD selection | 对齐 |
| TruthfulQA bucket | LLaMA-7B `[16,32)` 偶数层 | 已更新配置 `[16,18,...,30]` | 对齐 |
| Mature layer | final layer | `mature_layer: -1` | 对齐 |
| APC / relative top | 0.1 | `relative_top: 0.1` | 对齐 |
| TruthfulQA-MC post-softmax | 论文发现不使用 post-softmax 更好 | 脚本用 contrastive logits 直接评分 | 对齐 |
| MC1/MC2/MC3 | 官方指标 | 已实现均值汇总与逐题记录 | 对齐 |
| LLaMA 权重 | LLaMA family | 默认 `huggyllama/llama-7b` | 近似，需要说明 |
| GPU 环境 | V100, fp16, batch size 1 | 当前无 GPU | 待补 |

## 6. GPU 可用后的命令模板

### 6.1 安装环境

```powershell
conda env create -f environment.yml
conda activate dola-nlp
```

### 6.2 运行 TruthfulQA-MC

```powershell
python scripts/run_hf_mc_eval.py --config configs/hf_truthfulqa.yaml --method all
```

输出：

```text
outputs/hf_mc_eval.csv
outputs/hf_mc_eval_summary.csv
```

### 6.3 只跑 vanilla baseline

```powershell
python scripts/run_hf_mc_eval.py --config configs/hf_truthfulqa.yaml --method vanilla --output outputs/truthfulqa_vanilla.csv
```

### 6.4 只跑 DoLa

```powershell
python scripts/run_hf_mc_eval.py --config configs/hf_truthfulqa.yaml --method dola --output outputs/truthfulqa_dola.csv
```

## 7. 结果记录模板

GPU 实验完成后，在报告中填入：

| Model | Dataset | Method | Candidate Layers | Metric | Score | Notes |
|---|---|---|---|---|---:|---|
| LLaMA-7B | TruthfulQA-MC | vanilla | none | MC1 | TBD | official metric |
| LLaMA-7B | TruthfulQA-MC | vanilla | none | MC2 | TBD | official metric |
| LLaMA-7B | TruthfulQA-MC | vanilla | none | MC3 | TBD | official metric |
| LLaMA-7B | TruthfulQA-MC | DoLa | `[16,32)` even | MC1 | TBD | dynamic JSD |
| LLaMA-7B | TruthfulQA-MC | DoLa | `[16,32)` even | MC2 | TBD | dynamic JSD |
| LLaMA-7B | TruthfulQA-MC | DoLa | `[16,32)` even | MC3 | TBD | dynamic JSD |

环境记录：

| 项目 | 值 |
|---|---|
| GPU | TBD |
| CUDA | TBD |
| PyTorch | TBD |
| Transformers | TBD |
| dtype | fp16 |
| batch size | 1 |
| max examples | 817 |

## 8. 和论文结果不一致时如何解释

可能原因：

1. 模型权重不同  
   论文使用 LLaMA family；HuggingFace 上可访问的 `huggyllama/llama-7b` 或其他替代模型可能和原始权重/tokenizer 存在差异。

2. 指标不同  
   当前脚本已输出 MC1/MC2/MC3；如果和论文仍不同，应继续检查 prompt、tokenization 和 answer scoring。

3. 数据处理不同  
   prompt、answer scoring、tokenization、是否包含 all true/false choices 都会影响分数。

4. layer bucket 选择不同  
   TruthfulQA 应使用高层 `[16,32)`；FACTOR/GSM8K 应使用低层。混用会导致结果偏差。

5. 推理实现不同  
   官方 DoLa 在不同任务中对 post-softmax、APC、repetition penalty 处理不同。

报告中推荐表述：

> 本项目优先对齐了 DoLa 的核心机制、candidate layer bucket、mature layer、relative top、TruthfulQA-MC contrastive likelihood 和 MC1/MC2/MC3 指标实现。由于当前无 GPU，尚未完成真实 LLaMA-7B benchmark；后续 GPU 实验将运行这些指标并与论文 Table 1 做数值对比。

## 9. 当前已完成的第二步成果

已完成：

- 从论文 PDF 抽取官方实验设置。
- 更新 `configs/hf_truthfulqa.yaml` 为 LLaMA-7B TruthfulQA 高层 bucket。
- 新增 `configs/hf_factor_llama7b.yaml` 作为 FACTOR/GSM8K 低层 bucket 模板。
- 调整 `scripts/run_hf_mc_eval.py` 方法命名为 `vanilla` / `dola`，避免把多选似然评分误称为 greedy/beam/sampling。
- 明确当前未完成项：FACTOR loader、GPU 跑分、与论文 Table 1 的真实数值对比。

## 10. 下一步建议

在没有 GPU 的情况下，还可以继续做：

1. 写 FACTOR 数据 loader。
2. 做 layer/logits 可视化脚本模板。
3. 将 MC1/MC2/MC3 指标说明整合进 `report/report.md` 和 PPT。

GPU 可用后优先做：

1. 跑 LLaMA-7B TruthfulQA-MC。
2. 填写结果记录模板。
3. 与论文 Table 1 的 TruthfulQA-MC 行对齐分析。
