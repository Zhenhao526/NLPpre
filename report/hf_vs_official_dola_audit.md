# 官方 DoLa 与自写 HuggingFace TruthfulQA-MC 实现差异审计

本文档审计当前 `scripts/run_hf_mc_eval.py` 与官方 DoLa `tfqa_mc_eval.py` 的差异，目标是解释为什么自写 HF 结果中 DoLa 低于 vanilla，而官方 DoLa 结果中 DoLa 明显高于 baseline。

## 1. 当前结果现象

官方 DoLa LLaMA-7B TruthfulQA-MC：

| run | MC1 | MC2 | MC3 | n |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.2392 | 0.3925 | 0.1807 | 790 |
| DoLa high-layer | 0.3278 | 0.6540 | 0.3289 | 790 |

自写 HF LLaMA-7B TruthfulQA-MC：

| method | MC1 | MC2 | MC3 | n |
| --- | ---: | ---: | ---: | ---: |
| vanilla | 0.1983 | 0.3575 | 0.1757 | 817 |
| DoLa | 0.1579 | 0.3472 | 0.1331 | 817 |

这两个结果不能直接比较。它们不仅模型相同，评测入口、数据构造、prompt、scoring 和 DoLa 细节都有差异。

## 1.1 已完成的本地修复

本轮已在本机完成以下代码级修复；这些修复不依赖 GPU，但最终数值仍需在 3090 服务器上重新跑 LLaMA-7B 验证。

- `scripts/run_hf_mc_eval.py` 支持 `dataset_source: official_csv`，可直接读取 `data/official_truthfulqa/TruthfulQA.csv`。
- 新增 `scripts/truthfulqa_official_mc.py`，集中实现官方 CSV 的 MC1/MC2 target 构造和官方风格 QA prompt。
- LLaMA-7B TruthfulQA 配置已从 HF 817 条 multiple-choice 切换到官方 CSV 790 条。
- LLaMA-7B TruthfulQA 配置已将 `relative_top` 从 `0.1` 改为 `0.0`，与官方主命令默认设置一致。
- 自写 HF DoLa scoring 新增 `dola_score_mode: official_like`，采用 `log_softmax(final) - log_softmax(premature)` 后直接累加目标 token 分数。
- DoLa premature layer 选择从“每个答案一个层”改为“每个 token 一个层”，输出中会保存 token-level selected layers。
- 输出新增 `masked_vocab_entries`、`masked_target_tokens`、`num_target_tokens`、`prompt_chars` 等诊断字段。
- Pythia/GPT-2 旧配置显式保留 `prompt_style: simple` 和 `dola_score_mode: simple`，避免历史补充实验语义被改变。
- `scripts/run_llama7b_truthfulqa_next_steps.sh` 已同步改成官方 CSV、official prompt、`relative_top=0.0` 和 790 条分片。

## 2. 高优先级问题

### 2.1 数据集入口不一致：790 条 vs 817 条

官方脚本读取原始 `TruthfulQA.csv`，本项目本地该文件有 790 行。字段包括：

- `Question`
- `Best Answer`
- `Correct Answers`
- `Incorrect Answers`
- `Best Incorrect Answer`

自写 HF 脚本读取 HuggingFace `truthfulqa/truthful_qa` 的 `multiple_choice` 配置，validation 为 817 条，直接使用 `mc1_targets` 和 `mc2_targets`。

影响：

- 样本数量不同，baseline 已经不可直接对齐。
- 官方脚本会基于原始 CSV 构造 MC1/MC2 候选答案；HF 数据集的候选集合和顺序可能已经过二次处理。
- 当前自写 HF baseline 低于官方 baseline：0.1983 vs 0.2392，说明差异不只影响 DoLa，也影响普通 likelihood scoring。

涉及代码：

- 自写 HF 数据加载：[scripts/run_hf_mc_eval.py](../scripts/run_hf_mc_eval.py) 第 262-263 行。
- 官方运行脚本：[scripts/run_official_dola_truthfulqa.sh](../scripts/run_official_dola_truthfulqa.sh) 第 37-45 行。

修复建议：

1. 新增 `scripts/run_hf_truthfulqa_csv_eval.py`，直接读取 `TruthfulQA.csv`。
2. 用与官方一致的字段构造 MC1/MC2。
3. 先只跑 vanilla，要求 baseline 接近官方 0.2392 后再调 DoLa。

### 2.2 Prompt 不一致：自写 HF 是零样本短 prompt，官方是 TruthfulQA QA prompt

自写 HF 当前 prompt 为：

```text
Question: {question}
Answer:
```

然后拼接候选答案：

```text
Question: {question}
Answer: {choice}
```

官方 `tfqa_mc_eval.py` 使用 `build_prompt_and_answer()` 构造输入：先拼接一段 TruthfulQA instruction 和若干示例 Q/A，再追加当前问题与候选答案。也就是说，官方 TruthfulQA-MC 使用带示例的 QA prompt，而不是当前这个最短零样本 prompt。

影响：

- LLaMA 对 prompt 格式非常敏感。
- 候选答案的前置空格、换行和 few-shot 示例会改变 token likelihood。
- baseline 不对齐时，DoLa 结果没有解释力。

涉及代码：

- 自写 HF prompt 构造：[scripts/run_hf_mc_eval.py](../scripts/run_hf_mc_eval.py) 第 146-148 行。

修复建议：

1. 从官方 `tfqa_mc_eval.py` 复刻 prompt 构造函数。
2. 在输出 CSV 中保存 `full_input_text` 或 prompt hash，便于逐样本排查。
3. 在相同 prompt 下先比较 vanilla 的前 10 条分数和预测。

### 2.3 Relative-top 配置与官方主结果不一致，且当前实现会严重扭曲分数

当前自写 HF 在 DoLa 后执行：

```python
probs = torch.softmax(final_logits.float(), dim=-1)
threshold = torch.max(probs, dim=-1, keepdim=True).values * relative_top
mask = probs < threshold
return contrastive_logits.masked_fill(mask, -1e9)
```

配置中 `relative_top = 0.1`。但官方 DoLa README 给出的 TruthfulQA-MC 主实验命令没有传入 `--relative_top`；官方 `tfqa_mc_eval.py` 的默认值是 `relative_top=0.0`，因此官方主结果实际没有启用 relative-top filtering。

当前 HF 输出显示：

- DoLa 817 条中，MC1 分数含 `-1e9/-2e9/...` 的样本有 781 条。
- DoLa 817 条中，MC2 分数含 `-1e9/-2e9/...` 的样本有 803 条。
- 这些大负数来自 continuation token 被 relative-top mask 掉后又参与整句答案 log-prob 求和。

这会系统性惩罚较长答案或包含低 final-layer token 概率的答案，导致 DoLa 分数被严重扭曲。示例：

```text
idx=0, DoLa mc1_scores = [-1000000000.0, -13.7871, -15.2297, -2000000000.0]
```

这不是正常的语言模型似然尺度。

影响：

- DoLa 分数被 `-1e9` 主导，不再反映 contrastive likelihood。
- MC2 概率质量会被 mask 形状支配，而不是被答案事实性支配。
- 这很可能是自写 HF DoLa 低于 vanilla 的最直接原因。

涉及代码：

- filter 实现：[scripts/run_hf_mc_eval.py](../scripts/run_hf_mc_eval.py) 第 96-106 行。
- filter 调用：[scripts/run_hf_mc_eval.py](../scripts/run_hf_mc_eval.py) 第 172-174 行。
- 当前配置：[configs/hf_truthfulqa.yaml](../configs/hf_truthfulqa.yaml) 第 9 行。

修复建议：

1. 先把 LLaMA-7B TruthfulQA-MC 配置改成 `relative_top: 0.0`，与官方主命令对齐。
2. 若后续要研究 relative-top，再复刻官方 DoLa 的 `get_relative_top_filter()` 逻辑，不要简单用 `-1e9` 进入整句答案求和。
3. 若保留过滤，至少输出每个答案被 mask 的 token 数量，避免 silent failure。

### 2.4 DoLa 分数公式与官方实现不一致

当前自写 HF 直接使用：

```python
contrastive_logits = final_logits - contrast_alpha * premature_logits
log_probs = torch.log_softmax(contrastive_logits, dim=-1)
score = sum(log_probs[target_tokens])
```

官方 DoLa 实现不是这个最简公式的直接等价形式。官方实现基于改造过的 `transformers-4.28.1`，在 forward 内部返回 early-exit logits，并用 `lm_score` 对多个候选答案进行 scoring。在官方 `lm_score()` 中，TruthfulQA-MC 主命令使用 `post_softmax=False`，核心分数是：

```python
final_logits = final_logits.log_softmax(dim=-1)
base_logits = base_logits.log_softmax(dim=-1)
diff_logits = final_logits - base_logits
log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
```

也就是说，官方主结果是“两个层的 log-prob 差值直接累加”，不是“raw logits 相减后再做一次 log_softmax”。

影响：

- 当前实现把 hidden state 手动过 `lm_head`，并对 raw-logit contrast 再做 `log_softmax`，与官方 `log_softmax(final) - log_softmax(base)` 不一致。
- 在多选 likelihood 中，“先 raw logits contrast 再 normalize”与官方“先按层归一化再相减并直接累加”会给出不同排序。
- 此项会影响 DoLa，但不影响 vanilla；因此它是解释 DoLa 差异的核心候选原因之一。

涉及代码：

- 手写 logits：[scripts/run_hf_mc_eval.py](../scripts/run_hf_mc_eval.py) 第 84-93 行。
- DoLa scoring：[scripts/run_hf_mc_eval.py](../scripts/run_hf_mc_eval.py) 第 167-175 行。

修复建议：

1. 优先把官方 `dola.py` 的 `lm_score` 逻辑移植成可读版本，尤其是 `post_softmax=False` 的分支。
2. 保留当前实现为 `method=dola_simple`，新增 `method=dola_official_like`。
3. 同一批 10 条样本同时输出 official JSON 和 HF CSV 的逐答案分数，逐项对齐。

### 2.5 Premature layer 选择粒度不一致

当前 HF 实现先对一个候选答案的全部 continuation token 求平均 JSD，然后为整个答案选择一个 premature layer：

```python
scores.append((js_divergence(...).mean().item(), layer))
return max(scores)[1]
```

当前输出中，817 条 DoLa 的 `selected_layer_mode` 全部是 16.0。这说明动态层选择实际退化成了固定 layer 16。

官方 DoLa 的核心是对每个解码位置使用 mature layer 与候选 premature layers 的 JS divergence 动态选择。对多选 scoring 而言，也应至少在 token 级别保留 selected layer，而不是整个答案共用一个层。

影响：

- 当前“动态 DoLa”基本变成 `final - layer16`。
- 无法复现官方 high-layer bucket 的动态选层行为。
- 对长答案尤其不合理，因为不同 token 可能需要不同 premature layer。

涉及代码：

- layer selection：[scripts/run_hf_mc_eval.py](../scripts/run_hf_mc_eval.py) 第 119-131 行。
- selected layer 汇总：[scripts/run_hf_mc_eval.py](../scripts/run_hf_mc_eval.py) 第 314、333-334 行。

修复建议：

1. 改成 token-level selection：每个 token position 选一个 premature layer。
2. 输出 `selected_layers_json` 应记录 token-level layer 序列，而不是每个候选答案一个层。
3. 统计 layer 分布。如果仍全部为 16，说明 JSD 计算或 hidden-state 索引仍有问题。

## 3. 中优先级差异

### 3.1 Hidden-state layer index 可能与官方 layer 编号存在 off-by-one 风险

HF `outputs.hidden_states` 通常包含 embedding 输出作为 index 0，随后第 1..N 项对应 transformer block 输出。当前实现将 `mature_layer=-1` 归一化为 `num_layers`，candidate `[16,18,...,30]` 原样使用。

这在 HF hidden-state 语义下大体合理，但官方 early-exit layer 编号是在改造后的 LLaMA forward 内部定义的。二者是否完全一致需要用同一输入比较 layer logits。

修复建议：

- 做一个 1 条样本的 layer dump，对比官方 layer 16/18/.../32 与 HF hidden_states index 16/18/.../32 的 target token logits。
- 如果存在 off-by-one，应统一到官方编号。

### 3.2 缓存 key 只用 choice，理论上不稳健

当前每个 question 内部的 `score_cache` 使用 `choice` 作为 key：

```python
cache[choice] = score_choice(...)
```

由于 cache 在每个 question 内重新创建，这不会跨问题污染。但若同一问题中 MC1/MC2 有同文案 choice，复用分数是合理的。该问题不是当前结果异常的主因。

### 3.3 指标计算本身基本正确

`scripts/truthfulqa_metrics.py` 中 MC1/MC2/MC3 的定义与 TruthfulQA 多选指标一致：

- MC1：唯一最佳答案分数高于所有错误答案。
- MC2：正确答案集合的归一化概率质量。
- MC3：正确答案中有多少比例高于所有错误答案。

因此当前主要问题不在指标聚合，而在输入、prompt 和 DoLa scoring。

## 4. 建议修复路线

### Step 1：先修 vanilla 对齐

目标：自写 HF vanilla baseline 接近官方 baseline 0.2392。

要做：

1. 改读原始 `TruthfulQA.csv`。
2. 复刻官方 prompt 与 MC 候选构造。
3. 输出前 10 条逐答案分数，与官方 JSON 或官方脚本临时输出对比。

只有 vanilla 对齐后，DoLa 对齐才有意义。

### Step 2：禁用当前 relative-top，做 DoLa sanity check

目标：确认 `-1e9` mask 是否是 DoLa 低分主因。

要做：

1. 将 LLaMA-7B TruthfulQA-MC 配置改为 `relative_top: 0.0`，先与官方主实验命令保持一致。
2. 跑 100 条和全量。
3. 检查输出中是否还出现 `-1e9/-2e9`。
4. 若 DoLa 明显回升，说明当前 `relative_top=0.1` 是主要干扰源；若仍不对齐，再继续查 scoring 与 prompt。

### Step 3：实现 official-like DoLa scoring

目标：复刻官方 `lm_score` 语义。

要做：

1. 移植官方 DoLa scoring 的关键逻辑。
2. 改成 token-level premature layer selection。
3. 将当前简化实现保留为 `dola_simple`，避免历史结果不可追踪。

### Step 4：逐样本差异报告

目标：解释结果差异，而不是只给总分。

建议输出字段：

- question
- answer choice
- vanilla score
- DoLa simple score
- DoLa official-like score
- selected layer per token
- masked token count
- prompt text hash
- target token ids

## 5. 当前结论

当前自写 HF 实现不能作为官方 DoLa 复现结果使用，只能作为诊断和补充实验。它存在三个主要问题：

1. 数据和 prompt 没有对齐官方 TruthfulQA-MC。
2. 自写配置启用了官方主结果未启用的 `relative_top=0.1`，并把大量答案 token 置为 `-1e9`，严重扭曲多选 likelihood。
3. DoLa layer selection 和 scoring 是简化版，未复刻官方 token-level dynamic early-exit 逻辑。

因此，当前“自写 HF DoLa 低于 vanilla”不能说明 DoLa 方法无效。更合理的解释是：自写实现尚未对齐官方评测链路，尤其是 relative-top 和 token-level scoring 存在实现偏差。
