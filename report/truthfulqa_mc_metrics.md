# TruthfulQA MC1 / MC2 / MC3 指标说明

本项目已在 `scripts/truthfulqa_metrics.py` 和 `scripts/run_hf_mc_eval.py` 中实现 TruthfulQA 官方多选指标。

## 1. 输入格式

HuggingFace `truthful_qa` 的 `multiple_choice` split 中，每条样本包含：

- `question`
- `mc1_targets`
- `mc2_targets`

其中 target 字段包含：

```text
choices: answer candidates
labels: 1 for true answer, 0 for false answer
```

## 2. MC1

MC1 是 winner-takes-all 指标。

定义：

```text
MC1 = 1 if the single best true answer scores higher than every false answer else 0
```

它只使用 `mc1_targets`。官方数据中 `mc1_targets` 通常只有一个 true answer。

直观解释：

- 模型必须把最佳正确答案排在所有错误答案前面。
- 这个指标比较严格，也更容易受分数波动影响。

## 3. MC2

MC2 衡量模型给所有正确答案分配的归一化概率质量。

设每个候选答案的 log-likelihood score 为 `s_i`，候选集合中正确答案索引集合为 `T`，则：

```text
MC2 = sum_{i in T} exp(s_i) / sum_j exp(s_j)
```

实现中使用稳定 softmax：

```text
exp(s_i - max(s))
```

直观解释：

- 如果模型把概率质量集中到所有 true answers 上，MC2 高。
- 它比 MC1 更稳定，因为不只看单个 winner。

## 4. MC3

MC3 衡量每个正确答案是否都能超过所有错误答案。

定义：

```text
MC3 = average_{i in true answers} 1[s_i > max false score]
```

直观解释：

- 如果有多个正确说法，MC3 统计其中有多少比例排在所有 false answers 前面。
- 它比 MC1 更细粒度，比 MC2 更关注排序。

## 5. 脚本输出

运行：

```powershell
python scripts/run_hf_mc_eval.py --config configs/hf_truthfulqa.yaml --method all
```

逐题输出：

```text
outputs/hf_mc_eval.csv
```

包含字段：

- `MC1`
- `MC2`
- `MC3`
- `prediction_mc1`
- `best_answer_mc1`
- `mc1_scores_json`
- `mc2_scores_json`
- `selected_layers_json`

汇总输出：

```text
outputs/hf_mc_eval_summary.csv
```

包含：

```text
method, MC1, MC2, MC3, n
```

## 6. 本机测试

不需要 GPU，可以运行：

```powershell
python scripts/test_truthfulqa_metrics.py
```

当前测试结果：

```text
TruthfulQA MC metric smoke tests passed.
```

## 7. 与论文对齐情况

已对齐：

- `mc1_targets` / `mc2_targets`
- MC1 / MC2 / MC3 计算逻辑
- LLaMA-7B TruthfulQA candidate layer bucket `[16, 32)` 偶数层
- mature layer = final layer
- relative top = 0.0 for the official-style LLaMA-7B TruthfulQA-MC run

已完成的官方 LLaMA-7B 主复现：

- 官方 DoLa `tfqa_mc_eval.py`
- `huggyllama/llama-7b`
- baseline: MC1 0.2392, MC2 0.3925, MC3 0.1807, n=790
- DoLa high-layer: MC1 0.3278, MC2 0.6540, MC3 0.3289, n=790

已完成的 HF official-style LLaMA-7B 复核：

- 自写 `scripts/run_hf_mc_eval.py`
- 原始 `TruthfulQA.csv`，official-style prompt，`relative_top=0.0`
- vanilla: MC1 0.2392, MC2 0.3925, MC3 0.1807, n=790
- DoLa: MC1 0.3038, MC2 0.6445, MC3 0.3148, n=790
- 诊断字段显示 `masked_target_tokens=0`

已完成的官方 FACTOR 补充复现：

- 官方 DoLa `factor_eval.py`
- `huggyllama/llama-7b`
- FACTOR News: baseline accuracy 0.5859, DoLa accuracy 0.6149, n=1036
- FACTOR Wiki: baseline accuracy 0.5862, DoLa accuracy 0.6219, n=2994

仍待补充：

- HF official-style 与官方 DoLa high-layer 的残余 tokenization/scoring 差异核对
