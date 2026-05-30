# 官方 DoLa 与 HF official-style TruthfulQA-MC 最终对比

本文档只记录 3090 服务器上的最终结果和仍需解释的残余差异。

## 1. 最终结果

官方 DoLa LLaMA-7B TruthfulQA-MC：

| run | MC1 | MC2 | MC3 | n |
|---|---:|---:|---:|---:|
| baseline | 0.2392 | 0.3925 | 0.1807 | 790 |
| DoLa high-layer | 0.3278 | 0.6540 | 0.3289 | 790 |

HF official-style LLaMA-7B TruthfulQA-MC：

| method | MC1 | MC2 | MC3 | n |
|---|---:|---:|---:|---:|
| vanilla | 0.2392 | 0.3925 | 0.1807 | 790 |
| DoLa | 0.3038 | 0.6445 | 0.3148 | 790 |

FACTOR 官方 DoLa LLaMA-7B：

| Dataset | Method | Accuracy | n |
|---|---|---:|---:|
| News | baseline | 0.5859 | 1036 |
| News | DoLa | 0.6149 | 1036 |
| Wiki | baseline | 0.5862 | 2994 |
| Wiki | DoLa | 0.6219 | 2994 |

## 2. 当前对齐状态

HF official-style 入口已经完成以下对齐：

- 使用原始 `TruthfulQA.csv`，共 790 条样本。
- 使用官方风格 QA prompt。
- 使用 LLaMA-7B TruthfulQA 高层 candidate bucket：`[16,18,20,22,24,26,28,30]`。
- 使用 final layer 作为 mature layer。
- 使用 `relative_top=0.0`，与官方 TruthfulQA-MC 主命令一致。
- 使用 official-like token-level DoLa scoring。
- 输出诊断字段显示 `masked_target_tokens=0`。

因此，HF official-style vanilla 已与官方 baseline 对齐；HF DoLa 也明显高于 vanilla，说明自写入口已经能够复现 DoLa 在 TruthfulQA-MC 上的主要提升趋势。

## 3. 残余差异

HF official-style DoLa 仍略低于官方 DoLa high-layer：

| Metric | official DoLa | HF official-style DoLa | Difference |
|---|---:|---:|---:|
| MC1 | 0.3278 | 0.3038 | -0.0240 |
| MC2 | 0.6540 | 0.6445 | -0.0095 |
| MC3 | 0.3289 | 0.3148 | -0.0141 |

可能原因包括：

1. 官方仓库使用 patched `transformers-4.28.1`，early-exit logits 的返回位置可能与 HF hidden states 手动投影不完全等价。
2. 官方 `lm_score` 对 prompt prefix、continuation token 边界和候选答案拼接的处理可能还有细节差异。
3. token-level premature layer selection 的 tie-breaking、JSD 计算细节和 layer index 语义可能仍需逐样本对齐。

## 4. 后续检查重点

后续如需进一步逼近官方 DoLa 结果，应优先做：

1. 选取固定 10 条样本，导出官方 `lm_score` 与 HF official-style 的逐候选答案分数。
2. 对同一 prompt 比较 layer 16/18/.../32 的 target token logits，检查 hidden-state index 是否与官方 early-exit layer 编号完全一致。
3. 对比 tokenization 边界，确认 prompt prefix 与 candidate continuation 的切分完全一致。
4. 输出 token-level selected layer 序列，确认动态层选择与官方实现一致。
