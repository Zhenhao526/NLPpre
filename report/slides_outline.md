# DoLa 课堂展示提纲（10-15 分钟）

## 1. 背景与问题（1.5 分钟）

- LLM hallucination：语言流畅但事实错误。
- 自回归 decoding 的风险：每一步只优化局部下一个 token 概率，错误事实会被后续上下文放大。
- 任务目标：不重新训练模型，仅修改推理阶段 decoding，提高 factuality。

## 2. DoLa 核心思想（2 分钟）

- Transformer 不同层表达的信息不同。
- Premature layer 更容易保留表面模式、高频先验或未成熟预测。
- Mature layer 给出最终语言模型分布。
- 通过 layer contrast 抑制 premature 分布中的虚假高置信倾向。

公式：

```text
z_DoLa = z_M - alpha * z_l
p_DoLa = softmax(z_DoLa / T)
```

## 3. Decoding Pipeline（2 分钟）

- 输入 prompt。
- 前向传播并保留 hidden states。
- 对 candidate layers 和 mature layer 投影到词表 logits。
- 选择 premature layer。
- 计算 contrastive logits。
- greedy/sampling 得到最终输出。

建议展示图：`figures/layer_probability_trace.png`。

## 4. 实验设置（2 分钟）

- 本机环境：Python 3.13，无 `nvidia-smi`，未安装 CUDA 版 PyTorch。
- 本机演示数据：英文事实问答 + 中文事实问答 + 困难混淆样本，共 17 条。
- 官方主实验：双 RTX 3090，LLaMA-7B，TruthfulQA-MC 与 FACTOR News/Wiki，官方 DoLa 仓库。
- HF official-style 复核：双 RTX 3090，LLaMA-7B，TruthfulQA-MC 790 条。
- Baselines：Greedy、Beam Search、Sampling。
- 参数分析：Layer selection、Temperature。
- 真实模型入口：`scripts/run_hf_mc_eval.py` 支持 TruthfulQA 官方 MC1/MC2/MC3。

## 5. 结果（2 分钟）

展示 `figures/baseline_vs_dola.png`。

| Method | Accuracy | Truthfulness |
|---|---:|---:|
| Greedy | 0.706 | 0.706 |
| Beam Search | 0.706 | 0.706 |
| Sampling | 0.412 | 0.412 |
| DoLa | 0.882 | 0.882 |

官方 DoLa LLaMA-7B TruthfulQA-MC：

| Method | MC1 | MC2 | MC3 | n |
|---|---:|---:|---:|---:|
| baseline | 0.2392 | 0.3925 | 0.1807 | 790 |
| DoLa high-layer | 0.3278 | 0.6540 | 0.3289 | 790 |

结论：官方实现下 DoLa 在三个指标上均高于 baseline，可作为本项目主复现实验结果。这里不额外声称统计显著性。

HF official-style TruthfulQA-MC 复核：

| Method | MC1 | MC2 | MC3 | n |
|---|---:|---:|---:|---:|
| vanilla | 0.2392 | 0.3925 | 0.1807 | 790 |
| DoLa | 0.3038 | 0.6445 | 0.3148 | 790 |

结论：修复数据、prompt、relative-top 和 scoring 后，自写 HF vanilla 已与官方 baseline 对齐，DoLa 也明显高于 vanilla。

官方 DoLa LLaMA-7B FACTOR：

| Dataset | Method | Accuracy | n |
|---|---|---:|---:|
| News | baseline | 0.5859 | 1036 |
| News | DoLa | 0.6149 | 1036 |
| Wiki | baseline | 0.5862 | 2994 |
| Wiki | DoLa | 0.6219 | 2994 |

结论：DoLa 在 FACTOR News/Wiki 上均优于 baseline，说明收益可迁移到 TruthfulQA 之外的事实性 benchmark。

## 6. 分析（3 分钟）

- 成功案例：Chicago world's fair、W/Tungsten、cobalamin/Vitamin B12。
- 失败案例：Robert Hooke vs. Leeuwenhoek、Sucre/Bolivia。
- 为什么有效：减去 premature layer 中的高频陷阱先验。
- 为什么失败：当错误选项和正确答案在同一事实邻域中都很强时，layer contrast 不足以引入外部知识。
- Temperature：温度越高，多样性上升，但 factuality 下降；DoLa 在各温度下仍优于普通 sampling。

## 7. 结论（1 分钟）

- DoLa 优点：无需训练、可插入推理、对事实性有帮助。
- DoLa 局限：增加计算开销；依赖层选择；不能创造模型不知道的知识。
- 后续方向：HF/官方残余差异分析、中文事实性 benchmark、RAG + DoLa、效率和 KV cache 分析。
