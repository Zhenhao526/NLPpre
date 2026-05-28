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
- Colab GPU 补充实验：T4 16GB，Pythia-1.4B，TruthfulQA-MC validation 817 条。
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

Colab Pythia-1.4B 全量 TruthfulQA-MC：

| Method | MC1 | MC2 | MC3 | n |
|---|---:|---:|---:|---:|
| vanilla | 0.2081 | 0.3609 | 0.1879 | 817 |
| DoLa | 0.1628 | 0.3672 | 0.1311 | 817 |

结论：DoLa 在 Pythia-1.4B 上仅小幅提升 MC2，MC1/MC3 下降；该结果证明真实 GPU 流程打通，但不等价于 LLaMA 主结果复现。

## 6. 分析（3 分钟）

- 成功案例：Chicago world's fair、W/Tungsten、cobalamin/Vitamin B12。
- 失败案例：Robert Hooke vs. Leeuwenhoek、Sucre/Bolivia。
- 为什么有效：减去 premature layer 中的高频陷阱先验。
- 为什么失败：当错误选项和正确答案在同一事实邻域中都很强时，layer contrast 不足以引入外部知识。
- Temperature：温度越高，多样性上升，但 factuality 下降；DoLa 在各温度下仍优于普通 sampling。

## 7. 结论（1 分钟）

- DoLa 优点：无需训练、可插入推理、对事实性有帮助。
- DoLa 局限：增加计算开销；依赖层选择；不能创造模型不知道的知识。
- 后续方向：24GB+ GPU 上补 LLaMA-7B、中文事实性 benchmark、RAG + DoLa、效率和 KV cache 分析。
