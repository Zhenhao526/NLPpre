# DoLa Beamer 讲稿

适用文件：`report/dola_beamer.pdf`  
建议时长：10-15 分钟

## 第 1 页：标题页

各位老师同学好，我汇报的题目是“基于 DoLa 解码策略的大语言模型事实性增强方法复现与分析”。

这个工作围绕 ICLR 2024 的 DoLa 方法展开。DoLa 的核心特点是：不重新训练模型，而是在推理阶段利用不同层之间的 logits 差异来改善事实性。目前项目包含三部分结果：第一是本机可复现实验，用来解释 DoLa 的机制；第二是在双 RTX 3090 服务器上使用官方 DoLa 仓库完成的 LLaMA-7B TruthfulQA-MC 和 FACTOR News/Wiki 复现；第三是自写 HuggingFace official-style TruthfulQA-MC 全量复核。

## 第 2 页：汇报结构

这次汇报分成五个部分。

第一部分介绍 hallucination 问题和项目目标；第二部分介绍 DoLa 的核心方法和公式；第三部分说明当前实验设置；第四部分展示 baseline、官方 benchmark、layer selection 和 temperature 结果；最后是案例分析、结论和下一步工作。

## 第 3 页：问题背景：Hallucination

大语言模型的一个重要问题是 hallucination，也就是模型输出看起来很自然、很合理，但事实是错的。

比如问澳大利亚首都，模型可能回答 Sydney。这个答案很常见，因为 Sydney 是澳大利亚曝光度最高的城市之一，但事实答案是 Canberra。

这种问题和自回归解码有关。模型每一步都在预测下一个 token，它优化的是局部概率，而不是事实正确性。如果某个错误答案在语言上更常见、更流畅，它就可能被模型优先生成。更大的模型虽然记住了更多知识，但也更擅长生成看起来可信的文本，所以模型变大并不能彻底解决 hallucination。

因此，这个项目关注的问题是：能不能不重新训练模型，只通过 decoding 策略提升 factuality。

## 第 4 页：项目目标与当前状态

这一页说明当前项目完成了什么，以及限制在哪里。

目前已经完成了 DoLa decoding pipeline 的实现，并对比了 Greedy、Beam Search、Sampling 和 DoLa 四种方法。同时做了 layer selection 分析、temperature 分析，以及成功和失败案例分析。报告、图表、脚本和 Beamer PPT 都已经生成。

限制也需要明确说明：本机没有 NVIDIA GPU，所以本机结果用于解释机制，不作为 7B benchmark。真实主结果来自 3090 服务器上的官方 DoLa LLaMA-7B TruthfulQA-MC、FACTOR News/Wiki，以及修复后的 HF official-style TruthfulQA-MC 复核。

## 第 5 页：DoLa 核心思想

DoLa 的出发点是：Transformer 不同层表达的信息不同。

较早的层通常更接近词法、局部搭配和高频模式；成熟层更接近最终语言建模输出，但它也可能混入语言流畅性偏置。DoLa 的做法不是引入一个额外模型，而是在同一个模型内部对比不同层。

直观地说，如果某个错误答案在 premature layer 中已经很强，说明它可能来自表面模式或高频先验；而正确答案如果更多来自成熟上下文证据，那么用成熟层 logits 减去 premature layer logits，就可能削弱这个错误答案的优势。

这就是 DoLa 的核心：layer contrast。

## 第 6 页：数学形式

这一页是 DoLa 的数学表达。

给定当前位置 `t`，第 `l` 层的 hidden state 记作 `h_t^l`。通过语言模型输出头 `W_lm`，可以得到这一层对应的词表 logits，也就是 `z_t^l`。

设成熟层是 `M`，候选的 premature layer 是 `l`，DoLa 的 contrastive logits 定义为：

`z_DoLa = z_M - alpha * z_l`。

这里 `alpha` 是对比强度，实验中设为 1.0。之后再除以 temperature 做 softmax，得到最终用于解码的分布。

这个公式的直觉是：成熟层给出最终预测，premature layer 提供需要被惩罚的未成熟预测倾向。二者做差后，某些“流畅但错误”的 token 会被降低相对概率。

## 第 7 页：DoLa Decoding Pipeline

这一页展示完整流程。

首先输入 prompt，然后模型前向传播，同时保留不同层的 hidden states。接着把 candidate layers 和 mature layer 的 hidden states 都投影到词表 logits。

之后选择一个 premature layer，再计算 `z_DoLa = z_M - alpha z_l`。最后用 greedy 或 sampling 从 contrastive logits 中生成输出。

本项目实现了两套入口：本机脚本用透明的 layer/logits 模拟这个流程，便于在没有 GPU 的环境下复现和分析；真实模型脚本则调用 HuggingFace 模型的 hidden states 和输出 embedding head，用于在 GPU 上跑 TruthfulQA。

## 第 8 页：实验设置

当前实验数据是多选事实问答，共 17 条。

其中包括 6 条英文事实问答、6 条中文事实问答，以及 5 条困难混淆样本。每条样本都有问题、候选选项、正确答案和一个常见错误陷阱。

对比方法包括 Greedy、Beam Search、Sampling 和 DoLa。固定参数包括随机种子 42，mature layer 为 12，candidate layers 为 `[2, 4, 6, 8, 10]`，relative top 为 0.08，对比强度 alpha 为 1.0。

这个实验的定位是教学型可复现分析，重点是验证 DoLa 的机制和分析维度，而不是替代真实大模型 benchmark。

## 第 9 页：当前环境

这里是当前机器环境。

系统是 Windows 10，Python 版本是 3.13.12。当前没有检测到 `nvidia-smi`，说明没有可用的 NVIDIA GPU 命令行环境。已经安装了 `numpy`、`pandas` 和 `matplotlib`，但没有安装真实大模型推理需要的 `torch`、`transformers` 和 `datasets`。

所以本机结果来自本机脚本。真实模型部分只报告 3090 服务器上的 LLaMA-7B：官方 DoLa TruthfulQA-MC、官方 FACTOR News/Wiki，以及修复后的 HF official-style TruthfulQA-MC 复核。

## 第 10 页：Baseline 对比结果

这一页是主要结果。

在当前本机实验中，Greedy 和 Beam Search 的 accuracy 都是 0.706，Sampling 是 0.412，DoLa 达到 0.882。Truthfulness 指标和 accuracy 一致，因为这里是多选事实问答，选中正确答案就记为 truthful。

可以看到，DoLa 相比 Greedy 和 Beam Search 有明显提升。Sampling 表现较差，是因为随机性引入了更多错误选项。

这里需要强调，这个结果说明的是：在当前可控实验中，layer contrast 确实能够削弱一部分错误先验，提高正确答案排名。它主要用于解释机制，不能单独代表真实大模型 benchmark。

## 第 11 页：官方 LLaMA-7B TruthfulQA-MC 主结果

这一页是最重要的真实模型主结果。

我使用官方 DoLa 仓库的 `tfqa_mc_eval.py`，模型是本地下载的 `huggyllama/llama-7b`。baseline 的 MC1、MC2、MC3 分别是 0.2392、0.3925、0.1807；DoLa high-layer 的结果分别是 0.3278、0.6540、0.3289。

这里 DoLa 的 early-exit layers 是 `16,18,20,22,24,26,28,30,32`，其中 32 是 mature layer，前面的偶数层是 candidate premature layers。可以看到，DoLa 在三个指标上都超过 baseline，方向和论文主结论一致。这里我只说“超过”或“高于”，不额外声称统计显著性。

因此这部分可以作为项目的主复现实验结果。

## 第 12 页：自写 HF official-style 复核

这一页解释为什么我要保留自写 HuggingFace 入口。

这个自写 HF 入口读取原始 `TruthfulQA.csv` 的 790 条样本，使用官方风格 QA prompt，设置 `relative_top=0.0`，并使用 token-level official-like DoLa scoring。

全量复核后，vanilla 的 MC1、MC2、MC3 是 0.2392、0.3925、0.1807，已经和官方 baseline 对齐；DoLa 的结果是 0.3038、0.6445、0.3148，明显高于 vanilla。

但它仍然略低于官方 DoLa high-layer，所以后续如果继续追求更严格复现，应检查 tokenization 边界、early-exit logits 获取方式，以及官方 `lm_score` 里的细节。

## 第 13 页：官方 LLaMA-7B FACTOR 结果

这一页是 TruthfulQA 之外的第二个官方事实性 benchmark。

我继续使用官方 DoLa 仓库的 `factor_eval.py`，模型仍然是本地下载的 `huggyllama/llama-7b`。和 TruthfulQA 不同，FACTOR 按论文设置使用低层 candidate bucket，early-exit layers 是 `0,2,4,6,8,10,12,14,32`，其中 32 是 mature layer。

结果上，News 数据集 baseline accuracy 是 0.5859，DoLa 是 0.6149；Wiki 数据集 baseline accuracy 是 0.5862，DoLa 是 0.6219。两个子集上 DoLa 都高于 baseline。

这说明 DoLa 的收益不只出现在 TruthfulQA-MC，也能迁移到 FACTOR 这种事实性判别任务。结合上一页 TruthfulQA-MC 的结果，可以更有力地说明官方实现下 DoLa 对 factuality benchmark 是有效的。

## 第 14 页：Layer Selection 分析

这一页分析 premature layer 的影响。

左图是固定不同 premature layer 时的 DoLa accuracy。当前小数据集上，不同层的 accuracy 都是 0.882。但右边和相关统计显示，越接近 mature layer，对比后的 answer probability 越低。

这说明一个问题：如果 premature layer 太接近 mature layer，两个分布差异变小，DoLa 可利用的 contrast 信号也会减弱。

从方法理解上看，layer selection 很关键。层太早可能语义不足，层太晚可能和最终层太像。真实模型上应该继续观察 JS divergence、answer logits 和 trap logits 随层数的变化。

## 第 15 页：Temperature 分析

这一页展示 temperature sweep。

当 temperature 从 0.1 增加到 1.0，普通 Sampling 的准确率从 0.721 下降到 0.449。DoLa+Sampling 也下降，但在每个 temperature 下都高于普通 Sampling。

这说明 temperature 提高后，输出多样性增强，但事实性下降。这是 sampling 类方法常见的 trade-off。

DoLa 可以在一定程度上缓解这个问题，因为它先通过 layer contrast 调整了 token 分布，再进行采样。但如果 temperature 太高，随机性还是会削弱 factuality。

## 第 16 页：DoLa 成功案例

这一页列了三个成功案例。

第一个是 1893 年世界博览会的问题。Baseline 受 Paris 的世界博览会高频先验影响，回答 Paris；DoLa 修正为 Chicago。

第二个是化学符号 W。因为 W 和 Tungsten 的英文表面形式不匹配，baseline 容易选 Tin；DoLa 修正为 Tungsten。

第三个是 cobalamin。Vitamin C 是更常见的维生素事实，但 cobalamin 实际上是 Vitamin B12，DoLa 把答案修正回来。

这些案例的共同点是：错误答案大多来自高频实体、表面联想或常见事实模板。DoLa 的作用就是削弱 premature layer 中这类陷阱先验。

## 第 17 页：DoLa 失败案例

这一页是失败案例。

第一个问题是“谁在观察软木时引入了 cell 这个词”。正确答案是 Robert Hooke，但 DoLa 仍然输出 Anton van Leeuwenhoek。原因是这两个名字都和显微镜强相关，错误答案不是简单的表面幻觉，而是同一事实邻域内的混淆。

第二个问题是 Sucre 作为 constitutional capital 对应哪个国家。正确答案是 Bolivia，但 DoLa 输出 Peru。这需要具体的地理政治知识。如果模型内部对正确事实的证据本来就弱，DoLa 不能凭空补知识。

所以 DoLa 的边界很清楚：它是重排模型内部预测分布的方法，不是检索系统。如果模型不知道事实，或者多个候选项在内部表示上都很强，就需要 RAG、验证器或外部知识。

## 第 18 页：结论

总结来看，DoLa 通过同一模型内部的 layer contrast 改变 decoding 分布。

在本机实验中，DoLa 把 accuracy 和 truthfulness 从 0.706 提升到 0.882。官方 LLaMA-7B TruthfulQA-MC 实验中，DoLa 在 MC1、MC2、MC3 上都高于 baseline；官方 FACTOR 实验中，DoLa 在 News 和 Wiki accuracy 上也都高于 baseline。HF official-style 复核中，vanilla 与官方 baseline 对齐，DoLa 也明显高于 vanilla。这些 3090 服务器结果共同支持论文主结论。

同时，失败案例也说明了它的局限：DoLa 不能创造模型没有掌握的知识，也不能完全解决复杂实体混淆。

一句话总结就是：DoLa 是一种低成本抑制“流畅但错误”生成倾向的方法，但它不是 factuality 问题的完整解法。

## 第 19 页：下一步提升方向

官方 FACTOR Wiki 和 News 已经完成，HF official-style 的主要偏差也已经修复，所以下一步最重要的是分析 HF 复核结果与官方 DoLa high-layer 之间的残余差异。

这部分主要包括 tokenization 边界、early-exit logits 获取方式、动态 premature layer selection 和官方 `lm_score` 的细节。因为目前 HF vanilla 已经对齐官方 baseline，残余差异更集中在 DoLa scoring 细节上。

第三步是加强 logits 可视化，尤其是成功和失败样例中 answer/trap logits 随层变化的曲线。

第四步可以扩展中文事实性数据集，比如历史、地理、文学和科学常识。

最后还可以做效率分析，比如 latency、tokens/s、显存占用，以及 candidate layers 数量对速度的影响。如果时间允许，可以再和 RAG、CoT 或 self-consistency 做扩展对比。

## 第 20 页：代码与输出

这一页列出项目文件。

本机实验脚本是 `scripts/run_local_dola_demo.py`；自写真实模型入口是 `scripts/run_hf_mc_eval.py`；官方 TruthfulQA 复现入口是 `scripts/run_official_dola_truthfulqa.sh`，官方 FACTOR 复现入口是 `scripts/run_official_dola_factor.sh`。官方 TruthfulQA 主结果在 `outputs/official_dola_truthfulqa_summary.csv`，HF official-style 复核结果在 `outputs/hf_official_fix/llama7b_official_style_full_summary.csv`，官方 FACTOR 结果在 `outputs/official_dola_factor/factor_summary.csv`。

报告是 `report/report.md`，当前 PPT 是 `report/dola_beamer.pdf`。论文参考是 ICLR 2024 的 DoLa，链接在页面底部。

## 第 21 页：结束页

我的汇报到这里结束，谢谢大家。

如果有问题，可以从三个角度讨论：第一，DoLa 的 layer contrast 为什么有效；第二，官方 LLaMA-7B 主结果和 HF official-style 复核的差异；第三，后续如何做残余实现差异分析、RAG 或中文事实性评测。
