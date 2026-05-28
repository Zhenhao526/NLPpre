# 《DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models》
# 自然语言处理课程大作业任务书（完整版）

---

## 一、课题名称

**基于 DoLa 解码策略的大语言模型事实性增强方法复现与分析**

论文：

- DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models
- ICLR 2024

论文地址：
https://openreview.net/forum?id=Th6NyL07na

官方代码：
https://github.com/voidism/DoLa

---

# 二、项目背景与研究意义

近年来，大语言模型（Large Language Models, LLMs）快速发展，在：

- 对话系统
- 问答系统
- 翻译
- 文本生成
- 编程辅助

等任务中表现优异。

然而，大模型仍然存在严重的：

# Hallucination（幻觉）

问题，即：

- 模型生成不存在的事实；
- 编造知识；
- 给出逻辑合理但客观错误的信息；
- 对不确定问题“强行回答”。

这类问题会严重影响：

- 医疗
- 法律
- 教育
- 搜索
- Agent 系统

等实际应用。

---

## DoLa 方法的核心思想

DoLa（Decoding by Contrasting Layers）提出：

> 不重新训练模型，仅在推理阶段修改 decoding 策略，
即可显著提升模型事实性。

论文发现：

- Transformer 不同层包含不同类型的信息；
- 中层更偏向事实知识；
- 高层更偏向语言流畅性与模式延续。

因此：

通过“对比不同层的 logits”，
可以抑制语言模型“习惯性胡编”。

---

# 三、大作业总体目标

本项目要求学生：

1. 阅读并理解 DoLa 论文；
2. 掌握大模型推理与 decoding 技术；
3. 运行并复现官方实验；
4. 分析 DoLa 的有效性与局限性；
5. 完成实验扩展；
6. 撰写完整科研风格实验报告；
7. 完成课堂展示。

---

# 四、项目任务要求

---

# 第一部分：论文阅读与方法理解

## 任务目标

深入理解论文的：

- 研究背景
- 方法设计
- 数学原理
- 实验逻辑

---

## 必须完成内容

### 1. Hallucination 问题分析

需要说明：

- 什么是 Hallucination
- 为什么 autoregressive decoding 容易产生幻觉
- 为什么更大的模型仍然存在幻觉

需要结合：

- 示例
- 论文实验
- 自己理解

进行分析。

---

### 2. DoLa 方法理解

需要详细解释：

## （1）Layer Contrast

包括：

- 为什么不同层表示不同信息
- 中层与高层的差异
- 为什么采用 logits difference

---

## （2）DoLa Decoding Pipeline

需要画图说明：

- 输入
- hidden states
- logits
- contrastive logits
- final decoding

---

## （3）Candidate Layer Selection

分析：

- 为什么选择中间层
- 不同 layer 的作用
- layer 如何影响 factuality

---

## （4）数学公式推导

需要解释：

- logits difference
- normalization
- decoding objective

要求：

- 给出公式；
- 解释公式中每个变量；
- 说明公式直觉意义。

---

# 第二部分：环境搭建与代码运行

---

## 环境要求

建议环境：

| 项目 | 推荐 |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| CUDA | 11.8 |
| GPU | RTX3090 / RTX4090 / A100 |
| 显存 | ≥24GB |
| Linux | Ubuntu 20.04+ |

---

## 必须完成

### 1. 成功运行官方代码

需要：

- 下载模型；
- 配置环境；
- 成功完成 inference。

---

### 2. 提供完整环境配置

必须提交：

- requirements.txt
- conda env
- CUDA 版本
- GPU 信息

---

### 3. README

必须包含：

- 环境安装步骤；
- 数据下载方式；
- 运行命令；
- 结果复现方法。

---

# 第三部分：基础实验复现

---

# 任务1：Baseline 对比实验

必须比较：

| 方法 |
|---|
| Greedy Decoding |
| Beam Search |
| Sampling |
| DoLa |

---

## 至少完成一个 benchmark

推荐：

| 数据集 | 推荐程度 |
|---|---|
| TruthfulQA | 强制推荐 |
| FACTOR | 推荐 |
| GSM8K | 可选 |
| StrategyQA | 可选 |

---

## 必须提交

### （1）实验结果表格

至少包含：

| Method | Accuracy | Truthfulness | Notes |
|---|---|---|---|

---

### （2）实验截图

包括：

- 推理日志；
- GPU 使用；
- 输出结果。

---

### （3）结果分析

需要分析：

- 为什么 DoLa 有提升；
- 提升来自哪里；
- 哪些任务提升明显；
- 哪些任务效果一般。

---

# 第四部分：参数分析实验

---

## 必须完成至少两种参数分析

---

# 分析1：Layer Selection

分析：

- 不同 candidate layer 的影响；
- 哪些 layer 最有效；
- layer 是否具有规律。

建议：

- 测试 4~6 组 layer。

---

# 分析2：Temperature

测试：

- 0.1
- 0.3
- 0.7
- 1.0

分析：

- factuality
- diversity
- hallucination

之间关系。

---

# 分析3：Top-k / Top-p

研究：

- sampling 与 DoLa 的关系；
- decoding randomness 对 factuality 的影响。

---

# 分析4：模型规模

可选：

- 7B
- 13B
- 34B

分析：

- 模型越大 DoLa 是否越有效。

---

# 第五部分：案例分析（重点）

---

## 必须完成

至少分析：

| 类型 | 数量 |
|---|---|
| DoLa 成功案例 | ≥5 |
| DoLa 失败案例 | ≥5 |

---

## 分析内容

需要人工分析：

- 为什么 baseline 出错；
- 为什么 DoLa 修复；
- 为什么 DoLa 失败；
- 是否存在新的 hallucination。

---

## 推荐展示形式

建议：

- 表格
- 可视化
- attention 分析
- logits 分析

---

# 第六部分：扩展实验（重要加分项）

---

## 扩展方向1：中文事实性测试

自行构建：

- 中文知识问答；
- 中文事实验证。

推荐方向：

- 中国历史
- 地理
- 时事
- 科学常识

---

## 扩展方向2：不同模型对比

测试：

| 模型 |
|---|
| LLaMA2 |
| Qwen |
| ChatGLM |
| Mistral |

分析：

- DoLa 是否具有泛化性。

---

## 扩展方向3：与其他方法对比

建议比较：

| 方法 |
|---|
| Contrastive Decoding |
| Self-Consistency |
| Chain-of-Thought |
| RAG |

---

## 扩展方向4：推理效率分析

研究：

- latency
- 显存占用
- tokens/s
- KV cache

---

# 第七部分：代码规范要求

---

## 必须满足

### 1. 可运行

助教需要：

- clone 后直接运行；
- 成功复现实验。

---

### 2. 目录结构规范

建议：

```text
project/
│
├── README.md
├── requirements.txt
├── scripts/
├── data/
├── outputs/
├── figures/
└── report/
```

---

### 3. 代码注释

要求：

- 关键函数必须有注释；
- 自己修改部分必须标注。

---

### 4. 实验可复现

必须：

- 固定随机种子；
- 标注参数；
- 提供运行命令。

---

# 第八部分：实验报告要求（核心）

---

## 报告长度

建议：

- 15~25 页

---

# 报告结构（强制）

---

## 1. Introduction

需要介绍：

- Hallucination
- factuality
- decoding methods
- DoLa 动机

---

## 2. Related Work

至少包括：

- Contrastive Decoding
- Chain-of-Thought
- RAG
- Factuality Enhancement

---

## 3. Method

必须：

- 详细介绍 DoLa；
- 给出数学公式；
- 给出流程图；
- 给出算法伪代码。

---

## 4. Experiment

必须包含：

- 数据集；
- 模型；
- GPU；
- 参数；
- benchmark；
- evaluation metrics。

---

## 5. Result

必须：

- 表格；
- 曲线；
- 可视化；
- 与论文结果比较。

---

## 6. Analysis

重点部分。

必须分析：

- 哪些任务有效；
- 哪些任务失败；
- 为什么有效；
- 为什么失败；
- layer 的作用；
- decoding 的本质。

---

## 7. Conclusion

总结：

- DoLa 优点；
- DoLa 缺点；
- 自己发现的问题；
- 未来改进方向。

---

# 第九部分：课堂展示要求

---

## 时间

10~15 分钟。

---

## PPT 必须包括

| 内容 | 是否必须 |
|---|---|
| 背景 | 是 |
| 方法 | 是 |
| 公式 | 是 |
| 实验 | 是 |
| 分析 | 是 |
| 扩展实验 | 建议 |
| 总结 | 是 |

---

## 展示重点

老师更关注：

- 是否真正理解论文；
- 是否真正复现实验；
- 是否有深入分析。

不仅仅是“跑通代码”。

---

# 第十部分：评分标准（详细版）

| 项目 | 比例 |
|---|---|
| 成功运行代码 | 10% |
| 基础实验复现 | 20% |
| 方法理解 | 15% |
| 参数分析 | 15% |
| 案例分析 | 15% |
| 扩展实验 | 15% |
| 报告与展示 | 10% |

---

# 第十一部分：推荐实验方案

---

# 方案A：低算力（推荐）

适合：

- 单卡 24GB

建议：

- 使用 7B 模型；
- 只跑 TruthfulQA；
- 只做 inference。

工作量适中。

---

# 方案B：中等算力

适合：

- 双3090
- A100

建议：

- 多 benchmark；
- 多模型；
- 参数分析。

---

# 方案C：高阶研究型

适合：

- 有科研经验；
- 有较强 GPU 资源。

建议：

- 中文扩展；
- 新 decoding；
- 改进 DoLa。

---

# 第十二部分：建议时间安排

| 周次 | 内容 |
|---|---|
| 第1周 | 阅读论文 |
| 第2周 | 跑通代码 |
| 第3周 | baseline 实验 |
| 第4周 | 参数分析 |
| 第5周 | 扩展实验 |
| 第6周 | 报告撰写 |
| 第7周 | PPT 准备 |

---

# 第十三部分：推荐阅读资料

---

## 必读论文

1. DoLa (ICLR 2024)
2. Contrastive Decoding
3. TruthfulQA
4. Self-Consistency Improves Chain of Thought

---

## 推荐资源

- HuggingFace Transformers
- OpenReview
- PapersWithCode
- 官方 GitHub

---

# 第十四部分：最终目标

---

## 最低要求

- 跑通官方代码；
- 成功复现至少一个 benchmark。

---

## 良好要求

- 完成参数分析；
- 能解释 layer 行为。

---

## 优秀要求

- 做中文扩展；
- 对论文提出批判性分析；
- 提出新的改进方法。

---

# 第十五部分：附录（建议）

建议附录包含：

- 完整运行命令；
- 实验日志；
- GPU 信息；
- 参数配置；
- 错误排查记录；
- 失败实验。

---

# 十六、总结

本项目不仅要求：

- “会运行代码”

更强调：

- 理解论文；
- 分析问题；
- 发现局限；
- 形成科研思维。

目标是：

让学生真正掌握：

- 大模型 decoding；
- factuality；
- Hallucination；
- NLP 论文复现方法；
- 科研实验分析能力。
