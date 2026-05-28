# DoLa 事实性增强复现与分析

本项目根据 `DoLa_大作业任务书_完整版.md` 搭建。目录包含两类实验：

- 本机可复现实验：`scripts/run_local_dola_demo.py`，无需 GPU 和 Transformers，用透明的 layer/logits 模拟复现 DoLa 的核心对比思想，并生成 baseline、layer selection、temperature、案例分析和图表。
- 真实模型入口：`scripts/run_hf_mc_eval.py`，用于在 Python 3.10/3.11 + PyTorch + CUDA 环境中运行 HuggingFace causal LM 的 TruthfulQA 多选评测。

论文与官方资源：

- DoLa paper: https://openreview.net/forum?id=Th6NyL07na
- Official code: https://github.com/voidism/DoLa

## 目录结构

```text
.
├── README.md
├── requirements.txt
├── environment.yml
├── configs/
│   ├── hf_truthfulqa.yaml
│   └── local_demo.json
├── data/
│   └── local_factual_qa.jsonl
├── figures/
│   ├── baseline_vs_dola.png
│   ├── layer_probability_trace.png
│   ├── layer_selection_sweep.png
│   └── temperature_sweep.png
├── outputs/
├── report/
│   ├── report.md
│   └── slides_outline.md
└── scripts/
    ├── collect_env.py
    ├── run_hf_mc_eval.py
    └── run_local_dola_demo.py
```

## 环境安装

当前机器实测为 Python 3.13，未检测到 `nvidia-smi`，且未安装 `torch/transformers`。因此本仓库默认先跑本机可复现实验；真实 7B 模型建议使用单卡 24GB 以上 GPU。

推荐真实复现实验环境：

```powershell
conda env create -f environment.yml
conda activate dola-nlp
```

如果只跑本机演示，当前环境需要：

```powershell
python -m pip install pandas matplotlib numpy pyyaml tqdm
```

## 本机复现实验

```powershell
python scripts/run_local_dola_demo.py
python scripts/collect_env.py
```

主要输出：

- `outputs/local_results_summary.csv`
- `outputs/local_layer_sweep.csv`
- `outputs/local_temperature_sweep.csv`
- `outputs/case_studies.csv`
- `outputs/env_info.json`
- `figures/*.png`

当前本机结果：

| Method | Accuracy | Truthfulness | Notes |
|---|---:|---:|---|
| Greedy | 0.706 | 0.706 | final-layer argmax |
| Beam Search | 0.706 | 0.706 | top-2 final-layer beam |
| Sampling | 0.412 | 0.412 | top-p sampling over final logits |
| DoLa | 0.882 | 0.882 | final logits minus selected premature logits |

## 真实模型 TruthfulQA 复现

配置文件：`configs/hf_truthfulqa.yaml`。

```powershell
python scripts/run_hf_mc_eval.py --config configs/hf_truthfulqa.yaml --method all
```

官方设置对齐说明见 `report/official_alignment_plan.md`。当前 TruthfulQA 配置已按论文中的 LLaMA-7B 高层 bucket 对齐：candidate layers 为 `[16, 18, 20, 22, 24, 26, 28, 30]`，mature layer 为 final layer，relative top 为 `0.1`。
MC1/MC2/MC3 指标说明见 `report/truthfulqa_mc_metrics.md`，本机可用 `python scripts/test_truthfulqa_metrics.py` 验证指标计算。

默认模型为 `huggyllama/llama-7b`。如果使用 gated LLaMA/Llama-2 权重，需要先登录 HuggingFace 并接受模型许可。显存不足时可改用更小的 causal LM，但报告中应说明模型差异。

官方代码复现可参考：

```powershell
git clone https://github.com/voidism/DoLa third_party/DoLa
cd third_party/DoLa
pip install -r requirements.txt
```

官方实现中常见约定是：`--early-exit-layers -1` 表示 final-layer baseline；两个 layer 表示 DoLa-static；多个 layer 时最后一个为 mature layer，前面为 candidate premature layers。

## 方法摘要

令 mature layer 的 logits 为 `z_M`，candidate premature layer 的 logits 为 `z_l`。DoLa 用层间差分得到：

```text
z_DoLa = z_M - alpha * z_l
```

其中 `alpha` 是对比强度。直觉上，早/中层更容易携带未成熟的表面模式或高频先验；成熟层包含更完整的上下文预测。对两者做差可以削弱“流畅但不真实”的 token 倾向，增强事实相关 token 的相对概率。

## 可复现性

- 随机种子：`configs/local_demo.json` 中固定为 `42`。
- 本机结果由 `scripts/run_local_dola_demo.py` 一键生成。
- 环境信息由 `scripts/collect_env.py` 写入 `outputs/env_info.json`。
- 所有实验参数写在 `configs/` 下。

## 局限

本机演示不是 7B LLM 的真实推理结果，而是为了在无 GPU/无 Transformers 的机器上完整展示 DoLa 的 decoding pipeline、参数分析和案例分析。正式提交时，如有 GPU，应补跑 `scripts/run_hf_mc_eval.py` 或官方 DoLa 仓库，并把真实日志、GPU 使用截图和输出结果放入 `outputs/` 与报告附录。
