# Lightning AI 免费 GPU 运行指南

目标：使用本 GitHub 仓库在 Lightning AI Studio 上先跑免费 GPU smoke test，再视显存和剩余额度运行 LLaMA-7B TruthfulQA-MC。

仓库地址：

```text
https://github.com/Zhenhao526/NLPpre.git
```

## 1. 推荐运行顺序

不要一开始直接跑 LLaMA-7B。建议顺序：

```text
1. 克隆仓库
2. 安装依赖
3. 跑本机小模型 smoke test：gpt2 + 10 条 TruthfulQA
4. 确认输出 MC1/MC2/MC3
5. 如果 GPU 显存 >= 24GB，再跑 LLaMA-7B 10 条
6. 最后跑 LLaMA-7B 全量 817 条
```

这样可以避免免费 GPU 时间浪费在环境错误、模型下载失败或显存不足上。

## 2. 创建 Lightning Studio

在 Lightning AI 中新建 Studio 后，选择带 GPU 的实例。优先级：

```text
L4 24GB / A10G 24GB / A100 40GB / A100 80GB
```

如果只有 T4 16GB：

```text
只建议跑 gpt2 smoke test 或更小模型，不建议跑 LLaMA-7B DoLa。
```

## 3. 克隆仓库

在 Lightning Studio terminal 中执行：

```bash
git clone https://github.com/Zhenhao526/NLPpre.git
cd NLPpre
```

如果仓库是 private，Lightning 可能需要 GitHub 授权。可选方式：

```bash
git clone https://<YOUR_GITHUB_TOKEN>@github.com/Zhenhao526/NLPpre.git
```

注意：不要把 token 写入代码或提交到仓库。

## 4. 安装依赖

Lightning 的 PyTorch/CUDA 通常已经预装。先检查：

```bash
python --version
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

再安装项目依赖：

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

如果安装 `torch` 很慢或冲突，可以只装非 torch 依赖：

```bash
python -m pip install transformers datasets accelerate sentencepiece protobuf pandas pyyaml tqdm matplotlib
```

## 5. 先跑 GPT-2 smoke test

这个测试只跑 10 条 TruthfulQA，用于验证：

- 数据集能下载
- 模型能加载
- DoLa hidden states 能取出
- MC1/MC2/MC3 能输出
- 结果文件能写入

命令：

```bash
python scripts/run_hf_mc_eval.py \
  --config configs/hf_truthfulqa_smoke_gpt2.yaml \
  --method all \
  --output outputs/lightning_smoke_gpt2.csv
```

成功后应生成：

```text
outputs/lightning_smoke_gpt2.csv
outputs/lightning_smoke_gpt2_summary.csv
```

查看汇总：

```bash
cat outputs/lightning_smoke_gpt2_summary.csv
```

## 6. 检查 GPU 显存

```bash
nvidia-smi
```

如果显存小于 24GB，不建议跑 LLaMA-7B DoLa。原因是 DoLa 需要 `output_hidden_states=True`，并对候选层投影 logits，显存压力比普通推理更高。

## 7. LLaMA-7B 小样本测试

如果 GPU 显存足够，先把 `configs/hf_truthfulqa.yaml` 复制成小样本配置：

```bash
cp configs/hf_truthfulqa.yaml configs/hf_truthfulqa_llama7b_10.yaml
python - <<'PY'
from pathlib import Path
p = Path("configs/hf_truthfulqa_llama7b_10.yaml")
text = p.read_text()
text = text.replace("max_examples: 817", "max_examples: 10")
p.write_text(text)
PY
```

运行：

```bash
python scripts/run_hf_mc_eval.py \
  --config configs/hf_truthfulqa_llama7b_10.yaml \
  --method all \
  --output outputs/lightning_llama7b_10.csv
```

查看：

```bash
cat outputs/lightning_llama7b_10_summary.csv
```

## 8. LLaMA-7B 全量 TruthfulQA-MC

确认 10 条样本无错误后，运行全量：

```bash
python scripts/run_hf_mc_eval.py \
  --config configs/hf_truthfulqa.yaml \
  --method all \
  --output outputs/lightning_llama7b_truthfulqa.csv
```

输出：

```text
outputs/lightning_llama7b_truthfulqa.csv
outputs/lightning_llama7b_truthfulqa_summary.csv
```

需要保存的文件：

```text
outputs/lightning_llama7b_truthfulqa_summary.csv
outputs/lightning_llama7b_truthfulqa.csv
```

还要保存 GPU 信息：

```bash
nvidia-smi > outputs/lightning_nvidia_smi.txt
python scripts/collect_env.py
```

## 9. 结果下载

在 Lightning 文件浏览器中下载：

```text
outputs/lightning_smoke_gpt2_summary.csv
outputs/lightning_llama7b_truthfulqa_summary.csv
outputs/lightning_nvidia_smi.txt
outputs/env_info.json
```

如果要把结果推回 GitHub：

```bash
git add outputs/lightning_* outputs/env_info.json
git commit -m "Add Lightning AI GPU evaluation outputs"
git push
```

## 10. 常见问题

### 10.1 Private repo 克隆失败

解决：

- 在 Lightning 中绑定 GitHub。
- 或使用 GitHub personal access token 克隆。
- 或暂时把仓库设为 public，跑完后再改 private。

### 10.2 HuggingFace 模型下载失败

`huggyllama/llama-7b` 可能下载较大。可以先用 smoke test 验证代码：

```bash
python scripts/run_hf_mc_eval.py --config configs/hf_truthfulqa_smoke_gpt2.yaml --method all
```

如果使用 gated LLaMA/Llama-2 模型，需要：

```bash
huggingface-cli login
```

### 10.3 CUDA out of memory

处理顺序：

1. 先跑 `max_examples: 10`。
2. 确认 GPU 显存是否 >= 24GB。
3. 改用更小模型做演示。
4. 换 A100 40GB 或 80GB。

### 10.4 免费额度不够

优先保留这些结果：

```text
gpt2 smoke test
LLaMA-7B 10 条 sanity check
GPU 信息截图 / nvidia-smi
```

即使全量没跑完，也能证明真实 GPU 环境已经接入，后续只差机时。

## 11. 报告中可写的表述

如果只完成 smoke test：

> 已在 Lightning AI 免费 GPU 环境中完成 GPT-2 + TruthfulQA-MC smoke test，验证了数据加载、hidden states 获取、DoLa contrastive scoring 和 MC1/MC2/MC3 指标输出流程。受免费 GPU 额度限制，LLaMA-7B 全量 TruthfulQA-MC 仍待后续补跑。

如果完成 LLaMA-7B 10 条：

> 已在 Lightning AI GPU 环境中完成 LLaMA-7B 的 10 条 TruthfulQA-MC sanity check，确认 DoLa pipeline 可以在真实模型上运行。全量 817 条评测预计还需要额外 4-6 GPU 小时。

如果完成全量：

> 已在 Lightning AI GPU 环境中完成 LLaMA-7B TruthfulQA-MC 全量评测，并输出 MC1/MC2/MC3 指标。结果文件已保存至 `outputs/lightning_llama7b_truthfulqa_summary.csv`。
