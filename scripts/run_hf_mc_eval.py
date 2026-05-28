"""TruthfulQA-MC DoLa evaluation for HuggingFace causal LMs.

This script is intended for the real reproduction environment described in the
README. It evaluates TruthfulQA-style multiple-choice items by scoring each
answer continuation and optionally replacing final-layer logits with DoLa
contrastive logits at continuation-token positions.

It reports the official TruthfulQA multiple-choice metrics:
MC1, MC2, and MC3.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from truthfulqa_metrics import compute_mc_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "hf_truthfulqa.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "hf_mc_eval.csv"
DATASET_ALIASES = {
    "truthful_qa": "truthfulqa/truthful_qa",
}


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get(name, name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_layer_index(layer: int, num_layers: int) -> int:
    # Hidden states include embeddings at index 0 and transformer block outputs
    # at indices 1..num_layers. -1 means the final transformer output.
    return num_layers if layer == -1 else layer


def get_num_transformer_layers(model: AutoModelForCausalLM) -> int:
    layer_paths = [
        ("model", "layers"),
        ("gpt_neox", "layers"),
        ("transformer", "h"),
    ]
    for path in layer_paths:
        module: Any = model
        for attr in path:
            if not hasattr(module, attr):
                break
            module = getattr(module, attr)
        else:
            return len(module)

    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(model.config, attr, None)
        if value is not None:
            return int(value)

    raise ValueError(f"Cannot infer transformer layer count for {type(model).__name__}.")


def logits_from_hidden(model: AutoModelForCausalLM, hidden: torch.Tensor) -> torch.Tensor:
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError(f"Model {type(model).__name__} has no output embedding layer.")
    return output_embeddings(hidden)


def log_softmax_from_hidden(model: AutoModelForCausalLM, hidden: torch.Tensor) -> torch.Tensor:
    logits = logits_from_hidden(model, hidden)
    return torch.log_softmax(logits.float(), dim=-1)


def apply_relative_top_filter(
    final_logits: torch.Tensor,
    contrastive_logits: torch.Tensor,
    relative_top: float,
) -> torch.Tensor:
    if relative_top <= 0:
        return contrastive_logits
    probs = torch.softmax(final_logits.float(), dim=-1)
    threshold = torch.max(probs, dim=-1, keepdim=True).values * relative_top
    mask = probs < threshold
    return contrastive_logits.masked_fill(mask, -1e9)


def js_divergence(log_probs_a: torch.Tensor, log_probs_b: torch.Tensor) -> torch.Tensor:
    probs_a = log_probs_a.exp()
    probs_b = log_probs_b.exp()
    mean = 0.5 * (probs_a + probs_b)
    log_mean = torch.log(mean.clamp_min(1e-12))
    return 0.5 * torch.sum(probs_a * (log_probs_a - log_mean), dim=-1) + 0.5 * torch.sum(
        probs_b * (log_probs_b - log_mean), dim=-1
    )


def select_premature_layer(
    model: AutoModelForCausalLM,
    hidden_states: tuple[torch.Tensor, ...],
    candidate_layers: list[int],
    mature_layer: int,
    token_positions: torch.Tensor,
) -> int:
    mature_log_probs = log_softmax_from_hidden(model, hidden_states[mature_layer][:, token_positions, :])
    scores = []
    for layer in candidate_layers:
        layer_log_probs = log_softmax_from_hidden(model, hidden_states[layer][:, token_positions, :])
        scores.append((float(js_divergence(mature_log_probs, layer_log_probs).mean().item()), layer))
    return max(scores)[1]


def score_choice(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    choice: str,
    method: str,
    candidate_layers: list[int],
    mature_layer: int,
    relative_top: float,
    contrast_alpha: float,
    device: str,
) -> tuple[float, int | None]:
    prompt = f"Question: {question}\nAnswer:"
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    full_ids = tokenizer(prompt + " " + choice, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    continuation_start = prompt_ids.shape[1]
    if full_ids.shape[1] <= continuation_start:
        return float("-inf"), None

    labels = full_ids[:, 1:]
    with torch.no_grad():
        outputs = model(full_ids, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    token_positions = torch.arange(continuation_start - 1, full_ids.shape[1] - 1, device=device)
    target_ids = labels[:, token_positions]

    if method in {"vanilla", "greedy", "beam", "sampling"}:
        log_probs = torch.log_softmax(outputs.logits[:, token_positions, :].float(), dim=-1)
        return float(log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).sum().item()), None

    if method != "dola":
        raise ValueError(f"Unknown method: {method}")

    premature_layer = select_premature_layer(model, hidden_states, candidate_layers, mature_layer, token_positions)
    final_logits = logits_from_hidden(model, hidden_states[mature_layer][:, token_positions, :]).float()
    premature_logits = logits_from_hidden(model, hidden_states[premature_layer][:, token_positions, :]).float()
    # TruthfulQA-MC in the paper scores answer likelihoods with contrastive
    # logits directly rather than applying post-softmax layer probabilities.
    contrastive_logits = final_logits - contrast_alpha * premature_logits
    contrastive_logits = apply_relative_top_filter(final_logits, contrastive_logits, relative_top)
    log_probs = torch.log_softmax(contrastive_logits, dim=-1)
    score = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1).sum().item()
    return float(score), premature_layer


def get_truthfulqa_targets(example: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    question = example["question"]
    mc1_targets = example.get("mc1_targets")
    mc2_targets = example.get("mc2_targets")
    if mc1_targets is None or mc2_targets is None:
        raise ValueError("TruthfulQA multiple_choice examples must contain mc1_targets and mc2_targets.")
    return question, mc1_targets, mc2_targets


def score_target_choices(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    choices: list[str],
    method: str,
    candidate_layers: list[int],
    mature_layer: int,
    relative_top: float,
    contrast_alpha: float,
    device: str,
    cache: dict[str, tuple[float, int | None]],
) -> tuple[list[float], list[int | None]]:
    scores: list[float] = []
    selected_layers: list[int | None] = []
    for choice in choices:
        if choice not in cache:
            cache[choice] = score_choice(
                model,
                tokenizer,
                question,
                choice,
                method,
                candidate_layers,
                mature_layer,
                relative_top,
                contrast_alpha,
                device,
            )
        score, selected_layer = cache[choice]
        scores.append(score)
        selected_layers.append(selected_layer)
    return scores, selected_layers


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HuggingFace LM with DoLa on multiple-choice QA.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=["vanilla", "dola", "all"], default="all")
    args = parser.parse_args()

    config = read_config(args.config)
    seed_everything(int(config["seed"]))
    device = str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA, but torch.cuda.is_available() is false.")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    if device != "cuda":
        model.to(device)
    model.eval()

    num_layers = get_num_transformer_layers(model)
    mature_layer = normalize_layer_index(int(config["mature_layer"]), num_layers)
    candidate_layers = [normalize_layer_index(int(x), num_layers) for x in config["candidate_premature_layers"]]
    candidate_layers = [x for x in candidate_layers if 0 <= x < mature_layer]
    if not 0 <= mature_layer <= num_layers:
        raise ValueError(f"mature_layer={mature_layer} is outside hidden-state range 0..{num_layers}.")
    if not candidate_layers:
        raise ValueError("No valid candidate_premature_layers remain after layer normalization.")

    dataset_name = normalize_dataset_name(str(config["dataset_name"]))
    dataset = load_dataset(dataset_name, config["dataset_config"], split=config["split"])
    max_examples = config.get("max_examples")
    if max_examples:
        dataset = dataset.select(range(min(int(max_examples), len(dataset))))

    methods = ["vanilla", "dola"] if args.method == "all" else [args.method]
    rows: list[dict[str, Any]] = []
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        question, mc1_targets, mc2_targets = get_truthfulqa_targets(example)
        mc1_choices = list(mc1_targets["choices"])
        mc1_labels = [int(x) for x in mc1_targets["labels"]]
        mc2_choices = list(mc2_targets["choices"])
        mc2_labels = [int(x) for x in mc2_targets["labels"]]
        for method in methods:
            score_cache: dict[str, tuple[float, int | None]] = {}
            mc1_scores, mc1_selected_layers = score_target_choices(
                model,
                tokenizer,
                question,
                mc1_choices,
                method,
                candidate_layers,
                mature_layer,
                float(config["relative_top"]),
                float(config["contrast_alpha"]),
                device,
                score_cache,
            )
            mc2_scores, mc2_selected_layers = score_target_choices(
                model,
                tokenizer,
                question,
                mc2_choices,
                method,
                candidate_layers,
                mature_layer,
                float(config["relative_top"]),
                float(config["contrast_alpha"]),
                device,
                score_cache,
            )
            metrics = compute_mc_metrics(mc1_scores, mc1_labels, mc2_scores, mc2_labels)
            mc1_pred_idx = int(metrics["mc1_pred_idx"])
            mc1_best_idx = int(metrics["mc1_best_idx"])
            selected_layers = [x for x in mc1_selected_layers + mc2_selected_layers if x is not None]
            rows.append(
                {
                    "idx": idx,
                    "method": method,
                    "question": question,
                    "MC1": metrics["MC1"],
                    "MC2": metrics["MC2"],
                    "MC3": metrics["MC3"],
                    "prediction_mc1": mc1_choices[mc1_pred_idx],
                    "best_answer_mc1": mc1_choices[mc1_best_idx],
                    "mc1_pred_idx": mc1_pred_idx,
                    "mc1_best_idx": mc1_best_idx,
                    "mc1_choices_json": json.dumps(mc1_choices, ensure_ascii=False),
                    "mc1_labels_json": json.dumps(mc1_labels, ensure_ascii=False),
                    "mc1_scores_json": json.dumps(mc1_scores, ensure_ascii=False),
                    "mc2_choices_json": json.dumps(mc2_choices, ensure_ascii=False),
                    "mc2_labels_json": json.dumps(mc2_labels, ensure_ascii=False),
                    "mc2_scores_json": json.dumps(mc2_scores, ensure_ascii=False),
                    "selected_layers_json": json.dumps(selected_layers, ensure_ascii=False),
                    "selected_layer_mode": int(pd.Series(selected_layers).mode().iloc[0]) if selected_layers else None,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    summary = df.groupby("method").agg(MC1=("MC1", "mean"), MC2=("MC2", "mean"), MC3=("MC3", "mean"), n=("MC1", "size")).reset_index()
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
