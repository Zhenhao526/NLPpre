"""Run a deterministic DoLa-style reproduction without a GPU.

The script uses a small factual QA set and a transparent layer simulator. It is
not a replacement for LLM inference; it is a reproducible teaching experiment
that mirrors DoLa's core operation: contrast a mature layer with an earlier
layer and decode from the contrastive logits.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "data" / "local_factual_qa.jsonl"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "local_demo.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs"
DEFAULT_FIGURES = PROJECT_ROOT / "figures"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(temperature, 1e-8)
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def entropy(probs: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(clipped * np.log(clipped)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * np.log(p / q)))


def normalize(values: np.ndarray) -> np.ndarray:
    if len(values) <= 1:
        return np.zeros_like(values)
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    if math.isclose(min_v, max_v):
        return np.zeros_like(values)
    return (values - min_v) / (max_v - min_v)


def lexical_bias(question: str, choice: str) -> float:
    """Small bias for surface-form overlap between a question and a choice."""
    q = question.lower()
    c = choice.lower()
    if not c:
        return 0.0
    char_hits = sum(1 for ch in set(c) if ch.strip() and ch in q)
    return char_hits / max(len(set(c)), 1)


def item_noise(item_id: str, choice: str, layer: int, seed: int) -> float:
    # Stable across Python invocations, unlike built-in hash randomization.
    key = f"{seed}:{item_id}:{choice}:{layer}"
    value = sum((idx + 1) * ord(ch) for idx, ch in enumerate(key))
    rng = random.Random(value)
    return rng.uniform(-0.04, 0.04)


def factual_strength(item: dict[str, Any], layer: int, mature_layer: int) -> float:
    """Later layers contain stronger task-specific answer evidence."""
    progress = layer / max(mature_layer, 1)
    base = 1.0 / (1.0 + math.exp(-8.0 * (progress - 0.58)))
    if item["id"].startswith("hard_"):
        hard = 0.12
    elif item["id"].startswith("fail_"):
        hard = 0.48
    else:
        hard = 1.0
    return hard * base


def language_prior(layer: int, mature_layer: int) -> float:
    """A broad prior for salient but sometimes wrong completions."""
    progress = layer / max(mature_layer, 1)
    return 0.9 - 0.22 * progress + 0.08 * math.sin(math.pi * progress)


def base_choice_score(item: dict[str, Any], choice: str) -> float:
    choices = item["choices"]
    answer = item["answer"]
    trap = item["popular_trap"]
    idx = choices.index(choice)
    if choice == answer:
        return 0.25
    if choice == trap:
        return 0.55
    return 0.1 - 0.03 * idx


def simulate_layer_logits(
    item: dict[str, Any],
    layer: int,
    mature_layer: int,
    seed: int,
) -> np.ndarray:
    """Create transparent per-layer logits for one multiple-choice item."""
    logits: list[float] = []
    f_strength = factual_strength(item, layer, mature_layer)
    prior = language_prior(layer, mature_layer)
    for choice in item["choices"]:
        score = base_choice_score(item, choice)
        score += 1.55 * f_strength if choice == item["answer"] else 0.0
        score += 0.9 * prior if choice == item["popular_trap"] else 0.0
        if item["id"].startswith("hard_") and choice == item["popular_trap"]:
            score += 1.25 * (layer / max(mature_layer, 1)) ** 1.4
        score += 0.12 * lexical_bias(item["question"], choice)
        score += item_noise(item["id"], choice, layer, seed)
        logits.append(score)
    return np.array(logits, dtype=float)


def layer_logit_table(
    item: dict[str, Any],
    layers: list[int],
    mature_layer: int,
    seed: int,
) -> dict[int, np.ndarray]:
    return {layer: simulate_layer_logits(item, layer, mature_layer, seed) for layer in layers}


def apply_relative_top_filter(
    final_logits: np.ndarray,
    contrastive_logits: np.ndarray,
    relative_top: float,
) -> np.ndarray:
    """Keep choices whose final-layer probability is near the top choice."""
    if relative_top <= 0:
        return contrastive_logits
    probs = softmax(final_logits)
    threshold = float(np.max(probs) * relative_top)
    filtered = contrastive_logits.copy()
    filtered[probs < threshold] = -1e9
    return filtered


def select_premature_layer(
    layer_logits: dict[int, np.ndarray],
    candidate_layers: list[int],
    mature_layer: int,
) -> int:
    mature_probs = softmax(layer_logits[mature_layer])
    scored = []
    for layer in candidate_layers:
        probs = softmax(layer_logits[layer])
        scored.append((js_divergence(mature_probs, probs), layer))
    return max(scored)[1]


def decode_greedy(layer_logits: dict[int, np.ndarray], mature_layer: int) -> tuple[int, dict[str, Any]]:
    return int(np.argmax(layer_logits[mature_layer])), {"decoder": "greedy"}


def decode_beam(layer_logits: dict[int, np.ndarray], mature_layer: int, beam_size: int = 2) -> tuple[int, dict[str, Any]]:
    logits = layer_logits[mature_layer]
    ranked = np.argsort(logits)[::-1][:beam_size]
    probs = softmax(logits)
    # Length/fluent-prior stand-in: beam search often amplifies high-probability
    # continuations, so choose the highest final-layer probability in the beam.
    best = int(ranked[np.argmax(probs[ranked])])
    return best, {"decoder": "beam", "beam_size": beam_size}


def decode_sampling(
    layer_logits: dict[int, np.ndarray],
    mature_layer: int,
    rng: np.random.Generator,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[int, dict[str, Any]]:
    logits = layer_logits[mature_layer]
    probs = softmax(logits, temperature=temperature)
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    keep = order[cumulative <= top_p]
    if len(keep) == 0 or keep[-1] != order[0]:
        keep = np.append(keep, order[len(keep)])
    masked = np.zeros_like(probs)
    masked[keep] = probs[keep]
    masked = masked / np.sum(masked)
    pred = int(rng.choice(len(probs), p=masked))
    return pred, {"decoder": "sampling", "temperature": temperature, "top_p": top_p}


def decode_dola(
    layer_logits: dict[int, np.ndarray],
    candidate_layers: list[int],
    mature_layer: int,
    relative_top: float,
    contrast_alpha: float,
    fixed_premature_layer: int | None = None,
) -> tuple[int, dict[str, Any]]:
    premature_layer = fixed_premature_layer
    if premature_layer is None:
        premature_layer = select_premature_layer(layer_logits, candidate_layers, mature_layer)
    final_logits = layer_logits[mature_layer]
    premature_logits = layer_logits[premature_layer]
    contrastive_logits = final_logits - contrast_alpha * premature_logits
    contrastive_logits = apply_relative_top_filter(final_logits, contrastive_logits, relative_top)
    return int(np.argmax(contrastive_logits)), {
        "decoder": "dola",
        "premature_layer": premature_layer,
        "relative_top": relative_top,
        "contrast_alpha": contrast_alpha,
    }


def evaluate_predictions(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby("method")
        .agg(
            Accuracy=("correct", "mean"),
            Truthfulness=("truthful", "mean"),
            AvgEntropy=("entropy", "mean"),
            AvgPrematureLayer=("premature_layer", "mean"),
        )
        .reset_index()
    )
    grouped["Accuracy"] = grouped["Accuracy"].round(3)
    grouped["Truthfulness"] = grouped["Truthfulness"].round(3)
    grouped["AvgEntropy"] = grouped["AvgEntropy"].round(3)
    grouped["Notes"] = grouped["method"].map(
        {
            "Greedy": "final-layer argmax",
            "Beam Search": "top-2 final-layer beam",
            "Sampling": "top-p sampling over final logits",
            "DoLa": "final logits minus selected premature logits",
        }
    )
    return grouped[["method", "Accuracy", "Truthfulness", "AvgEntropy", "AvgPrematureLayer", "Notes"]]


def run_base_experiment(data: list[dict[str, Any]], config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    layers = list(map(int, config["layers"]))
    mature_layer = int(config["mature_layer"])
    candidate_layers = list(map(int, config["candidate_premature_layers"]))
    rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []

    for item in data:
        table = layer_logit_table(item, layers, mature_layer, seed)
        for layer in layers:
            probs = softmax(table[layer])
            answer_idx = item["choices"].index(item["answer"])
            trap_idx = item["choices"].index(item["popular_trap"])
            layer_rows.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "layer": layer,
                    "answer_prob": probs[answer_idx],
                    "trap_prob": probs[trap_idx],
                    "entropy": entropy(probs),
                    "pred": item["choices"][int(np.argmax(probs))],
                    "correct": item["choices"][int(np.argmax(probs))] == item["answer"],
                }
            )

        methods = {
            "Greedy": decode_greedy(table, mature_layer),
            "Beam Search": decode_beam(table, mature_layer),
            "Sampling": decode_sampling(table, mature_layer, rng),
            "DoLa": decode_dola(
                table,
                candidate_layers,
                mature_layer,
                float(config["relative_top"]),
                float(config["contrast_alpha"]),
            ),
        }
        for method, (pred_idx, meta) in methods.items():
            probs = softmax(table[mature_layer])
            pred = item["choices"][pred_idx]
            rows.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "method": method,
                    "question": item["question"],
                    "answer": item["answer"],
                    "popular_trap": item["popular_trap"],
                    "prediction": pred,
                    "correct": pred == item["answer"],
                    "truthful": pred == item["answer"],
                    "entropy": entropy(probs),
                    "premature_layer": meta.get("premature_layer", np.nan),
                    "note": item["note"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(layer_rows)


def run_layer_sweep(data: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    seed = int(config["seed"])
    layers = list(map(int, config["layers"]))
    mature_layer = int(config["mature_layer"])
    candidate_layers = [layer for layer in layers if layer != mature_layer]
    rows = []
    for fixed_layer in candidate_layers:
        correct = []
        selected_answer_prob = []
        for item in data:
            table = layer_logit_table(item, layers, mature_layer, seed)
            pred_idx, _ = decode_dola(
                table,
                candidate_layers,
                mature_layer,
                float(config["relative_top"]),
                float(config["contrast_alpha"]),
                fixed_premature_layer=fixed_layer,
            )
            correct.append(item["choices"][pred_idx] == item["answer"])
            answer_idx = item["choices"].index(item["answer"])
            contrastive = table[mature_layer] - table[fixed_layer]
            selected_answer_prob.append(softmax(contrastive)[answer_idx])
        rows.append(
            {
                "premature_layer": fixed_layer,
                "accuracy": float(np.mean(correct)),
                "avg_contrastive_answer_prob": float(np.mean(selected_answer_prob)),
            }
        )
    return pd.DataFrame(rows)


def run_temperature_sweep(data: list[dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    seed = int(config["seed"])
    layers = list(map(int, config["layers"]))
    mature_layer = int(config["mature_layer"])
    candidate_layers = list(map(int, config["candidate_premature_layers"]))
    trials = int(config["sampling_trials"])
    rows = []
    for temp in config["temperatures"]:
        rng = np.random.default_rng(seed + int(float(temp) * 1000))
        baseline_correct = []
        dola_correct = []
        baseline_unique = defaultdict(set)
        dola_unique = defaultdict(set)
        for item in data:
            table = layer_logit_table(item, layers, mature_layer, seed)
            premature = select_premature_layer(table, candidate_layers, mature_layer)
            contrastive = apply_relative_top_filter(
                table[mature_layer],
                table[mature_layer] - float(config["contrast_alpha"]) * table[premature],
                float(config["relative_top"]),
            )
            for _ in range(trials):
                base_probs = softmax(table[mature_layer], temperature=float(temp))
                base_idx = int(rng.choice(len(base_probs), p=base_probs))
                baseline_correct.append(item["choices"][base_idx] == item["answer"])
                baseline_unique[item["id"]].add(item["choices"][base_idx])

                dola_probs = softmax(contrastive, temperature=float(temp))
                dola_idx = int(rng.choice(len(dola_probs), p=dola_probs))
                dola_correct.append(item["choices"][dola_idx] == item["answer"])
                dola_unique[item["id"]].add(item["choices"][dola_idx])
        rows.append(
            {
                "temperature": float(temp),
                "method": "Sampling",
                "accuracy": float(np.mean(baseline_correct)),
                "diversity": float(np.mean([len(v) for v in baseline_unique.values()])),
            }
        )
        rows.append(
            {
                "temperature": float(temp),
                "method": "DoLa+Sampling",
                "accuracy": float(np.mean(dola_correct)),
                "diversity": float(np.mean([len(v) for v in dola_unique.values()])),
            }
        )
    return pd.DataFrame(rows)


def pick_case_studies(predictions: pd.DataFrame) -> pd.DataFrame:
    pivot = predictions.pivot_table(
        index=["id", "category", "question", "answer", "popular_trap", "note"],
        columns="method",
        values="prediction",
        aggfunc="first",
    ).reset_index()
    success = pivot[(pivot["Greedy"] != pivot["answer"]) & (pivot["DoLa"] == pivot["answer"])].copy()
    failure = pivot[pivot["DoLa"] != pivot["answer"]].copy()
    success["case_type"] = "DoLa success"
    failure["case_type"] = "DoLa failure"
    selected = pd.concat([success.head(3), failure.head(3)], ignore_index=True)
    cols = ["case_type", "id", "category", "question", "answer", "popular_trap", "Greedy", "DoLa", "note"]
    return selected[cols]


def save_figures(
    summary: pd.DataFrame,
    layer_sweep: pd.DataFrame,
    temp_sweep: pd.DataFrame,
    layer_trace: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("default")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    order = ["Greedy", "Beam Search", "Sampling", "DoLa"]
    summary_plot = summary.set_index("method").loc[order]
    ax.bar(summary_plot.index, summary_plot["Accuracy"], color=["#6b7280", "#8b5cf6", "#f59e0b", "#0ea5e9"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Baseline Decoding vs. DoLa")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "baseline_vs_dola.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(layer_sweep["premature_layer"], layer_sweep["accuracy"], marker="o", color="#0f766e")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Fixed premature layer")
    ax.set_ylabel("DoLa accuracy")
    ax.set_title("Layer Selection Sweep")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "layer_selection_sweep.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for method, group in temp_sweep.groupby("method"):
        ax.plot(group["temperature"], group["accuracy"], marker="o", label=method)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Accuracy")
    ax.set_title("Temperature Sensitivity")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "temperature_sweep.png", dpi=180)
    plt.close(fig)

    trace = layer_trace.groupby("layer")[["answer_prob", "trap_prob"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(trace["layer"], trace["answer_prob"], marker="o", label="Answer probability", color="#16a34a")
    ax.plot(trace["layer"], trace["trap_prob"], marker="s", label="Trap probability", color="#dc2626")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Average probability")
    ax.set_title("Middle-layer Factual Signal vs. Late-layer Trap Prior")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "layer_probability_trace.png", dpi=180)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-style Markdown table."""
    if df.empty:
        return "_No rows._"
    text_df = df.copy()
    for column in text_df.columns:
        text_df[column] = text_df[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(text_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in text_df.iterrows():
        cells = [str(row[column]).replace("\n", " ") for column in headers]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown_report(
    output_dir: Path,
    summary: pd.DataFrame,
    layer_sweep: pd.DataFrame,
    temp_sweep: pd.DataFrame,
    cases: pd.DataFrame,
) -> None:
    report = output_dir / "local_experiment_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Local DoLa Reproduction Summary\n\n")
        f.write("This file is generated by `scripts/run_local_dola_demo.py`.\n\n")
        f.write("## Baseline Results\n\n")
        f.write(dataframe_to_markdown(summary))
        f.write("\n\n## Layer Selection\n\n")
        f.write(dataframe_to_markdown(layer_sweep))
        f.write("\n\n## Temperature Sweep\n\n")
        f.write(dataframe_to_markdown(temp_sweep))
        f.write("\n\n## Case Studies\n\n")
        f.write(dataframe_to_markdown(cases))
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local DoLa reproduction demo.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    data = read_jsonl(args.data)
    config = read_config(args.config)
    random.seed(int(config["seed"]))
    np.random.seed(int(config["seed"]))

    predictions, layer_trace = run_base_experiment(data, config)
    summary = evaluate_predictions(predictions)
    layer_sweep = run_layer_sweep(data, config)
    temp_sweep = run_temperature_sweep(data, config)
    cases = pick_case_studies(predictions)

    predictions.to_csv(args.output_dir / "local_predictions.csv", index=False, encoding="utf-8-sig")
    layer_trace.to_csv(args.output_dir / "local_layer_trace.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / "local_results_summary.csv", index=False, encoding="utf-8-sig")
    layer_sweep.to_csv(args.output_dir / "local_layer_sweep.csv", index=False, encoding="utf-8-sig")
    temp_sweep.to_csv(args.output_dir / "local_temperature_sweep.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(args.output_dir / "case_studies.csv", index=False, encoding="utf-8-sig")

    save_figures(summary, layer_sweep, temp_sweep, layer_trace, args.figure_dir)
    write_markdown_report(args.output_dir, summary, layer_sweep, temp_sweep, cases)

    log_lines = [
        "Local DoLa demo run log",
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        f"Seed: {config['seed']}",
        f"Examples: {len(data)}",
        f"Mature layer: {config['mature_layer']}",
        f"Candidate premature layers: {config['candidate_premature_layers']}",
        f"Relative top: {config['relative_top']}",
        f"Contrast alpha: {config['contrast_alpha']}",
        "",
        summary.to_string(index=False),
        "",
        "Generated files:",
        "- outputs/local_predictions.csv",
        "- outputs/local_results_summary.csv",
        "- outputs/local_layer_sweep.csv",
        "- outputs/local_temperature_sweep.csv",
        "- outputs/case_studies.csv",
        "- figures/baseline_vs_dola.png",
        "- figures/layer_selection_sweep.png",
        "- figures/temperature_sweep.png",
        "- figures/layer_probability_trace.png",
    ]
    (args.output_dir / "local_run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print("Local DoLa demo finished.")
    print(f"Examples: {len(data)}")
    print(f"Outputs: {args.output_dir}")
    print(f"Figures: {args.figure_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
