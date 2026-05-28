"""Estimate local CPU time for LLaMA-7B style TruthfulQA-MC evaluation.

This is a hardware-side estimate. It does not download or run a 7B model.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "outputs" / "cpu_time_estimate.json"


def matmul_benchmark(size: int = 2048, repeats: int = 5) -> dict[str, float]:
    rng = np.random.default_rng(42)
    a = rng.standard_normal((size, size), dtype=np.float32)
    b = rng.standard_normal((size, size), dtype=np.float32)
    _ = a @ b
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = a @ b
        times.append(time.perf_counter() - start)
    best = min(times)
    flops = 2.0 * size**3
    return {
        "matrix_size": size,
        "best_seconds": best,
        "median_seconds": float(np.median(times)),
        "estimated_fp32_gflops_best": flops / best / 1e9,
        "estimated_fp32_gflops_median": flops / float(np.median(times)) / 1e9,
    }


def memory_benchmark(size_mb: int = 1024, repeats: int = 5) -> dict[str, float]:
    count = size_mb * 1024 * 1024 // np.dtype(np.float32).itemsize
    src = np.ones(count, dtype=np.float32)
    dst = np.zeros_like(src)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        np.copyto(dst, src)
        times.append(time.perf_counter() - start)
    best = min(times)
    bytes_copied = src.nbytes
    return {
        "size_mb": size_mb,
        "best_seconds": best,
        "median_seconds": float(np.median(times)),
        "estimated_copy_gbps_best": bytes_copied / best / 1e9,
        "estimated_copy_gbps_median": bytes_copied / float(np.median(times)) / 1e9,
    }


def estimate_truthfulqa_time() -> dict[str, object]:
    # TruthfulQA validation has 817 questions. The MC dataset usually has many
    # candidate answers per question; the exact count varies by item. We use a
    # conservative range because the current machine has no model installed.
    examples = 817
    candidate_answers_per_example_range = [8, 16]
    avg_tokens_per_candidate_range = [8, 20]
    methods = 2  # vanilla + DoLa

    # Practical CPU generation/scoring throughput for a 7B dense model on a
    # 6-core desktop CPU is usually below llama.cpp int4 throughput if using
    # PyTorch FP32, and around the lower single-digit token/s range with
    # optimized quantized inference. Use a range for honest planning.
    cpu_tokens_per_second_range = {
        "pytorch_fp32_or_fp16_emulated": [0.05, 0.3],
        "optimized_int4_llamacpp_reference": [1.0, 4.0],
    }

    total_token_range = [
        examples * candidate_answers_per_example_range[0] * avg_tokens_per_candidate_range[0] * methods,
        examples * candidate_answers_per_example_range[1] * avg_tokens_per_candidate_range[1] * methods,
    ]

    estimates = {}
    for mode, speed_range in cpu_tokens_per_second_range.items():
        fastest_hours = total_token_range[0] / speed_range[1] / 3600.0
        slowest_hours = total_token_range[1] / speed_range[0] / 3600.0
        estimates[mode] = {
            "estimated_hours_range": [fastest_hours, slowest_hours],
            "estimated_days_range": [fastest_hours / 24.0, slowest_hours / 24.0],
        }

    return {
        "examples": examples,
        "methods": methods,
        "candidate_answers_per_example_range": candidate_answers_per_example_range,
        "avg_tokens_per_candidate_range": avg_tokens_per_candidate_range,
        "estimated_scored_tokens_range": total_token_range,
        "cpu_tokens_per_second_assumptions": cpu_tokens_per_second_range,
        "estimates": estimates,
    }


def main() -> None:
    result = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "matmul_benchmark": matmul_benchmark(),
        "memory_benchmark": memory_benchmark(),
        "truthfulqa_llama7b_cpu_estimate": estimate_truthfulqa_time(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
