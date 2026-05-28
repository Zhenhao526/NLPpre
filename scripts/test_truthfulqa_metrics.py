"""Smoke tests for official TruthfulQA MC metrics."""

from __future__ import annotations

from truthfulqa_metrics import compute_mc_metrics


def approx_equal(left: float, right: float, eps: float = 1e-9) -> bool:
    return abs(left - right) <= eps


def main() -> None:
    # MC1 true answer beats all false answers.
    metrics = compute_mc_metrics(
        mc1_scores=[3.0, 1.0, 0.0],
        mc1_labels=[1, 0, 0],
        mc2_scores=[3.0, 2.0, 1.0, 0.0],
        mc2_labels=[1, 1, 0, 0],
    )
    assert metrics["MC1"] == 1.0
    assert metrics["MC3"] == 1.0
    assert 0.0 < metrics["MC2"] < 1.0

    # MC1 fails when a false answer has the highest score. MC3 counts the
    # fraction of true answers that beat every false answer.
    metrics = compute_mc_metrics(
        mc1_scores=[2.0, 3.0, 1.0],
        mc1_labels=[1, 0, 0],
        mc2_scores=[4.0, 2.0, 3.0, 1.0],
        mc2_labels=[1, 1, 0, 0],
    )
    assert metrics["MC1"] == 0.0
    assert approx_equal(float(metrics["MC3"]), 0.5)
    assert 0.0 < metrics["MC2"] < 1.0

    print("TruthfulQA MC metric smoke tests passed.")


if __name__ == "__main__":
    main()
