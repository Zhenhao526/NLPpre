"""Summarize official DoLa JSON outputs from tfqa_mc_eval.py or factor_eval.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    row: dict[str, object] = {"run": path.stem}
    if {"total_mc1", "total_mc2", "total_mc3"}.issubset(data):
        row.update(
            {
                "MC1": data["total_mc1"],
                "MC2": data["total_mc2"],
                "MC3": data["total_mc3"],
                "n": len(data.get("question", [])),
            }
        )
    elif "is_correct" in data:
        values = data["is_correct"]
        row.update({"accuracy": sum(values) / len(values), "n": len(values)})
    else:
        raise ValueError(f"Unsupported official DoLa JSON format: {path}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize official DoLa JSON result files.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = pd.DataFrame([summarize(path) for path in args.inputs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
