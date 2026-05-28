"""Merge sharded HuggingFace MC evaluation CSV files and recompute summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded run_hf_mc_eval outputs.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.inputs]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["idx", "method"]).drop_duplicates(["idx", "method"], keep="last")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    summary = (
        df.groupby("method")
        .agg(MC1=("MC1", "mean"), MC2=("MC2", "mean"), MC3=("MC3", "mean"), n=("MC1", "size"))
        .reset_index()
    )
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(summary.to_string(index=False))
    print(f"Wrote {args.output}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
