"""Collect multiple *_summary.csv files into one comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def infer_run_name(path: Path) -> str:
    name = path.name
    if name.endswith("_summary.csv"):
        name = name[: -len("_summary.csv")]
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize multiple HF MC eval summary CSV files.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        df = pd.read_csv(path)
        run = infer_run_name(path)
        for row in df.to_dict(orient="records"):
            row["run"] = run
            rows.append(row)

    out = pd.DataFrame(rows)
    cols = ["run", "method", "MC1", "MC2", "MC3", "n"]
    out = out[[c for c in cols if c in out.columns]]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
