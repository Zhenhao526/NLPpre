"""Collect reproducibility environment information."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "outputs" / "env_info.json"


def run(command: list[str]) -> dict[str, str | int]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {"returncode": 127, "stdout": "", "stderr": f"command not found: {command[0]} ({exc.errno})"}


def module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", "installed")


def main() -> None:
    info = {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "processor": platform.processor(),
        "machine": platform.machine(),
        "nvidia_smi_path": shutil.which("nvidia-smi"),
        "nvidia_smi": run(["nvidia-smi"]),
        "packages": {
            "numpy": module_version("numpy"),
            "pandas": module_version("pandas"),
            "matplotlib": module_version("matplotlib"),
            "torch": module_version("torch"),
            "transformers": module_version("transformers"),
            "datasets": module_version("datasets"),
            "accelerate": module_version("accelerate"),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(info, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(info, indent=2, ensure_ascii=True))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
