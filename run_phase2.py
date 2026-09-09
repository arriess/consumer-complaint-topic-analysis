"""
Run the complete Phase 2 analysis pipeline in the required order.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEPS = [
    ROOT / "src" / "01_acquire_validate.py",
    ROOT / "src" / "02_preprocess.py",
    ROOT / "src" / "03_vectorize_and_model.py",
]


def main() -> None:
    for step in STEPS:
        print(f"\n{'=' * 72}")
        print(f"RUNNING: {step.name}")
        print(f"{'=' * 72}\n")
        subprocess.run([sys.executable, str(step)], check=True)

    print("\n[OK] Phase 2 pipeline completed successfully.")


if __name__ == "__main__":
    main()
