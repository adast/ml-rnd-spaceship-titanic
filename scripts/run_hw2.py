"""Run all four HW2 models on the frozen development folds."""

import argparse
from pathlib import Path

from spaceship_hw2 import run

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "reports/hw2")
    args = parser.parse_args()
    run(root, args.output)
