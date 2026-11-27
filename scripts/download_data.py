#!/usr/bin/env python3
"""Загрузить фиксированные публичные CSV и проверить их SHA-256."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    provenance = json.loads((root / "data/PROVENANCE.json").read_text())
    for name, entry in provenance["files"].items():
        destination = root / "data/raw" / name
        if destination.exists():
            body = destination.read_bytes()
        elif args.verify_only:
            raise FileNotFoundError(destination)
        else:
            url = provenance[
                "train_url" if name == "train.csv" else "test_url"
            ]
            request = urllib.request.Request(
                url, headers={"User-Agent": "spaceship-hw1/0.1"}
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read()
        if (
            len(body) != entry["bytes"]
            or hashlib.sha256(body).hexdigest() != entry["sha256"]
        ):
            raise ValueError(
                f"Dataset checksum mismatch: {name}; refusing unrecorded data"
            )
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".csv.part")
            temporary.write_bytes(body)
            temporary.replace(destination)
        print(f"Проверен {name}: {len(body)} байт; SHA256 {entry['sha256']}")


if __name__ == "__main__":
    main()
