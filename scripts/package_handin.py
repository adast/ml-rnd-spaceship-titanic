#!/usr/bin/env python3
"""Обновить список контрольных сумм и собрать ZIP для сдачи."""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=root.parent / (root.name + ".zip")
    )
    args = parser.parse_args()
    command = [
        "git", "ls-files", "-z", "--cached", "--others", "--exclude-standard",
    ]
    listed = subprocess.check_output(command, cwd=root).decode().split("\0")
    names = set(filter(None, listed)) - {"PACKAGE_MANIFEST.json"}
    names.update(
        [
            "data/raw/train.csv",
            "data/raw/test.csv",
            "artifacts/baseline.joblib",
        ]
    )
    manifest = []
    for name in sorted(names):
        body = (root / name).read_bytes()
        manifest.append(
            {
                "path": name,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        )
    (root / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.part")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name in sorted(names | {"PACKAGE_MANIFEST.json"}):
            archive.write(root / name, arcname=root.name + "/" + name)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC verification failed")
        for entry in manifest:
            body = archive.read(root.name + "/" + entry["path"])
            if (
                len(body) != entry["bytes"]
                or hashlib.sha256(body).hexdigest() != entry["sha256"]
            ):
                raise ValueError(
                    f"Archive integrity mismatch: {entry['path']}"
                )
    temporary.replace(destination)
    print(f"Проверено контрольных сумм: {len(manifest)}; архив: {destination}")


if __name__ == "__main__":
    main()
