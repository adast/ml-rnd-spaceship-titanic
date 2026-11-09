#!/usr/bin/env python3
"""Собрать README из шаблона и измеренных результатов эксперимента."""

import json
import re
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    fragments = json.loads(
        (root / "reports/report_fragments.json").read_text()
    )
    summary = json.loads((root / "reports/summary.json").read_text())
    audit = json.loads((root / "reports/data_audit.json").read_text())
    metadata = json.loads((root / "reports/run_metadata.json").read_text())
    baseline = next(
        x for x in summary["models"] if x["model"] == summary["primary_model"]
    )
    filled = {
        "RESULTS_TABLE": fragments["results_table"],
        "CONTRAST_TABLE": fragments["contrast_table"],
        "ERROR_TABLE": fragments["error_table"],
        "ERROR_TYPE_TABLE": fragments["error_type_table"],
        "TYPICAL_ERROR_TABLE": fragments["typical_error_table"],
        "N_DEV": str(audit["split"]["development_rows"]),
        "N_HOLDOUT": str(audit["split"]["final_holdout_rows"]),
        "N_DEV_GROUPS": str(audit["split"]["development_groups"]),
        "N_HOLDOUT_GROUPS": str(audit["split"]["holdout_groups"]),
        "N_NEGATIVE": str(audit["target_counts_development"]["0"]),
        "N_POSITIVE": str(audit["target_counts_development"]["1"]),
        "ACCURACY": f"{baseline['fold_accuracy_mean'] * 100:.3f}%",
        "FOLD_SD": f"{baseline['fold_accuracy_sd'] * 100:.3f}",
        "ERROR_COUNT": str(fragments["primary_errors"]),
        "RANDOM_ACCURACY": f"{fragments['random_mean_accuracy'] * 100:.3f}%",
        "FIT_SECONDS": f"{metadata['total_seconds']:.2f}",
        "NUMERIC_DUPLICATES": str(
            audit["duplicate_predictor_records_development"]
        ),
        "HYPOTHESIS_FOLDS": str(summary["hypothesis"]["improved_folds"]),
        "PYTHON_VERSION": metadata["python"],
        "CONCLUSIONS": "Выводы пока не добавлены.",
    }
    conclusions = root / "docs/CONCLUSIONS.md"
    if conclusions.exists() and conclusions.read_text().strip():
        filled["CONCLUSIONS"] = conclusions.read_text().strip()
    text = (root / "docs/README.template.md").read_text()
    for name, value in filled.items():
        text = text.replace("{{" + name + "}}", value)
    # Шаблон находится в docs/, а готовый README в корне проекта.

    def rebase_link(match):
        target = match.group(1)
        if target.startswith("../"):
            target = target[3:]
        elif "://" not in target and (root / "docs" / target).is_file():
            target = "docs/" + target
        return "](" + target + ")"

    text = re.sub(r"\]\(([^)]+)\)", rebase_link, text)
    if "{{" in text:
        raise ValueError("An unresolved README field remains")
    (root / "README.md").write_text(text)
    print("README собран из измеренных результатов")


if __name__ == "__main__":
    main()
