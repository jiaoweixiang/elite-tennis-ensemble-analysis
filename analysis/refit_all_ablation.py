"""Refit all 34 fixed-member leave-one-indicator-out comparisons (2025 test AUC).

Each indicator is deleted in turn; the primary model's members and threshold
stay fixed while feature screening, learners and Platt calibration are refitted.
The resulting 2025 deletion-refit AUC decreases are the source of Figure 4b and
of the all-indicator ranking computed by ablation_ranks.py.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from reproduce import ROOT, REF


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1, help="Number of independent refits to run concurrently")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/feature_ablation_refits")
    parser.add_argument("--check-reference", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    features = list(pd.read_csv(ROOT / "data/processed/train_grouped.csv").columns[1:35])
    primary = json.loads((REF / "primary_results.json").read_text(encoding="utf-8"))["full"]
    test_full = primary["test_2025"]

    def fit(item: tuple[int, str]) -> dict:
        index, feature = item
        directory = args.output / f"feature_{index + 1:02d}"
        command = [sys.executable, str(Path(__file__).with_name("refit_primary.py")),
                   "--drop-feature", feature, "--output", str(directory)]
        env = os.environ.copy()
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
            env[variable] = "1"
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"Refit failed for {feature}:\n{completed.stdout}\n{completed.stderr}")
        result = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        test = result["test_2025"]
        return {"feature": feature,
                "test_full_auc": test_full["auc"], "test_without_auc": test["auc"],
                "test_auc_drop": test_full["auc"] - test["auc"],
                "selected_member_rule": "fixed AUC-selected final membership and threshold"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        rows = list(executor.map(fit, enumerate(features)))
    table = pd.DataFrame(rows)
    table = table.sort_values("test_auc_drop", ascending=False, kind="stable")
    path = args.output / "feature_ablation.csv"
    table.to_csv(path, index=False)
    if args.check_reference:
        expected = pd.read_csv(REF / "feature_ablation.csv", encoding="utf-8-sig")
        table = table.set_index("feature").loc[expected.feature].reset_index()
        for column in table.select_dtypes(include=np.number):
            difference = np.max(np.abs(table[column].to_numpy(float) - expected[column].to_numpy(float)))
            assert difference < 1e-8, (column, difference)
        print(f"All {len(table)} feature-removal rows match the deposited panel source")
    print(f"Feature-removal data saved to {path}")


if __name__ == "__main__":
    main()
