"""Refit all 34 fixed-member leave-one-indicator-out comparisons.

This is the numeric source for the manuscript's feature-removal panel. The
ensemble members and Youden thresholds stay fixed at the primary model's
training-only choices while feature screening, learners and Platt calibration
are fitted again after each indicator is removed.
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
    outer_full = primary["outer"]
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
        outer, test = result["outer"], result["test_2025"]
        return {"feature": feature,
                "outer_full_auc": outer_full["auc"], "outer_without_auc": outer["auc"],
                "outer_auc_drop": outer_full["auc"] - outer["auc"],
                "test_full_auc": test_full["auc"], "test_without_auc": test["auc"],
                "test_auc_drop": test_full["auc"] - test["auc"],
                "outer_brier_increase": outer["brier"] - outer_full["brier"],
                "test_brier_increase": test["brier"] - test_full["brier"],
                "selected_member_rule": "fixed AUC-selected membership within each fold"}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        rows = list(executor.map(fit, enumerate(features)))
    table = pd.DataFrame(rows)
    table["outer_auc_drop_rank"] = table.outer_auc_drop.rank(ascending=False, method="first").astype(int)
    table = table.sort_values("outer_auc_drop_rank")
    path = args.output / "feature_ablation.csv"
    table.to_csv(path, index=False)
    if args.check_reference:
        expected = pd.read_csv(REF / "feature_ablation.csv", encoding="utf-8-sig")
        assert table.feature.tolist() == expected.feature.tolist()
        for column in table.select_dtypes(include=np.number):
            difference = np.max(np.abs(table[column].to_numpy(float) - expected[column].to_numpy(float)))
            assert difference < 1e-8, (column, difference)
        print(f"All {len(table)} feature-removal rows match the deposited panel source")
    print(f"Feature-removal data saved to {path}")


if __name__ == "__main__":
    main()
