"""Refit the three classification comparators reported in Table 4."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold

from models import BASELINE_NAMES, baseline_pipeline
from reproduce import ROOT, REF, metrics, verify_inputs, youden_threshold


FAMILIES = {"Logistic L2": list(range(0, 3)),
            "Logistic L1": list(range(3, 6)),
            "RF classifier": [6]}
REFERENCE_NAMES = {"Logistic L2": "logistic_l2",
                   "Logistic L1": "logistic_l1",
                   "RF classifier": "random_forest"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/refit_baselines")
    parser.add_argument("--check-reference", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data/processed/train_grouped.csv")
    test = pd.read_csv(ROOT / "data/processed/test_grouped.csv")
    verify_inputs(train, test)
    features = list(train.columns[1:35])
    x, y = train[features].to_numpy(float), train.iloc[:, 0].to_numpy(int)
    xt, yt = test[features].to_numpy(float), test.iloc[:, 0].to_numpy(int)
    groups = train.cv_group.to_numpy(str)
    all_rows = np.arange(len(train))

    def split(indices: np.ndarray):
        indices = np.asarray(indices, dtype=int)
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
        for left, right in cv.split(x[indices], y[indices], groups[indices]):
            fitting, heldout = indices[left], indices[right]
            assert not set(groups[fitting]).intersection(groups[heldout])
            yield fitting, heldout

    def fit_predict(fitting: np.ndarray, target: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.column_stack([
                baseline_pipeline(name).fit(x[fitting], y[fitting]).predict_proba(target)[:, 1]
                for name in BASELINE_NAMES
            ])

    def choose(oof: np.ndarray, labels: np.ndarray) -> dict:
        selections = {}
        for family, candidates in FAMILIES.items():
            selected = min(candidates, key=lambda j: (brier_score_loss(labels, oof[:, j]), j))
            selections[family] = {"candidate": BASELINE_NAMES[selected], "column": selected,
                                  "oof_brier": brier_score_loss(labels, oof[:, selected]),
                                  "threshold": youden_threshold(labels, oof[:, selected])}
        return selections

    names = list(FAMILIES)
    outer_candidate_scores = np.full((len(train), len(BASELINE_NAMES)), np.nan)
    outer_scores = np.full((len(train), len(names)), np.nan)
    outer_predictions = np.zeros((len(train), len(names)), dtype=int)
    outer_choices = []
    for fold, (fitting, heldout) in enumerate(split(all_rows), 1):
        inner = np.full((len(fitting), len(BASELINE_NAMES)), np.nan)
        local = {int(row): j for j, row in enumerate(fitting)}
        for inner_fitting, inner_heldout in split(fitting):
            inner[[local[int(row)] for row in inner_heldout]] = fit_predict(inner_fitting, x[inner_heldout])
        assert np.isfinite(inner).all()
        choices = choose(inner, y[fitting])
        outer_choices.append(choices)
        candidates = fit_predict(fitting, x[heldout])
        outer_candidate_scores[heldout] = candidates
        for j, family in enumerate(names):
            chosen = choices[family]
            outer_scores[heldout, j] = candidates[:, chosen["column"]]
            outer_predictions[heldout, j] = outer_scores[heldout, j] >= chosen["threshold"]
        print(f"Outer fold {fold}/5 complete", flush=True)

    final_choices = choose(outer_candidate_scores, y)
    test_candidates = fit_predict(all_rows, xt)
    test_scores = np.column_stack([test_candidates[:, final_choices[family]["column"]]
                                   for family in names])
    test_predictions = np.column_stack([
        test_scores[:, j] >= final_choices[family]["threshold"]
        for j, family in enumerate(names)
    ]).astype(int)
    rows = []
    for j, family in enumerate(names):
        for sample, labels, scores, predictions in (
            ("outer", y, outer_scores[:, j], outer_predictions[:, j]),
            ("2025", yt, test_scores[:, j], test_predictions[:, j]),
        ):
            rows.append({"model": family, "sample": sample,
                         **metrics(labels, scores, predictions)})
    pd.DataFrame(rows).to_csv(args.output / "baseline_performance.csv", index=False)
    for label, source, scores, predictions in (
        ("outer", train, outer_scores, outer_predictions),
        ("test", test, test_scores, test_predictions),
    ):
        frame = pd.DataFrame({"match_id": source.match_id, "label": source.iloc[:, 0]})
        for j, family in enumerate(names):
            key = REFERENCE_NAMES[family]
            frame[key + "_score"] = scores[:, j]
            frame[key + "_pred"] = predictions[:, j]
        frame.to_csv(args.output / f"{label}_predictions.csv", index=False)
    (args.output / "selections.json").write_text(json.dumps({"outer": outer_choices,
                                                              "final": final_choices}, indent=2), encoding="utf-8")

    if args.check_reference:
        for label in ("outer", "test"):
            actual = pd.read_csv(args.output / f"{label}_predictions.csv")
            expected = pd.read_csv(REF / f"selection_comparators_{label}_predictions.csv")
            assert actual.match_id.equals(expected.match_id)
            for family in names:
                key = REFERENCE_NAMES[family]
                difference = float(np.max(np.abs(actual[key + "_score"] - expected[key + "_score"])))
                assert difference < 1e-8, (label, family, difference)
                assert np.array_equal(actual[key + "_pred"], expected[key + "_pred"])
        print("All three classification-comparator predictions match the deposited records")
    print(f"Comparator results saved to {args.output}")


if __name__ == "__main__":
    main()
