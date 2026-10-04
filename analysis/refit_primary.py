"""Refit the primary grouped, calibrated AUC-selection workflow from set tables.

This is intentionally separate from the quick numerical audit in reproduce.py.
It retrains all 12 base learners in every outer, inner and calibration fold;
depending on the computer, a complete run can take considerably longer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import warnings
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import _SigmoidCalibration
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import rankdata

from models import NAMES, pipeline
from reproduce import ROOT, REF, COMBOS, WEIGHTS, metrics, select_auc, verify_inputs, youden_threshold


def select_f1(matrix: np.ndarray, labels: np.ndarray) -> dict:
    """Sensitivity comparator: maximize training OOF F1, then accuracy.

    Each subset uses the same training-only Youden threshold rule as the
    primary AUC selector. Original subset order resolves any remaining tie.
    """
    scores = matrix @ WEIGHTS
    order = np.argsort(-scores, axis=0, kind="stable")
    sorted_scores = np.take_along_axis(scores, order, axis=0)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels, axis=0)
    fp = np.cumsum(1 - sorted_labels, axis=0)
    distinct = np.vstack((sorted_scores[:-1] != sorted_scores[1:],
                          np.ones((1, scores.shape[1]), dtype=bool)))
    j = np.where(distinct, tp / labels.sum() - fp / (len(labels) - labels.sum()), -np.inf)
    pos = np.argmax(np.vstack((np.zeros((1, scores.shape[1])), j)), axis=0)
    columns = np.arange(scores.shape[1])
    thresholds = np.where(pos == 0, np.inf, sorted_scores[np.maximum(pos - 1, 0), columns])
    pred = scores >= thresholds
    true_positive = np.sum(pred & (labels[:, None] == 1), axis=0)
    false_positive = np.sum(pred & (labels[:, None] == 0), axis=0)
    false_negative = labels.sum() - true_positive
    f1 = 2 * true_positive / (2 * true_positive + false_positive + false_negative)
    accuracy = np.mean(pred == labels[:, None], axis=0)
    ties = np.flatnonzero(f1 == f1.max())
    ties = ties[accuracy[ties] == accuracy[ties].max()]
    index = int(ties[0])
    selected_scores = scores[:, index]
    ranks = rankdata(selected_scores, method="average")
    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    auc = (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return {"index": index, "columns": list(COMBOS[index]),
            "members": [NAMES[i] for i in COMBOS[index]],
            "inner_auc": float(auc), "inner_f1": float(f1[index]),
            "threshold_youden": float(thresholds[index])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("full", "two", "eight", "backhand"), default="full")
    parser.add_argument("--criterion", choices=("auc", "f1"), default="auc")
    parser.add_argument("--drop-feature", help="Fixed-member refit after removing one named indicator")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-reference", action="store_true",
                        help="Require numerical agreement with the deposited predictions")
    args = parser.parse_args()
    if args.criterion == "f1" and (args.variant != "full" or args.drop_feature):
        parser.error("F1-selection sensitivity is defined only for the full indicator set")
    if args.drop_feature and args.variant != "full":
        parser.error("--drop-feature requires --variant full")
    if args.drop_feature and args.check_reference and args.drop_feature != "Backhand Returns":
        parser.error("A reference check is available only for Backhand Returns")
    if args.output is None:
        suffix = ("refit_drop_" + re.sub(r"[^a-z0-9]+", "_", args.drop_feature.lower()).strip("_")
                  if args.drop_feature else
                  "refit_f1" if args.criterion == "f1" else
                  "refit" if args.variant == "full" else f"refit_{args.variant}")
        args.output = ROOT / "outputs" / suffix
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "cache"
    cache.mkdir(exist_ok=True)

    train = pd.read_csv(ROOT / "data/processed/train_grouped.csv")
    test = pd.read_csv(ROOT / "data/processed/test_grouped.csv")
    verify_inputs(train, test)
    if args.drop_feature and args.drop_feature not in train.columns[1:35]:
        parser.error(f"Unknown indicator: {args.drop_feature}")
    removed = {"full": [], "backhand": ["Backhand Returns"],
               "two": ["Service Breaks", "Total Points Won"],
               "eight": ["Service Breaks", "Total Points Won", "Double Faults", "Unforced Errors",
                         "Backhand Return Win ", "Forehand Return Win ", "1st Serve Pts Won ",
                         "2nd Serve Pts Won "]}[args.variant]
    if args.drop_feature:
        removed = [args.drop_feature]
    fixed_members = args.variant == "backhand" or bool(args.drop_feature)
    selector = select_auc if args.criterion == "auc" else select_f1
    features = [name for name in train.columns[1:35] if name not in removed]
    x, y = train[features].to_numpy(float), train.iloc[:, 0].to_numpy(int)
    xt, yt = test[features].to_numpy(float), test.iloc[:, 0].to_numpy(int)
    groups = train.cv_group.to_numpy(str)
    fingerprint = hashlib.sha256((ROOT / "data/processed/train_grouped.csv").read_bytes()
                                 + (ROOT / "data/processed/test_grouped.csv").read_bytes()
                                 + (Path(__file__).with_name("models.py")).read_bytes()
                                 + Path(__file__).read_bytes()
                                 + "|".join(features).encode("utf-8")).hexdigest()

    def split(indices: np.ndarray, n: int):
        indices = np.asarray(indices, dtype=int)
        cv = StratifiedGroupKFold(n, shuffle=True, random_state=42)
        for a, b in cv.split(x[indices], y[indices], groups[indices]):
            fitting, heldout = indices[a], indices[b]
            assert not set(groups[fitting]).intersection(groups[heldout])
            assert len(np.unique(y[fitting])) == 2
            yield fitting, heldout

    def fit_block(indices: np.ndarray, target: np.ndarray, tag: str, save_models=False):
        indices = np.asarray(indices, dtype=int)
        digest = hashlib.sha256((fingerprint + "|" + ",".join(map(str, indices))
                                 + "|" + hashlib.sha256(target.tobytes()).hexdigest()).encode()).hexdigest()[:24]
        path = cache / f"{digest}.npz"
        if path.exists() and not save_models:
            return np.load(path)["probabilities"], None
        local = {int(row): j for j, row in enumerate(indices)}
        raw_oof = np.full((len(indices), len(NAMES)), np.nan)
        raw_target = np.empty((len(target), len(NAMES)))
        fitted, calibration = {}, {}
        cal_folds = list(split(indices, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for column, name in enumerate(NAMES):
                for fitting, heldout in cal_folds:
                    model = pipeline(name).fit(x[fitting], y[fitting])
                    raw_oof[[local[int(row)] for row in heldout], column] = model.decision_function(x[heldout])
                fitted[name] = pipeline(name).fit(x[indices], y[indices])
                raw_target[:, column] = fitted[name].decision_function(target)
                calibration[name] = _SigmoidCalibration().fit(raw_oof[:, column], y[indices])
        assert np.isfinite(raw_oof).all()
        probability = np.column_stack([calibration[name].predict(raw_target[:, j])
                                       for j, name in enumerate(NAMES)])
        assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
        np.savez_compressed(path, probabilities=probability)
        print(f"{tag}: {len(indices)} fitting sets, {len(target)} predictions", flush=True)
        return probability, {"models": fitted, "calibrators": calibration} if save_models else None

    indices = np.arange(len(train))
    member_outer = np.full((len(train), len(NAMES)), np.nan)
    outer_score = np.full(len(train), np.nan)
    outer_pred = np.zeros(len(train), dtype=int)
    audit_singles = args.variant == "full" and args.criterion == "auc" and not args.drop_feature
    if audit_singles:
        single_outer_pred = np.zeros((len(train), len(NAMES)), dtype=int)
        single_fold_thresholds = np.full((5, len(NAMES)), np.nan)
    records = []
    primary = json.loads((REF / "primary_results.json").read_text(encoding="utf-8"))["full"]
    for fold, (fitting, heldout) in enumerate(split(indices, 5), 1):
        if fixed_members:
            choice = primary["outer_folds"][fold - 1]["selection"]
        else:
            inner = np.full((len(fitting), len(NAMES)), np.nan)
            local = {int(row): j for j, row in enumerate(fitting)}
            for subfold, (a, b) in enumerate(split(fitting, 5), 1):
                p, _ = fit_block(a, x[b], f"outer {fold}, inner {subfold}")
                inner[[local[int(row)] for row in b]] = p
            assert np.isfinite(inner).all()
            choice = selector(inner, y[fitting])
        probabilities, _ = fit_block(fitting, x[heldout], f"outer {fold}, final fit")
        member_outer[heldout] = probabilities
        if audit_singles:
            single_fold_thresholds[fold - 1] = [youden_threshold(y[fitting], inner[:, j])
                                                 for j in range(len(NAMES))]
            single_outer_pred[heldout] = probabilities >= single_fold_thresholds[fold - 1]
        score = probabilities[:, choice["columns"]].mean(axis=1)
        pred = score >= choice["threshold_youden"]
        outer_score[heldout], outer_pred[heldout] = score, pred
        records.append({"fold": fold, "members": choice["members"],
                        "inner_auc": choice["inner_auc"],
                        "threshold": choice["threshold_youden"],
                        "validation": metrics(y[heldout], score, pred)})
        print(f"outer {fold}/5: {choice['members']}", flush=True)

    final = primary["final_selection"] if fixed_members else selector(member_outer, y)
    test_members, bundle = fit_block(indices, xt, "all-development final fit", save_models=True)
    test_score = test_members[:, final["columns"]].mean(axis=1)
    test_pred = test_score >= final["threshold_youden"]
    if audit_singles:
        single_final_thresholds = [youden_threshold(y, member_outer[:, j]) for j in range(len(NAMES))]
        single_rows = []
        expected_singles = json.loads((REF / "single_model_results.json").read_text(encoding="utf-8"))["models"]
        for j, name in enumerate(NAMES):
            for sample, labels, scores, predictions in (
                ("outer", y, member_outer[:, j], single_outer_pred[:, j]),
                ("2025", yt, test_members[:, j], test_members[:, j] >= single_final_thresholds[j]),
            ):
                measured = metrics(labels, scores, predictions)
                expected = expected_singles[name]["full"]["outer" if sample == "outer" else "test_2025"]
                for key in ("auc", "accuracy", "precision", "f1", "balanced_accuracy", "brier"):
                    assert abs(measured[key] - expected[key]) < 1e-8, (name, sample, key)
                single_rows.append({"model": name, "sample": sample, **measured})
        pd.DataFrame(single_rows).to_csv(args.output / "single_model_performance_refit.csv", index=False)
        pd.DataFrame(single_fold_thresholds, columns=NAMES).to_csv(
            args.output / "single_model_outer_thresholds.csv", index=False)
        pd.DataFrame({"model": NAMES, "development_oof_threshold": single_final_thresholds}).to_csv(
            args.output / "single_model_final_thresholds.csv", index=False)
    result = {"variant": args.variant, "removed_features": removed,
              "selection_rule": ("fixed original AUC-selected membership and thresholds" if fixed_members else
                                 "max grouped inner OOF AUC; ties by fewer members then original order" if args.criterion == "auc" else
                                 "max grouped inner OOF F1; ties by accuracy then original order"),
              "calibration": "three-fold match-grouped OOF sigmoid (Platt) per learner",
              "outer": metrics(y, outer_score, outer_pred),
              "outer_folds": records,
              "final_selection": {k: v for k, v in final.items() if k != "all_auc"},
              "test_2025": metrics(yt, test_score, test_pred)}
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame({"match_id": train.match_id, "label": y,
                  "score": outer_score, "prediction": outer_pred}).to_csv(args.output / "outer_predictions.csv", index=False)
    pd.DataFrame({"match_id": test.match_id, "label": yt,
                  "score": test_score, "prediction": test_pred.astype(int)}).to_csv(args.output / "test_predictions.csv", index=False)
    np.savez_compressed(args.output / "member_predictions.npz",
                        outer_members=member_outer, train_platt_oof=member_outer, test_members=test_members)
    joblib.dump({**bundle, "features": features, "selection": result["final_selection"]}, args.output / "final_models.joblib")

    if args.check_reference:
        if args.criterion == "f1":
            expected_outer = pd.read_csv(REF / "selection_comparators_outer_predictions.csv")
            expected_test = pd.read_csv(REF / "selection_comparators_test_predictions.csv")
            for label, reference, scores, predictions in (
                ("outer", expected_outer, outer_score, outer_pred),
                ("2025", expected_test, test_score, test_pred),
            ):
                difference = float(np.max(np.abs(reference.f1_selector_score - scores)))
                print(f"{label} F1-selection score maximum absolute difference: {difference:.3g}")
                assert difference < 1e-8
                assert np.array_equal(reference.f1_selector_pred.to_numpy(int), predictions.astype(int))
            expected = result
        elif args.drop_feature == "Backhand Returns" or args.variant == "backhand":
            reference_outer = pd.read_csv(REF / "backhand_ablation_outer_predictions.csv")
            reference_test = pd.read_csv(REF / "backhand_ablation_test_predictions.csv")
            for label, old, new in (("outer", reference_outer.without_backhand, outer_score),
                                    ("test", reference_test.without_backhand, test_score)):
                delta = float(np.max(np.abs(old - new)))
                print(f"{label} score maximum absolute difference: {delta:.3g}")
                assert delta < 1e-8
            expected = primary
        elif args.variant == "full":
            saved = np.load(REF / "prediction_matrices.npz")
            for key, new in (("outer_members", member_outer), ("test_members", test_members)):
                delta = float(np.max(np.abs(saved[key] - new)))
                print(f"{key} maximum absolute difference: {delta:.3g}")
                assert delta < 1e-8
            expected = json.loads((REF / "primary_results.json").read_text(encoding="utf-8"))["full"]
        else:
            key = "two_feature" if args.variant == "two" else "eight_feature"
            reference_outer = pd.read_csv(REF / f"{key}_outer_predictions.csv")
            reference_test = pd.read_csv(REF / f"{key}_test_predictions.csv")
            for label, old, new in (("outer", reference_outer.score, outer_score),
                                    ("test", reference_test.score, test_score)):
                delta = float(np.max(np.abs(old - new)))
                print(f"{label} score maximum absolute difference: {delta:.3g}")
                assert delta < 1e-8
            expected = json.loads((REF / f"{key}_results.json").read_text(encoding="utf-8"))
        assert final["members"] == expected["final_selection"]["members"]
        if fixed_members:
            ablation = json.loads((REF / "backhand_ablation.json").read_text(encoding="utf-8"))
            assert abs(result["outer"]["auc"] - ablation["outer"]["without_backhand_auc"]) < 1e-8
            assert abs(result["test_2025"]["auc"] - ablation["test_2025"]["without_backhand_auc"]) < 1e-8
        else:
            assert abs(result["outer"]["auc"] - expected["outer"]["auc"]) < 1e-8
            assert abs(result["test_2025"]["auc"] - expected["test_2025"]["auc"]) < 1e-8
    print(f"Refit complete: {args.output}")


if __name__ == "__main__":
    main()
