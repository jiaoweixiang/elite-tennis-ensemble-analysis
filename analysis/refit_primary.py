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
import warnings
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import _SigmoidCalibration
from sklearn.model_selection import StratifiedGroupKFold

from models import NAMES, pipeline
from reproduce import ROOT, REF, metrics, select_auc, verify_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("full", "two", "eight", "backhand"), default="full")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-reference", action="store_true",
                        help="Require numerical agreement with the deposited predictions")
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / ("outputs/refit" if args.variant == "full" else f"outputs/refit_{args.variant}")
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "cache"
    cache.mkdir(exist_ok=True)

    train = pd.read_csv(ROOT / "data/processed/train_grouped.csv")
    test = pd.read_csv(ROOT / "data/processed/test_grouped.csv")
    verify_inputs(train, test)
    removed = {"full": [], "backhand": ["Backhand Returns"],
               "two": ["Service Breaks", "Total Points Won"],
               "eight": ["Service Breaks", "Total Points Won", "Double Faults", "Unforced Errors",
                         "Backhand Return Win ", "Forehand Return Win ", "1st Serve Pts Won ",
                         "2nd Serve Pts Won "]}[args.variant]
    features = [name for name in train.columns[1:35] if name not in removed]
    x, y = train[features].to_numpy(float), train.iloc[:, 0].to_numpy(int)
    xt, yt = test[features].to_numpy(float), test.iloc[:, 0].to_numpy(int)
    groups = train.cv_group.to_numpy(str)
    fingerprint = hashlib.sha256((ROOT / "data/processed/train_grouped.csv").read_bytes()
                                 + (ROOT / "data/processed/test_grouped.csv").read_bytes()
                                 + (Path(__file__).with_name("models.py")).read_bytes()).hexdigest()

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
    records = []
    primary = json.loads((REF / "primary_results.json").read_text(encoding="utf-8"))["full"]
    for fold, (fitting, heldout) in enumerate(split(indices, 5), 1):
        if args.variant == "backhand":
            choice = primary["outer_folds"][fold - 1]["selection"]
        else:
            inner = np.full((len(fitting), len(NAMES)), np.nan)
            local = {int(row): j for j, row in enumerate(fitting)}
            for subfold, (a, b) in enumerate(split(fitting, 5), 1):
                p, _ = fit_block(a, x[b], f"outer {fold}, inner {subfold}")
                inner[[local[int(row)] for row in b]] = p
            assert np.isfinite(inner).all()
            choice = select_auc(inner, y[fitting])
        probabilities, _ = fit_block(fitting, x[heldout], f"outer {fold}, final fit")
        member_outer[heldout] = probabilities
        score = probabilities[:, choice["columns"]].mean(axis=1)
        pred = score >= choice["threshold_youden"]
        outer_score[heldout], outer_pred[heldout] = score, pred
        records.append({"fold": fold, "members": choice["members"],
                        "inner_auc": choice["inner_auc"],
                        "threshold": choice["threshold_youden"],
                        "validation": metrics(y[heldout], score, pred)})
        print(f"outer {fold}/5: {choice['members']}", flush=True)

    final = primary["final_selection"] if args.variant == "backhand" else select_auc(member_outer, y)
    test_members, bundle = fit_block(indices, xt, "all-development final fit", save_models=True)
    test_score = test_members[:, final["columns"]].mean(axis=1)
    test_pred = test_score >= final["threshold_youden"]
    result = {"variant": args.variant, "removed_features": removed,
              "selection_rule": "max grouped inner OOF AUC; ties by fewer members then original order",
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
        if args.variant == "full":
            saved = np.load(REF / "prediction_matrices.npz")
            for key, new in (("outer_members", member_outer), ("test_members", test_members)):
                delta = float(np.max(np.abs(saved[key] - new)))
                print(f"{key} maximum absolute difference: {delta:.3g}")
                assert delta < 1e-8
            expected = json.loads((REF / "primary_results.json").read_text(encoding="utf-8"))["full"]
        elif args.variant == "backhand":
            reference_outer = pd.read_csv(REF / "backhand_ablation_outer_predictions.csv")
            reference_test = pd.read_csv(REF / "backhand_ablation_test_predictions.csv")
            for label, old, new in (("outer", reference_outer.without_backhand, outer_score),
                                    ("test", reference_test.without_backhand, test_score)):
                delta = float(np.max(np.abs(old - new)))
                print(f"{label} score maximum absolute difference: {delta:.3g}")
                assert delta < 1e-8
            expected = primary
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
        if args.variant == "backhand":
            ablation = json.loads((REF / "backhand_ablation.json").read_text(encoding="utf-8"))
            assert abs(result["outer"]["auc"] - ablation["outer"]["without_backhand_auc"]) < 1e-8
            assert abs(result["test_2025"]["auc"] - ablation["test_2025"]["without_backhand_auc"]) < 1e-8
        else:
            assert abs(result["outer"]["auc"] - expected["outer"]["auc"]) < 1e-8
            assert abs(result["test_2025"]["auc"] - expected["test_2025"]["auc"]) < 1e-8
    print(f"Refit complete: {args.output}")


if __name__ == "__main__":
    main()
