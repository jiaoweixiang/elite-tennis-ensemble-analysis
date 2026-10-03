"""Refit the documented 33-setting hyperparameter sensitivity analysis."""

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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from models import NAMES, pipeline
from reproduce import ROOT, REF, metrics, select_auc, verify_inputs


GRID = {
    "LR": [("default", {})],
    "Ridge": [("alpha1", {"score__estimator__model__alpha": 1.0}),
              ("alpha0.1", {"score__estimator__model__alpha": 0.1}),
              ("alpha10", {"score__estimator__model__alpha": 10.0})],
    "Lasso": [("alpha0.1", {"score__estimator__model__alpha": 0.1}),
              ("alpha0.01", {"score__estimator__model__alpha": 0.01}),
              ("alpha1", {"score__estimator__model__alpha": 1.0})],
    "RF": [("leaf1", {"score__estimator__min_samples_leaf": 1}),
           ("leaf3", {"score__estimator__min_samples_leaf": 3}),
           ("leaf5", {"score__estimator__min_samples_leaf": 5})],
    "GB": [("lr0.1", {"score__estimator__learning_rate": 0.1}),
           ("lr0.03", {"score__estimator__learning_rate": 0.03}),
           ("lr0.2", {"score__estimator__learning_rate": 0.2})],
    "SVR": [("C1", {"score__estimator__model__C": 1.0}),
            ("C0.1", {"score__estimator__model__C": 0.1}),
            ("C10", {"score__estimator__model__C": 10.0})],
    "AdaB": [("trees100", {"score__estimator__n_estimators": 100}),
             ("trees50", {"score__estimator__n_estimators": 50}),
             ("trees200", {"score__estimator__n_estimators": 200})],
    "DT": [("unlimited", {"score__estimator__max_depth": None}),
           ("depth3", {"score__estimator__max_depth": 3}),
           ("depth5", {"score__estimator__max_depth": 5})],
    "KNN": [("k5", {"score__estimator__model__n_neighbors": 5}),
            ("k3", {"score__estimator__model__n_neighbors": 3}),
            ("k9", {"score__estimator__model__n_neighbors": 9})],
    "MLP": [("h64", {"score__estimator__model__hidden_layer_sizes": (64,)}),
            ("h32", {"score__estimator__model__hidden_layer_sizes": (32,)}),
            ("h128", {"score__estimator__model__hidden_layer_sizes": (128,)})],
    "GNB": [("smooth1e-9", {"score__estimator__model__var_smoothing": 1e-9}),
            ("smooth1e-10", {"score__estimator__model__var_smoothing": 1e-10}),
            ("smooth1e-8", {"score__estimator__model__var_smoothing": 1e-8})],
    "LDA": [("svd", {"score__estimator__model__solver": "svd",
                     "score__estimator__model__shrinkage": None}),
            ("lsqr_auto", {"score__estimator__model__solver": "lsqr",
                           "score__estimator__model__shrinkage": "auto"})],
}
assert list(GRID) == NAMES and sum(map(len, GRID.values())) == 33


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/refit_tuning")
    parser.add_argument("--check-reference", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "cache"
    cache.mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / "data/processed/train_grouped.csv")
    test = pd.read_csv(ROOT / "data/processed/test_grouped.csv")
    verify_inputs(train, test)
    feature_names = list(train.columns[1:35])
    x, y = train[feature_names].to_numpy(float), train.iloc[:, 0].to_numpy(int)
    xt, yt = test[feature_names].to_numpy(float), test.iloc[:, 0].to_numpy(int)
    groups = train.cv_group.to_numpy(str)
    fingerprint = hashlib.sha256((ROOT / "data/processed/train_grouped.csv").read_bytes()
                                 + Path(__file__).read_bytes()).hexdigest()

    def split(indices, k):
        indices = np.asarray(indices, dtype=int)
        cv = StratifiedGroupKFold(k, shuffle=True, random_state=42)
        for a, b in cv.split(x[indices], y[indices], groups[indices]):
            left, right = indices[a], indices[b]
            assert not set(groups[left]).intersection(groups[right])
            yield left, right

    def choose_and_fit(indices, name):
        indices = np.asarray(indices, dtype=int)
        key = hashlib.sha256((fingerprint + name + ",".join(map(str, indices))).encode()).hexdigest()[:24]
        path = cache / f"{key}.joblib"
        if path.exists():
            return joblib.load(path)
        folds = list(split(indices, 3))
        local = {int(row): j for j, row in enumerate(indices)}
        candidates = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for label, settings in GRID[name]:
                oof = np.full(len(indices), np.nan)
                for fitting, heldout in folds:
                    model = pipeline(name).set_params(**settings).fit(x[fitting], y[fitting])
                    oof[[local[int(row)] for row in heldout]] = model.decision_function(x[heldout])
                candidates.append((roc_auc_score(y[indices], oof), label, settings, oof))
            best = int(np.argmax([item[0] for item in candidates]))
            _, label, settings, oof = candidates[best]
            model = pipeline(name).set_params(**settings).fit(x[indices], y[indices])
            calibration = _SigmoidCalibration().fit(oof, y[indices])
        bundle = {"name": name, "selected_setting": label, "model": model,
                  "calibration": calibration,
                  "candidates": [{"setting": c[1], "raw_oof_auc": c[0]} for c in candidates]}
        joblib.dump(bundle, path)
        return bundle

    def tuned_block(indices, target):
        chosen = [choose_and_fit(indices, name) for name in NAMES]
        return np.column_stack([item["calibration"].predict(item["model"].decision_function(target))
                                for item in chosen])

    indices = np.arange(len(train))
    outer_member = np.full((len(train), len(NAMES)), np.nan)
    outer_score = np.full(len(train), np.nan)
    outer_pred = np.zeros(len(train), dtype=int)
    fold_rows = []
    for fold, (fitting, heldout) in enumerate(split(indices, 5), 1):
        inner = np.full((len(fitting), len(NAMES)), np.nan)
        local = {int(row): j for j, row in enumerate(fitting)}
        for subfold, (a, b) in enumerate(split(fitting, 5), 1):
            p = tuned_block(a, x[b])
            inner[[local[int(row)] for row in b]] = p
            print(f"outer {fold}, inner {subfold}", flush=True)
        choice = select_auc(inner, y[fitting])
        p = tuned_block(fitting, x[heldout])
        outer_member[heldout] = p
        score = p[:, choice["columns"]].mean(axis=1)
        pred = score >= choice["threshold_youden"]
        outer_score[heldout], outer_pred[heldout] = score, pred
        fold_rows.append({"fold": fold, "members": choice["members"],
                          "inner_auc": choice["inner_auc"]})
        print(f"outer {fold} selection: {choice['members']}", flush=True)
    final = select_auc(outer_member, y)
    test_member = tuned_block(indices, xt)
    pd.DataFrame([{"learner": name, "setting": label, "parameters": json.dumps(settings, sort_keys=True)}
                  for name, candidates in GRID.items() for label, settings in candidates]).to_csv(
                      args.output / "evaluated_settings.csv", index=False)
    pd.DataFrame([{"learner": name,
                   "selected_setting": choose_and_fit(indices, name)["selected_setting"]}
                  for name in NAMES]).to_csv(args.output / "final_selected_settings.csv", index=False)
    test_score = test_member[:, final["columns"]].mean(axis=1)
    test_pred = test_score >= final["threshold_youden"]
    result = {"outer": metrics(y, outer_score, outer_pred), "outer_folds": fold_rows,
              "final_selection": {k: v for k, v in final.items() if k != "all_auc"},
              "test_2025": metrics(yt, test_score, test_pred)}
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(args.output / "member_predictions.npz", outer_platt=outer_member, test_platt=test_member)
    pd.DataFrame({"match_id": train.match_id, "label": y, "score": outer_score,
                  "prediction": outer_pred}).to_csv(args.output / "outer_predictions.csv", index=False)
    pd.DataFrame({"match_id": test.match_id, "label": yt, "score": test_score,
                  "prediction": test_pred.astype(int)}).to_csv(args.output / "test_predictions.csv", index=False)
    if args.check_reference:
        old = np.load(REF / "tuning_base_predictions.npz")
        for name, current in (("outer_platt", outer_member), ("test_platt", test_member)):
            delta = float(np.max(np.abs(old[name] - current)))
            print(f"{name} maximum absolute difference: {delta:.3g}")
            assert delta < 1e-8
        expected = json.loads((REF / "tuning_results.json").read_text(encoding="utf-8"))
        assert final["members"] == expected["final_selection"]["members"]
        assert abs(result["outer"]["auc"] - expected["outer"]["auc"]) < 1e-8
        assert abs(result["test_2025"]["auc"] - expected["test_2025"]["auc"]) < 1e-8
    print(f"Tuning sensitivity refit complete: {args.output}")


if __name__ == "__main__":
    main()
