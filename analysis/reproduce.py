"""Recalculate the numerical results from the deposited model outputs.

The input tables and frozen member predictions are checked before use. For an
independent rerun of model fitting, see ``analysis/refit_primary.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve, r2_score,
)
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REF = DATA / "reference"
NAMES = ["LR", "Ridge", "Lasso", "RF", "GB", "SVR", "AdaB", "DT", "KNN", "MLP", "GNB", "LDA"]
COMBOS = [c for size in range(1, 13) for c in itertools.combinations(range(12), size)]
WEIGHTS = np.zeros((12, len(COMBOS)))
for j, combo in enumerate(COMBOS):
    WEIGHTS[list(combo), j] = 1 / len(combo)


def read_json(name: str) -> dict:
    return json.loads((REF / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(train: pd.DataFrame, test: pd.DataFrame) -> None:
    reference = read_json("primary_results.json")
    assert sha256(DATA / "processed/train_grouped.csv") == reference["train_hash_sha256"]
    assert sha256(DATA / "processed/test_grouped.csv") == reference["test_hash_sha256"]
    assert len(train) == 199 and len(test) == 40
    assert (train.iloc[:, 0] == 1).sum() == 163 and (test.iloc[:, 0] == 1).sum() == 30
    assert train.match_id.nunique() == 59 and test.match_id.nunique() == 12
    assert not set(train.match_id).intersection(test.match_id)
    assert set(train.year) == {2019, 2020, 2021, 2023, 2024}
    assert set(test.year) == {2025}
    assert list(train.columns[1:35]) == list(test.columns[1:35])
    for table in (train, test):
        depth = table[["Forecourt Short Balls", "Midcourt Balls", "Backcourt Deep Balls"]]
        assert np.allclose(depth.sum(axis=1), 1, atol=0.011)


def youden_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Training-only Youden J with the original stable tie rule."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_y = scores[order], y[order]
    tp, fp = np.cumsum(sorted_y), np.cumsum(1 - sorted_y)
    distinct = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    j = np.where(distinct, tp / (y == 1).sum() - fp / (y == 0).sum(), -np.inf)
    pos = int(np.argmax(np.r_[0.0, j]))
    return float(sorted_scores[pos - 1]) if pos else float("inf")


def select_auc(matrix: np.ndarray, y: np.ndarray) -> dict:
    scores = matrix @ WEIGHTS
    ranks = rankdata(scores, axis=0, method="average")
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    auc = (ranks[y == 1].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    ties = np.flatnonzero(np.isclose(auc, auc.max(), atol=1e-12, rtol=0))
    index = int(ties[0])  # COMBOS is ordered by size, then model order.
    return {
        "index": index, "columns": list(COMBOS[index]),
        "members": [NAMES[i] for i in COMBOS[index]],
        "inner_auc": float(auc[index]), "auc_tie_count": int(len(ties)),
        "threshold_youden": youden_threshold(y, scores[:, index]),
        "all_auc": auc,
    }


def metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "auc": roc_auc_score(y, score),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "brier": brier_score_loss(y, score),
    }


def reconstruct_primary(train: pd.DataFrame, test: pd.DataFrame, out: Path) -> tuple[np.ndarray, np.ndarray]:
    arrays = np.load(REF / "prediction_matrices.npz")
    expected = read_json("primary_results.json")["full"]
    y, yt = train.iloc[:, 0].to_numpy(int), test.iloc[:, 0].to_numpy(int)
    groups = train.cv_group.to_numpy(str)
    outer_scores, outer_preds = np.full(len(y), np.nan), np.zeros(len(y), dtype=int)
    fold_rows = []
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=42).split(train.iloc[:, 1:35], y, groups)
    for fold, (fitting, heldout) in enumerate(outer, 1):
        assert not set(groups[fitting]).intersection(groups[heldout])
        choice = expected["outer_folds"][fold - 1]["selection"]
        p = arrays["outer_members"][heldout][:, choice["columns"]].mean(axis=1)
        outer_scores[heldout], outer_preds[heldout] = p, p >= choice["threshold_youden"]
        fold_rows.append({"fold": fold, "members": "+".join(choice["members"]),
                          "inner_auc": choice["inner_auc"], "n_validation_sets": len(heldout),
                          **metrics(y[heldout], p, outer_preds[heldout])})
    assert np.isfinite(outer_scores).all()
    final = select_auc(arrays["train_platt_oof"], y)
    assert final["members"] == expected["final_selection"]["members"]
    assert abs(final["inner_auc"] - expected["final_selection"]["inner_auc"]) < 1e-10
    assert abs(final["threshold_youden"] - expected["final_selection"]["threshold_youden"]) < 1e-10
    test_scores = arrays["test_members"][:, final["columns"]].mean(axis=1)
    test_preds = test_scores >= final["threshold_youden"]
    old_outer = pd.read_csv(REF / "outer_predictions.csv")
    old_test = pd.read_csv(REF / "test_predictions.csv")
    assert np.allclose(outer_scores, old_outer.auc_selector_score, atol=1e-12)
    assert np.allclose(test_scores, old_test.auc_selector_score, atol=1e-12)
    rows = [{"sample": "2019-2024 grouped outer", "sets": len(y), "matches": train.match_id.nunique(),
             "members": "reselected in each outer fold", **metrics(y, outer_scores, outer_preds)},
            {"sample": "2025 temporal", "sets": len(yt), "matches": test.match_id.nunique(),
             "members": "+".join(final["members"]), **metrics(yt, test_scores, test_preds)}]
    pd.DataFrame(rows).to_csv(out / "primary_performance.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "outer_folds.csv", index=False)
    pd.DataFrame({"subset_order": np.arange(1, 4096),
                  "members": ["+".join(NAMES[i] for i in c) for c in COMBOS],
                  "size": [len(c) for c in COMBOS], "development_oof_auc": final["all_auc"]}).to_csv(
                      out / "ensemble_candidates_4095.csv", index=False)
    pd.DataFrame({"match_id": train.match_id, "label": y,
                  "predicted_p_win": outer_scores, "predicted_win": outer_preds}).to_csv(
                      out / "primary_outer_predictions.csv", index=False)
    pd.DataFrame({"match_id": test.match_id, "label": yt,
                  "predicted_p_win": test_scores, "predicted_win": test_preds.astype(int)}).to_csv(
                      out / "primary_2025_predictions.csv", index=False)
    (out / "final_selection.json").write_text(json.dumps({k: v for k, v in final.items() if k != "all_auc"}, indent=2), encoding="utf-8")
    return outer_scores, test_scores


def reproduce_shap(test: pd.DataFrame, out: Path) -> None:
    files = [np.load(REF / f"full_auc_SHAP_seed{seed}.npz") for seed in (42, 142, 242)]
    features = files[0]["features"]
    assert all(np.array_equal(z["features"], features) for z in files)
    absolute = np.abs(np.stack([z["values"] for z in files])).mean(axis=0)
    importance = absolute.mean(axis=0)
    order = np.argsort(-importance, kind="stable")
    pd.DataFrame({"rank": np.arange(1, len(order) + 1), "feature": features[order],
                  "mean_absolute_SHAP": importance[order]}).to_csv(out / "ensemble_shap_ranks.csv", index=False)
    mean_values = np.stack([z["values"] for z in files]).mean(axis=0)
    rows = []
    for j, feature in enumerate(features):
        for i in range(len(test)):
            rows.append({"feature": feature, "rank": int(np.flatnonzero(order == j)[0] + 1),
                         "set_row": i + 1, "match_id": test.match_id.iat[i],
                         "set_result": int(test.iloc[i, 0]),
                         "feature_value": float(test[feature].iat[i]),
                         "mean_SHAP": float(mean_values[i, j])})
    pd.DataFrame(rows).to_csv(out / "ensemble_shap_per_set.csv", index=False)
    backhand = int(np.flatnonzero(features == "Backhand Returns")[0])
    by_match = [np.flatnonzero(test.match_id.to_numpy(str) == g) for g in np.unique(test.match_id)]
    rng = np.random.RandomState(42)
    ranks = []
    backhand_magnitudes = []
    for _ in range(2000):
        draw = np.concatenate([by_match[j] for j in rng.randint(len(by_match), size=len(by_match))])
        imp = absolute[draw].mean(axis=0)
        ranks.append(int(np.flatnonzero(np.argsort(-imp, kind="stable") == backhand)[0] + 1))
        backhand_magnitudes.append(float(imp[backhand]))
    pd.DataFrame({"resample": np.arange(1, 2001), "backhand_rank": ranks,
                  "backhand_mean_absolute_SHAP": backhand_magnitudes}).to_csv(
        out / "backhand_match_resample_ranks.csv", index=False)
    original = read_json("primary_results.json")["full"]["shap_2025"]
    assert np.mean(np.asarray(ranks) <= 6) == original["backhand_top6_match_bootstrap_frequency"]


def reproduce_exclusions(train: pd.DataFrame, test: pd.DataFrame, out: Path) -> None:
    rows = []
    for name, prefix in (("two score-related measures", "two_feature"),
                         ("eight outcome-linked measures", "eight_feature")):
        for sample, file, data in (("outer", f"{prefix}_outer_predictions.csv", train),
                                   ("2025", f"{prefix}_test_predictions.csv", test)):
            table = pd.read_csv(REF / file)
            score = table.score.to_numpy(float)
            pred_col = "pred" if "pred" in table else "prediction"
            pred = table[pred_col].to_numpy(int)
            y = data.iloc[:, 0].to_numpy(int)
            assert len(table) == len(y) and np.array_equal(table.label, y)
            rows.append({"analysis": name, "sample": sample, **metrics(y, score, pred)})
    pd.DataFrame(rows).to_csv(out / "feature_exclusion_performance.csv", index=False)


def calibration_diagnostics(train: pd.DataFrame, test: pd.DataFrame,
                            outer_score: np.ndarray, test_score: np.ndarray, out: Path) -> None:
    rows = []
    for sample, table, p in (("2019-2024 outer CV", train, outer_score),
                             ("2025 temporal test", test, test_score)):
        y = table.iloc[:, 0].to_numpy(int)
        order = np.argsort(p, kind="stable")
        groups = [("Overall", np.arange(len(y)))]
        groups += [(f"Q{i}", ids) for i, ids in enumerate(np.array_split(order, 5), 1)]
        for label, ids in groups:
            rows.append({"evaluation_sample": sample, "probability_bin": label,
                         "sets": len(ids), "wins": int(y[ids].sum()),
                         "matches": table.iloc[ids].match_id.nunique(),
                         "mean_predicted_probability": p[ids].mean(),
                         "observed_win_rate": y[ids].mean(),
                         "brier": np.mean((p[ids] - y[ids]) ** 2) if label == "Overall" else np.nan,
                         "minimum_probability": p[ids].min(), "maximum_probability": p[ids].max()})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "calibration_diagnostics.csv", index=False)
    old = pd.read_csv(REF / "calibration_table_source.csv")
    assert np.array_equal(frame.sets, old["Sets"])
    assert np.allclose(frame.mean_predicted_probability, old["Mean predicted p(win)"], atol=1e-12)
    assert np.allclose(frame.observed_win_rate, old["Observed win rate"], atol=1e-12)


def other_results(train: pd.DataFrame, test: pd.DataFrame, out: Path) -> None:
    arrays = np.load(REF / "prediction_matrices.npz")
    singles = read_json("single_model_results.json")["models"]
    outer_thresholds = pd.read_csv(REF / "single_model_outer_thresholds.csv")[NAMES].to_numpy(float)
    final_thresholds = pd.read_csv(REF / "single_model_final_thresholds.csv")
    assert final_thresholds.model.tolist() == NAMES and outer_thresholds.shape == (5, len(NAMES))
    outer_predictions = np.zeros_like(arrays["outer_members"], dtype=int)
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
        train.iloc[:, 1:35], train.iloc[:, 0], train.cv_group)
    for fold, (_, heldout) in enumerate(folds):
        outer_predictions[heldout] = arrays["outer_members"][heldout] >= outer_thresholds[fold]
    rows = []
    roc_rows = []
    for j, name in enumerate(NAMES):
        false_positive, true_positive, thresholds = roc_curve(
            train.iloc[:, 0], arrays["outer_members"][:, j])
        for k, (fpr, tpr, threshold) in enumerate(zip(false_positive, true_positive, thresholds)):
            roc_rows.append({"model": name, "point_index": k,
                             "fpr": fpr, "tpr": tpr, "score_threshold": threshold})
        for sample, mat, data, predictions in (
            ("outer", arrays["outer_members"], train, outer_predictions[:, j]),
            ("2025", arrays["test_members"], test,
             arrays["test_members"][:, j] >= final_thresholds.development_oof_threshold.iat[j]),
        ):
            y = data.iloc[:, 0].to_numpy(int)
            reference = singles[name]["full"]["outer" if sample == "outer" else "test_2025"]
            calculated = metrics(y, mat[:, j], predictions)
            for measure in ("auc", "brier", "accuracy", "f1", "precision", "recall"):
                old = reference["recall_win" if measure == "recall" else measure]
                assert abs(calculated[measure] - old) < 1e-10, (name, sample, measure)
            rows.append({"model": name, "sample": sample, **calculated,
                         "oof_pseudo_r2": r2_score(y, mat[:, j]) if sample == "outer" else np.nan})
    pd.DataFrame(rows).to_csv(out / "single_model_performance.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(out / "single_model_roc_coordinates.csv", index=False)
    tuning = read_json("tuning_results.json")
    pd.DataFrame([
        {"sample": sample, "auc": tuning[key]["auc"], "members": "+".join(tuning["final_selection"]["members"])}
        for sample, key in (("outer", "outer"), ("2025", "test_2025"))
    ]).to_csv(out / "tuning_sensitivity.csv", index=False)
    backhand = read_json("backhand_ablation.json")
    pd.DataFrame([{"sample": label, **backhand[key]} for label, key in
                  (("outer", "outer"), ("2025", "test_2025"))]).to_csv(
                      out / "backhand_ablation.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "processed/train_grouped.csv")
    test = pd.read_csv(DATA / "processed/test_grouped.csv")
    verify_inputs(train, test)
    outer_score, test_score = reconstruct_primary(train, test, args.output)
    reproduce_shap(test, args.output)
    reproduce_exclusions(train, test, args.output)
    calibration_diagnostics(train, test, outer_score, test_score, args.output)
    other_results(train, test, args.output)
    print(f"Verified inputs and wrote numerical results to {args.output}")


if __name__ == "__main__":
    main()
