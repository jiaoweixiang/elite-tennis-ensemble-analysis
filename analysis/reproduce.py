"""Recalculate the manuscript's numerical results from the deposited model outputs.

The input tables and frozen member predictions are checked before use. For an
independent rerun of model fitting, see ``analysis/refit_primary.py``.

Outputs (in outputs/):
  primary_performance.csv        Tables S2-S3 ensemble rows (pooled outer and 2025)
  outer_folds.csv                Table S2 fold rows
  final_selection.json           Table S2 final selection row
  ensemble_candidates_4095.csv   Figure 3c
  primary_outer_predictions.csv, primary_2025_predictions.csv
  ensemble_shap_ranks.csv        Figure 4c-d; Table S7 leading six indicators
  ensemble_shap_per_set.csv      Figure 4c, 4e
  backhand_match_resample_ranks.csv  Table S5 (2025 SHAP rows); Figure 4f
  backhand_ablation.csv          Table S5 fixed-member dAUC
  calibration_diagnostics.csv    Table S4 quintile rows
  single_model_performance.csv   Table S3 learner rows; Figure 3a, 3d
  single_model_roc_coordinates.csv   Figure 3b
  comparator_performance.csv     Table S3 baseline rows; Table S7 F1 selection
  tuning_sensitivity.csv         Table S7 hyperparameter tuning
  feature_exclusion_performance.csv  Table S7 twelve-indicator exclusion
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
OUTER_COLUMNS = ["auc", "accuracy", "f1", "brier"]
TEST_COLUMNS = ["auc", "accuracy", "precision", "recall", "f1", "brier"]


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


def pick(values: dict, columns) -> dict:
    return {column: values[column] for column in columns}


def match_groups(match_id: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(match_id == g) for g in sorted(set(match_id))]


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
        fold_rows.append({"fold": fold, "matches": train.match_id.iloc[heldout].nunique(),
                          "sets": len(heldout), "members": "+".join(choice["members"]),
                          "inner_auc": choice["inner_auc"], "threshold": choice["threshold_youden"],
                          **pick(metrics(y[heldout], p, outer_preds[heldout]), OUTER_COLUMNS)})
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
    outer_metrics, test_metrics = metrics(y, outer_scores, outer_preds), metrics(yt, test_scores, test_preds)
    rows = [{"sample": "2019-2024 grouped outer", "sets": len(y), "matches": train.match_id.nunique(),
             "members": "selected independently in each fold", **pick(outer_metrics, OUTER_COLUMNS)},
            {"sample": "2025 temporal", "sets": len(yt), "matches": test.match_id.nunique(),
             "members": "+".join(final["members"]), **pick(test_metrics, TEST_COLUMNS)}]
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


def reproduce_exclusion(test: pd.DataFrame, out: Path) -> None:
    """Table S7 twelve-indicator exclusion: 2025 metrics, AUC difference, Backhand ranks."""
    reference = read_json("twelve_feature_results.json")
    table = pd.read_csv(REF / "twelve_feature_test_predictions.csv")
    full = pd.read_csv(REF / "test_predictions.csv")
    y = test.iloc[:, 0].to_numpy(int)
    assert np.array_equal(table.label, y) and np.array_equal(full.label, y)
    score, pred = table.score.to_numpy(float), table.prediction.to_numpy(int)
    full_score = full.auc_selector_score.to_numpy(float)
    groups = match_groups(table.match_id.to_numpy(str))
    rng = np.random.default_rng(20261111)
    draws = []
    for _ in range(2000):
        ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[ix])) < 2:
            continue
        draws.append(roc_auc_score(y[ix], full_score[ix]) - roc_auc_score(y[ix], score[ix]))
    difference = roc_auc_score(y, full_score) - roc_auc_score(y, score)
    low, high = np.quantile(draws, [0.025, 0.975])
    saved = reference["full_minus_restricted_test_auc"]
    assert np.allclose([difference, low, high], [saved["estimate"], *saved["CI95"]], atol=1e-12)
    ranks = pd.read_csv(REF / "revision2026/twelve_agreement/topAUC40_backhand_ranks.csv")
    measured = pick(metrics(y, score, pred), ["auc", "accuracy", "f1"])
    for key, value in measured.items():
        assert abs(value - reference["test_2025"][key]) < 1e-12
    row = {"analysis": "twelve indicators excluded", "sample": "2025",
           "members": "+".join(reference["final_selection"]["members"]), **measured,
           "full_minus_restricted_auc": difference, "difference_CI95_low": low, "difference_CI95_high": high,
           "backhand_shap_rank": int(ranks.backhand_rank.iat[0]),
           "backhand_top6_count_among_top40": int((ranks.backhand_rank <= 6).sum())}
    assert row["backhand_shap_rank"] == reference["shap_2025"]["backhand_rank_selected_ensemble"]
    assert row["backhand_top6_count_among_top40"] == reference["shap_2025"]["backhand_top6_count_among_top40"]
    pd.DataFrame([row]).to_csv(out / "feature_exclusion_performance.csv", index=False)


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
                         "brier": np.mean((p[ids] - y[ids]) ** 2),
                         "minimum_probability": p[ids].min(), "maximum_probability": p[ids].max()})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "calibration_diagnostics.csv", index=False)
    old = pd.read_csv(REF / "calibration_table_source.csv")
    assert np.array_equal(frame.sets, old["Sets"])
    assert np.allclose(frame.mean_predicted_probability, old["Mean predicted p(win)"], atol=1e-12)
    assert np.allclose(frame.observed_win_rate, old["Observed win rate"], atol=1e-12)
    assert np.allclose(frame.brier, old["Brier score"], atol=1e-12)


def single_learners(train: pd.DataFrame, test: pd.DataFrame, out: Path) -> None:
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
    y, yt = train.iloc[:, 0].to_numpy(int), test.iloc[:, 0].to_numpy(int)
    rows, roc_rows = [], []
    for j, name in enumerate(NAMES):
        false_positive, true_positive, thresholds = roc_curve(y, arrays["outer_members"][:, j])
        for k, (fpr, tpr, threshold) in enumerate(zip(false_positive, true_positive, thresholds)):
            roc_rows.append({"model": name, "point_index": k,
                             "fpr": fpr, "tpr": tpr, "score_threshold": threshold})
        outer = pick(metrics(y, arrays["outer_members"][:, j], outer_predictions[:, j]), ["auc", "accuracy", "f1"])
        test_pred = arrays["test_members"][:, j] >= final_thresholds.development_oof_threshold.iat[j]
        current = pick(metrics(yt, arrays["test_members"][:, j], test_pred), TEST_COLUMNS)
        for key, value in outer.items():
            assert abs(value - singles[name]["outer"][key]) < 1e-10, (name, "outer", key)
        for key, value in current.items():
            assert abs(value - singles[name]["test_2025"][key]) < 1e-10, (name, "2025", key)
        rows.append({"model": name, "oof_pseudo_r2": r2_score(y, arrays["outer_members"][:, j]),
                     **{f"outer_{k}": v for k, v in outer.items()},
                     **{f"test_2025_{k}": v for k, v in current.items()}})
    pd.DataFrame(rows).to_csv(out / "single_model_performance.csv", index=False)
    pd.DataFrame(roc_rows).to_csv(out / "single_model_roc_coordinates.csv", index=False)


def comparators(out: Path) -> None:
    """Table S3 classification comparators and the Table S7 F1-selection workflow."""
    outer, test = pd.read_csv(REF / "outer_predictions.csv"), pd.read_csv(REF / "test_predictions.csv")
    rows = []
    for label, key in (("Baseline: Logistic L2", "logistic_l2"), ("Baseline: Logistic L1", "logistic_l1"),
                       ("Baseline: RF classifier", "random_forest"), ("F1-selection ensemble", "f1_selector")):
        o = metrics(outer.label.to_numpy(int), outer[key + "_score"].to_numpy(float), outer[key + "_pred"].to_numpy(int))
        t = metrics(test.label.to_numpy(int), test[key + "_score"].to_numpy(float), test[key + "_pred"].to_numpy(int))
        rows.append({"model": label, **{f"outer_{k}": o[k] for k in ("auc", "accuracy", "f1")},
                     **{f"test_2025_{k}": t[k] for k in TEST_COLUMNS}})
    pd.DataFrame(rows).to_csv(out / "comparator_performance.csv", index=False)


def tuning(out: Path) -> None:
    """Table S7 hyperparameter-tuning column."""
    result = read_json("tuning_results.json")
    pd.DataFrame([{"members": "+".join(result["final_selection"]["members"]),
                   **{f"outer_{k}": result["outer"][k] for k in ("auc", "accuracy", "f1")},
                   **{f"test_2025_{k}": result["test_2025"][k] for k in ("auc", "accuracy", "f1")},
                   "backhand_shap_rank": result["shap_2025"]["ensemble_backhand_rank"],
                   **{f"setting_{name}": value for name, value in result["final_settings"].items()}}]).to_csv(
        out / "tuning_sensitivity.csv", index=False)


def backhand_ablation(out: Path) -> None:
    """Table S5 fixed-member dAUC, using the same 2,000 paired match resamples as Figure 4b."""
    p = pd.read_csv(REF / "revision2026/primary_ablation/test_ablation_predictions.csv")
    ab = pd.read_csv(REF / "backhand_ablation_test_predictions.csv")
    column = {c.strip(): c for c in p.columns}["Backhand Returns"]
    assert np.allclose(p[column], ab.without_backhand, atol=1e-12) and np.allclose(p.full, ab.full, atol=1e-12)
    y, full, without = p.label.to_numpy(int), p.full.to_numpy(float), p[column].to_numpy(float)
    groups = match_groups(p.match_id.to_numpy(str))
    rng = np.random.default_rng(20261112)
    for _ in range(2000):
        rng.integers(0, 59, 59)  # advance the generator to the state used for the reported intervals
    draws = []
    for _ in range(2000):
        ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[ix])) < 2:
            continue
        draws.append(roc_auc_score(y[ix], full[ix]) - roc_auc_score(y[ix], without[ix]))
    low, high = np.quantile(draws, [0.025, 0.975])
    source = pd.read_csv(REF / "revision2026/primary_ablation/fig4b_aligned_95CI_source.csv")
    saved = source[source.feature == "Backhand Returns"].iloc[0]
    delta = roc_auc_score(y, full) - roc_auc_score(y, without)
    assert np.allclose([delta, low, high], [saved.delta_auc, saved.CI2_5, saved.CI97_5], atol=1e-12)
    pd.DataFrame([{"sample": "2025", "full_auc": roc_auc_score(y, full), "without_backhand_auc": roc_auc_score(y, without),
                   "auc_delta_full_minus_without": delta, "CI95_low": low, "CI95_high": high,
                   "valid_resamples": len(draws)}]).to_csv(out / "backhand_ablation.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA / "processed/train_grouped.csv")
    test = pd.read_csv(DATA / "processed/test_grouped.csv")
    verify_inputs(train, test)
    outer_score, test_score = reconstruct_primary(train, test, args.output)
    reproduce_shap(test, args.output)
    reproduce_exclusion(test, args.output)
    calibration_diagnostics(train, test, outer_score, test_score, args.output)
    single_learners(train, test, args.output)
    comparators(args.output)
    tuning(args.output)
    backhand_ablation(args.output)
    print(f"Verified inputs and wrote numerical results to {args.output}")


if __name__ == "__main__":
    main()
