"""Recalculate 2025 match-cluster intervals with the selected model held fixed."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score

from reproduce import ROOT, REF


def main() -> None:
    frame = pd.read_csv(REF / "test_predictions.csv")
    groups = frame.match_id.unique()
    group_values = frame.match_id.to_numpy()
    y_all = frame.label.to_numpy(int)
    scores = frame.auc_selector_score.to_numpy(float)
    predictions = frame.auc_selector_pred.to_numpy(int)
    rng = np.random.default_rng(20260928)
    sampled_metrics = {name: [] for name in ("auc", "accuracy", "f1", "brier")}
    for _ in range(5000):
        sampled_groups = rng.choice(groups, len(groups), replace=True)
        positions = np.concatenate([np.flatnonzero(group_values == group) for group in sampled_groups])
        y, score, predicted = y_all[positions], scores[positions], predictions[positions]
        if len(np.unique(y)) < 2:
            continue
        sampled_metrics["auc"].append(roc_auc_score(y, score))
        sampled_metrics["accuracy"].append(accuracy_score(y, predicted))
        sampled_metrics["f1"].append(f1_score(y, predicted))
        sampled_metrics["brier"].append(brier_score_loss(y, score))
    functions = {"auc": (roc_auc_score, scores),
                 "accuracy": (accuracy_score, predictions),
                 "f1": (f1_score, predictions),
                 "brier": (brier_score_loss, scores)}
    rows = []
    for name, (function, values) in functions.items():
        low, high = np.quantile(sampled_metrics[name], [0.025, 0.975])
        rows.append({"metric": name, "estimate": function(y_all, values),
                     "lower_95": low, "upper_95": high,
                     "valid_draws": len(sampled_metrics[name])})
    result = pd.DataFrame(rows)
    reference = json.loads((REF / "test_metric_cluster_bootstrap.json").read_text(encoding="utf-8"))
    for row in result.itertuples():
        expected = reference[row.metric]
        assert np.allclose([row.estimate, row.lower_95, row.upper_95],
                           [expected["estimate"], *expected["interval"]], atol=1e-12)
        assert row.valid_draws == expected["n"]
    output = ROOT / "outputs/test_metric_cluster_bootstrap.csv"
    output.parent.mkdir(exist_ok=True)
    result.to_csv(output, index=False)
    print("Recalculated 5,000 match-cluster draws and four 2025 metric intervals", flush=True)


if __name__ == "__main__":
    main()
