"""Recalculate member-level permutation prediction sensitivity (Figure 4A)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from reproduce import ROOT, REF


def main() -> None:
    refit = ROOT / "outputs/refit"
    bundle = joblib.load(refit / "final_models.joblib")
    names = bundle["selection"]["members"]
    features = bundle["features"]
    train = pd.read_csv(ROOT / "data/processed/train_grouped.csv")[features].to_numpy(float)
    test = pd.read_csv(ROOT / "data/processed/test_grouped.csv")[features].to_numpy(float)

    def predict(x):
        return np.column_stack([
            bundle["calibrators"][name].predict(bundle["models"][name].decision_function(x))
            for name in names
        ])

    rng = np.random.RandomState(42)
    rows = []
    for subset, x in (("development", train), ("2025", test)):
        original = predict(x)
        for column, feature in enumerate(features):
            differences = np.zeros((30, len(names)))
            for repeat in range(30):
                changed = x.copy()
                changed[:, column] = x[rng.permutation(len(x)), column]
                differences[repeat] = np.mean(np.abs(predict(changed) - original), axis=0)
            for member, name in enumerate(names):
                rows.append({"subset": subset, "feature": feature, "model": name,
                             "mean_abs_probability_change": differences[:, member].mean(),
                             "sd_across_permutations": differences[:, member].std(ddof=1),
                             "n_sets": len(x), "repeats": 30})
    result = pd.DataFrame(rows)
    totals = result.loc[result.subset == "development"].groupby("model").mean_abs_probability_change.sum()
    result["within_model_relative_sensitivity"] = result.mean_abs_probability_change / result.model.map(totals)
    result["probability_points"] = 100 * result.mean_abs_probability_change
    reference = pd.read_csv(REF / "prediction_sensitivity.csv")
    assert result[["subset", "feature", "model"]].equals(reference[["subset", "feature", "model"]])
    for column in ("mean_abs_probability_change", "sd_across_permutations",
                   "within_model_relative_sensitivity", "probability_points"):
        delta = np.max(np.abs(result[column] - reference[column]))
        assert delta < 1e-10, (column, delta)
    output = refit / "prediction_sensitivity.csv"
    result.to_csv(output, index=False)
    print(f"Recalculated {len(result)} member-feature sensitivity rows", flush=True)


if __name__ == "__main__":
    main()
