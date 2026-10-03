"""Recalculate 2025 KernelSHAP values for the refitted final ensemble."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import joblib
import numpy as np
import pandas as pd
import shap

from reproduce import ROOT, REF


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refit-dir", type=Path, default=ROOT / "outputs/refit")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/refit/shap")
    parser.add_argument("--runs", type=int, choices=(3, 30), default=3,
                        help="Three archived complete runs or all 30 repeatability runs")
    parser.add_argument("--check-reference", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.refit_dir / "final_models.joblib")
    train = pd.read_csv(ROOT / "data/processed/train_grouped.csv")
    test = pd.read_csv(ROOT / "data/processed/test_grouped.csv")
    features = bundle["features"]
    x = train[features].to_numpy(float)
    xt = test[features].to_numpy(float)
    members = bundle["selection"]["members"]
    kept = np.unique(np.concatenate([bundle["models"][name].named_steps["screen"].keep_
                                     for name in members]))
    names = np.asarray([features[j] for j in kept])
    mean_train = x.mean(axis=0)

    def predict(z):
        restored = np.tile(mean_train, (len(z), 1))
        restored[:, kept] = z
        return np.column_stack([
            bundle["calibrators"][name].predict(bundle["models"][name].decision_function(restored))
            for name in members
        ]).mean(axis=1)

    fitted_predictions = pd.read_csv(args.refit_dir / "test_predictions.csv").score.to_numpy(float)
    assert np.max(np.abs(predict(xt[:, kept]) - fitted_predictions)) < 1e-10
    background = shap.sample(x[:, kept], 50, random_state=42)
    matrices = []
    repeat_rows = []
    for seed in (42 + 100 * i for i in range(args.runs)):
        np.random.seed(seed)
        explainer = shap.KernelExplainer(predict, background)
        values = np.asarray(explainer.shap_values(xt[:, kept], nsamples=2048, l1_reg=0, silent=True))
        assert values.shape == (40, len(kept))
        assert np.max(np.abs(values.sum(axis=1) + explainer.expected_value - fitted_predictions)) < 1e-6
        np.savez_compressed(args.output / f"seed{seed}.npz", values=values, features=names,
                            base=float(explainer.expected_value))
        if args.check_reference and seed in (42, 142, 242):
            old = np.load(REF / f"full_auc_SHAP_seed{seed}.npz")
            assert np.array_equal(old["features"], names)
            delta = float(np.max(np.abs(old["values"] - values)))
            print(f"seed {seed} maximum SHAP difference: {delta:.3g}", flush=True)
            assert delta < 1e-6
        matrices.append(values)
        run_order = np.argsort(-np.abs(values).mean(axis=0), kind="stable")
        repeat_rows.append({"seed": seed,
                            "backhand_rank": int(np.where(names[run_order] == "Backhand Returns")[0][0] + 1),
                            "top_six": " | ".join(names[run_order[:6]])})
    importance = np.abs(np.stack(matrices)).mean(axis=(0, 1))
    order = np.argsort(-importance, kind="stable")
    pd.DataFrame({"rank": np.arange(1, len(order) + 1), "feature": names[order],
                  "mean_absolute_SHAP": importance[order]}).to_csv(args.output / "feature_ranks.csv", index=False)
    repeats = pd.DataFrame(repeat_rows)
    repeats.to_csv(args.output / "repeat_ranks.csv", index=False)
    if args.check_reference and args.runs == 30:
        reference = pd.read_csv(REF / "shap_repeat30_ranks.csv")
        assert repeats.equals(reference), "Repeatability ranks differ from the deposited record"
        print("All 30 complete-run backhand ranks match the deposited record", flush=True)
    print(f"SHAP values saved to {args.output}")


if __name__ == "__main__":
    main()
