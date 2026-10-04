"""Summarize rank agreement across 30 complete KernelSHAP runs."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from reproduce import ROOT, REF


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    directory = args.directory or ROOT / "outputs/shap_repeat30"
    if args.directory is None and not (directory / "seed42.npz").exists():
        directory = ROOT / "outputs/refit/shap"
    seeds = [42 + 100 * i for i in range(30)]
    data = [np.load(directory / f"seed{seed}.npz") for seed in seeds]
    features = data[0]["features"].astype(str)
    assert all(np.array_equal(item["features"], features) for item in data)
    values = np.stack([item["values"] for item in data])
    importance = np.abs(values).mean(axis=1)
    order = np.argsort(-importance, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(30)[:, None], order] = np.arange(1, len(features) + 1)
    backhand = list(features).index("Backhand Returns")
    correlations, jaccards = [], []
    for left, right in itertools.combinations(range(30), 2):
        correlations.append(spearmanr(importance[left], importance[right]).statistic)
        first, second = set(order[left, :6]), set(order[right, :6])
        jaccards.append(len(first & second) / len(first | second))
    summary = {
        "runs": 30,
        "backhand_rank_min": int(ranks[:, backhand].min()),
        "backhand_rank_max": int(ranks[:, backhand].max()),
        "backhand_top6_frequency": float(np.mean(ranks[:, backhand] <= 6)),
        "pairwise_rank_spearman_median": float(np.median(correlations)),
        "pairwise_rank_spearman_min": float(np.min(correlations)),
        "pairwise_rank_spearman_max": float(np.max(correlations)),
        "pairwise_top6_jaccard_median": float(np.median(jaccards)),
        "mean_abs_shap_30_vs_archived3_max_absolute_difference": float(
            np.max(np.abs(importance.mean(axis=0) - importance[:3].mean(axis=0)))),
        "point_shap_30_vs_archived3_max_absolute_difference": float(
            np.max(np.abs(values.mean(axis=0) - values[:3].mean(axis=0)))),
    }
    expected = json.loads((REF / "shap_repeat30_summary.json").read_text(encoding="utf-8"))
    assert [summary["backhand_rank_min"], summary["backhand_rank_max"]] == expected["backhand_rank_range"]
    assert summary["backhand_top6_frequency"] == expected["backhand_top6_frequency"]
    assert np.isclose(summary["pairwise_rank_spearman_median"],
                      expected["pairwise_rank_spearman_median"])
    assert np.isclose(summary["pairwise_top6_jaccard_median"],
                      expected["pairwise_top6_jaccard_median"])
    for key in ("mean_abs_shap_30_vs_archived3_max_absolute_difference",
                "point_shap_30_vs_archived3_max_absolute_difference"):
        assert np.isclose(summary[key], expected[key])
    assert np.allclose([summary["pairwise_rank_spearman_min"],
                        summary["pairwise_rank_spearman_max"]],
                       expected["pairwise_rank_spearman_range"])
    output = directory / "repeatability_summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame({"seed": seeds, "backhand_rank": ranks[:, backhand]}).to_csv(
        directory / "backhand_ranks.csv", index=False)
    pd.DataFrame(ranks, columns=features).assign(seed=seeds).to_csv(
        directory / "all_feature_ranks_by_run.csv", index=False)
    print(f"Summarized {len(correlations)} SHAP-run pairs: {summary}", flush=True)


if __name__ == "__main__":
    main()
