"""Rebuild the top-40 ensemble ranking comparison from member SHAP arrays."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from reproduce import ROOT, REF, NAMES, COMBOS, WEIGHTS


def rank_rows(importance: np.ndarray) -> np.ndarray:
    order = np.argsort(-importance, axis=1, kind="stable")
    ranks = np.empty_like(order)
    ranks[np.arange(len(order))[:, None], order] = np.arange(1, order.shape[1] + 1)
    return ranks


def jaccard(a: np.ndarray, b: np.ndarray, n: int) -> float:
    left, right = set(np.flatnonzero(a <= n)), set(np.flatnonzero(b <= n))
    return len(left & right) / len(left | right)


def main(output: Path | None = None) -> None:
    output = output or ROOT / "outputs"
    output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(ROOT / "data/processed/train_grouped.csv")
    y = train.iloc[:, 0].to_numpy(int)
    oof = np.load(REF / "prediction_matrices.npz")["train_platt_oof"]
    scores = oof @ WEIGHTS
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    ranks = rankdata(scores, axis=0, method="average")
    auc = (ranks[y == 1].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    order = np.argsort(-np.round(auc, 12), kind="stable")
    selected = order[:40]

    values = []
    feature_names = None
    for name in NAMES:
        seeds = []
        for seed in (42, 142, 242):
            item = np.load(REF / "member_shap" / f"shap_{name}_seed{seed}.npz")
            labels = np.asarray([f.strip() for f in item["features"].astype(str)])
            if feature_names is None:
                feature_names = labels
            assert np.array_equal(feature_names, labels)
            seeds.append(item["values"])
        values.append(np.stack(seeds))
    values = np.stack(values)  # learner x seed x 2025 set x feature
    assert values.shape == (12, 3, 40, 34)

    weights = WEIGHTS[:, selected].T
    combined = np.einsum("km,msnf->ksnf", weights, values)
    ensemble_importance = np.abs(combined).mean(axis=(1, 2))
    single_importance = np.abs(values).mean(axis=(1, 2))
    ensemble_ranks = rank_rows(ensemble_importance)
    single_ranks = rank_rows(single_importance)

    selections = []
    rank_table = []
    for rank, index in enumerate(selected, 1):
        selections.append({"development_auc_order": rank, "subset_order": int(index + 1),
                           "members": "+".join(NAMES[j] for j in COMBOS[index]),
                           "development_oof_auc": auc[index]})
        for feature, position, magnitude in zip(feature_names, ensemble_ranks[rank - 1],
                                                ensemble_importance[rank - 1]):
            rank_table.append({"development_auc_order": rank, "feature": feature,
                               "shap_rank": int(position), "mean_absolute_SHAP": magnitude,
                               "top6": int(position <= 6), "top10": int(position <= 10)})
    pd.DataFrame(selections).to_csv(output / "top40_ensembles.csv", index=False)
    frame = pd.DataFrame(rank_table)
    frame.to_csv(output / "top40_feature_ranks.csv", index=False)
    old = pd.read_csv(REF / "top40_feature_ranks.csv")
    assert np.array_equal(frame.shap_rank, old.shap_rank)
    assert np.max(np.abs(frame.mean_absolute_SHAP - old.mean_absolute_shap)) < 1e-12

    pairs = []
    for cohort, rank_matrix, magnitudes, names in (
        ("12 single models", single_ranks, single_importance, NAMES),
        ("top 40 ensembles", ensemble_ranks, ensemble_importance, list(map(str, range(1, 41)))),
    ):
        for i, j in itertools.combinations(range(len(rank_matrix)), 2):
            pairs.append({"cohort": cohort, "item_1": names[i], "item_2": names[j],
                          "top6_jaccard": jaccard(rank_matrix[i], rank_matrix[j], 6),
                          "top10_jaccard": jaccard(rank_matrix[i], rank_matrix[j], 10),
                          "full_rank_spearman": spearmanr(magnitudes[i], magnitudes[j]).statistic})
    pair_table = pd.DataFrame(pairs)
    pair_table.to_csv(output / "rank_agreement_pairs.csv", index=False)
    pair_table.groupby("cohort")[["top6_jaccard", "top10_jaccard", "full_rank_spearman"]].median().to_csv(
        output / "rank_agreement_medians.csv")
    bh = list(feature_names).index("Backhand Returns")
    assert int((ensemble_ranks[:, bh] <= 6).sum()) == 22
    assert int((ensemble_ranks[:, bh] <= 10).sum()) == 36
    assert int((single_ranks[:, bh] <= 6).sum()) == 3
    print(f"Rebuilt 40 ensemble rankings and {len(pairs)} pair comparisons")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    main(parser.parse_args().output)
