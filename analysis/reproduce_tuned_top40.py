"""Rebuild rank agreement for the 33-setting tuning sensitivity analysis."""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from reproduce import ROOT, REF, NAMES, COMBOS, WEIGHTS
from reproduce_top40 import jaccard, rank_rows


def main() -> None:
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    train = pd.read_csv(ROOT / "data/processed/train_grouped.csv")
    y = train.iloc[:, 0].to_numpy(int)
    member_scores = np.load(REF / "tuning_base_predictions.npz")["outer_platt"]
    scores = member_scores @ WEIGHTS
    positive, negative = int(y.sum()), int((1 - y).sum())
    sum_ranks = rankdata(scores, axis=0, method="average")[y == 1].sum(axis=0)
    auc = (sum_ranks - positive * (positive + 1) / 2) / (positive * negative)
    selected = np.argsort(-np.round(auc, 12), kind="stable")[:40]

    members = []
    names = None
    for learner in NAMES:
        runs = []
        for seed in (42, 142, 242):
            item = np.load(REF / "tuned_member_shap" / f"shap_{learner}_seed{seed}.npz")
            labels = np.asarray([label.strip() for label in item["features"].astype(str)])
            if names is None:
                names = labels
            assert np.array_equal(names, labels)
            runs.append(item["values"])
        members.append(np.stack(runs))
    members = np.stack(members)
    assert members.shape == (12, 3, 40, 34)
    combined = np.einsum("km,msnf->ksnf", WEIGHTS[:, selected].T, members)
    ensemble_importance = np.abs(combined).mean(axis=(1, 2))
    single_importance = np.abs(members).mean(axis=(1, 2))
    ensemble_ranks = rank_rows(ensemble_importance)
    single_ranks = rank_rows(single_importance)

    rows = []
    for order, index in enumerate(selected, 1):
        for feature, rank, magnitude in zip(names, ensemble_ranks[order - 1],
                                            ensemble_importance[order - 1]):
            rows.append({"auc_order": order, "feature": feature, "shap_rank": int(rank),
                         "mean_absolute_shap": magnitude, "top6": int(rank <= 6),
                         "top10": int(rank <= 10)})
    rank_table = pd.DataFrame(rows)
    reference = pd.read_csv(REF / "tuning_top40_feature_ranks.csv")
    assert np.array_equal(rank_table.shap_rank, reference.shap_rank)
    assert np.max(np.abs(rank_table.mean_absolute_shap - reference.mean_absolute_shap)) < 1e-12
    rank_table.to_csv(output / "tuning_top40_feature_ranks.csv", index=False)
    pd.DataFrame({"learner": NAMES, "backhand_rank": single_ranks[:, list(names).index("Backhand Returns")]})\
        .to_csv(output / "tuning_single_backhand_ranks.csv", index=False)

    agreements = []
    for cohort, ranks, magnitudes in (("tuned single learners", single_ranks, single_importance),
                                      ("tuned top 40 ensembles", ensemble_ranks, ensemble_importance)):
        for left, right in itertools.combinations(range(len(ranks)), 2):
            agreements.append({"cohort": cohort, "left": left + 1, "right": right + 1,
                               "top6_jaccard": jaccard(ranks[left], ranks[right], 6),
                               "top10_jaccard": jaccard(ranks[left], ranks[right], 10),
                               "full_rank_spearman": spearmanr(magnitudes[left], magnitudes[right]).statistic})
    agreement = pd.DataFrame(agreements)
    agreement.to_csv(output / "tuning_rank_agreement_pairs.csv", index=False)
    medians = agreement.groupby("cohort")[["top6_jaccard", "top10_jaccard", "full_rank_spearman"]].median()
    medians.to_csv(output / "tuning_rank_agreement_medians.csv")
    import json
    expected = json.loads((REF / "tuning_top40_summary.json").read_text(encoding="utf-8"))
    for cohort, label in (("tuned single learners", "all_12_single_models"),
                          ("tuned top 40 ensembles", "top40_ensembles")):
        row = medians.loc[cohort]
        assert np.allclose(row.to_numpy(float), [expected[label]["top6_jaccard_median"],
                                                expected[label]["top10_jaccard_median"],
                                                expected[label]["full_rank_spearman_median"]])
    print("Rebuilt tuned top-40 rankings and 846 pair comparisons", flush=True)


if __name__ == "__main__":
    main()
