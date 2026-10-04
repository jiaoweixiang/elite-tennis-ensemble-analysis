# Elite tennis ensemble analysis

This repository contains the set-level data and analysis code for a retrospective study of Novak Djokovic's Grand Slam hard-court matches. The development period is 2019–2024 (199 sets from 59 matches; no eligible 2022 matches), and 2025 is a later-year test period (40 sets from 12 matches). All predictors summarize the completed set; the analysis is not a pre-match forecasting model.

The current workflow fits 12 learners, calibrates each learner's score with a match-grouped Platt sigmoid, and selects from all 4,095 non-empty, equal-weight learner subsets by development out-of-fold ROC-AUC. Feature screening and standardization are fitted within training folds. Selection is repeated within five match-grouped outer folds; the final model is then selected using all development sets and evaluated on 2025 sets. The classification threshold is chosen from training predictions by Youden's J statistic. No 2025 labels are used for model or threshold fitting. The later-year outcomes were examined during manuscript revision, so this is a temporal test rather than a wholly untouched confirmatory cohort.

## Reproduce the numbers

The deposited indicator dictionary and the generated `outputs/descriptive_correlations.csv` cover the manuscript's variable definitions, distribution checks and initial outcome correlations. The script also regenerates four 2025 metric intervals from 5,000 match-cluster resamples, holding the fitted model and threshold fixed.

Use Python 3.12 in a fresh environment:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` verifies SHA-256 checksums, reconstructs all 239 set rows and 34 candidate indicators from the six source workbooks, and checks the reconstructed tables against the deposited model inputs. It then recalculates the reported AUC, accuracy, F1, Brier score, 4,095-subset development selection scores, single-model OOF pseudo-R² and ROC coordinates, per-set ensemble SHAP values, match-resampling ranks, and comparisons among the top 40 ensembles and 12 single learners. Numerical CSV and JSON files are written to `outputs/`; no plotting software is needed. The quick statistical run uses deposited per-learner predictions and SHAP arrays, not saved headline values, to recompute the statistics.

To rerun model fitting from the corrected set tables, use:

```bash
python analysis/refit_primary.py --check-reference
python analysis/refit_primary.py --variant two --check-reference
python analysis/refit_primary.py --variant eight --check-reference
python analysis/refit_primary.py --variant backhand --check-reference
python analysis/refit_baselines.py --check-reference
python analysis/refit_sensitivity.py
```

The full, two-feature and eight-feature commands repeat five outer folds, five inner folds, three match-grouped calibration folds, and AUC selection over 4,095 subsets. Results, per-set predictions and fitted final learners are written to `outputs/refit*`. The full refit also writes independently recalculated metrics for all 12 single learners. `--check-reference` compares the new predictions and selected members with the deposited analysis. The classification-comparator command independently refits L2 logistic regression, L1 logistic regression and a random-forest classifier, writing their Table 4 metrics and per-set predictions to `outputs/refit_baselines/`. The two-feature variant removes `Service Breaks` and `Total Points Won`; the eight-feature variant also removes `Double Faults`, `Unforced Errors`, `Backhand Return Win`, `Forehand Return Win`, `1st Serve Pts Won` and `2nd Serve Pts Won`. Both reselect their ensemble members. The backhand ablation removes only `Backhand Returns`, retaining the members and thresholds already selected in the full model before refitting. These variants are sensitivity analyses, not the primary model.

To reproduce the complete 34-indicator feature-removal panel and the F1-selection comparator in Table 9, run:

```bash
python analysis/refit_all_ablation.py --jobs 4 --check-reference
python analysis/refit_primary.py --criterion f1 --output outputs/refit_f1 --check-reference
```

The first command refits each indicator removal with the original fold-specific members and thresholds, writes `outputs/feature_ablation_refits/feature_ablation.csv`, and checks all 34 rows against the deposited panel source. It is more computationally demanding than `run_all.py`; set `--jobs 1` on a low-memory computer. The second command repeats grouped F1-based subset selection as a sensitivity comparison; its `results.json` and per-set prediction CSVs provide the Table 9 F1-selection values. The primary analysis continues to select by AUC.

The sensitivity command permutes each indicator 30 times in the refitted ensemble members and verifies the resulting prediction changes against the deposited Figure 4A values.

To recalculate the final model's three complete KernelSHAP runs after the primary refit:

```bash
python -m pip install -r requirements-shap.txt
python analysis/refit_shap.py --check-reference
python analysis/refit_shap.py --runs 30 --output outputs/shap_repeat30 --check-reference
python analysis/summarize_shap_repeats.py
```

Each run explains all 40 test sets using 50 development-background observations and a `nsamples=2048` KernelSHAP evaluation budget. The default checks the three archived SHAP matrices; `--runs 30` also checks the recorded rank from each complete repeat. The separate 2,000 whole-match resamples reuse SHAP values to assess uncertainty due to test-match composition; they are not 2,000 new model fits or SHAP runs.

The hyperparameter sensitivity analysis can also be refitted from the processed data:

```bash
python analysis/refit_tuning.py --check-reference
```

This evaluates 33 documented learner settings within the grouped training folds, then repeats calibration and selection over 4,095 subsets. It is separate from the fixed-parameter primary analysis. The command checks the resulting outer and 2025 predictions against the deposited sensitivity analysis.

## Results checked by the scripts

| Evaluation | ROC-AUC | Accuracy | F1 | Brier |
| --- | ---: | ---: | ---: | ---: |
| Pooled match-grouped outer predictions | 0.9656 | 0.8844 | 0.9256 | 0.0606 |
| 2025 temporal test | 0.9733 | 0.9000 | 0.9355 | 0.0888 |

The final development selection has seven members: LR, Ridge, Lasso, gradient boosting, AdaBoost, decision tree and Gaussian naive Bayes. Its *development selection* AUC is 0.9702; this value is not a held-out performance estimate. `Backhand Returns` ranks sixth by mean absolute SHAP in the selected ensemble. Across 2,000 whole-match resamples it enters the top six in 58.6% of draws. It appears in the top six of three of the 12 individual learners. Among the 40 highest-development-AUC combinations, median pairwise top-ten feature-set Jaccard similarity is 1.000, versus 0.381 among the single learners. These rank comparisons describe this dataset and fitted model family; they do not establish a causal role for any indicator.

## Repository layout

| Path | Contents |
| --- | --- |
| `data/raw/` | Six corrected, author-compiled annual source workbooks (2019, 2020, 2021, 2023, 2024, 2025) |
| `data/processed/` | Corrected set-level tables with match identifiers and the 34 candidate indicators |
| `data/indicator_dictionary.csv` | Operational definitions and manuscript indicator codes |
| `data/reference/` | Deposited base predictions, SHAP arrays, sensitivity outputs and numerical audit records |
| `data/reference/feature_ablation.csv` | Complete 34-indicator feature-removal panel source, checked by independent refitting |
| `data/sha256.csv` | Checksums and byte counts for deposited data files |
| `analysis/models.py` | Documented fixed settings and fold-local feature screen for 12 learners |
| `analysis/build_processed.py` | Rebuilds the model tables directly from annual workbooks and the set-to-source map |
| `analysis/descriptive.py` | Recalculates distribution tests and outcome correlations |
| `analysis/bootstrap_ci.py` | Recalculates 2025 match-cluster metric intervals |
| `analysis/reproduce.py` | Recalculation of numerical outputs from deposited predictions |
| `analysis/reproduce_top40.py` | Recalculation of top-40 ensemble and single-learner SHAP agreement |
| `analysis/reproduce_tuned_top40.py` | Recalculation of tuned top-40 and single-learner SHAP agreement |
| `analysis/refit_primary.py` | Full model refit, including two- and eight-feature exclusions |
| `analysis/refit_baselines.py` | Independent refits of the three classification comparators in Table 4 |
| `analysis/refit_all_ablation.py` | Independent fixed-member refits of all 34 single-indicator removals |
| `analysis/refit_sensitivity.py` | Member-level permutation sensitivity for the primary refit |
| `analysis/refit_shap.py` | Full SHAP rerun for the final primary model |
| `analysis/summarize_shap_repeats.py` | Rank agreement across 30 complete SHAP runs |
| `analysis/refit_tuning.py` | Grouped hyperparameter sensitivity refit |

The source corrections and variable definitions are described in [`data/README.md`](data/README.md). The repository is intended to support numerical checking of the manuscript; generated figures are not required to run the analysis.

## Citation and archival record

`CITATION.cff` describes this software repository. It contains no DOI because none has been registered. A DOI can later be obtained by connecting the public repository to Zenodo and archiving a versioned GitHub release. The release should be made only after the manuscript, source-file rights, authorship and citation details have been confirmed; the DOI and release version can then be added to `CITATION.cff`.
