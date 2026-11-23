# Elite tennis ensemble analysis

Version `1.0.2` supports the manuscript **Interpretable ensemble machine learning for exploratory single-player technical–tactical analysis in elite tennis**. The version DOI will be assigned by Zenodo on deposition.

The retrospective single-player study uses 34 indicators (Table 1) from 239 completed Grand Slam hard-court sets: 199 development sets from 59 matches in 2019–2024 and 40 temporal-test sets from 12 matches in 2025 (30 won, 10 lost). There were no eligible 2022 matches. The primary workflow uses fold-local screening, 12 learners, grouped Platt calibration and development-AUC selection among all 4,095 equal-weight subsets.

## Reproduce the reported numbers

Python 3.12 with the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` verifies checksums, rebuilds the model inputs from the six source workbooks and recalculates every reported number from the deposited predictions and SHAP arrays. Results are written to `outputs/`. It does not retrain models; see *Optional model refitting*.

## Where each result is produced

| Manuscript item | Output file(s) |
| --- | --- |
| Table S1, Figure 2 | `outputs/descriptive_correlations.csv`; note: `outputs/revision2026/partial_spearman.csv` |
| Table S2 | `outputs/outer_folds.csv`, `outputs/primary_performance.csv`, `outputs/final_selection.json` |
| Table S3 | `outputs/single_model_performance.csv`, `outputs/comparator_performance.csv`, `outputs/primary_performance.csv`, `outputs/test_metric_cluster_bootstrap.csv` |
| Table S4 | `outputs/calibration_diagnostics.csv`, `outputs/calibration_intercept_slope.csv` |
| Table S5 | `outputs/backhand_match_resample_ranks.csv`, `outputs/ensemble_shap_ranks.csv`, `outputs/backhand_ablation.csv`, `outputs/single_model_shap_ranks.csv`; KernelSHAP reruns: `data/reference/shap_repeat30_summary.json` |
| Table S6, Figure 5 | `outputs/revision2026/four_group_agreement.csv`, `outputs/revision2026/paired_agreement_differences.csv`, `outputs/rank_agreement_medians.csv`, `outputs/top40_feature_ranks.csv`, `outputs/single_model_shap_ranks.csv` |
| Table S7 | exclusion: `outputs/feature_exclusion_performance.csv`; selection criterion: `outputs/comparator_performance.csv`; tuning: `outputs/tuning_sensitivity.csv`, `outputs/revision2026/tuned_rank_agreement.csv` |
| Table S8 | `outputs/revision2026/opponent_adjustment.csv`; `data/reference/revision2026/opponent/` |
| Table S9 | `outputs/revision2026/court_depth_omission.csv` |
| Table S10 | `outputs/revision2026/return_representations_mean_mask.csv`, `outputs/return_representation_diagnostics.json` |
| Table S11 | `outputs/revision2026/observer_agreement.csv`, `outputs/recoding_origin_check.json` |
| Figure 3 | `outputs/single_model_performance.csv`, `outputs/single_model_roc_coordinates.csv`, `outputs/ensemble_candidates_4095.csv` |
| Figure 4 | 4a `data/reference/prediction_sensitivity.csv`; 4b `outputs/revision2026/figure4b_ablation.csv`; 4c–e `outputs/ensemble_shap_per_set.csv`, `outputs/ensemble_shap_ranks.csv`; 4f `outputs/backhand_match_resample_ranks.csv`; 4g `outputs/single_model_shap_ranks.csv` |
| Main text | 2025 deletion-refit ranking of all 34 indicators: `outputs/revision2026/ablation_test_rank_all34.csv`; external count check: `outputs/revision2026/external_recording_agreement.csv` |

`outputs/source_value_differences.csv`, `outputs/rebuilt_*_from_workbooks.csv` and `outputs/recoding_origin_check.json` are data-integrity checks (see `data/README.md`).

## Optional model refitting

Each command retrains the learners inside the match-grouped folds and, with `--check-reference`, asserts agreement with the deposited predictions. `requirements-shap.txt` adds SHAP for the explanation refits.

```bash
python analysis/refit_primary.py --check-reference                   # primary workflow (Tables S2-S3)
python analysis/refit_baselines.py --check-reference                 # classification comparators (Table S3)
python analysis/refit_primary.py --criterion f1 --check-reference    # F1 selection (Table S7)
python analysis/refit_primary.py --variant twelve --check-reference  # twelve-indicator exclusion (Table S7)
python analysis/refit_exclusion_shap.py --check-reference            # exclusion SHAP ranks (Table S7)
python analysis/refit_tuning.py --check-reference                    # hyperparameter tuning (Table S7)
python analysis/refit_primary.py --variant backhand --check-reference  # Backhand Returns ablation (Table S5)
python analysis/refit_all_ablation.py --check-reference              # 34 deletion refits (Figure 4b)
python analysis/refit_sensitivity.py                                 # Figure 4a (after refit_primary.py)
python analysis/refit_shap.py --check-reference                      # ensemble KernelSHAP (after refit_primary.py)
python analysis/refit_shap.py --runs 30 && python analysis/summarize_shap_repeats.py  # Table S5 reruns
python analysis/refit_revision_opponent.py                           # Table S8
python analysis/refit_revision_representations.py court_drop_only              # Table S9
python analysis/refit_revision_representations.py return_share_without_total   # Table S10
python analysis/refit_revision_representations.py return_rates_without_total   # Table S10
```

The twelve-indicator exclusion removes Service Breaks, Total Points Won, Aces, Net Points Won, Double Faults, Unforced Errors, Backhand Return Win, Forehand Return Win, 1st Serve Pts Won, 2nd Serve Pts Won, Backhand Returns Won and Forehand Returns Won, and repeats screening, calibration and selection. For Table S10, ΔAUC is the training-mean mask: the focal predictor is replaced by its development-set mean while the fitted learners, screening, calibration and ensemble members remain fixed. For Table S9, ΔAUC deletes Backhand Returns and refits with the selected members fixed.

## Repository layout

| Path | Contents |
| --- | --- |
| `data/raw/` | Six de-identified annual source workbooks |
| `data/processed/` | Authoritative 199-set and 40-set model inputs |
| `data/indicator_dictionary.csv` | Names, definitions, codes and model-column names (Table 1) |
| `data/reference/` | Deposited predictions, explanations and reference results |
| `data/reference/revision2026/` | Source arrays and records for Tables S3 and S5–S11 |
| `analysis/` | Reproduction and refit scripts |
| `supplementary/` | Table 1 and Tables S1–S11 |
| `figures/` | Figures 1–5 |
| `validation/` | Logs of the validation runs listed in `VALIDATION_REPORT.json` |
| `data/sha256.csv`, `MANIFEST_SHA256.csv` | Data and complete-release integrity manifests |

## Provenance and citation

Version `1.0.2` extends the published release [v1.0.1, DOI 10.5281/zenodo.23133624](https://doi.org/10.5281/zenodo.23133624); the release-series DOI is [10.5281/zenodo.23132418](https://doi.org/10.5281/zenodo.23132418). Third-party records retain their upstream attribution and licence; see `LICENSE`, `data/README.md` and `data/reference/revision2026/opponent/sources/`. No match footage or observer identities are included.

After changing any released file, run `python update_manifests.py` and then `python run_all.py`.
