# Elite Tennis Ensemble Analysis

This repository contains the reproducible analysis code for a case study of technical-tactical determinants of set outcomes in elite tennis, using Novak Djokovic's Grand Slam hard-court matches as the study sample.

The workflow screens 12 candidate machine-learning algorithms, evaluates all 4,095 non-empty model subsets using leakage-safe out-of-fold (OOF) predictions, selects a parsimonious four-model ensemble, and applies feature-level ablation and SHAP analysis to identify key technical-tactical determinants.

## Repository Contents

```text
.
├── train_en.csv                       # Training data, 199 sets
├── val_en.csv                         # Independent test data, 40 sets
├── tennis_workflow_v4_en.py           # Main reproducible workflow
├── plot_model_ablation.py             # Exhaustive 4,095-subset ensemble ablation figures
├── plot_importance_heatmap.py         # Native importance heatmap for selected ensemble members
├── plot_shap_dependence.py            # SHAP dependence plots for the top ensemble indicators
├── run_all.py                         # One-command reproduction script
├── requirements.txt                   # Python dependencies
└── README.md
```

## Environment

The code was tested with Python 3.10. Install dependencies with:

```bash
pip install -r requirements.txt
```

The main dependencies are `numpy`, `pandas`, `scikit-learn`, `shap`, `matplotlib`, `seaborn`, and `openpyxl`.

## Reproduce the Full Analysis

Run the complete workflow from the repository root:

```bash
python run_all.py
```

This executes the scripts in the required order:

1. `tennis_workflow_v4_en.py`
2. `plot_model_ablation.py`
3. `plot_importance_heatmap.py`
4. `plot_shap_dependence.py`

## Expected Key Results

With `random_state = 42`, the global OOF optimum among all 4,095 non-empty subsets of the 12 candidate algorithms is:

```text
LR + Ridge + Lasso + KNN
OOF F1 = 0.9723
OOF Accuracy = 0.9548
Test Accuracy = 0.9000
Test F1 = 0.9355
Test AUC = 0.9767
```

The SHAP dependence analysis should reproduce the following empirical thresholds:

```text
Total Returns       ~39 per set
Service Breaks      >=1 per set
Total Points Won    ~16 per set
Double Faults       ~3 per set
Backhand Returns    ~22 per set
Unforced Errors     ~15 per set
```

## Generated Outputs

Running `python run_all.py` generates:

- `tennis_workflow_results_en.xlsx`
- ROC and SHAP summary figures from the main workflow
- exhaustive model-subset ablation figures
- native importance heatmap
- SHAP dependence plots for the top ensemble indicators
- `model_ablation_all_4095.csv`

Generated output files are ignored by `.gitignore` so that the repository remains focused on code and source data.

## Data

The two CSV files contain de-identified set-level technical-tactical indicators used in the analysis:

- `train_en.csv`: training set
- `val_en.csv`: independent test set

The first column is the binary set outcome label, and the remaining columns are technical-tactical indicators.

## Citation

If using this repository, please cite the associated manuscript once available.

