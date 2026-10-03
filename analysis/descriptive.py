"""Recalculate distribution checks and set-outcome correlations (Table 2)."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, shapiro, spearmanr

from reproduce import ROOT, verify_inputs


def main() -> None:
    train = pd.read_csv(ROOT / "data/processed/train_grouped.csv")
    test = pd.read_csv(ROOT / "data/processed/test_grouped.csv")
    verify_inputs(train, test)
    y = train["Set Result"].to_numpy(int)
    rows = []
    for name in train.columns[1:35]:
        x = train[name].to_numpy(float)
        later = test[name].to_numpy(float)
        w, normality_p = shapiro(x)
        method = "Pearson" if normality_p >= 0.05 else "Spearman"
        coefficient, correlation_p = (pearsonr if method == "Pearson" else spearmanr)(x, y)
        rows.append({
            "indicator": name,
            "development_median": float(np.median(x)),
            "development_q1": float(np.quantile(x, 0.25)),
            "development_q3": float(np.quantile(x, 0.75)),
            "test_median": float(np.median(later)),
            "test_q1": float(np.quantile(later, 0.25)),
            "test_q3": float(np.quantile(later, 0.75)),
            "shapiro_w": float(w),
            "shapiro_p": float(normality_p),
            "correlation_method": method,
            "correlation_coefficient": float(coefficient),
            "correlation_p": float(correlation_p),
        })
    output = ROOT / "outputs/descriptive_correlations.csv"
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Recalculated {len(rows)} indicator distributions and correlations", flush=True)


if __name__ == "__main__":
    main()
