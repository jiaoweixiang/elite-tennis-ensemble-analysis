# -*- coding: utf-8 -*-
"""
Native-importance heatmap for the selected ensemble members (signed)
====================================================================
Reads tennis_workflow_results_en.xlsx:
  * sheet "6-Native Importance (norm)"  -> magnitude of native importance
  * sheet "2-Correlation"               -> sign (Direction vs outcome)
and draws a vector-editable heatmap for LR, Ridge, Lasso, KNN.

Magnitudes in sheet 6 are absolute (|coef| etc.). We attach a sign from the
indicator's correlation direction with the outcome (sheet 2): a "Positive"
indicator keeps +, a "Negative" indicator becomes -, so the diverging colour
scale is meaningful (red = associated with losing, blue = with winning).

Indicator selection (exactly 10): start threshold = 0.20 on the |importance|
across the four models; if fewer than 10 indicators exceed it, lower the
threshold stepwise until >=10 do, then keep the 10 with the largest
cross-model maximum |importance|.

Colour: blue - white(0) - yellow - red (white pinned at 0; red/yellow = positive, blue = negative).
Note: sheet 6 now reports a single, uniform importance for every model --
permutation importance on the standardized pipelines (n_repeats=30, training
set; negative values set to 0), making the four columns directly comparable.

Output: figure_importance_heatmap.pdf  (vector, embedded editable fonts)
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

XLSX_IN   = "tennis_workflow_results_en.xlsx"
SHEET_IMP = "6-Native Importance (norm)"
SHEET_COR = "2-Correlation"
PDF_OUT   = "figure_importance_heatmap.pdf"
MODELS    = ["LR", "Ridge", "Lasso", "KNN"]
START_THR, STEP, N_TOP = 0.20, 0.01, 10
CMAP = LinearSegmentedColormap.from_list(
    "BuW_YlRd", ["#2166AC", "#ABD9E9", "#FFFFFF", "#FEE08B", "#B2182B"])


def main():
    imp = pd.read_excel(XLSX_IN, sheet_name=SHEET_IMP, index_col=0)
    cor = pd.read_excel(XLSX_IN, sheet_name=SHEET_COR)
    sign = {r["Feature"]: (1.0 if str(r["Direction"]).strip().lower().startswith("pos") else -1.0)
            for _, r in cor.iterrows()}

    sub = imp.loc[MODELS]                              # 4 x n_features, magnitudes
    maxis = sub.max(axis=0)                            # per-indicator max |importance|
    thr = START_THR
    while (maxis > thr).sum() < N_TOP and thr > -1:
        thr -= STEP
    feats = maxis[maxis > thr].sort_values(ascending=False).index.tolist()[:N_TOP]
    print(f"Indicators > {START_THR} in any of 4 models: {int((maxis > START_THR).sum())}; "
          f"threshold lowered to {round(thr,3)} to reach {N_TOP}.")

    mag = sub[feats].T.values.astype(float)           # 10 x 4 magnitudes
    s = np.array([sign.get(f, 1.0) for f in feats])[:, None]
    M = mag * s                                        # signed
    vmax = float(np.nanmax(np.abs(M))) or 1.0
    norm = Normalize(-vmax, vmax)                      # 0 -> white

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.titlesize": 12.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.dpi": 600})
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    qm = ax.pcolormesh(M, cmap=CMAP, norm=norm, edgecolors="white", linewidth=1.2)
    ax.set_box_aspect(1)                               # square plotting area

    nrow, ncol = M.shape
    ax.set_xticks(np.arange(ncol) + 0.5); ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_yticks(np.arange(nrow) + 0.5); ax.set_yticklabels(feats, fontsize=10)
    ax.invert_yaxis()
    ax.set_title("Signed permutation importance of selected ensemble members", pad=10)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(nrow):
        for j in range(ncol):
            frac = norm(M[i, j])
            tc = "white" if (frac > 0.80 or frac < 0.20) else "#222222"
            ax.text(j + 0.5, i + 0.5, f"{M[i, j]:+.3f}", ha="center", va="center",
                    fontsize=9, color=tc)

    cb = fig.colorbar(qm, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Signed importance  (red = toward winning, blue = toward losing; white = 0)", fontsize=9)
    try:
        cb.solids.set_rasterized(False)
    except Exception:
        pass
    fig.text(0.5, -0.015,
             "Magnitude = permutation importance per model on its standardized pipeline (sheet 6; "
             "n_repeats=30, negatives set to 0);\nsign = correlation direction with the outcome (sheet 2).",
             ha="center", va="top", fontsize=7.6, color="#666")
    fig.tight_layout()
    fig.savefig(PDF_OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PDF_OUT}")


if __name__ == "__main__":
    main()
