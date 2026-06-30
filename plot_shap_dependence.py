# -*- coding: utf-8 -*-
"""
Single-indicator SHAP dependence plots (one per top-6 ensemble indicator)
=========================================================================
Reproduces, in English, the style of the reference dependence figure:
  * x-axis  = the indicator's value on the test set
  * y-axis  = the SHAP value of that indicator (ensemble KernelSHAP)
  * colour  = the most-interacting other indicator
  * dashed vertical line = threshold located by the "max-jump" rule
    (the x where the SHAP curve, sorted by x, makes its single largest step)

SHAP values are not stored in the results workbook, so this script recomputes
the ensemble SHAP from scratch with the SAME settings as the main pipeline
(seed 42, de-collinearised 31 features, ensemble = LR + Ridge + Lasso + KNN,
KernelExplainer with background 50, nsamples 150).

Outputs: figure_dependence_<rank>_<indicator>.pdf  (vector, editable fonts)
Dependencies: numpy, pandas, scikit-learn, shap, matplotlib
"""
import warnings, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
import shap

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
TRAIN_CSV, VAL_CSV = "train_en.csv", "val_en.csv"
RESULTS_XLSX, SHEET8 = "tennis_workflow_results_en.xlsx", "8-Ensemble SHAP (All)"
CORR_THRESHOLD = 0.90
N_TOP = 6
SHAP_BG, SHAP_NSAMPLES = 50, 150
DEP_CMAP = LinearSegmentedColormap.from_list("blue_purple_red", ["#1E88E5", "#8E24AA", "#E53935"])


def clean(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.replace("、", "", regex=False)
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().all():
            df[c] = LabelEncoder().fit_transform(df[c].fillna("missing"))
        elif df[c].isna().any():
            df[c] = df[c].fillna(df[c].mean())
    return df


def load_and_decorrelate():
    tr, va = clean(pd.read_csv(TRAIN_CSV)), clean(pd.read_csv(VAL_CSV))
    y = tr.iloc[:, 0].astype(int).values
    X0 = tr.iloc[:, 1:].replace([np.inf, -np.inf], np.nan)
    X0 = X0.fillna(X0.mean())
    Xv0 = va.iloc[:, 1:].replace([np.inf, -np.inf], np.nan)
    for c in Xv0.columns:
        Xv0[c] = Xv0[c].fillna(X0[c].mean())
    feat0 = list(X0.columns)
    Xa, Xva = X0.values.astype(float), Xv0.values.astype(float)
    n = len(feat0)
    A = np.abs(np.corrcoef(Xa.T)) >= CORR_THRESHOLD
    seen, clusters = [False] * n, []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
        while stack:
            u = stack.pop()
            if seen[u]:
                continue
            seen[u] = True
            comp.append(u)
            for v in range(n):
                if v != u and A[u, v] and not seen[v]:
                    stack.append(v)
        clusters.append(comp)
    tcorr = np.array([abs(np.corrcoef(Xa[:, j], y)[0, 1]) for j in range(n)])
    keep = sorted({(max(c, key=lambda j: tcorr[j]) if len(c) > 1 else c[0]) for c in clusters})
    feat = [feat0[i] for i in keep]
    return Xa[:, keep], y, Xva[:, keep], feat


def ensemble_predict_factory(X, y):
    def sc(m):
        return Pipeline([("s", StandardScaler()), ("m", m)])
    models = {"LR": sc(LinearRegression()), "Ridge": sc(Ridge(alpha=1.0)),
              "Lasso": sc(Lasso(alpha=0.1)), "KNN": sc(KNeighborsRegressor(n_neighbors=5))}
    for m in models.values():
        m.fit(X, y)
    return lambda A: np.mean([m.predict(A) for m in models.values()], axis=0)


def pick_interaction(i, sv, X):
    """Indicator that best explains the SHAP values of indicator i (max |corr|)."""
    best_j, best_r = None, -1
    for j in range(X.shape[1]):
        if j == i:
            continue
        if np.std(X[:, j]) < 1e-12:
            continue
        r = abs(np.corrcoef(X[:, j], sv[:, i])[0, 1])
        if np.isfinite(r) and r > best_r:
            best_r, best_j = r, j
    return best_j


def max_jump_threshold(x, yv):
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], yv[order]
    if len(xs) < 2:
        return float(np.median(x))
    k = int(np.argmax(np.abs(np.diff(ys))))
    return float(xs[k])


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:40]


def main():
    X, y, Xv, feat = load_and_decorrelate()
    predict = ensemble_predict_factory(X, y)
    bg = shap.sample(X, min(SHAP_BG, len(X)), random_state=SEED)
    sv = np.array(shap.KernelExplainer(predict, bg).shap_values(Xv, nsamples=SHAP_NSAMPLES, silent=True))
    if sv.ndim == 3:
        sv = sv[..., -1]

    # The six indicators (and their order) are read directly from the paper's Sheet 8
    # ensemble-SHAP ranking. No feature names are hard-coded: change the data -> Sheet 8
    # changes -> these dependence plots follow automatically. Only the workbook/sheet names
    # and N_TOP are configuration. (Reading the ranking from Sheet 8 also keeps the selection
    # consistent with the paper, independent of the small KernelSHAP sampling noise here.)
    s8 = pd.read_excel(RESULTS_XLSX, sheet_name=SHEET8).sort_values("Rank")
    top_names = [nm for nm in s8["Feature"].tolist() if nm in feat][:N_TOP]
    if len(top_names) < N_TOP:
        raise ValueError(f"Sheet 8 yielded only {len(top_names)} usable indicators; "
                         f"check '{RESULTS_XLSX}' / sheet '{SHEET8}'.")
    order = [feat.index(nm) for nm in top_names]
    print("Top-%d indicators (read from Sheet 8):" % N_TOP, [feat[i] for i in order])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12.5, "axes.labelsize": 11.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600})

    outputs = []
    for rank, i in enumerate(order, start=1):
        x = Xv[:, i].astype(float)
        yv = sv[:, i].astype(float)
        j = pick_interaction(i, sv, Xv)
        thr = max_jump_threshold(x, yv)

        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        scv = ax.scatter(x, yv, c=Xv[:, j], cmap=DEP_CMAP, s=48,
                         edgecolor="white", linewidth=0.4, zorder=3)
        scv.set_rasterized(False)
        ax.axhline(0, color="#BBBBBB", lw=0.8, zorder=1)
        ax.axvline(thr, color="#2C3E80", ls=(0, (5, 4)), lw=1.3, zorder=2)
        ymin, ymax = ax.get_ylim()
        ax.text(thr, ymax, f"  Threshold \u2248 {thr:.3f}", rotation=90,
                va="top", ha="left", fontsize=9, color="#2C3E80")

        cb = fig.colorbar(scv, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(feat[j], fontsize=10)
        cb.outline.set_visible(False)
        cb.solids.set_rasterized(False)

        ax.set_xlabel(feat[i])
        ax.set_ylabel(f"SHAP value for\n{feat[i]}")
        ax.set_title(f"Dependence Plot (Top {rank}/{N_TOP}): {feat[i]}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        fname = f"figure_dependence_{rank}_{slug(feat[i])}.pdf"
        fig.savefig(fname, bbox_inches="tight")
        plt.close(fig)
        outputs.append(fname)
        print("saved", fname, "| colour =", feat[j], "| threshold =", round(thr, 3))
    return outputs


if __name__ == "__main__":
    main()
