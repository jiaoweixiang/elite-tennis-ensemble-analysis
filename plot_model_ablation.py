# -*- coding: utf-8 -*-
"""
Model-size ablation visualization over all 4095 non-empty ensemble subsets.
Reuses the main pipeline to recompute OOF/test scores and renders the ablation panel.
"""
import os
import numpy as np
import pandas as pd
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import tennis_workflow_v4_en as tw
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

C_ALL      = "#9ec9e2"
C_BESTK    = "#1f4e78"
C_GLOBAL   = "#d1495b"
C_IN       = "#1f4e78"
C_OUT      = "#f2f2f2"


def enumerate_all(names, OOF, TEST, y, yv):
    n = len(names)
    total = (1 << n) - 1
    print(f"Enumerating all {total} non-empty subsets ...")
    rows = []
    cnt = 0
    for k in range(1, n + 1):
        for combo in combinations(range(n), k):
            cnt += 1
            idx = list(combo)
            oof_avg = OOF[:, idx].mean(axis=1)
            thr = tw.youden_threshold(y, oof_avg)
            oof_pred = (oof_avg >= thr).astype(int)
            test_avg = TEST[:, idx].mean(axis=1)
            test_pred = (test_avg >= thr).astype(int)
            try:
                tauc = roc_auc_score(yv, test_avg)
            except Exception:
                tauc = np.nan
            rows.append({
                "k": k,
                "members": "+".join(names[i] for i in idx),
                "mask": tuple(1 if i in idx else 0 for i in range(n)),
                "OOF_F1": f1_score(y, oof_pred),
                "OOF_Acc": accuracy_score(y, oof_pred),
                "Test_Acc": accuracy_score(yv, test_pred),
                "Test_F1": f1_score(yv, test_pred),
                "Test_AUC": tauc,
            })
        if cnt % 1000 < k:
            print(f"  ... {cnt}/{total}")
    df = pd.DataFrame(rows)
    print(f"Done: {len(df)} combinations.")
    return df


def pick_best(df):
    order = df.sort_values(["OOF_F1", "OOF_Acc"], ascending=False)
    best = order.iloc[0]
    idx_by_k = (df.sort_values(["OOF_F1", "OOF_Acc"], ascending=False)
                  .groupby("k", as_index=False).head(1)
                  .sort_values("k"))
    return best, idx_by_k


def fig_strip(df, best, best_by_k, names):
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ks = sorted(df["k"].unique())

    viol_data, viol_pos = [], []
    for k in ks:
        vals = df.loc[df["k"] == k, "OOF_F1"].values
        if len(vals) >= 8:
            viol_data.append(vals)
            viol_pos.append(k)
    if viol_data:
        vp = ax.violinplot(viol_data, positions=viol_pos, widths=0.78,
                           showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor("#cfe3f2"); b.set_edgecolor("none"); b.set_alpha(0.55)

    for k in ks:
        vals = df.loc[df["k"] == k, "OOF_F1"].values
        x = k + rng.uniform(-0.32, 0.32, size=len(vals))
        ax.scatter(x, vals, s=8, color=C_ALL, alpha=0.45,
                   edgecolors="none", zorder=2)

    bx = best_by_k["k"].values
    by = best_by_k["OOF_F1"].values
    ax.plot(bx, by, "-", color=C_BESTK, lw=1.6, zorder=4)
    ax.scatter(bx, by, s=46, color=C_BESTK, zorder=5,
               label="Best subset at each model count")

    ax.scatter([best["k"]], [best["OOF_F1"]], marker="*", s=420,
               color=C_GLOBAL, edgecolors="white", linewidths=1.2, zorder=6,
               label=f"Global optimum: {{{best['members'].replace('+', ', ')}}}")
    ax.annotate(f"k = {int(best['k'])}\nOOF F1 = {best['OOF_F1']:.4f}",
                xy=(best["k"], best["OOF_F1"]),
                xytext=(best["k"] + 1.4, best["OOF_F1"] - 0.045),
                fontsize=9, color=C_GLOBAL, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_GLOBAL, lw=1.2))

    ax.set_xticks(ks)
    ax.set_xlabel("Number of base models in the ensemble (k)", fontsize=11)
    ax.set_ylabel("Out-of-fold F1 (training, 5-fold)", fontsize=11)
    ax.set_title("Exhaustive model-ablation over all 4,095 ensemble subsets",
                 fontsize=12, fontweight="bold")
    ax.text(0.985, 0.04,
            r"$N=2^{12}-1=4{,}095$ model combinations",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color="#555", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.85))
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    ax.grid(axis="y", ls=":", color="#ccc", alpha=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("figure3A_ablation_strip_by_k.pdf"); fig.savefig("figure3A_ablation_strip_by_k.png", dpi=150)
    plt.close(fig)
    print("  saved figure3A_ablation_strip_by_k.pdf")


def fig_membership(df, best, names):
    order = df.sort_values(["OOF_F1", "OOF_Acc"], ascending=False).reset_index(drop=True)
    M = np.array(order["mask"].tolist())
    f1 = order["OOF_F1"].values

    fig, (axh, axb) = plt.subplots(
        1, 2, figsize=(8.6, 6.0), gridspec_kw={"width_ratios": [3.0, 1.0], "wspace": 0.06})

    axh.imshow(M, aspect="auto", interpolation="nearest",
               cmap=matplotlib.colors.ListedColormap([C_OUT, C_IN]),
               vmin=0, vmax=1)
    axh.set_xticks(range(len(names)))
    axh.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axh.set_ylabel("All 4,095 ensemble subsets\n(ranked by OOF F1, best at top)", fontsize=10)
    axh.set_yticks([0, len(order) - 1])
    axh.set_yticklabels(["best", "worst"], fontsize=9)
    best_row = int(order.index[(order["members"] == best["members"]).values][0])
    axh.axhline(best_row, color=C_GLOBAL, lw=1.4)
    axh.set_title("Model membership of every subset", fontsize=11, fontweight="bold")

    yy = np.arange(len(order))
    axb.plot(f1, yy, color=C_BESTK, lw=0.8)
    axb.fill_betweenx(yy, f1.min(), f1, color="#cfe3f2", alpha=0.7)
    axb.scatter([best["OOF_F1"]], [best_row], marker="*", s=160,
                color=C_GLOBAL, edgecolors="white", linewidths=0.8, zorder=5)
    axb.set_ylim(len(order) - 1, 0)
    axb.set_yticks([])
    axb.set_xlabel("OOF F1", fontsize=10)
    axb.set_title("Performance", fontsize=11, fontweight="bold")
    for s in ("top", "right"):
        axb.spines[s].set_visible(False)

    topN = max(1, int(0.05 * len(order)))
    freq = M[:topN].mean(axis=0)
    leg = [Patch(facecolor=C_IN, label="model included"),
           Patch(facecolor=C_OUT, edgecolor="#ccc", label="model excluded"),
           Patch(facecolor=C_GLOBAL, label=f"global optimum (k={int(best['k'])})")]
    axh.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.13),
               ncol=3, fontsize=8.5, frameon=False)
    order_str = ", ".join(f"{n}:{p:.0%}" for n, p in
                          sorted(zip(names, freq), key=lambda t: -t[1]))
    fig.text(0.5, 0.005,
             f"Inclusion frequency among the best 5% subsets - {order_str}",
             ha="center", fontsize=7.4, color="#555")
    fig.suptitle("Exhaustive model-ablation: which base models build the best ensembles",
                 fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig("figure3B_ablation_membership.pdf"); fig.savefig("figure3B_ablation_membership.png", dpi=150)
    plt.close(fig)
    print("  saved figure3B_ablation_membership.pdf")


def fig_rank_curve(df, best, names):
    order = df.sort_values(["OOF_F1", "OOF_Acc"], ascending=False).reset_index(drop=True)
    f1 = order["OOF_F1"].values
    ranks = np.arange(1, len(order) + 1)
    best_rank = int(order.index[(order["members"] == best["members"]).values][0]) + 1

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.fill_between(ranks, f1.min(), f1, color="#cfe3f2", alpha=0.6)
    ax.plot(ranks, f1, color=C_BESTK, lw=1.4)

    ax.scatter([best_rank], [best["OOF_F1"]], marker="*", s=380,
               color=C_GLOBAL, edgecolors="white", linewidths=1.2, zorder=6)
    ax.annotate(
        f"Selected ensemble  {{{best['members'].replace('+', ', ')}}}\n"
        f"rank {best_rank} of {len(order)}   |   OOF F1 = {best['OOF_F1']:.4f}",
        xy=(best_rank, best["OOF_F1"]),
        xytext=(len(order) * 0.18, best["OOF_F1"] - 0.07),
        fontsize=9.5, color=C_GLOBAL, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_GLOBAL, lw=1.3))

    ax.set_xlabel("Ensemble subset rank (sorted by OOF F1)", fontsize=11)
    ax.set_ylabel("Out-of-fold F1 (training, 5-fold)", fontsize=11)
    ax.set_title("All 4,095 ensemble subsets ranked by performance",
                 fontsize=12, fontweight="bold")
    ax.text(0.985, 0.92,
            f"N = {len(order):,} model combinations\n"
            f"OOF F1 range: {f1.min():.3f} - {f1.max():.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color="#555", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bbb", alpha=0.85))
    ax.set_xlim(0, len(order))
    ax.grid(ls=":", color="#ccc", alpha=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig("figure3C_ablation_rank_curve.pdf"); fig.savefig("figure3C_ablation_rank_curve.png", dpi=150)
    plt.close(fig)
    print("  saved figure3C_ablation_rank_curve.pdf")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    X0, y, Xv0, yv, feat0 = tw.load_data()
    keep, _ = tw.decorrelate(X0, y, feat0, tw.COLLINEAR_THRESH)
    X, Xv = X0[:, keep], Xv0[:, keep]
    names = list(tw.make_models().keys())
    print("Computing OOF/TEST score matrices (same as main pipeline) ...")
    OOF, TEST = tw.compute_scores(X, y, Xv, names)

    df = enumerate_all(names, OOF, TEST, y, yv)
    df.drop(columns=["mask"]).to_csv("model_ablation_all_4095.csv", index=False)
    best, best_by_k = pick_best(df)

    print("\n=== Best subset per ensemble size ===")
    print(best_by_k[["k", "members", "OOF_Acc", "OOF_F1",
                     "Test_Acc", "Test_F1", "Test_AUC"]].to_string(index=False))
    print(f"\nGlobal optimum: {{{best['members']}}}  (k={int(best['k'])}, "
          f"OOF F1={best['OOF_F1']:.4f}, OOF Acc={best['OOF_Acc']:.4f})")

    print("\nDrawing figures ...")
    fig_strip(df, best, best_by_k, names)
    fig_membership(df, best, names)
    fig_rank_curve(df, best, names)
    print("\n[Done] 3 PDFs + model_ablation_all_4095.csv")


if __name__ == "__main__":
    main()
