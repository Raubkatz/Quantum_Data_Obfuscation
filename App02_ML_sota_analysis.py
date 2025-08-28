"""
Non–Lie-Group Results: 1×N Heatmaps (same color scheme as older scripts)

Reads:
    experiment_results_medical_sota_comparisons_nu_100/final_results.pkl
        results[dataset_name][(method, param)] = {
            "Best CV Score", "Test Accuracy", "Test Precision",
            "Test Recall", "Test F1", "Training Time", "Params"
        }
        Baseline at key ("original","none").

Writes (under analysis_results_non_lie_heatmaps/):
    Per dataset / method:
        - acc_{dataset}_{method}.png/.eps      (1×N accuracy heatmap)
        - delta_{dataset}_{method}.png/.eps    (1×N Δ vs baseline heatmap)
        - metrics_{dataset}_{method}.csv       (ordered table used for plotting)
        - metrics_{dataset}_{method}.tex       (LaTeX table)

Color scheme:
    - Accuracy: viridis
    - Delta vs baseline: ListedColormap(['#3A2081', '#64A1CF']) with BoundaryNorm([-1, 0, 1])
"""

import os
import pickle
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ----------------------------
# Config
# ----------------------------
RESULTS_DIR = "experiment_results_medical_sota_comparisons_nu_100"
FINAL_FILE  = "final_results.pkl"
OUT_DIR     = "analysis_results_non_lie_heatmaps"
os.makedirs(OUT_DIR, exist_ok=True)

METHOD_LABELS = {
    "original":         "Baseline",
    "random_projection":"Random Projection",
    "pca_gaussian":     "PCA + Gaussian",
    "shuffle":          "Feature Shuffling",
    "gaussian_mech":    "Gaussian Mechanism",
}

PARAM_AXIS_LABEL = {
    "random_projection": "n_components",
    "pca_gaussian":      "noise std (latent)",
    "shuffle":           "shuffled fraction",
    "gaussian_mech":     "ε (smaller = stronger privacy)",
}

FONT_SIZE = 16
ANNOT_SIZE = 14

def pretty_param(method: str, p: Any) -> str:
    if method == "random_projection" and p is None:
        return "full"
    return str(p)

def is_numeric(vals: List[Any]) -> bool:
    try:
        _ = [float(v) for v in vals]
        return True
    except Exception:
        return False

# ----------------------------
# Load results
# ----------------------------
final_path = os.path.join(RESULTS_DIR, FINAL_FILE)
if not os.path.exists(final_path):
    raise FileNotFoundError(f"Results file not found: {final_path}")

with open(final_path, "rb") as f:
    results: Dict[str, Dict[Tuple[str, Any], Dict[str, Any]]] = pickle.load(f)

# ----------------------------
# Plot helpers (one-row heatmaps)
# ----------------------------
def plot_heatmap_row(values: List[float],
                     xlabels: List[str],
                     title: str,
                     cbar_label: str,
                     out_png: str,
                     out_eps: str,
                     cmap,
                     norm=None):
    """
    values: list length N -> will be plotted as a single row matrix of shape (1, N)
    """
    mat = np.array(values, dtype=float).reshape(1, -1)

    plt.figure(figsize=(max(8, 1.2*len(values)), 2.8))
    ax = sns.heatmap(
        mat,
        annot=True,
        fmt=".4f",
        cmap=cmap,
        norm=norm,
        cbar=True,
        xticklabels=xlabels,
        yticklabels=[""],
        annot_kws={"size": ANNOT_SIZE}
    )
    ax.set_title(title, fontsize=FONT_SIZE)
    ax.set_xlabel("")  # xlabels carry parameter values
    ax.set_ylabel("")
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=15)
    ax.tick_params(axis='x', labelrotation=0, labelsize=FONT_SIZE-2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.savefig(out_eps, bbox_inches="tight")
    plt.close()

# ----------------------------
# Main
# ----------------------------
for dataset, entries in results.items():
    # baseline
    baseline_key = ("original", "none")
    if baseline_key not in entries:
        candidates = [k for k in entries if k[0] == "original"]
        if not candidates:
            print(f"[WARN] No baseline for '{dataset}'. Skipping dataset.")
            continue
        baseline_key = candidates[0]
    baseline_acc = float(entries[baseline_key].get("Test Accuracy", np.nan))

    ds_dir = os.path.join(OUT_DIR, dataset.replace(" ", "_"))
    os.makedirs(ds_dir, exist_ok=True)

    # per method
    for method in ["random_projection", "pca_gaussian", "shuffle", "gaussian_mech"]:
        # collect rows
        rows = []
        for (m, p), met in entries.items():
            if m != method:
                continue
            acc = float(met.get("Test Accuracy", np.nan))
            dlt = acc - baseline_acc
            rows.append({
                "param": p,
                "param_label": pretty_param(method, p),
                "Test Accuracy": acc,
                "Delta Accuracy": dlt,
                "Best CV Score": float(met.get("Best CV Score", np.nan)),
                "Test Precision": float(met.get("Test Precision", np.nan)),
                "Test Recall": float(met.get("Test Recall", np.nan)),
                "Test F1": float(met.get("Test F1", np.nan)),
                "Training Time": float(met.get("Training Time", np.nan)),
            })
        if not rows:
            print(f"[INFO] No entries for {dataset} — {method}")
            continue

        df = pd.DataFrame(rows)

        # sort params
        if is_numeric(df["param"].tolist()):
            df = df.assign(_p=[float(p) for p in df["param"]]).sort_values("_p").drop(columns="_p")
        else:
            df = df.sort_values("param_label")

        # save tables
        pretty_axis = PARAM_AXIS_LABEL.get(method, "parameter")
        df_out = df[[
            "param_label", "Best CV Score", "Test Accuracy", "Delta Accuracy",
            "Test Precision", "Test Recall", "Test F1", "Training Time"
        ]].rename(columns={"param_label": pretty_axis})
        csv_path = os.path.join(ds_dir, f"metrics_{dataset.replace(' ','_')}_{method}.csv")
        tex_path = os.path.join(ds_dir, f"metrics_{dataset.replace(' ','_')}_{method}.tex")
        df_out.to_csv(csv_path, index=False)
        try:
            with open(tex_path, "w") as f:
                f.write(df_out.to_latex(index=False, float_format="%.4f", escape=False))
        except Exception as e:
            print(f"[WARN] LaTeX export failed for {dataset} — {method}: {e}")

        # one-row heatmaps
        xlabels = df["param_label"].tolist()
        acc_vals = df["Test Accuracy"].tolist()
        dlt_vals = df["Delta Accuracy"].tolist()

        # Accuracy heatmap (viridis)
        acc_png = os.path.join(ds_dir, f"acc_{dataset.replace(' ','_')}_{method}.png")
        acc_eps = os.path.join(ds_dir, f"acc_{dataset.replace(' ','_')}_{method}.eps")
        plot_heatmap_row(
            values=acc_vals,
            xlabels=xlabels,
            title=f"{METHOD_LABELS[method]} — {dataset}\nBaseline accuracy = {baseline_acc:.3f}",
            cbar_label="Accuracy",
            out_png=acc_png,
            out_eps=acc_eps,
            cmap="viridis",
            norm=None
        )

        # Δ vs baseline heatmap (two-color map, boundary at 0)
        delta_png = os.path.join(ds_dir, f"delta_{dataset.replace(' ','_')}_{method}.png")
        delta_eps = os.path.join(ds_dir, f"delta_{dataset.replace(' ','_')}_{method}.eps")
        two_color = mcolors.ListedColormap(['#3A2081', '#64A1CF'])
        bounds = [-1.0, 0.0, 1.0]
        norm = mcolors.BoundaryNorm(bounds, two_color.N)
        plot_heatmap_row(
            values=dlt_vals,
            xlabels=xlabels,
            title=f"{METHOD_LABELS[method]} — {dataset} (Δ vs baseline)",
            cbar_label="ΔAccuracy",
            out_png=delta_png,
            out_eps=delta_eps,
            cmap=two_color,
            norm=norm
        )

        # ----------------------------
        # NEW: Printout of accuracies
        # ----------------------------
        print(f"\n=== {dataset} | {METHOD_LABELS[method]} ===")
        for lbl, acc in zip(xlabels, acc_vals):
            print(f"  {lbl:>12s} : {acc:.4f}")

print("Done. Outputs written to:", OUT_DIR)
