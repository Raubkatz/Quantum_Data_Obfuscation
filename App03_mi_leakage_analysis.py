"""
MI Aggregation & Error-Bar Plots for Obfuscation Methods
========================================================

Author: Dr. techn. Sebastian Raubitzek MSc. BSc.
Affiliation: SBA Research, Complexity and Resilience Research Group

Purpose
-------
For each dataset and obfuscation method, run n_runs randomized evaluations,
aggregate **mean mutual information (MI)** across repeats for each
obfuscation-strength setting (x-axis), compute standard error of the mean (SEM),
and generate:
  (1) MI vs. strength curves (averages), and
  (2) MI vs. strength curves with error bars (SEM).

Scope
-----
Datasets: sklearn breast cancer, OpenML ilpd, heart, breast-cancer-coimbra.
Obfuscations:
  - Lie-group feature maps: SU, SL (if `SymmetryFeatureMaps` is available).
  - Random Projection (vary output dimensionality).
  - PCA + Gaussian Noise (vary std; multiple keep ratios).
  - Feature Shuffling (vary shuffled fraction).
  - Gaussian Mechanism (DP-style; vary ε and report MI vs σ and vs ε).

Notes
-----
- The train/test split is fixed to isolate randomness to the obfuscation/noise
  procedures. Each repeat sets independent RNG seeds for numpy and python's
  random to produce fresh projections/shuffles/noise.
- MI is estimated via equal-width binning + mutual_info_score (model-free).
- Outputs are written under:
    leakage_results_agg/
      └── {dataset}/
            mi_{method}_{axis}.csv
            mi_{method}_{axis}_avg.png
            mi_{method}_{axis}_avg_sem.png

Plotting
--------
- Marker for averaged points: 'x' (small crosses) to make error bars visible.
- Colors: ListedColormap(['#3A2081', '#64A1CF']).
- When >2 curves are needed (e.g., multiple keep ratios), lines cycle styles:
  ['-', '--', ':', '-.'] while reusing the two colors.

Dependencies
------------
- numpy, pandas, matplotlib, scikit-learn, scipy
- class_symmetry_feature_maps_noise.SymmetryFeatureMaps
"""

import os
import sys
import math
import json
import random
from copy import deepcopy as dc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.metrics import mutual_info_score


from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import mutual_info_score

# Optional Lie-group maps
try:
    from class_symmetry_feature_maps_noise import SymmetryFeatureMaps
    HAS_LIE = True
except Exception:
    HAS_LIE = False

# -------------------------
# Configuration
# -------------------------

OUT_ROOT = "leakage_results_agg"
os.makedirs(OUT_ROOT, exist_ok=True)

# number of repeated runs per configuration
N_RUNS = 10000

# Requested colormap & line styles
CMAP = mcolors.ListedColormap(['#3A2081', '#64A1CF'])
LINE_STYLES = ['-', '--', ':', '-.']

# DP Gaussian settings (match leakage assumptions)
DP_DELTA = 1e-5
CLIP_MIN = 0.0
CLIP_MAX = math.pi
SENS = (CLIP_MAX - CLIP_MIN)

# Binning for MI
BINS = 30

# Grids per method
LIE_GROUPS = ['SU', 'SL']            # if available
LIE_NOISES = [0.0, 0.001, 0.003, 0.01, 0.032, 0.1]
LIE_MULTIPLY = [1]                   # leakage comparison

# Random projection: explicit integer dims (no None)
def rp_dims_for(d: int):
    """Return explicit output dims for RP (avoid None)."""
    return sorted(set([max(2, min(d, d)), max(2, d // 2), max(2, min(8, d))]))

PCA_KEEP = [1.0, 0.75, 0.5]
PCA_STDS = [0.0, 0.01, 0.03, 0.05, 0.1]

SHUFFLE_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]

DP_EPSILONS = [0.5, 1.0, 2.0, 4.0]  # smaller ε => stronger privacy

# -------------------------
# Dataset loading
# -------------------------

def encode_non_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map categories to [0,1] evenly; leave numeric columns as-is."""
    for column in df.select_dtypes(include=['object', 'category']).columns:
        uniq = df[column].unique()
        if len(uniq) <= 1:
            df[column] = 0.0
            continue
        mapping = {v: i / (len(uniq) - 1) for i, v in enumerate(uniq)}
        df[column] = df[column].map(mapping)
    return df

def load_dataset_nu(data_nr: int):
    """
    0 -> sklearn breast cancer
    1 -> OpenML 'ilpd'
    2 -> OpenML 'heart'
    3 -> OpenML 'breast-cancer-coimbra'
    """
    if data_nr == 0:
        data_sk = load_breast_cancer()
        return data_sk.data, data_sk.target, "breast cancer"
    elif data_nr in [1, 2, 3]:
        data_names = {1: "ilpd", 2: "heart", 3: "breast-cancer-coimbra"}
        data_name = data_names[data_nr]
        data_sk = fetch_openml(name=data_name, version=1, as_frame=True)
        X, y = data_sk.data, data_sk.target
        if isinstance(X, pd.DataFrame):
            X = encode_non_numeric_features(X).values
        if isinstance(y, pd.Series) and y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        return X, y, data_name
    else:
        print('No valid data choice, exiting...')
        sys.exit(1)


def load_dataset(data_nr):
    if data_nr == 0:
        data_sk = load_breast_cancer()
        return data_sk.data, data_sk.target, "breast cancer"
    elif data_nr in range(1, 4):  # OpenML datasets
        data_names = ["ilpd",  #
                      "heart",
                      "breast-cancer-coimbra",  #
                      "diabetes",  #
                      ]
        data_name = data_names[data_nr - 3]
        data_sk = fetch_openml(name=data_name, version=1, as_frame=True)
        X = data_sk.data
        y = data_sk.target
        # Encode non-numeric features
        if isinstance(X, pd.DataFrame):
            X = encode_non_numeric_features(X)
        if isinstance(y, pd.Series) and y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)  # encode non-numeric labels
        return X, y, data_name
    else:
        print('No valid data choice, exiting...')
        sys.exit()

# -------------------------
# RNG utilities
# -------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)

# -------------------------
# MI estimation
# -------------------------

#def binned_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = BINS) -> float:
#    x_b = np.digitize(x, np.histogram_bin_edges(x, bins=bins)) - 1
#    y_b = np.digitize(y, np.histogram_bin_edges(y, bins=bins)) - 1
#    L = min(len(x_b), len(y_b))
#    return float(mutual_info_score(x_b[:L], y_b[:L]))  # typo fixed below!

# fix typo: correct function name
from sklearn.metrics import mutual_info_score as _mi_score
def binned_mutual_information(x: np.ndarray, y: np.ndarray, bins: int = BINS) -> float:
    x_b = np.digitize(x, np.histogram_bin_edges(x, bins=bins)) - 1
    y_b = np.digitize(y, np.histogram_bin_edges(y, bins=bins)) - 1
    L = min(len(x_b), len(y_b))
    return float(_mi_score(x_b[:L], y_b[:L]))

def mean_featurewise_mi(X: np.ndarray, X_obf: np.ndarray, bins: int = BINS) -> float:
    assert X.shape[0] == X_obf.shape[0]
    d = min(X.shape[1], X_obf.shape[1])
    mi_vals = [binned_mutual_information(X[:, j], X_obf[:, j], bins=bins) for j in range(d)]
    return float(np.nanmean(mi_vals))

# -------------------------
# Obfuscation transforms (stochastic)
# -------------------------

def transform_random_projection(X: np.ndarray, n_components: int, random_state=None) -> np.ndarray:
    """RP with explicit integer dims (>=2)."""
    d = X.shape[1]
    k = int(max(2, min(n_components, d)))
    rp = GaussianRandomProjection(n_components=k, random_state=random_state)
    return rp.fit_transform(X)

def transform_pca_noise(X: np.ndarray, noise_std: float, keep_ratio: float = 1.0, random_state=None) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    d = X.shape[1]
    k = max(1, int(math.ceil(d * keep_ratio)))
    pca = PCA(n_components=k, random_state=random_state)
    Z = pca.fit_transform(X)
    Z_noisy = Z + rng.normal(0.0, noise_std, size=Z.shape)
    return pca.inverse_transform(Z_noisy)

def transform_shuffle(X: np.ndarray, frac: float, random_state=None) -> np.ndarray:
    """Independent column permutations on a fraction of columns. frac=0.0 => no-op."""
    if frac <= 0.0:
        return X.copy()
    rng = np.random.RandomState(random_state)
    Xs = X.copy()
    d = X.shape[1]
    k = int(round(frac * d))
    k = max(0, min(k, d))
    if k == 0:
        return Xs
    cols = rng.choice(np.arange(d), size=k, replace=False)
    for j in cols:
        rng.shuffle(Xs[:, j])
    return Xs

def gaussian_mechanism_features(X: np.ndarray, epsilon: float, delta: float = DP_DELTA,
                                clip_min: float = CLIP_MIN, clip_max: float = CLIP_MAX,
                                random_state=None) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    Xc = X.copy()
    np.clip(Xc, clip_min, clip_max, out=Xc)
    eps = max(float(epsilon), 1e-8)
    sigma = SENS * math.sqrt(2 * math.log(1.25 / delta)) / eps
    return Xc + rng.normal(0.0, sigma, size=Xc.shape)

def dp_sigma_from_epsilon(eps: float, delta: float = DP_DELTA) -> float:
    eps = max(float(eps), 1e-8)
    return SENS * math.sqrt(2.0 * math.log(1.25 / float(delta))) / eps

def transform_lie_group(X: np.ndarray, noise_level: float,
                        group_family: str = 'SU', output_real: bool = True,
                        multiply: int = 1, random_state=None) -> np.ndarray:
    if not HAS_LIE:
        raise ImportError("SymmetryFeatureMaps not available.")
    # SymmetryFeatureMaps uses numpy RNG internally; we control global seed outside
    sfm = SymmetryFeatureMaps(X.shape[1])
    if multiply == 1:
        return np.array([
            sfm.apply_feature_map(x, group_family, output_real=output_real, noise_level=noise_level)
            for x in X
        ])
    Ys = []
    for x in X:
        variants = sfm.apply_feature_map(x, group_family, output_real=output_real,
                                         noise_level=noise_level, multiply=multiply)
        Ys.append(variants[0])
    return np.array(Ys)

# -------------------------
# Aggregation helpers
# -------------------------

def agg_stats(values: list) -> tuple:
    vals = np.asarray(values, dtype=float)
    mean = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return mean, sem

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_curve_csv(out_dir: str, filename: str, rows: list, columns: list):
    ensure_dir(out_dir)
    pd.DataFrame(rows, columns=columns).to_csv(os.path.join(out_dir, filename), index=False)

def plot_avg(x, y, xlabel, title, outfile, label=None, color=None, linestyle='-'):
    plt.figure()
    plt.plot(x, y, linestyle=linestyle, color=color if color else CMAP(0), marker='x')
    if label:
        plt.legend([label])
    plt.xlabel(xlabel)
    plt.ylabel("mean_mi")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()

def plot_avg_sem(x, y, yerr, xlabel, title, outfile, label=None, color=None, linestyle='-'):
    plt.figure()
    plt.errorbar(x, y, yerr=yerr, fmt='-x', color=color if color else CMAP(0),
                 ecolor=CMAP(1), capsize=3, linestyle=linestyle, label=label if label else None)
    if label:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("mean_mi")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()

def multi_curve_plot_avg(curves, xlabel, title, outfile):
    plt.figure()
    for i, c in enumerate(curves):
        color = CMAP(i % CMAP.N)
        linestyle = LINE_STYLES[i % len(LINE_STYLES)]
        plt.plot(c["x"], c["y"], linestyle=linestyle, color=color, marker='x', label=c["label"])
    plt.xlabel(xlabel)
    plt.ylabel("mean_mi")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()

def multi_curve_plot_avg_sem(curves, xlabel, title, outfile):
    plt.figure()
    for i, c in enumerate(curves):
        color = CMAP(i % CMAP.N)
        linestyle = LINE_STYLES[i % len(LINE_STYLES)]
        plt.errorbar(c["x"], c["y"], yerr=c["sem"], fmt='-x', color=color,
                     ecolor=CMAP((i+1) % CMAP.N), capsize=3,
                     linestyle=linestyle, label=c["label"])
    plt.xlabel(xlabel)
    plt.ylabel("mean_mi")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()

# -------------------------
# Main runner
# -------------------------

def run_for_dataset(data_nr: int):
    # Load and scale
    X, y, name = load_dataset(data_nr)
    scaler = MinMaxScaler(feature_range=(0, math.pi))
    Xs = scaler.fit_transform(X)

    # Fixed split to isolate randomness to transformations
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
    X_ref = X_test  # reference original

    out_dir = os.path.join(OUT_ROOT, name)
    ensure_dir(out_dir)

    # ----------------- Identity (sanity) -----------------
    mi_identity = mean_featurewise_mi(X_ref, X_ref)
    save_curve_csv(out_dir, "mi_identity.csv",
                   rows=[[0.0, mi_identity, 0.0]],
                   columns=["x", "mean_mi", "sem"])

    # ----------------- Random Projection -----------------
    d = X_ref.shape[1]
    rp_settings = rp_dims_for(d)
    rp_rows = []
    for ncomp in rp_settings:
        mi_runs = []
        for run in range(N_RUNS):
            set_seed(10_000 + run)
            X_obf = transform_random_projection(X_ref, n_components=ncomp,
                                                random_state=np.random.randint(0, 1_000_000))
            mi_runs.append(mean_featurewise_mi(X_ref, X_obf))
        mean_mi, sem = agg_stats(mi_runs)
        rp_rows.append([ncomp, mean_mi, sem])
    rp_rows_sorted = sorted(rp_rows, key=lambda r: r[0])
    save_curve_csv(out_dir, "mi_random_projection_n_components.csv",
                   rp_rows_sorted, ["n_components", "mean_mi", "sem"])
    x = [r[0] for r in rp_rows_sorted]; y = [r[1] for r in rp_rows_sorted]; e = [r[2] for r in rp_rows_sorted]
    plot_avg(x, y, "output dimensionality (n_components)",
             f"{name} – Random Projection: MI vs n_components",
             os.path.join(out_dir, "mi_random_projection_n_components_avg.png"))
    plot_avg_sem(x, y, e, "output dimensionality (n_components)",
                 f"{name} – Random Projection: MI vs n_components (±SEM)",
                 os.path.join(out_dir, "mi_random_projection_n_components_avg_sem.png"))

    # ----------------- PCA + Gaussian Noise -----------------
    pca_curves = []
    for keep in PCA_KEEP:
        rows = []
        for std in PCA_STDS:
            mi_runs = []
            for run in range(N_RUNS):
                set_seed(20_000 + run)
                X_obf = transform_pca_noise(X_ref, noise_std=std, keep_ratio=keep,
                                            random_state=np.random.randint(0, 1_000_000))
                mi_runs.append(mean_featurewise_mi(X_ref, X_obf))
            mean_mi, sem = agg_stats(mi_runs)
            rows.append([std, mean_mi, sem])
        rows = sorted(rows, key=lambda r: r[0])
        save_curve_csv(out_dir, f"mi_pca_gaussian_keep_{keep}.csv", rows, ["std", "mean_mi", "sem"])
        pca_curves.append({"label": f"keep={keep}", "x": [r[0] for r in rows],
                           "y": [r[1] for r in rows], "sem": [r[2] for r in rows]})
    multi_curve_plot_avg(pca_curves, "noise std (PCA space)",
                         f"{name} – PCA + Gaussian: MI vs std",
                         os.path.join(out_dir, "mi_pca_gaussian_avg.png"))
    multi_curve_plot_avg_sem(pca_curves, "noise std (PCA space)",
                             f"{name} – PCA + Gaussian: MI vs std (±SEM)",
                             os.path.join(out_dir, "mi_pca_gaussian_avg_sem.png"))

    # ----------------- Feature Shuffling -----------------
    rows = []
    for frac in SHUFFLE_FRACS:
        mi_runs = []
        for run in range(N_RUNS):
            set_seed(30_000 + run)
            X_obf = transform_shuffle(X_ref, frac=frac, random_state=np.random.randint(0, 1_000_000))
            mi_runs.append(mean_featurewise_mi(X_ref, X_obf))
        mean_mi, sem = agg_stats(mi_runs)
        rows.append([frac, mean_mi, sem])
    rows = sorted(rows, key=lambda r: r[0])
    save_curve_csv(out_dir, "mi_shuffle_frac.csv", rows, ["frac", "mean_mi", "sem"])
    x = [r[0] for r in rows]; y = [r[1] for r in rows]; e = [r[2] for r in rows]
    plot_avg(x, y, "shuffled fraction",
             f"{name} – Feature Shuffling: MI vs shuffled fraction",
             os.path.join(out_dir, "mi_shuffle_frac_avg.png"))
    plot_avg_sem(x, y, e, "shuffled fraction",
                 f"{name} – Feature Shuffling: MI vs shuffled fraction (±SEM)",
                 os.path.join(out_dir, "mi_shuffle_frac_avg_sem.png"))

    # ----------------- Gaussian Mechanism (DP) -----------------
    rows_eps = []
    rows_sigma = []
    for eps in DP_EPSILONS:
        mi_runs = []
        for run in range(N_RUNS):
            set_seed(40_000 + run)
            X_obf = gaussian_mechanism_features(X_ref, epsilon=eps,
                                                delta=DP_DELTA,
                                                clip_min=CLIP_MIN, clip_max=CLIP_MAX,
                                                random_state=np.random.randint(0, 1_000_000))
            mi_runs.append(mean_featurewise_mi(X_ref, X_obf))
        mean_mi, sem = agg_stats(mi_runs)
        rows_eps.append([eps, mean_mi, sem])
        rows_sigma.append([dp_sigma_from_epsilon(eps, DP_DELTA), mean_mi, sem])

    rows_eps = sorted(rows_eps, key=lambda r: r[0])
    rows_sigma = sorted(rows_sigma, key=lambda r: r[0])

    save_curve_csv(out_dir, "mi_gaussian_mech_eps.csv", rows_eps, ["eps", "mean_mi", "sem"])
    save_curve_csv(out_dir, "mi_gaussian_mech_sigma.csv", rows_sigma, ["sigma", "mean_mi", "sem"])

    xe = [r[0] for r in rows_eps]; ye = [r[1] for r in rows_eps]; ee = [r[2] for r in rows_eps]
    plot_avg(xe, ye, "privacy parameter ε (smaller = stronger privacy)",
             f"{name} – Gaussian Mechanism: MI vs ε",
             os.path.join(out_dir, "mi_gaussian_mech_eps_avg.png"))
    plot_avg_sem(xe, ye, ee, "privacy parameter ε (smaller = stronger privacy)",
                 f"{name} – Gaussian Mechanism: MI vs ε (±SEM)",
                 os.path.join(out_dir, "mi_gaussian_mech_eps_avg_sem.png"))

    xs = [r[0] for r in rows_sigma]; ys = [r[1] for r in rows_sigma]; es = [r[2] for r in rows_sigma]
    plot_avg(xs, ys, "Gaussian noise σ",
             f"{name} – Gaussian Mechanism: MI vs σ",
             os.path.join(out_dir, "mi_gaussian_mech_sigma_avg.png"))
    plot_avg_sem(xs, ys, es, "Gaussian noise σ",
                 f"{name} – Gaussian Mechanism: MI vs σ (±SEM)",
                 os.path.join(out_dir, "mi_gaussian_mech_sigma_avg_sem.png"))

    # ----------------- Lie-group maps (if available) -----------------
    if HAS_LIE:
        for grp in LIE_GROUPS:
            rows = []
            for nl in LIE_NOISES:
                mi_runs = []
                for run in range(N_RUNS):
                    set_seed(50_000 + run)
                    X_obf = transform_lie_group(X_ref, noise_level=nl, group_family=grp,
                                                output_real=True, multiply=1,
                                                random_state=np.random.randint(0, 1_000_000))
                    mi_runs.append(mean_featurewise_mi(X_ref, X_obf))
                mean_mi, sem = agg_stats(mi_runs)
                rows.append([nl, mean_mi, sem])
            rows = sorted(rows, key=lambda r: r[0])
            save_curve_csv(out_dir, f"mi_lie_{grp}_noise.csv", rows, ["noise", "mean_mi", "sem"])
            x = [r[0] for r in rows]; y = [r[1] for r in rows]; e = [r[2] for r in rows]
            plot_avg(x, y, "map noise", f"{name} – LIE {grp}: MI vs map noise",
                     os.path.join(out_dir, f"mi_lie_{grp}_noise_avg.png"))
            plot_avg_sem(x, y, e, "map noise", f"{name} – LIE {grp}: MI vs map noise (±SEM)",
                         os.path.join(out_dir, f"mi_lie_{grp}_noise_avg_sem.png"))

    # Manifest
    manifest = {
        "dataset": name,
        "n_runs": N_RUNS,
        "bins": BINS,
        "has_lie": HAS_LIE,
        "dp_delta": DP_DELTA,
        "clip_range": [CLIP_MIN, CLIP_MAX]
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

def main():
    # datasets 0..3 as requested
    for data_nr in [0, 1, 2, 3]:
        print(f"=== Processing dataset id {data_nr} ===")
        run_for_dataset(data_nr)
    print(f"Done. Aggregated outputs under: {OUT_ROOT}")

if __name__ == "__main__":
    main()
