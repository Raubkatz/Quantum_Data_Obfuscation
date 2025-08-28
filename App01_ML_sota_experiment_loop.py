"""
Author: Dr. techn. Sebastian Raubitzek MSc. BSc.; SBA Research, Complexity and Resilience Research Group

Machine Learning Experiments with Non-Group Obfuscation Techniques
==================================================================

This script performs machine learning experiments with LightGBM on several
medical datasets, evaluating the impact of feature obfuscation techniques that
are not based on Lie-group feature maps. The purpose is to compare state-of-the-art
feature-space obfuscations for privacy and robustness while keeping the
evaluation pipeline identical to Lie-group experiments.

Obfuscation Techniques:
-----------------------
1. Random Projection
   - Projects features into a lower-dimensional subspace using a Gaussian random matrix.
   - Widely used in privacy-preserving ML and compressed sensing to reduce
     dimensionality while approximately preserving distances. This aligns with
     obfuscation because it prevents exact feature recovery while retaining task utility.
   - Ref: Bingham & Mannila (2001), "Random projection in dimensionality reduction"
          Li et al. (2006), "Very sparse random projections"

2. PCA + Gaussian Noise
   - Applies PCA for dimensionality reduction, adds Gaussian noise in latent space,
     and reconstructs features.
   - Used in adversarial robustness and privacy contexts: PCA reduces redundancy
     while noise obfuscates fine-grained structure, limiting inversion/reconstruction attacks.
   - Ref: Abdi & Williams (2010), "Principal Component Analysis"
          Dwork & Roth (2014), "The Algorithmic Foundations of Differential Privacy"
          Shokri et al. (2017), "Membership inference attacks against ML models"

3. Feature Shuffling
   - Randomly permutes feature columns (per sample).
   - A simple anonymization baseline used in privacy benchmarks: destroys
     correlations between features while maintaining marginal distributions.
     This creates a strong obfuscation baseline for tabular data.
   - Ref: Li et al. (2019), "Privacy-Preserving Data Publishing: A Survey on Recent Developments"
          Domingo-Ferrer & Torra (2001), "A quantitative comparison of disclosure control methods"

4. Gaussian Mechanism (DP-style)
   - Adds calibrated Gaussian noise to features as a proxy for ε-Differential Privacy.
   - This is the canonical mechanism in DP, providing formal privacy guarantees
     against re-identification and inference. Here used directly on features to
     simulate DP-style obfuscation at different privacy budgets (ε).
   - Ref: Dwork & Roth (2014), "The Algorithmic Foundations of Differential Privacy"
          Abadi et al. (2016), "Deep learning with differential privacy"

Experiment Workflow:
--------------------
- Load and preprocess datasets (breast cancer, ilpd, heart, breast-cancer-coimbra, diabetes).
- Apply each obfuscation at multiple noise levels (5 levels).
- Train and evaluate LightGBM with Bayesian hyperparameter search.
- Save intermediate and final results into structured directories.

Output:
-------
Results are saved under:

    experiment_results_medical_sota_comparisons_nu_100/
        datasetname_noise_config_intermediate.pkl
        final_results.pkl

Each result entry includes:
- Best CV score
- Test Accuracy, Precision, Recall, F1
- Training time
- Obfuscation method and parameters

"""


import os, sys, time, pickle, math
import numpy as np
import pandas as pd
from copy import deepcopy as dc

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from lightgbm import LGBMClassifier
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ---------------------------
# Utilities
# ---------------------------

seed=137

def encode_non_numeric_features(df):
    for column in df.select_dtypes(include=['object', 'category']).columns:
        unique_values = df[column].unique()
        value_to_number = {value: idx / (len(unique_values) - 1) for idx, value in enumerate(unique_values)}
        df[column] = df[column].map(value_to_number)
    return df

def load_dataset_nu(data_nr):
    if data_nr == 0:
        data_sk = load_breast_cancer()
        return data_sk.data, data_sk.target, "breast cancer"
    elif data_nr in [1,2,3,4]:
        data_names = ["ilpd","heart","breast-cancer-coimbra","diabetes"]
        data_name = data_names[data_nr-1]
        data_sk = fetch_openml(name=data_name, version=1, as_frame=True)
        X, y = data_sk.data, data_sk.target
        if isinstance(X, pd.DataFrame):
            X = encode_non_numeric_features(X)
            X = X.values
        if isinstance(y, pd.Series) and y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        return X, y, data_name
    else:
        print("Invalid dataset id")
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



# ---------------------------
# Obfuscation transforms
# ---------------------------

def transform_random_projection(X, n_components=None, random_state=42):
    if n_components is None:
        n_components = max(2, X.shape[1]//2)
    rp = GaussianRandomProjection(n_components=n_components, random_state=random_state)
    return rp.fit_transform(X)

def transform_pca_noise(X, noise_std=0.05, keep_ratio=1.0, random_state=42):
    d = X.shape[1]
    k = max(1, int(math.ceil(d*keep_ratio)))
    pca = PCA(n_components=k, random_state=random_state)
    Z = pca.fit_transform(X)
    Z_noisy = Z + np.random.normal(0.0, noise_std, size=Z.shape)
    return pca.inverse_transform(Z_noisy)

def transform_shuffle(X, frac=0.5, random_state=42):
    rng = np.random.RandomState(random_state)
    Xs = X.copy()
    d = X.shape[1]
    k = max(1, int(round(frac*d)))
    cols = rng.choice(np.arange(d), size=k, replace=False)
    for j in cols:
        rng.shuffle(Xs[:,j])
    return Xs

def gaussian_mechanism_features(X, epsilon, delta=1e-5,
                                clip_min=0.0, clip_max=math.pi, random_state=42):
    rng = np.random.RandomState(random_state)
    Xc = X.copy()
    np.clip(Xc, clip_min, clip_max, out=Xc)
    S = (clip_max-clip_min)
    sigma = S * math.sqrt(2*math.log(1.25/delta)) / max(epsilon, 1e-8)
    return Xc + rng.normal(0.0, sigma, size=Xc.shape)


# ---------------------------
# LightGBM training
# ---------------------------

def train_lightgbm(X_train, X_test, y_train, y_test):
    param_grid = {
        'num_leaves': [20,40,60,80,100],
        'learning_rate': [0.01,0.05,0.1,0.2],
        'n_estimators': [100,200,300,400],
        'max_depth': [3,5,7,9,-1],
        'subsample': [0.7,0.8,0.9,1.0],
        'colsample_bytree': [0.7,0.8,0.9,1.0],
        'reg_alpha': [0.0,0.1,0.3],
        'reg_lambda': [0.0,0.1,0.3]
    }
    clf = LGBMClassifier(verbose=-1)
    search = BayesSearchCV(clf, param_grid, n_iter=100, cv=5, verbose=True)
    t0 = time.time()
    search.fit(X_train, y_train)
    t1 = time.time()
    model = search.best_estimator_
    preds = model.predict(X_test)
    return {
        "Best CV Score": search.best_score_,
        "Test Accuracy": accuracy_score(y_test, preds),
        "Test Precision": precision_score(y_test, preds, average='macro'),
        "Test Recall": recall_score(y_test, preds, average='macro'),
        "Test F1": f1_score(y_test, preds, average='macro'),
        "Training Time": t1-t0,
        "Params": search.best_params_
    }


# ---------------------------
# Experiment driver
# ---------------------------

results_dir = "experiment_results_medical_sota_comparisons_nu_100"
os.makedirs(results_dir, exist_ok=True)
experiment_results = {}

noise_levels = [0.0, 0.01, 0.03, 0.05, 0.1]  # 5 noise levels

for data_nr in range(4):  # run over 4 datasets
    X,y,name = load_dataset(data_nr)
    print("="*80)
    print(f" Dataset {data_nr}: {name}")
    scaler = MinMaxScaler(feature_range=(0,np.pi))
    Xs = scaler.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(Xs,y,test_size=0.2,random_state=seed)

    experiment_results[name] = {}

    # Train on original
    print(f"  Running Baseline Experiment...")
    base_metrics = train_lightgbm(X_train,X_test,y_train,y_test)

    print(f"    -> Done: Acc={base_metrics['Test Accuracy']:.3f}, F1={base_metrics['Test F1']:.3f}")

    experiment_results[name][("original","none")] = base_metrics
    with open(os.path.join(results_dir,f"{name}_original_intermediate.pkl"),'wb') as f:
        pickle.dump(experiment_results,f)

    # Random projection configs
    for d_out in [None, min(10,Xs.shape[1])]:
        print(f"  Running Random Projection (std={d_out})...")
        Xtr = transform_random_projection(X_train,n_components=d_out)
        Xte = transform_random_projection(X_test,n_components=d_out)
        metrics = train_lightgbm(Xtr,Xte,y_train,y_test)
        print(f"    -> Done: Acc={metrics['Test Accuracy']:.3f}, F1={metrics['Test F1']:.3f}")
        experiment_results[name][("random_projection",d_out)] = metrics

    # PCA + Gaussian
    for std in noise_levels:
        print(f"  Running PCA+Gaussian (std={std})...")
        Xtr = transform_pca_noise(X_train, noise_std=std, keep_ratio=1.0)
        Xte = transform_pca_noise(X_test, noise_std=std, keep_ratio=1.0)
        metrics = train_lightgbm(Xtr, Xte, y_train, y_test)
        print(f"    -> Done: Acc={metrics['Test Accuracy']:.3f}, F1={metrics['Test F1']:.3f}")
        experiment_results[name][("pca_gaussian", std)] = metrics

    # Shuffle
    for frac in [0.25,0.5,1.0]:
        print(f"  Running Shuffle (std={frac})...")
        Xtr = transform_shuffle(X_train,frac=frac)
        Xte = transform_shuffle(X_test,frac=frac)
        metrics = train_lightgbm(Xtr,Xte,y_train,y_test)
        print(f"    -> Done: Acc={metrics['Test Accuracy']:.3f}, F1={metrics['Test F1']:.3f}")
        experiment_results[name][("shuffle",frac)] = metrics

    # Gaussian mechanism
    for eps in [0.5,1.0,2.0,4.0]:
        print(f"  Running Gaussian Mechanism (std={frac})...")
        Xtr = gaussian_mechanism_features(X_train,epsilon=eps)
        Xte = gaussian_mechanism_features(X_test,epsilon=eps)
        metrics = train_lightgbm(Xtr,Xte,y_train,y_test)
        print(f"    -> Done: Acc={metrics['Test Accuracy']:.3f}, F1={metrics['Test F1']:.3f}")
        experiment_results[name][("gaussian_mech",eps)] = metrics

    # Intermediate save
    with open(os.path.join(results_dir,f"{name}_intermediate.pkl"),'wb') as f:
        pickle.dump(experiment_results,f)

# Final save
with open(os.path.join(results_dir,"final_results.pkl"),'wb') as f:
    pickle.dump(experiment_results,f)

print("All experiments finished.")
