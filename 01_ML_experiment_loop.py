"""
Author: Dr. techn. Sebastian Raubitzek MSc. BSc.; SBA Research, Complexity and Resilience Research Group

This script is designed to perform extensive machine learning experiments using LightGBM on several medical datasets.
The main focus of the script is on applying symmetry feature maps to transform the feature space and evaluate the
performance of models trained on these transformed features under various noise levels and multiplicative factors.

The script includes the following key functionalities:

1. **Data Loading and Preprocessing**:
   - Multiple datasets (breast cancer, ilpd, heart, breast-cancer-coimbra, diabetes) are loaded from the sklearn and
    OpenML libraries.
   - Non-numeric features are encoded using a min-max scaling approach to standardize the range of features between
    0 and pi.

2. **Feature Map Application**:
   - The script applies symmetry feature maps using the `SymmetryFeatureMaps` class, which transforms the original
    features into a new feature space.
   - This transformation is performed with various noise levels and multiplication factors to evaluate robustness and
    feature space alterations.

3. **Model Training and Evaluation**:
   - LightGBM is used as the classifier, with hyperparameter optimization performed using Bayesian search (via `BayesSearchCV`).
   - After hyperparameter optimization, the model is evaluated using various metrics including accuracy, precision,
    recall, F1 score, and training time.
   - The results for each dataset, noise level, and multiplication factor configuration are stored in a dictionary,
    which is periodically saved as a pickle file to prevent data loss.

4. **Results Storage**:
   - All experimental results are saved in a structured manner, with intermediate results being saved after each
    configuration to ensure that progress is not lost.

### Key Functions:
- `process_samples(X, feature_map)`: Applies the provided feature map, i.e. Lie-group transformation,
    to each sample in `X`.
- `process_samples_multiply(X, feature_map_multiply)`: Applies the feature map to each sample in `X`, repeating
    the transformation according to the multiplication factor.
- `encode_non_numeric_features(df)`: Encodes non-numeric features in a dataframe using a simple linear mapping from
    categories to numbers.
- `load_dataset(data_nr)`: Loads one of the predefined datasets based on the `data_nr` identifier. Depending on the
    value of `data_nr`, it either loads a dataset from sklearn or fetches one from OpenML.
- `train_lightgbm(X_train, X_test, y_train, y_test)`: Trains a LightGBM model using Bayesian optimization for
    hyperparameter tuning and evaluates its performance.

### Usage:
This script is designed to be run as a standalone experiment. It automatically loads datasets, applies feature
transformations, and trains LightGBM models under various conditions. The results are saved in a directory specified at
the beginning of the script, and intermediate results are saved frequently to prevent data loss.

### Parameters:
- `data_nr`: Integer value used to select the dataset to load. Ranges from 0 to 3.
- `noise_levels`: A numpy array of noise levels to be applied during the feature map transformation.
- `multpliers`: A list of integers representing the multiplication factors for the feature map transformation.

### Dependencies:
- `sklearn`: For dataset loading, preprocessing, and model evaluation.
- `lightgbm`: For training and optimizing the LightGBM classifier.
- `skopt`: For Bayesian hyperparameter search.
- `numpy`, `pandas`: For numerical computations and data handling.
- `seaborn`, `matplotlib`: For data visualization.
- `pickle`: For saving the results.

### Output:
The script produces a series of pickle files containing the results of each experiment, stored in a specified directory.
Additionally, it prints the progress and results of each experiment to the console.

"""

from sklearn.preprocessing import MinMaxScaler
import seaborn as sns;
sns.set_theme()
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml  # Modified to use fetch_openml
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import sys
from copy import deepcopy as dc
import pandas as pd
from skopt import BayesSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import pickle
from class_symmetry_feature_maps_noise import SymmetryFeatureMaps
import numpy as np
import os

def process_samples(X, feature_map):
    processed_samples = []
    for x in X:
        processed_samples.append(feature_map(x))
    return processed_samples

def process_samples_multiply(X, feature_map_multiply):
    multiplied_processed_samples = []
    for x in X:
        processed_samples = dc(feature_map_multiply(x))
        for i in range(len(processed_samples)):
            multiplied_processed_samples.append(processed_samples[i])
    return multiplied_processed_samples

def encode_non_numeric_features(df):
    for column in df.select_dtypes(include=['object', 'category']).columns:
        unique_values = df[column].unique()
        value_to_number = {value: idx / (len(unique_values) - 1) for idx, value in enumerate(unique_values)}
        df[column] = df[column].map(value_to_number)
    return df

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

def train_lightgbm(X_train, X_test, y_train, y_test):
    parameter_grid_lgbm_huge = {
        'num_leaves': [20, 31, 40, 50, 60, 70, 80, 90, 100],  # number of leaves in full tree
        'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2],  # step size for gradient descent
        'n_estimators': [50, 100, 150, 200, 250, 300, 400, 500],  # number of boosting iterations
        'max_depth': [3, 5, 7, 9, 11, 13, 15, -1],  # maximum tree depth, -1 means no limit
        'min_child_samples': [10, 20, 30, 40, 50, 60, 70],  # minimum number of data points in a leaf
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # subsample ratio of the training instances
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        # subsample ratio of columns when constructing each tree
        'reg_alpha': [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],  # L1 regularization term on weights
        'reg_lambda': [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]  # L2 regularization term on weights
    }

    classifier = LGBMClassifier(verbose=-1)
    bocv = BayesSearchCV(classifier, parameter_grid_lgbm_huge, n_iter=100, cv=5, verbose=False) #og n_iter=25
    start_time_hp = time.time()
    bocv.fit(X_train, y_train)
    end_time_hp = time.time()
    hp_search_time = end_time_hp - start_time_hp

    # Use the best estimator
    model = bocv.best_estimator_
    best_params = bocv.best_params_
    best_cv_score = bocv.best_score_

    predictions = model.predict(X_test)
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro')
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')

    # Return results
    return best_params, best_cv_score, accuracy, precision, recall, f1, hp_search_time


#If you want to run the experiment for the SL group, just replace instances of SU wiht SL
results_dir = "experiment_results_medical_SU_5_may_2024_nu_100"
os.makedirs(results_dir, exist_ok=True)
experiment_results = {}

data_nr = 1

# Generate 10 logarithmically spaced noise levels from 0.001 to 0.1
noise_levels = np.logspace(-3, -1, num=5)

# Add zero noise level to the array
noise_levels = np.insert(noise_levels, 0, 0.0)

print(f' Noise Levels: {noise_levels}')

multpliers = [1, 2, 3, 4, 5]

print(f' Multipliers: {multpliers}')

for i in range(4): #looping through 3 mediacl data sets.

    data_nr = i
    # Modified to use the security-based dataset from OpenML
    X, y, name = load_dataset(data_nr=data_nr)
    print('#####################################################################')
    print('#####################################################################')
    print('#####################################################################')
    print('#####################################################################')
    print('#####################################################################')
    print('#####################################################################')
    print(f'Dataset nr:{data_nr}, name: {name}\nOriginal run')

    experiment_results[name] = {}

    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)

    # Split the dataset first
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train and evaluate LightGBM with original features
    best_params, best_cv_score, accuracy, precision, recall, f1, hp_search_time = train_lightgbm(X_train, X_test, y_train, y_test)
    print(f"Accuracy with original features: {accuracy}")

    configuration_key = (0, 0)
    print(f"configuration_key {configuration_key}")

    # Store metrics in a DataFrame
    metrics_df = pd.DataFrame({
        'Best CV Score': [best_cv_score],
        'Test Accuracy': [accuracy],
        'Test Precision': [precision],
        'Test Recall': [recall],
        'Test F1': [f1],
        'Training Time': [hp_search_time],
        'Noise': [0.0]
    })

    experiment_results[name][configuration_key] = metrics_df

    # Intermediate save after each configuration
    intermediate_save_path = os.path.join(results_dir, f"{name}_{0}_{0}_intermediate.pkl")
    with open(intermediate_save_path, 'wb') as f:
        pickle.dump(experiment_results, f)
    print('intermediate saved')

    for multiply in multpliers:
        for noise_count in range(len(noise_levels)):
            noise = dc(noise_levels[noise_count])
            print('#####################################################################')
            print('#####################################################################')
            print(f'Dataset name: {name}')
            print(f'Noise Level:{noise}\nMultiplier:{multiply}')
            configuration_key = (noise_count, multiply)
            print(f"configuration_key {configuration_key}")

            if multiply == 1:

                # Initialize the SymmetryFeatureMaps class
                symmetry_feature_maps = SymmetryFeatureMaps(X_train.shape[1])

                # Define the feature map
                feature_map = lambda x: symmetry_feature_maps.apply_feature_map(x, 'SU', output_real=True, noise_level=noise)

                # Apply feature maps
                processed_X_train = dc(process_samples(X_train, feature_map))
                processed_X_test = dc(process_samples(X_test, feature_map))

                # Train and evaluate LightGBM with transformed features
                best_params, best_cv_score, accuracy, precision, recall, f1, hp_search_time = dc(train_lightgbm(processed_X_train, processed_X_test, y_train, y_test))
                print(f"Accuracy with transformed features: {accuracy}")

            else:
                # Initialize the SymmetryFeatureMaps class
                symmetry_feature_maps = SymmetryFeatureMaps(X_train.shape[1])

                # Define the feature map
                feature_map = lambda x: symmetry_feature_maps.apply_feature_map(x, 'SU', output_real=True, noise_level=noise)

                # Apply feature maps
                processed_X_train = dc(process_samples(X_train, feature_map))
                processed_X_test = dc(process_samples(X_test, feature_map))

                # For multiplied data, apply feature map with multiplication and adjust labels
                feature_map_multiply = lambda x: symmetry_feature_maps.apply_feature_map(x, 'SU', output_real=True,
                                                                                         noise_level=noise, multiply=multiply)
                processed_X_train_multiply = process_samples_multiply(X_train, feature_map_multiply)
                y_train_multiply = np.repeat(y_train, multiply)  # Repeat labels 10 times

                # Train and evaluate LightGBM with transformed and multiplied features
                best_params, best_cv_score, accuracy, precision, recall, f1, hp_search_time = dc(train_lightgbm(processed_X_train_multiply, processed_X_test, y_train_multiply,
                                                               y_test))
                print(f"Accuracy with transformed and multiplied features: {accuracy}")

            # Store metrics in a DataFrame
            metrics_df = pd.DataFrame({
                'Best CV Score': [best_cv_score],
                'Test Accuracy': [accuracy],
                'Test Precision': [precision],
                'Test Recall': [recall],
                'Test F1': [f1],
                'Training Time': [hp_search_time],
                'Noise': [noise],
            })

            experiment_results[name][configuration_key] = metrics_df

            # Intermediate save after each configuration
            intermediate_save_path = os.path.join(results_dir, f"{name}_{noise_count}_{multiply}_intermediate.pkl")
            with open(intermediate_save_path, 'wb') as f:
                pickle.dump(experiment_results, f)
            print('intermediate saved')

# Final save after all datasets and configurations are processed
final_save_path = os.path.join(results_dir, "final_results_SU.pkl")
with open(final_save_path, 'wb') as f:
    pickle.dump(experiment_results, f)
print('finally saved')
