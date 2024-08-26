"""
This module provides functions to load and preprocess datasets from sklearn's built-in datasets and OpenML. The module supports encoding non-numeric features and loading detailed dataset information, including feature names, number of samples, and a brief description.

### Key Functions:

1. **encode_non_numeric_features(X)**:
   - Encodes non-numeric features in a DataFrame using one-hot encoding (via `pd.get_dummies`). This function is useful for ensuring that all features are numeric before feeding them into machine learning models.

   **Parameters**:
   - `X` (pd.DataFrame): The input DataFrame with potentially non-numeric features.

   **Returns**:
   - pd.DataFrame: A DataFrame where all non-numeric features have been encoded as numeric via one-hot encoding.

2. **load_dataset(data_nr)**:
   - Loads a dataset based on the given identifier (`data_nr`). It supports both a built-in breast cancer dataset from sklearn and several OpenML datasets. Depending on the `data_nr` provided, the function either loads the breast cancer dataset or one of the specified OpenML datasets.

   **Parameters**:
   - `data_nr` (int): An integer identifier to choose the dataset to load. `0` loads the breast cancer dataset from sklearn, and `1-4` load specific datasets from OpenML.

   **Returns**:
   - tuple: A tuple containing the following elements:
     - `Data` (pd.DataFrame or np.ndarray): The features of the dataset.
     - `Target` (pd.Series or np.ndarray): The labels of the dataset.
     - `Name` (str): The name of the dataset.

   **Details**:
   - For `data_nr = 0`, the breast cancer dataset is loaded, and detailed information including feature names, number of features, number of classes, and a brief description is printed.
   - For `data_nr` in the range of 1 to 4, the corresponding OpenML dataset (ilpd, heart, breast-cancer-coimbra, diabetes) is loaded. The function handles both DataFrame and sparse matrix formats and prints detailed dataset information, including a URL link and a brief description.
   - If `data_nr` is out of range, the function will print an error message and exit.
"""

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import sys

def encode_non_numeric_features(X):
    # This function will encode non-numeric features if your dataset contains any
    return pd.get_dummies(X)

def load_dataset(data_nr):
    if data_nr == 0:
        data_sk = load_breast_cancer()
        dataset_info = {
            'Data': data_sk.data,
            'Target': data_sk.target,
            'Name': "Breast Cancer",
            'Feature Names': data_sk.feature_names,
            'Number of Features': data_sk.data.shape[1],
            'Number of Classes': len(set(data_sk.target)),
            'Number of Samples': data_sk.data.shape[0],
            'Description': data_sk.DESCR
        }
    elif data_nr in range(1, 5):  # OpenML datasets
        data_names = ["ilpd", "heart", "breast-cancer-coimbra", "diabetes"]
        data_name = data_names[data_nr]
        try:
            data_sk = fetch_openml(name=data_name, version=1, as_frame=True)
            X = data_sk.frame.drop(columns=data_sk.target_names)
            y = data_sk.frame[data_sk.target_names[0]]
        except ValueError:
            # Fallback for sparse datasets: load as dense array
            data_sk = fetch_openml(name=data_name, version=1, as_frame=False)
            X = data_sk.data.toarray()  # Convert sparse matrix to dense array
            y = data_sk.target
            feature_names = ['Feature_' + str(i) for i in range(X.shape[1])]
        else:
            feature_names = list(X.columns)
            X = encode_non_numeric_features(X)  # encode non-numeric features if DataFrame

        if isinstance(y, pd.Series) and y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)  # encode non-numeric labels

        dataset_info = {
            'Data': X,
            'Target': y,
            'Name': data_sk.details.get('name', 'Unknown dataset'),
            'ID': data_sk.details.get('id', 'N/A'),
            'Feature Names': feature_names,
            'Number of Features': X.shape[1],
            'Number of Classes': len(np.unique(y)),
            'Number of Samples': X.shape[0],
            'URL': data_sk.url,
            'Description': data_sk.details.get('description', 'Description not available.')
        }
    else:
        print('No valid data choice, exiting...')
        sys.exit()

    # Print all dataset information
    for key, value in dataset_info.items():
        if key not in ['Data', 'Target', 'Description']:
            print(f"{key}: {value}")
        elif key == 'Description':
            print(f"{key}:\n{value[:300]}...")  # Print the first 300 characters of the description

    # Return data, target, and name for compatibility with the rest of your code
    return dataset_info['Data'], dataset_info['Target'], dataset_info['Name']

# Example usage
for i in range(4):
    X, y, name = load_dataset(i)  # Load an OpenML dataset
