# Data Obfuscation and Symmetry-Based Feature Maps: A Machine Learning Approach

This repository contains Python scripts and associated tools for conducting machine learning experiments using symmetry-based feature maps and data obfuscation techniques. The project focuses on applying various group-theoretic transformations to datasets and evaluating the impact of these transformations on model performance under different noise levels.

## Authors: Dr. techn. Sebastian Raubitzek MSc. BSc.
#Preprint: https://www.preprints.org/manuscript/202407.1701/v1

## Overview

This repository provides a comprehensive suite of scripts designed to generate, analyze, and visualize the effects of symmetry-based transformations on various datasets. The scripts explore the impact of noise and multiplicative factors on machine learning models trained with these transformations.

The primary components of this project include:

1. **Dataset Preparation and Verification**: Scripts to load and preprocess datasets, ensuring they are suitable for the experiments.
2. **Symmetry Feature Map Application**: Scripts to apply group-theoretic transformations to the datasets, introducing noise and evaluating the robustness of these transformations.
3. **Machine Learning Experimentation**: Scripts to train and evaluate models using the transformed datasets, including hyperparameter tuning and performance metrics calculation.
4. **Analysis and Visualization**: Scripts to analyze the experimental results and generate visualizations, such as heatmaps, to illustrate the impact of transformations.

## Repository Structure

├── README.md

├── 00_check_datasets.py

├── 01_ML_experiment_loop.py

├── 02_analysis.py

├── add_check_symmetry_properties.py

├── class_symmetry_feature_maps_noise.py

├── func_sun.py

├── func_verification.py


All folders contain the experimental results from the preprint.




## Script Descriptions

1. **00_check_datasets.py**:
   - **Purpose**: Verifies and preprocesses datasets before they are used in the experiments. Ensures that datasets are loaded correctly and that any non-numeric features are appropriately encoded.
   - **Output**: Processed datasets ready for transformation and machine learning experiments.

2. **01_ML_experiment_loop.py**:
   - **Purpose**: The main loop for running machine learning experiments. It applies symmetry-based transformations to datasets, introduces noise, and trains models using LightGBM. The script includes hyperparameter optimization and evaluates performance metrics.
   - **Output**: A series of pickle files containing the results of each experiment, stored in the `finlamente_nu_noise/` directory.

3. **02_analysis.py**:
   - **Purpose**: Analyzes the results from the machine learning experiments. This script generates various statistical summaries and visualizations, including heatmaps, to assess the impact of different transformations.
   - **Output**: Heatmaps and LaTeX tables summarizing the experimental results, stored in the `heatmap_results_nu_noise/` directory.

4. **add_check_symmetry_properties.py**:
   - **Purpose**: Adds checks for symmetry properties in the feature maps and ensures that the transformations meet the required criteria.
   - **Output**: Validation results ensuring the integrity of the applied symmetry transformations.

5. **class_symmetry_feature_maps_noise.py**:
   - **Purpose**: Defines the `SymmetryFeatureMaps` class, which handles the application of group-based transformations to datasets. This class includes methods for adding noise and managing different group types.
   - **Output**: Transformed datasets with applied symmetry-based feature maps.

6. **func_sun.py**:
   - **Purpose**: Provides utility functions for generating and manipulating group elements, particularly for the SU(n) group and related symmetries.
   - **Output**: Group elements and associated transformations used in the feature maps.

7. **func_verification.py**:
   - **Purpose**: Contains functions for verifying the correctness and properties of the transformations applied to the datasets.
   - **Output**: Verification results used to validate the experimental process.

## Results

The results of the experiments, including all generated datasets, analysis outputs, and visualizations, are saved in the `finlamente_nu_noise/` and `heatmap_results_nu_noise/` directories.

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3.6 or higher
- numpy
- pandas
- scikit-learn
- lightgbm
- matplotlib
- seaborn
- scipy

## Usage

1. **Dataset Verification**: Run `00_check_datasets.py` to load and preprocess datasets, ensuring they are ready for the experiments.

2. **Run Experiments**: Use `01_ML_experiment_loop.py` to apply transformations, introduce noise, and train models on the datasets.

3. **Analyze Results**: Execute `02_analysis.py` to analyze the results of the experiments and generate visualizations.

4. **Check Symmetry Properties**: Use `add_check_symmetry_properties.py` to validate the properties of the applied transformations.

## License

This project is licensed under the terms of the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](http://creativecommons.org/licenses/by/4.0/).





