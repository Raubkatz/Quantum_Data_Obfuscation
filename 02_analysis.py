"""
Author: Dr. techn. Sebastian Raubitzek MSc. BSc.; SBA Research, Complexity and Resilience Research Group
This script processes experimental results stored in pickle files, generates heatmaps for the accuracy of various configurations,
and compares them against ground truth values for multiple datasets. The script is specifically tailored to handle results
from noise and multiplier experiments on several medical datasets.

### Key Features:
1. **Loading and Processing Results**:
   - The script loads results from pickle files stored in a specified directory. These results include accuracy metrics from
     experiments performed with different noise levels and multiplication factors.
   - The script identifies the type of analysis (SL or SU) based on the filename.

2. **Heatmap Generation**:
   - For each dataset, the script generates two heatmaps:
     1. **Accuracy Heatmap**: Displays the accuracy of the model for each noise and multiplier configuration.
     2. **Subtracted Accuracy Heatmap**: Shows the difference between the accuracy obtained and the ground truth value for the
        corresponding dataset, highlighting how noise and multiplier configurations impact the model's performance relative
        to the benchmark.

3. **LaTeX Table Generation**:
   - The script also converts the accuracy matrices into LaTeX tables for inclusion in reports or publications.
   - Two LaTeX tables are generated for each dataset:
     1. **Accuracy Table**: Represents the accuracy values for each combination of noise level and multiplier.
     2. **Subtracted Accuracy Table**: Represents the difference between the accuracy and the ground truth for each combination.

4. **Customizable Output**:
   - Heatmaps are saved in both PNG and EPS formats for each dataset and analysis type (SL or SU).
   - The LaTeX tables are saved as `.tex` files, making them easy to integrate into LaTeX documents.
   - The output directory is automatically created if it does not exist, ensuring that results are stored systematically.

### Detailed Workflow:
1. **Directory and File Handling**:
   - The script checks if the results directory exists. If it doesn't, an error message is printed and the script terminates.
   - If the directory exists, it lists all files and processes each one, assuming they contain experiment results stored in pickle format.

2. **Data Extraction and Matrix Initialization**:
   - For each dataset within a pickle file, the script initializes matrices (`accuracies_matrix` and `accuracies_matrix_diff`) to store the accuracy and the difference from ground truth.
   - It then iterates through each configuration (noise level and multiplier combination), extracts the accuracy, and updates the matrices accordingly.

3. **Heatmap Plotting**:
   - The script uses Seaborn and Matplotlib to generate heatmaps from the accuracy matrices.
   - A standard heatmap is generated to display accuracy values, and a differential heatmap is created to visualize deviations from the ground truth.
   - Custom colormaps are used for the differential heatmaps to clearly distinguish between positive and negative differences.

4. **Saving Outputs**:
   - The generated heatmaps are saved to the specified directory in both PNG and EPS formats.
   - LaTeX tables are generated from the accuracy matrices and saved as `.tex` files, ready for direct inclusion in LaTeX documents.

### Key Variables:
- `results_dir`: Directory path where the pickle files containing experimental results are stored.
- `save_dir`: Directory path where the generated heatmaps and LaTeX tables will be saved.
- `ground_truth_values`: Dictionary mapping dataset names to their respective ground truth accuracy values.
- `accuracies_matrix`: A matrix used to store accuracy values for different configurations.
- `accuracies_matrix_diff`: A matrix used to store the difference between accuracy values and the ground truth.

### Dependencies:
- `os`: For directory and file path operations.
- `pickle`: For loading and saving serialized Python objects.
- `numpy`: For numerical operations, particularly for initializing and manipulating matrices.
- `seaborn` and `matplotlib`: For generating heatmaps and visualizations.
- `pandas`: For creating DataFrames, which are used to organize and export data in tabular formats.
- `matplotlib.colors`: For custom color mapping in heatmaps.

### Output:
- **Heatmaps**: Visual representations of accuracy metrics for different noise levels and multipliers, saved as both PNG and EPS files.
- **LaTeX Tables**: Tabular representation of accuracy metrics and their differences from ground truth, saved as `.tex` files for easy integration into LaTeX documents.
- **Console Output**: The script prints progress and dataset names during execution, helping track the analysis process.

This script is an essential tool for visualizing and comparing the effects of noise and multipliers on model performance across different datasets.
"""

import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors

# Directory where pickle files are stored
#results_dir = "finlamente_nu_noise"
results_dir = "finalmente_nu_noise_100"
save_dir = "heatmap_results_nu_noise_100"  # Directory to save the heatmaps and CSVs
os.makedirs(save_dir, exist_ok=True)  # Create save_dir if it doesn't exist

# Dictionary to map dataset names to ground truth values
ground_truth_values = {
    "breast-cancer-coimbra": 0.8333333333333334,
    "breast cancer": 0.9736842105263158,
    "diabetes": 0.7467532467532467,
    "ilpd": 0.7435897435897436
}

# Check if directory exists
if os.path.exists(results_dir):
    # List all files in the directory
    files = os.listdir(results_dir)

    for file_name in files:
        # Construct full file path
        file_path = os.path.join(results_dir, file_name)

        print(file_name)
        analysis_type = "SL" if "SL" in file_name else "SU" if "SU" in file_name else "Unknown"
        print(analysis_type)

        # Load the content of the pickle file
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        # Iterate through the loaded data
        for dataset_name, configs in data.items():
            print(f"Dataset Name: {dataset_name}")

            # Initialize a 6x6 matrix to store accuracies
            accuracies_matrix = np.zeros((6, 5))
            accuracies_matrix_diff = np.zeros((6, 5))
            # Set the ground truth based on the dataset name
            #print(dataset_name)
            ground_truth = ground_truth_values.get(dataset_name)
            #print(ground_truth)

            noise_levels = []

            for config_key, metrics_df in configs.items():
                # Extract noise_count and multiply from the config_key
                noise_count, multiply = config_key
                print(f"noise_count {noise_count}, multiply {multiply}")

                # Assuming 'Test Accuracy' is a column in your DataFrame
                accuracy = metrics_df['Test Accuracy'].iloc[0]
                noise_level = metrics_df['Noise'].iloc[0]

                # Ensure that each noise level is only added once
                if noise_level not in noise_levels:
                    noise_levels.append(noise_level)


                # Update the matrix with the accuracy
                # Adjust indices if necessary, depending on how noise_count and multiply are defined
                if noise_count == 0 and multiply == 0:
                    print('Skip zero zero values')
                else:
                    accuracies_matrix[noise_count, multiply-1] = accuracy
                    accuracies_matrix_diff[noise_count, multiply-1] = accuracy - ground_truth

            # Sort noise_levels to ensure they're in ascending order
            noise_levels.sort()

            png_file_path = os.path.join(save_dir, f"{dataset_name}_{analysis_type}_heatmap.png")
            eps_file_path = os.path.join(save_dir, f"{dataset_name}_{analysis_type}_heatmap.eps")

            png_file_path_diff = os.path.join(save_dir, f"{dataset_name}_{analysis_type}_heatmap_diff.png")
            eps_file_path_diff = os.path.join(save_dir, f"{dataset_name}_{analysis_type}_heatmap_diff.eps")

            # Format noise levels with three decimal places
            formatted_noise_levels = [f"{level:.3f}" for level in noise_levels]

            ########################################################################################################
            # Create a DataFrame from the accuracies matrix
            formatted_accuracies = [[f"{accuracy:.3f}" for accuracy in row] for row in accuracies_matrix]

            df_accuracies = pd.DataFrame(formatted_accuracies, columns=[f"Multiplier {i}" for i in range(1, 6)],
                                         index=formatted_noise_levels)

            # Add a column for noise levels at the beginning
            df_accuracies.insert(0, 'Noise Level', formatted_noise_levels)

            # Add a row for multipliers at the beginning
            multipliers_row = ['--'] + [f"Multiplier {i}" for i in range(1, 6)]  # '--' for the (0,0) position
            df_with_multipliers = pd.DataFrame([multipliers_row], columns=df_accuracies.columns)
            df_accuracies = pd.concat([df_with_multipliers, df_accuracies], ignore_index=True)

            # Convert the entire DataFrame to object type to accommodate the string '--'
            df_accuracies = df_accuracies.astype('object')

            # Convert the DataFrame to LaTeX code
            latex_code = df_accuracies.to_latex(index=False, escape=False, float_format="%.3f")

            # Save the LaTeX code to a text file
            latex_file_path = os.path.join(save_dir, f"{dataset_name}_{analysis_type}_heatmap_table.tex")
            with open(latex_file_path, 'w') as latex_file:
                latex_file.write(latex_code)

            ########################################################################################################
            # Create a DataFrame from the accuracies matrix
            formatted_accuracies_diff = [[f"{accuracy:.3f}" for accuracy in row] for row in accuracies_matrix_diff]


            df_accuracies_diff = pd.DataFrame(formatted_accuracies_diff, columns=[f"Multiplier {i}" for i in range(1, 6)],
                                              index=formatted_noise_levels)

            # Add a column for noise levels at the beginning
            df_accuracies_diff.insert(0, 'Noise Level', formatted_noise_levels)

            # Add a row for multipliers at the beginning
            multipliers_row_diff = ['--'] + [f"Multiplier {i}" for i in range(1, 6)]  # '--' for the (0,0) position
            df_with_multipliers_diff = pd.DataFrame([multipliers_row_diff], columns=df_accuracies_diff.columns)
            df_accuracies_diff = pd.concat([df_with_multipliers_diff, df_accuracies_diff], ignore_index=True)

            # Convert the entire DataFrame to object type to accommodate the string '--'
            df_accuracies_diff = df_accuracies_diff.astype('object')

            # Convert the DataFrame to LaTeX code
            latex_code_diff = df_accuracies_diff.to_latex(index=False, escape=False, float_format="%.3f")

            # Save the LaTeX code to a text file
            latex_file_path_diff = os.path.join(save_dir, f"{dataset_name}_{analysis_type}_heatmap_table_diff_rb.tex")
            with open(latex_file_path_diff, 'w') as latex_file_diff:
                latex_file_diff.write(latex_code_diff)

            # Plotting the heatmap
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(accuracies_matrix, annot=True, fmt=".3f", cmap='viridis', xticklabels=range(1, 6), yticklabels=formatted_noise_levels, annot_kws={"size": 16})
            plt.title(f"Accuracy Heatmap for {dataset_name} and {analysis_type}\nBenchmark accuracy = {ground_truth:.3f}")
            plt.xlabel("Multiplier")
            plt.ylabel("Noise Level")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # Keep the noise level labels horizontal
            plt.savefig(png_file_path)
            plt.savefig(eps_file_path)
            plt.show()

            # Plotting the heatmap
            plt.figure(figsize=(10, 8))
            # Assuming 'accuracies_matrix_diff' is your data matrix
            # Create a custom colormap: solid red for negative, solid blue for positive
            #cmap = mcolors.ListedColormap(['orange', 'purple'])
            cmap = mcolors.ListedColormap(['#3A2081', '#64A1CF'])

            bounds = [-1.0, 0, 1.0]  # Define the boundaries for red and blue
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            # Define the center of your colormap, in this case, 0

            ax = sns.heatmap(accuracies_matrix_diff, annot=True, fmt=".3f", cmap=cmap, norm=norm,
                             xticklabels=range(1, 6), yticklabels=formatted_noise_levels, annot_kws={"size": 16})

            #ax = sns.heatmap(accuracies_matrix_diff, annot=True, fmt=".3f", cmap='viridis', xticklabels=range(1, 6), yticklabels=formatted_noise_levels)
            plt.title(f"Subtracted Accuracies for {dataset_name} and {analysis_type}", fontsize=16)
            plt.xlabel("Multiplier", fontsize=16)
            plt.ylabel("Noise Level", fontsize=16)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=16)  # Keep the noise level labels horizontal
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
            plt.tight_layout()
            plt.savefig(png_file_path_diff)
            plt.savefig(eps_file_path_diff)
            plt.show()

else:
    print(f"The directory {results_dir} does not exist. Please check the path and try again.")
