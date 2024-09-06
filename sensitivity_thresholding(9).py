import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load the datasets
file_paths = [
    '/content/filtered_002_TU2_Immune_1_thresholded_consensus_clustering_results 2.41.04 pm.csv',
    '/content/filtered_002_TU2_Immune_2_thresholded_consensus_clustering_results 2.41.04 pm.csv',
    '/content/filtered_CC_OC_585_TU1_Immune_1_thresholded_consensus_clustering_results 2.41.04 pm.csv',
    '/content/filtered_CC_OC_585_TU1_Immune_2_thresholded_consensus_clustering_results 2.41.04 pm.csv'
]

data_dfs = {f'file_{i+1}': pd.read_csv(file_path) for i, file_path in enumerate(file_paths)}

# Define the markers and the ground truth column
markers = ["CD66b_asNumeric", "CD56_asNumeric", "CD4_asNumeric", "CTLA4_asNumeric", "CD8_asNumeric", "CD20_asNumeric"]
ground_truth_column = 'cluster'

# Original thresholds provided for each file
thresholds = [
    {"CD66b_asNumeric": 5.64589286, "CD56_asNumeric": 8.38696429, "CD4_asNumeric": 4.38928571, "CTLA4_asNumeric": 1.94285714, "CD8_asNumeric": 3.52285714, "CD20_asNumeric": 3.1975},
    {"CD66b_asNumeric": 11.0958929, "CD56_asNumeric": 7.04696429, "CD4_asNumeric": 4.38928571, "CTLA4_asNumeric": 2.85714286, "CD8_asNumeric": 4.57142857, "CD20_asNumeric": 3.1975},
    {"CD66b_asNumeric": 6.04589286, "CD56_asNumeric": 18.7608929, "CD4_asNumeric": 2.65357143, "CTLA4_asNumeric": 2.86696429, "CD8_asNumeric": 6.20607143, "CD20_asNumeric": 1.3075},
    {"CD66b_asNumeric": 6.04589286, "CD56_asNumeric": 14.7992857, "CD4_asNumeric": 2.65357143, "CTLA4_asNumeric": 2.29357143, "CD8_asNumeric": 3.90607143, "CD20_asNumeric": 0.8875},
]

# Function to generate sensitivity thresholds
def generate_sensitivity_thresholds(original_thresholds):
    new_thresholds = []
    for percentage in [-30, -20, -10, -5, 5, 10]:
        modified_thresholds = {marker: value * (1 + percentage / 100) for marker, value in original_thresholds.items()}
        new_thresholds.append(modified_thresholds)
    return new_thresholds

# Function to compute metrics
def compute_metrics(actual_labels, predicted_labels):
    metrics = {}
    metrics['ARI'] = adjusted_rand_score(actual_labels, predicted_labels)
    metrics['NMI'] = normalized_mutual_info_score(actual_labels, predicted_labels)
    metrics['Accuracy'] = accuracy_score(actual_labels, predicted_labels)
    metrics['Precision'] = precision_score(actual_labels, predicted_labels, zero_division=0, average='weighted')
    metrics['Recall'] = recall_score(actual_labels, predicted_labels, zero_division=0, average='weighted')
    metrics['F1'] = f1_score(actual_labels, predicted_labels, zero_division=0, average='weighted')

    # AUC-ROC only if it's binary classification
    if len(set(actual_labels)) == 2:
        metrics['AUC_ROC'] = roc_auc_score(actual_labels, predicted_labels)
    else:
        metrics['AUC_ROC'] = float('nan')  # Not applicable for multiclass without probabilities

    return metrics

# Function to compare thresholds and compute metrics
def compare_thresholds_and_compute_metrics(data_df, method_name, ground_truth):
    metrics_results = {}
    for marker in markers:
        actual_labels = data_df[ground_truth]
        predicted_labels = data_df[f'{marker}_{method_name}_above_threshold']
        metrics_results[marker] = compute_metrics(actual_labels, predicted_labels)
    return metrics_results

# Function to apply Otsu's thresholding
def apply_otsu_thresholding(data_df, thresholds):
    for marker, threshold in thresholds.items():
        data_df[f'{marker}_otsu_above_threshold'] = data_df[marker] > threshold
    return data_df

# Function to apply GMM thresholding
def apply_gmm_thresholding(data_df, thresholds):
    for marker, threshold in thresholds.items():
        data_df[f'{marker}_gmm_above_threshold'] = data_df[marker] > threshold
    return data_df

# Function to plot marker histograms with the threshold lines
def plot_marker_histograms(data_df, thresholds, method_name, title):
    plt.figure(figsize=(12, 10))
    for i, marker in enumerate(markers, 1):
        plt.subplot(3, 2, i)
        plt.hist(data_df[marker], bins=30, color='blue', alpha=0.7)
        plt.axvline(x=thresholds[marker], color='red', linestyle='--', label=f'{method_name} Threshold')
        plt.title(f"{marker} - {method_name} Threshold")
        plt.xlabel(marker)
        plt.ylabel("Frequency")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Iterate over each file and perform sensitivity analysis
all_results = {}
for idx, (name, data_df) in enumerate(data_dfs.items()):
    print(f"Processing {name}...")

    # Original thresholds for this file
    original_thresholds = thresholds[idx]

    # Generate sensitivity thresholds (±5%, ±10%, ±20%, -30%)
    sensitivity_thresholds = generate_sensitivity_thresholds(original_thresholds)

    # Iterate over sensitivity thresholds and perform the thresholding analysis
    for threshold_variation, adjusted_thresholds in zip([-30, -20, -10, -5, 5, 10], sensitivity_thresholds):
        print(f"\nThreshold Variation: {threshold_variation}%")

        # Apply Otsu's thresholding with sensitivity
        data_df, _ = apply_otsu_thresholding(data_df.copy(), adjusted_thresholds)
        otsu_metrics = compare_thresholds_and_compute_metrics(data_df, 'otsu', ground_truth_column)

        # Apply GMM thresholding with sensitivity
        data_df, _ = apply_gmm_thresholding(data_df.copy(), adjusted_thresholds)
        gmm_metrics = compare_thresholds_and_compute_metrics(data_df, 'gmm', ground_truth_column)

        # Collect results for this variation
        all_results[f'{name}_variation_{threshold_variation}%'] = {
            'Otsu': otsu_metrics,
            'GMM': gmm_metrics,
        }

        # Plot the marker histograms with the adjusted thresholds
        plot_marker_histograms(data_df, adjusted_thresholds, 'otsu', f"{name} - Otsu's Threshold with {threshold_variation}% Variation")
        plot_marker_histograms(data_df, adjusted_thresholds, 'gmm', f"{name} - GMM Threshold with {threshold_variation}% Variation")

# Function to create a summary table from all_results
def create_summary_table(all_results):
    summary = []
    for dataset, methods in all_results.items():
        for method, metrics_dict in methods.items():
            for marker, metrics in metrics_dict.items():
                summary.append([dataset, marker, method] + list(metrics.values()))
    columns = ['Dataset', 'Marker', 'Method', 'ARI', 'NMI', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC_ROC']
    summary_df = pd.DataFrame(summary, columns=columns)
    return summary_df

# Generate the summary table
summary_df = create_summary_table(all_results)

# Display the summary table
print(summary_df)

# Optional: Save the summary to a CSV file for further analysis or inclusion in reports
summary_df.to_csv('sensitivity_analysis_thresholding_summary.csv', index=False)
