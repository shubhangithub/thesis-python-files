import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# Function to generate sensitivity thresholds
def generate_sensitivity_thresholds(original_thresholds):
    new_thresholds = []
    for percentage in [-30, -20, -10, -5, 5, 10]:
        modified_thresholds = {marker: value * (1 + percentage / 100) for marker, value in original_thresholds.items()}
        new_thresholds.append(modified_thresholds)
    return new_thresholds

# Function to get expressed markers based on thresholds
def get_expressed_markers(row, thresholds):
    expressed = [marker for marker in thresholds.keys() if row[marker] > thresholds[marker]]
    return expressed

# Function to filter clusters with less than 10 members
def filter_small_clusters(cells_df):
    cluster_counts = cells_df['cluster'].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= 10].index
    filtered_df = cells_df[cells_df['cluster'].isin(valid_clusters)].copy()
    return filtered_df

# Function to reclassify cells and filter clusters
def process_with_thresholds(cells_df, threshold_set, marker_columns):
    cells_df['expressed_markers'] = ''
    cells_df['cluster'] = 0
    unique_clusters = {}
    cluster_id = 1

    for index, row in cells_df.iterrows():
        expressed = get_expressed_markers(row, threshold_set)
        cells_df.at[index, 'expressed_markers'] = ', '.join(expressed)

        # Create a unique cluster ID based on the expressed markers
        expressed_key = ','.join(sorted(expressed))

        if expressed_key not in unique_clusters:
            unique_clusters[expressed_key] = cluster_id
            cluster_id += 1

        cells_df.at[index, 'cluster'] = unique_clusters[expressed_key]

    # Filter out clusters with less than 10 members
    filtered_cells_df = filter_small_clusters(cells_df)
    
    return filtered_cells_df

# Function for evaluating clustering performance
def evaluate_clustering(true_labels, predicted_labels, features):
    print('Evaluating clustering performance...')

    # ARI and NMI Scores
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    # Silhouette Score
    silhouette_avg = silhouette_score(features, predicted_labels) if len(set(predicted_labels)) > 1 else -1

    # Return the computed metrics
    return {
        'ARI': ari,
        'NMI': nmi,
        'Silhouette Score': silhouette_avg
    }

# Paths to your files (you should have one path per threshold set)
file_paths = [
    '/content/filtered_002_TU2_Immune_1_thresholded_consensus_clustering_results 2.41.04 pm.csv',
    '/content/filtered_002_TU2_Immune_2_thresholded_consensus_clustering_results 2.41.04 pm.csv',
    '/content/filtered_CC_OC_585_TU1_Immune_1_thresholded_consensus_clustering_results 2.41.04 pm.csv',
    '/content/filtered_CC_OC_585_TU1_Immune_2_thresholded_consensus_clustering_results 2.41.04 pm.csv'
]

# Corresponding original thresholds for each file
thresholds = [
    {"CD66b_asNumeric": 5.64589286, "CD56_asNumeric": 8.38696429, "CD4_asNumeric": 4.38928571, "CTLA4_asNumeric": 1.94285714, "CD8_asNumeric": 3.52285714, "CD20_asNumeric": 3.1975},
    {"CD66b_asNumeric": 11.0958929, "CD56_asNumeric": 7.04696429, "CD4_asNumeric": 4.38928571, "CTLA4_asNumeric": 2.85714286, "CD8_asNumeric": 4.57142857, "CD20_asNumeric": 3.1975},
    {"CD66b_asNumeric": 6.04589286, "CD56_asNumeric": 18.7608929, "CD4_asNumeric": 2.65357143, "CTLA4_asNumeric": 2.86696429, "CD8_asNumeric": 6.20607143, "CD20_asNumeric": 1.3075},
    {"CD66b_asNumeric": 6.04589286, "CD56_asNumeric": 14.7992857, "CD4_asNumeric": 2.65357143, "CTLA4_asNumeric": 2.29357143, "CD8_asNumeric": 3.90607143, "CD20_asNumeric": 0.8875},
]

# Generate new thresholds for sensitivity analysis for each file
all_sensitivity_thresholds = [generate_sensitivity_thresholds(threshold_set) for threshold_set in thresholds]

# Features for clustering performance evaluation
marker_columns = ['CD66b_asNumeric', 'CD56_asNumeric', 'CD4_asNumeric', 'CTLA4_asNumeric', 'CD8_asNumeric', 'CD20_asNumeric']

# Process each file with its corresponding thresholds
for file_idx, file_path in enumerate(file_paths):
    print(f"\nProcessing file: {file_path}")

    # Load the file
    cells = pd.read_csv(file_path)
    true_labels = cells['cluster'].values  # Assuming this is your ground truth clustering
    features = cells[marker_columns].values

    # Get all clustering method columns from your file
    clustering_columns = [col for col in cells.columns if '_Cluster' in col]

    # Get the sensitivity thresholds for the current file
    sensitivity_thresholds = all_sensitivity_thresholds[file_idx]

    # Process each sensitivity threshold set
    for j, threshold_set in enumerate(sensitivity_thresholds):
        print(f"\nProcessing with sensitivity variation {[-30, -20, -10, -5, 5, 10][j]}%...")

        # Reclassify cells and filter small clusters
        processed_cells = process_with_thresholds(cells.copy(), threshold_set, marker_columns)

        # Loop through each clustering method and compare performance
        for cluster_col in clustering_columns:
            print(f"\nComparing with clustering method: {cluster_col}")

            predicted_labels = processed_cells[cluster_col].values
            evaluation_results = evaluate_clustering(true_labels, predicted_labels, features)

            # Print the evaluation results
            print(f"Threshold Variation {[-30, -20, -10, -5, 5, 10][j]}% and Clustering Method {cluster_col}:")
            print(f"ARI: {evaluation_results['ARI']:.4f}, NMI: {evaluation_results['NMI']:.4f}, Silhouette Score: {evaluation_results['Silhouette Score']:.4f}")

