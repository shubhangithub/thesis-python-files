import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef

# Define performance metric functions
def marker_wise_accuracy(true, predicted):
    return accuracy_score(true, predicted)

def full_profile_accuracy(true, predicted):
    return np.mean(np.all(true == predicted, axis=1))

def marker_wise_confusion_matrix(true, predicted):
    return confusion_matrix(true, predicted)

def full_profile_confusion_matrix(true, predicted):
    return confusion_matrix(np.all(true == predicted, axis=1), np.ones_like(true[:, 0]))

def marker_wise_precision(true, predicted):
    return precision_score(true, predicted, average='binary')

def full_profile_precision(true, predicted):
    return precision_score(np.all(true == predicted, axis=1), np.ones_like(true[:, 0]))

def marker_wise_recall(true, predicted):
    return recall_score(true, predicted, average='binary')

def full_profile_recall(true, predicted):
    return recall_score(np.all(true == predicted, axis=1), np.ones_like(true[:, 0]))

def marker_wise_f1_score(true, predicted):
    return f1_score(true, predicted, average='binary')

def full_profile_f1_score(true, predicted):
    return f1_score(np.all(true == predicted, axis=1), np.ones_like(true[:, 0]))

def marker_wise_auc_roc(true, predicted):
    return roc_auc_score(true, predicted)

def full_profile_auc_roc(true, predicted):
    return roc_auc_score(np.all(true == predicted, axis=1), np.ones_like(true[:, 0]))

def marker_wise_mcc(true, predicted):
    return matthews_corrcoef(true, predicted)

def full_profile_mcc(true, predicted):
    return matthews_corrcoef(np.all(true == predicted, axis=1), np.ones_like(true[:, 0]))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
import ast


import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef

# Define performance metric functions (same as before)
# ...

# Function to transform markers into a binary format
def transform_markers_to_binary(marker_column, markers):
    def transform_row(marker_string):
        if pd.isna(marker_string):
            # No markers expressed
            return [0] * len(markers)
        else:
            # Binary vector based on which markers are expressed
            return [1 if marker in marker_string else 0 for marker in markers]

    # Apply the transformation
    return marker_column.apply(transform_row)

# Main function to evaluate performance across different methods and files
def evaluate_performance(file_path, method_name, predicted_column_name):
    print(f"Processing file: {file_path}")
    
    # Load dataset
    df = pd.read_csv(file_path)

    # Marker names to be checked
    marker_names = ['CD66b', 'CD56', 'CD4', 'CTLA4', 'CD8', 'CD20']

    # Transform expressed_markers into a binary format
    df['expressed_markers_binary'] = transform_markers_to_binary(df['expressed_markers'], marker_names)
    
    # Print the temporary structure after transformation (for debugging)
    print("Transformed expressed_markers (first 10 rows):\n", df[['expressed_markers', 'expressed_markers_binary']].head(10))

    # Transform predicted markers into a binary format
    df['predicted_markers_binary'] = transform_markers_to_binary(df[predicted_column_name], marker_names)

    # Print the transformed predicted markers for debugging
    print("Transformed predicted_markers (first 10 rows):\n", df[['predicted_markers_binary']].head(10))

    # Save the DataFrame with both original and binary columns
    save_file_path = file_path.replace(".csv", "_modified.csv")
    df.to_csv(save_file_path, index=False)
    print(f"Saved modified DataFrame to {save_file_path}")

    # Convert both true and predicted values to binary arrays
    true_values = np.array(df['expressed_markers_binary'].to_list())
    predicted_values = np.array(df['predicted_markers_binary'].to_list())

    print(f"Evaluating method: {method_name}")

    # Marker-wise evaluations (one for each marker)
    for marker_idx, marker_name in enumerate(marker_names):
        print(f"\n--- Marker-wise evaluations for {marker_name} ---")

        # Accuracy
        acc = marker_wise_accuracy(true_values[:, marker_idx], predicted_values[:, marker_idx])
        print(f"Marker-wise Accuracy for {marker_name}: {acc}")

        # Precision
        prec = marker_wise_precision(true_values[:, marker_idx], predicted_values[:, marker_idx])
        print(f"Marker-wise Precision for {marker_name}: {prec}")

        # Recall
        rec = marker_wise_recall(true_values[:, marker_idx], predicted_values[:, marker_idx])
        print(f"Marker-wise Recall for {marker_name}: {rec}")

        # F1 Score
        f1 = marker_wise_f1_score(true_values[:, marker_idx], predicted_values[:, marker_idx])
        print(f"Marker-wise F1 Score for {marker_name}: {f1}")

        # AUC-ROC
        auc_roc = marker_wise_auc_roc(true_values[:, marker_idx], predicted_values[:, marker_idx])
        print(f"Marker-wise AUC-ROC for {marker_name}: {auc_roc}")

        # MCC
        mcc = marker_wise_mcc(true_values[:, marker_idx], predicted_values[:, marker_idx])
        print(f"Marker-wise MCC for {marker_name}: {mcc}")

    # Full-profile evaluations (entire set of markers)
    print(f"\n--- Full-profile evaluations ---")

    # Accuracy
    full_acc = full_profile_accuracy(true_values, predicted_values)
    print(f"Full-profile Accuracy: {full_acc}")

    # Precision
    full_prec = full_profile_precision(true_values, predicted_values)
    print(f"Full-profile Precision: {full_prec}")

    # Recall
    full_rec = full_profile_recall(true_values, predicted_values)
    print(f"Full-profile Recall: {full_rec}")

    # F1 Score
    full_f1 = full_profile_f1_score(true_values, predicted_values)
    print(f"Full-profile F1 Score: {full_f1}")

    # AUC-ROC
    full_auc_roc = full_profile_auc_roc(true_values, predicted_values)
    print(f"Full-profile AUC-ROC: {full_auc_roc}")

    # MCC
    full_mcc = full_profile_mcc(true_values, predicted_values)
    print(f"Full-profile MCC: {full_mcc}")

# Example usage for different methods
files = [
    '/content/002_TU2_Immune_2_NEW_thresholded_encoded.csv',
    '/content/CC_OC_585_TU1_Immune_1_NEW_thresholded_encoded.csv'
]

methods = {
    'Otsu': 'otsu_expressed_markers',
    'IsoData': 'isodata_expressed_markers',
    'GMM': 'gmm_expressed_markers',
    'Cross-Entropy': 'cross_entropy_expressed_markers'
}

# Run evaluation for all files and methods
for file_path in files:
    for method_name, predicted_column_name in methods.items():
        evaluate_performance(file_path, method_name, predicted_column_name)
