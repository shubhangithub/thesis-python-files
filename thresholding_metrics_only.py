import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef

# True and predicted values should be numpy arrays for each function.

# Accuracy
def marker_wise_accuracy(true, predicted):
    return accuracy_score(true, predicted)

def full_profile_accuracy(true, predicted):
    return np.mean(np.all(true == predicted, axis=1))

# Confusion Matrix
def marker_wise_confusion_matrix(true, predicted):
    return confusion_matrix(true, predicted)

def full_profile_confusion_matrix(true, predicted):
    return confusion_matrix(np.all(true == predicted, axis=1), np.ones_like(true[:, 0]))

# Precision
def marker_wise_precision(true, predicted):
    return precision_score(true, predicted)

def full_profile_precision(true, predicted):
    return precision_score(np.all(true == predicted, axis=1), np.ones_like(true[:, 0]))

# Recall
def marker_wise_recall(true, predicted):
    return recall_score(true, predicted)

def full_profile_recall(true, predicted):
    return recall_score(np.all(true == predicted, axis=1), np.ones_like(true[:, 0]))

# F1 Score
def marker_wise_f1_score(true, predicted):
    return f1_score(true, predicted)

def full_profile_f1_score(true, predicted):
    return f1_score(np.all(true == predicted, axis=1), np.ones_like(true[:, 0]))

# AUC-ROC
def marker_wise_auc_roc(true, predicted):
    return roc_auc_score(true, predicted)

def full_profile_auc_roc(true, predicted):
    return roc_auc_score(np.all(true == predicted, axis=1), np.ones_like(true[:, 0]))

# Matthews Correlation Coefficient (MCC)
def marker_wise_mcc(true, predicted):
    return matthews_corrcoef(true, predicted)

def full_profile_mcc(true, predicted):
    return matthews_corrcoef(np.all(true == predicted, axis=1), np.ones_like(true[:, 0]))

