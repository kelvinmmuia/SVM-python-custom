# main_script.py

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC as SklearnSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Import your custom SVM
from svm import SVM

def load_arff_like_data(file_path):
    """
    Loads data from an ARFF-like file, parsing attributes and data.
    """
    attributes = []
    data_lines = []
    is_data_section = False
    target_col_name = None 

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.lower().startswith('@relation'):
                continue
            elif line.lower().startswith('@attribute'):
                parts = line.split()
                col_name = parts[1].strip("'")
                attributes.append(col_name)
                target_col_name = col_name 
            elif line.lower().startswith('@data'):
                is_data_section = True
                continue
            elif is_data_section:
                data_lines.append(line.split(','))

    df = pd.DataFrame(data_lines, columns=attributes)
    feature_cols = [col for col in attributes if col != target_col_name]

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df[feature_cols].values
    y_raw = df[target_col_name].values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"Target classes: {label_encoder.classes_} mapped to {np.unique(y)}")
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col_name}")

    return X, y, feature_cols, target_col_name, label_encoder

# --- Main script execution ---

# 1. Load the dataset
file_path = 'dataset'
X, y, feature_names, target_name, label_encoder = load_arff_like_data(file_path)

# 2. Initialize K-Fold cross-validation
n_splits = 10
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# 3. Define kernel configurations and parameters for comparison
# We'll use common parameters for a fairer comparison where possible.
# gamma_for_comparison: sklearn's 'auto' is 1/n_features. Custom SVM defaults to this if gamma=None.
gamma_for_comparison = 1.0 / X.shape[1] 
degree_for_comparison = 3
# For poly/sigmoid, sklearn's coef0 defaults to 0.0. Custom SVM defaults to 1.0.
# We will use 0.0 for both for consistency in this comparison.
coef0_for_comparison = 0.0 

kernels_to_test = ['linear', 'rbf', 'poly', 'sigmoid']

# SMO parameters for custom SVM (for non-linear kernels)
# (epochs and learning_rate are for SGD in linear custom SVM)
custom_svm_epochs_sgd = 1000 
custom_svm_lr_sgd = 0.001
custom_svm_smo_max_passes = 50 # Increased for potentially better SMO convergence
custom_svm_smo_tol = 1e-3

results_summary = [] # To store average metrics for each kernel/SVM type

# 4. Perform cross-validation for each kernel
for kernel_name in kernels_to_test:
    print(f"\n--- Testing Kernel: {kernel_name.upper()} ---")

    # Lists to store metrics for the current kernel across all folds
    custom_fold_accuracies, custom_fold_precisions, custom_fold_recalls = [], [], []
    sklearn_fold_accuracies, sklearn_fold_precisions, sklearn_fold_recalls = [], [], []

    fold_num = 1
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Scale features for this fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"  Fold {fold_num}/{n_splits} for kernel '{kernel_name}'...")

        # --- Custom SVM ---
        current_custom_gamma = gamma_for_comparison if kernel_name != 'linear' else None
        current_custom_degree = degree_for_comparison # SVM __init__ uses it if kernel is 'poly'
        current_custom_coef0 = coef0_for_comparison   # SVM __init__ uses it if kernel is 'poly' or 'sigmoid'
        
        try:
            custom_svm = SVM(C=1.0, mode='classification', kernel=kernel_name,
                               gamma=current_custom_gamma,
                               degree=current_custom_degree,
                               coef0=current_custom_coef0)
            
            custom_svm.fit(X_train_scaled, y_train, 
                           batch_size=32, learning_rate=custom_svm_lr_sgd, epochs=custom_svm_epochs_sgd, # For SGD (linear)
                           smo_max_passes=custom_svm_smo_max_passes, smo_tol=custom_svm_smo_tol) # For SMO
            y_pred_custom = custom_svm.predict(X_test_scaled)
            
            custom_fold_accuracies.append(accuracy_score(y_test, y_pred_custom))
            custom_fold_precisions.append(precision_score(y_test, y_pred_custom, pos_label=1, zero_division=0))
            custom_fold_recalls.append(recall_score(y_test, y_pred_custom, pos_label=1, zero_division=0))
        except NotImplementedError as e:
            print(f"    Custom SVM (Fold {fold_num}, {kernel_name}): Not implemented - {e}")
        except Exception as e:
            print(f"    Custom SVM (Fold {fold_num}, {kernel_name}): Error - {e}")
            # Optionally append NaNs or handle error to avoid skewing if some folds fail
            # For simplicity, we just skip appending if an error occurs for this fold for custom SVM.

        # --- Scikit-learn SVM ---
        current_sklearn_gamma = gamma_for_comparison if kernel_name != 'linear' else 'auto' # 'auto' is 1/n_features
        current_sklearn_degree = degree_for_comparison
        current_sklearn_coef0 = coef0_for_comparison
        
        try:
            sklearn_svm = SklearnSVC(C=1.0, kernel=kernel_name, 
                                     gamma=current_sklearn_gamma, 
                                     degree=current_sklearn_degree,
                                     coef0=current_sklearn_coef0,
                                     random_state=42,
                                     probability=False) # Probability=True can slow down, not needed for these metrics
            sklearn_svm.fit(X_train_scaled, y_train)
            y_pred_sklearn = sklearn_svm.predict(X_test_scaled)

            sklearn_fold_accuracies.append(accuracy_score(y_test, y_pred_sklearn))
            sklearn_fold_precisions.append(precision_score(y_test, y_pred_sklearn, pos_label=1, zero_division=0))
            sklearn_fold_recalls.append(recall_score(y_test, y_pred_sklearn, pos_label=1, zero_division=0))
        except Exception as e:
            print(f"    Sklearn SVM (Fold {fold_num}, {kernel_name}): Error - {e}")
            # Skip appending for Sklearn SVM for this fold if an error occurs.

        fold_num += 1

    # Calculate and store average metrics for Custom SVM for the current kernel
    if custom_fold_accuracies: # Check if list is not empty (i.e., at least one fold succeeded)
        results_summary.append({
            'Kernel': kernel_name, 'SVM Type': 'Custom',
            'Avg Accuracy': np.mean(custom_fold_accuracies),
            'Avg Precision': np.mean(custom_fold_precisions),
            'Avg Recall (Sensitivity)': np.mean(custom_fold_recalls)
        })
    else: # Handle case where all folds failed for custom SVM
        results_summary.append({
            'Kernel': kernel_name, 'SVM Type': 'Custom',
            'Avg Accuracy': np.nan, 'Avg Precision': np.nan, 'Avg Recall (Sensitivity)': np.nan
        })

    # Calculate and store average metrics for Sklearn SVM for the current kernel
    if sklearn_fold_accuracies:
        results_summary.append({
            'Kernel': kernel_name, 'SVM Type': 'Sklearn',
            'Avg Accuracy': np.mean(sklearn_fold_accuracies),
            'Avg Precision': np.mean(sklearn_fold_precisions),
            'Avg Recall (Sensitivity)': np.mean(sklearn_fold_recalls)
        })
    else: # Handle case where all folds failed for Sklearn SVM
        results_summary.append({
            'Kernel': kernel_name, 'SVM Type': 'Sklearn',
            'Avg Accuracy': np.nan, 'Avg Precision': np.nan, 'Avg Recall (Sensitivity)': np.nan
        })

# 5. Display results in a table
results_df = pd.DataFrame(results_summary)
print("\n\n--- Overall Performance Comparison ---")

# Format numerical columns for better readability
float_format_cols = ['Avg Accuracy', 'Avg Precision', 'Avg Recall (Sensitivity)']
for col in float_format_cols:
    results_df[col] = results_df[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")

print(results_df.to_string())