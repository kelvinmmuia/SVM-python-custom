# svm.py

import numpy as np
from kernels import linear, rbf, polynomial, sigmoid # Import new kernels

class SVM:
    def __init__(self, C=1.0, mode='classification', kernel='linear', 
                 gamma=None, degree=3, coef0=1.0): # Added kernel params
        self.C = C
        self.mode = mode
        if mode == 'regression' and kernel != 'linear':
            raise NotImplementedError("Regression is only implemented for linear kernel with SGD.")
        
        self.kernel_name = kernel
        self.gamma = gamma      # For RBF, Polynomial, Sigmoid
        self.degree = degree    # For Polynomial
        self.coef0 = coef0      # For Polynomial, Sigmoid

        self.w = None           # For linear case: weight vector
        self.b = 0.0            # Bias
        
        self.alpha = None       # Coefficients for support vectors (kernelized)
        self.X_sv = None        # Support vectors (kernelized)
        self.y_sv = None        # Labels of support vectors (kernelized, as -1 or 1)

    def _kernel(self, X1, X2=None):
        X2_use = X1 if X2 is None else X2 # Ensure X2 is not None for kernel functions
        
        if self.kernel_name == 'linear':
            return linear(X1, X2_use)
        elif self.kernel_name == 'rbf':
            return rbf(X1, X2_use, gamma=self.gamma)
        elif self.kernel_name == 'poly':
            return polynomial(X1, X2_use, degree=self.degree, gamma=self.gamma, coef0=self.coef0)
        elif self.kernel_name == 'sigmoid':
            return sigmoid(X1, X2_use, gamma=self.gamma, coef0=self.coef0)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel_name}")

    def _fit_smo(self, X, y_orig, max_passes=100, tol=1e-3):
        """
        Simplified SMO algorithm for SVM classification.
        y_orig: original labels (e.g., 0, 1)
        max_passes: max number of iterations over alphas without changes.
        tol: tolerance for KKT conditions.
        """
        n_samples, n_features = X.shape
        y = np.where(y_orig == 0, -1, 1) # SMO works with -1, 1 labels

        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        
        # Precompute Kernel Matrix (can be memory intensive for large datasets)
        K = self._kernel(X, X)

        passes = 0
        epoch = 0 # total iterations over the dataset
        max_epochs_smo = 1000 # Safeguard for total iterations

        while passes < max_passes and epoch < max_epochs_smo:
            num_changed_alphas = 0
            for i in range(n_samples):
                # Prediction error for sample i
                # f_i = sum(self.alpha * y * K[:,i]) + self.b is wrong for K shape
                f_i = np.sum(self.alpha * y * K[i, :]) + self.b # K[i,:] or K[:,i] depending on K definition
                                                              # If K_ij = K(xi,xj), then K[i,:] is K(xi, x_all)
                E_i = f_i - y[i]

                # Check KKT conditions violation
                if (y[i] * E_i < -tol and self.alpha[i] < self.C) or \
                   (y[i] * E_i > tol and self.alpha[i] > 0):
                    
                    # Select j randomly, != i
                    j_list = list(range(n_samples))
                    j_list.pop(i)
                    if not j_list: continue # Only one sample
                    j = np.random.choice(j_list)

                    f_j = np.sum(self.alpha * y * K[j, :]) + self.b
                    E_j = f_j - y[j]

                    alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                    # Compute L and H (bounds for alpha_j)
                    if y[i] != y[j]:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)

                    if abs(L - H) < 1e-5: # If L equals H
                        continue

                    # eta = 2*K_ij - K_ii - K_jj
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= -1e-5: # eta should be < 0 for maximum. Use small neg tolerance.
                        continue

                    # Update alpha_j
                    self.alpha[j] = alpha_j_old - (y[j] * (E_i - E_j)) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    # Check if change in alpha_j is significant
                    if abs(self.alpha[j] - alpha_j_old) < tol / 10.0: # More stringent for individual alpha change
                        self.alpha[j] = alpha_j_old # Revert if change too small
                        continue 

                    # Update alpha_i
                    self.alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # Update bias b
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i,i] \
                         - y[j] * (self.alpha[j] - alpha_j_old) * K[i,j]
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i,j] \
                         - y[j] * (self.alpha[j] - alpha_j_old) * K[j,j]

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else: # Both on bounds
                        self.b = (b1 + b2) / 2.0
                    
                    num_changed_alphas += 1
            
            epoch += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0 # Reset passes if alphas changed
        
        # Store support vectors
        sv_indices = np.where(self.alpha > tol)[0]
        if len(sv_indices) > 0:
            self.X_sv = X[sv_indices]
            self.y_sv = y[sv_indices] # Store y as -1, 1
            self.alpha = self.alpha[sv_indices]
        else: # No support vectors found
            print("Warning: No support vectors found. Model may not be properly trained.")
            self.X_sv = np.array([]).reshape(0, n_features)
            self.y_sv = np.array([])
            self.alpha = np.array([])
            # self.b is as computed

        self.w = None # Clear linear weights as we used SMO

    def fit(self, X, y, batch_size=32, learning_rate=0.001, epochs=1000, 
            smo_max_passes=20, smo_tol=1e-3): # Added SMO params
        """
        Fit the SVM model.
        For kernel='linear', uses SGD for primal form by default.
        For other kernels (or if self.force_smo_linear is True, not implemented here),
        uses SMO for dual form (classification only).
        """
        if self.kernel_name == 'linear' and self.mode == 'classification' and not hasattr(self, '_force_smo_linear'):
            # Primal SGD for linear classification
            n_samples, n_features = X.shape
            y_train = np.where(y == 0, -1, 1)

            self.w = np.zeros(n_features)
            self.b = 0.0

            for epoch in range(epochs):
                idx = np.random.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    batch = idx[start:end]
                    Xb, yb = X[batch], y_train[batch]
                    
                    margins = yb * (Xb @ self.w + self.b)
                    mask = margins < 1
                    
                    grad_w = self.w - self.C * np.sum(yb[mask, None] * Xb[mask], axis=0)
                    grad_b = -self.C * np.sum(yb[mask])
                    
                    self.w -= learning_rate * grad_w
                    self.b -= learning_rate * grad_b
            self.alpha = None # Ensure kernel attributes are None

        elif self.kernel_name == 'linear' and self.mode == 'regression':
            # Primal SGD for linear regression (Ridge-like)
            n_samples, n_features = X.shape
            y_train = y.copy() # Use original y for regression

            self.w = np.zeros(n_features)
            self.b = 0.0
            # Your existing SGD for regression
            for _ in range(epochs):
                idx = np.random.permutation(n_samples)
                for start in range(0, n_samples, batch_size):
                    end = start + batch_size
                    batch = idx[start:end]
                    Xb, yb = X[batch], y_train[batch]

                    preds = Xb @ self.w + self.b
                    errors = preds - yb
                    # Gradient from Ridge: L = C * sum(errors^2) + 0.5 * ||w||^2
                    # dL/dw = 2*C * Xb.T @ errors + w
                    # dL/db = 2*C * sum(errors)
                    grad_w = self.w + 2 * self.C * (Xb.T @ errors) # Sum over batch implicitly by Xb.T @ errors
                    grad_b = 2 * self.C * errors.sum()
                    
                    self.w -= learning_rate * grad_w
                    self.b -= learning_rate * grad_b
            self.alpha = None # Ensure kernel attributes are None

        elif self.mode == 'classification': # Use SMO for other kernels in classification
            self._fit_smo(X, y, max_passes=smo_max_passes, tol=smo_tol)
        else:
            raise NotImplementedError(f"Kernel '{self.kernel_name}' for mode '{self.mode}' is not supported with this solver.")

    def predict(self, X):
        if self.w is not None: # Linear SVM (primal) was trained
            scores = X @ self.w + self.b
            if self.mode == 'classification':
                return np.where(scores >= 0, 1, 0)
            else: # Regression
                return scores
        elif self.alpha is not None and self.X_sv is not None and self.y_sv is not None: # Kernel SVM (dual)
            if self.X_sv.shape[0] == 0: # No support vectors
                print("Warning: Predicting with no support vectors (kernelized model). Using bias only.")
                scores = np.full(X.shape[0], self.b)
            else:
                scores = np.zeros(X.shape[0])
                for i in range(X.shape[0]):
                    # Kernel values between all support vectors and the i-th test sample
                    kernel_vals = self._kernel(self.X_sv, X[i, :].reshape(1, -1)) # Shape (n_sv, 1)
                    scores[i] = np.sum(self.alpha * self.y_sv * kernel_vals.flatten()) + self.b
            
            if self.mode == 'classification':
                return np.where(scores >= 0, 1, 0) # Output 0 or 1
            else: # Should not be reached if regression for kernels is not implemented in fit
                return scores 
        else:
            raise RuntimeError("SVM model not trained or training failed.")