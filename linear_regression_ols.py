"""
Linear Regression from Scratch - Normal Equation (OLS / Closed Form)
======================================================================
Topic: Week 2 - Regression Models
Project: ML From Scratch (WNCC, IIT Bombay)

This script implements simple/multiple linear regression using the
Normal Equation (Ordinary Least Squares), without using sklearn.

Theory
------
Model:        y_hat = X @ w
Cost (MSE):   J(w) = (1/2m) * ||X w - y||^2
Closed form:  w = (X^T X)^-1 X^T y

This works well when the number of features is small and X^T X is
invertible. For large feature spaces, gradient descent (see
linear_regression_gd.py) is preferred.
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionOLS:
    """Linear Regression solved via the Normal Equation."""

    def __init__(self):
        self.weights = None  # includes bias as w[0]

    def _add_bias(self, X):
        # Prepend a column of ones to X for the bias/intercept term
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def fit(self, X, y):
        X_b = self._add_bias(X)
        y = y.reshape(-1, 1)
        # Normal equation: w = (X^T X)^-1 X^T y
        # np.linalg.pinv used instead of inv() for numerical stability
        self.weights = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X):
        X_b = self._add_bias(X)
        return (X_b @ self.weights).flatten()

    def score(self, X, y):
        """R^2 score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Demo: synthetic single-feature dataset  y = 4 + 3x + noise
    # ------------------------------------------------------------
    rng = np.random.default_rng(42)
    X = 2 * rng.random((100, 1))
    y = 4 + 3 * X.flatten() + rng.normal(0, 1, 100)

    model = LinearRegressionOLS()
    model.fit(X, y)
    y_pred = model.predict(X)

    print("Learned weights (bias, slope):", model.weights.flatten())
    print("MSE:", mean_squared_error(y, y_pred))
    print("R^2 score:", model.score(X, y))

    # Plot
    plt.figure(figsize=(6, 4))
    plt.scatter(X, y, alpha=0.6, label="Data")
    order = np.argsort(X.flatten())
    plt.plot(X.flatten()[order], y_pred[order], color="red", label="OLS fit")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression via Normal Equation")
    plt.legend()
    plt.tight_layout()
    plt.savefig("linear_regression_ols_demo.png", dpi=150)
    print("Plot saved to linear_regression_ols_demo.png")
