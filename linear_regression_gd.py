"""
Linear Regression from Scratch - Batch Gradient Descent
=========================================================
Topic: Week 2 - Gradient Descent Algorithms
Project: ML From Scratch (WNCC, IIT Bombay)

This corresponds to "LinearRegression2" in the lab sequence: instead of
solving the Normal Equation directly, we minimize the MSE cost function
iteratively using gradient descent. This generalizes better to large
datasets / many features, and is the same idea used later for logistic
regression and neural networks.

Theory
------
Cost:      J(w) = (1/2m) * sum( (X w - y)^2 )
Gradient:  dJ/dw = (1/m) * X^T (X w - y)
Update:    w := w - alpha * dJ/dw
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionGD:
    """Linear Regression trained with batch gradient descent."""

    def __init__(self, learning_rate=0.1, n_iters=1000, verbose=False):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.verbose = verbose
        self.weights = None
        self.cost_history = []

    def _add_bias(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def _compute_cost(self, X_b, y):
        m = len(y)
        preds = X_b @ self.weights
        return (1 / (2 * m)) * np.sum((preds - y) ** 2)

    def fit(self, X, y):
        X_b = self._add_bias(X)
        m, n = X_b.shape
        y = y.reshape(-1, 1)

        self.weights = np.zeros((n, 1))
        self.cost_history = []

        for i in range(self.n_iters):
            preds = X_b @ self.weights
            error = preds - y
            gradient = (1 / m) * (X_b.T @ error)
            self.weights -= self.lr * gradient

            cost = (1 / (2 * m)) * np.sum(error ** 2)
            self.cost_history.append(cost)

            if self.verbose and i % 100 == 0:
                print(f"Iter {i:4d} | Cost: {cost:.4f}")

        return self

    def predict(self, X):
        X_b = self._add_bias(X)
        return (X_b @ self.weights).flatten()

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


def normalize(X):
    """Feature scaling (standardization) - important for GD convergence."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1
    return (X - mu) / sigma, mu, sigma


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = 2 * rng.random((100, 1))
    y = 4 + 3 * X.flatten() + rng.normal(0, 1, 100)

    X_norm, mu, sigma = normalize(X)

    model = LinearRegressionGD(learning_rate=0.5, n_iters=500, verbose=True)
    model.fit(X_norm, y)
    y_pred = model.predict(X_norm)

    print("\nLearned weights (bias, slope, in normalized space):", model.weights.flatten())
    print("Final cost:", model.cost_history[-1])
    print("R^2 score:", model.score(X_norm, y))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    order = np.argsort(X.flatten())
    axes[0].scatter(X, y, alpha=0.6, label="Data")
    axes[0].plot(X.flatten()[order], y_pred[order], color="red", label="GD fit")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("y")
    axes[0].set_title("Linear Regression via Gradient Descent")
    axes[0].legend()

    axes[1].plot(model.cost_history)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Cost J(w)")
    axes[1].set_title("Convergence of Gradient Descent")

    plt.tight_layout()
    plt.savefig("linear_regression_gd_demo.png", dpi=150)
    print("Plot saved to linear_regression_gd_demo.png")
