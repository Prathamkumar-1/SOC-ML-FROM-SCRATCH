"""
Logistic Regression from Scratch - Gradient Descent
=====================================================
Topic: Week 2 - Classification via Logistic Regression
Project: ML From Scratch (WNCC, IIT Bombay)

Theory
------
Hypothesis:  h(x) = sigmoid(X w) = 1 / (1 + e^(-X w))
Cost (binary cross-entropy / log loss):
    J(w) = -(1/m) * sum( y*log(h) + (1-y)*log(1-h) )
Gradient (same form as linear regression, just with sigmoid hypothesis):
    dJ/dw = (1/m) * X^T (h - y)
Update:
    w := w - alpha * dJ/dw
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    # Clip to avoid overflow in exp for very negative/positive z
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


class LogisticRegressionGD:
    """Binary Logistic Regression trained with batch gradient descent."""

    def __init__(self, learning_rate=0.1, n_iters=1000, verbose=False):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.verbose = verbose
        self.weights = None
        self.cost_history = []

    def _add_bias(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def fit(self, X, y):
        X_b = self._add_bias(X)
        m, n = X_b.shape
        y = y.reshape(-1, 1)

        self.weights = np.zeros((n, 1))
        self.cost_history = []
        eps = 1e-9  # avoid log(0)

        for i in range(self.n_iters):
            z = X_b @ self.weights
            h = sigmoid(z)
            error = h - y
            gradient = (1 / m) * (X_b.T @ error)
            self.weights -= self.lr * gradient

            cost = -(1 / m) * np.sum(
                y * np.log(h + eps) + (1 - y) * np.log(1 - h + eps)
            )
            self.cost_history.append(cost)

            if self.verbose and i % 100 == 0:
                print(f"Iter {i:4d} | Cost: {cost:.4f}")

        return self

    def predict_proba(self, X):
        X_b = self._add_bias(X)
        return sigmoid(X_b @ self.weights).flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


def normalize(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1
    return (X - mu) / sigma, mu, sigma


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Demo: synthetic 2D binary classification dataset
    # ------------------------------------------------------------
    rng = np.random.default_rng(1)
    n = 200
    class0 = rng.normal(loc=[2, 2], scale=1.0, size=(n // 2, 2))
    class1 = rng.normal(loc=[6, 6], scale=1.0, size=(n // 2, 2))
    X = np.vstack([class0, class1])
    y = np.hstack([np.zeros(n // 2), np.ones(n // 2)])

    X_norm, mu, sigma = normalize(X)

    model = LogisticRegressionGD(learning_rate=0.5, n_iters=500, verbose=True)
    model.fit(X_norm, y)

    print("\nLearned weights (bias, w1, w2):", model.weights.flatten())
    print("Final cost:", model.cost_history[-1])
    print("Training accuracy:", model.accuracy(X_norm, y))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Decision boundary plot
    axes[0].scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Class 0", alpha=0.6)
    axes[0].scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Class 1", alpha=0.6)

    x1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x1_norm = (x1_vals - mu[0]) / sigma[0]
    w0, w1, w2 = model.weights.flatten()
    # decision boundary: w0 + w1*x1_norm + w2*x2_norm = 0  ->  x2_norm = -(w0 + w1*x1_norm)/w2
    x2_norm = -(w0 + w1 * x1_norm) / w2
    x2_vals = x2_norm * sigma[1] + mu[1]
    axes[0].plot(x1_vals, x2_vals, color="black", label="Decision boundary")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_title("Logistic Regression Decision Boundary")
    axes[0].legend()

    axes[1].plot(model.cost_history)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Cost J(w) (log loss)")
    axes[1].set_title("Convergence of Gradient Descent")

    plt.tight_layout()
    plt.savefig("logistic_regression_demo.png", dpi=150)
    print("Plot saved to logistic_regression_demo.png")
