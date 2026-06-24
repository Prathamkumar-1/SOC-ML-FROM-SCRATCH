"""
Anomaly Detection from Scratch (Gaussian Model)
==================================================
Week 3 - ML From Scratch (WNCC, IIT Bombay)

Fits an independent Gaussian to each feature, computes the joint
probability of each point, and flags points below a probability
threshold (epsilon) as anomalies.
"""

import numpy as np


class GaussianAnomalyDetector:
    def __init__(self, epsilon=1e-4):
        self.epsilon = epsilon
        self.mu = None
        self.sigma2 = None

    def fit(self, X):
        self.mu = X.mean(axis=0)
        self.sigma2 = X.var(axis=0)
        return self

    def _gaussian_pdf(self, X):
        # Product of independent per-feature Gaussians
        coef = 1.0 / np.sqrt(2 * np.pi * self.sigma2)
        exponent = -((X - self.mu) ** 2) / (2 * self.sigma2)
        probs = coef * np.exp(exponent)
        return np.prod(probs, axis=1)

    def predict(self, X):
        p = self._gaussian_pdf(X)
        return (p < self.epsilon).astype(int), p  # 1 = anomaly


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    normal_data = rng.normal(loc=[5, 5], scale=1.0, size=(200, 2))
    anomalies = np.array([[15, 15], [0, 15], [-5, -5]])
    X = np.vstack([normal_data, anomalies])

    detector = GaussianAnomalyDetector(epsilon=1e-3)
    detector.fit(normal_data)  # fit only on "normal" training data
    preds, probs = detector.predict(X)

    print("Number of points flagged as anomalies:", preds.sum())
    print("Anomalous points:\n", X[preds == 1])
