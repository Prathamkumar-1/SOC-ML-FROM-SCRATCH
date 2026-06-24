"""
K-Means Clustering from Scratch
=================================
Week 3 - ML From Scratch (WNCC, IIT Bombay)
"""

import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=0):
        self.k = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.rng = np.random.default_rng(random_state)
        self.centroids = None
        self.labels = None

    def fit(self, X):
        n_samples = X.shape[0]
        # Initialize centroids by picking random data points
        init_idx = self.rng.choice(n_samples, self.k, replace=False)
        self.centroids = X[init_idx].copy()

        for _ in range(self.max_iters):
            # Assign each point to nearest centroid
            distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
            labels = np.argmin(distances, axis=1)

            # Recompute centroids
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else self.centroids[j]
                for j in range(self.k)
            ])

            shift = np.linalg.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break

        self.labels = labels
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(distances, axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    c1 = rng.normal(loc=[0, 0], scale=0.6, size=(60, 2))
    c2 = rng.normal(loc=[5, 5], scale=0.6, size=(60, 2))
    c3 = rng.normal(loc=[0, 5], scale=0.6, size=(60, 2))
    X = np.vstack([c1, c2, c3])

    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X)

    print("Final centroids:\n", model.centroids)
    print("Cluster sizes:", np.bincount(model.labels))
