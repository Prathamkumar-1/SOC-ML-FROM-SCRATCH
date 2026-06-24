"""
Decision Tree Classifier from Scratch
=======================================
Week 3 - ML From Scratch (WNCC, IIT Bombay)

A simple CART-style decision tree using Gini impurity for binary/multi-class
classification. Kept intentionally simple/readable rather than optimized.
"""

import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # set only for leaf nodes

    def is_leaf(self):
        return self.value is not None


def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)


class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        n_samples, n_features = X.shape
        parent_gini = gini(y)

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                left_gini = gini(y[left_mask])
                right_gini = gini(y[right_mask])
                weighted_gini = (left_mask.sum() * left_gini + right_mask.sum() * right_gini) / n_samples
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, t

        return best_feat, best_thresh, best_gain

    def _grow_tree(self, X, y, depth):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        # Stopping conditions
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain <= 0:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_node = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=feat, threshold=thresh, left=left_node, right=right_node)

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    # Demo on a simple synthetic 2-class dataset
    rng = np.random.default_rng(0)
    n = 200
    class0 = rng.normal(loc=[2, 2], scale=1.0, size=(n // 2, 2))
    class1 = rng.normal(loc=[6, 6], scale=1.0, size=(n // 2, 2))
    X = np.vstack([class0, class1])
    y = np.hstack([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])

    # simple train/test split
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]

    tree = DecisionTreeClassifier(max_depth=4)
    tree.fit(X[train_idx], y[train_idx])

    print("Train accuracy:", tree.accuracy(X[train_idx], y[train_idx]))
    print("Test accuracy:", tree.accuracy(X[test_idx], y[test_idx]))
