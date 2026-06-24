"""
PCA from Scratch + Face Recognition Mid-Term Project
=======================================================
Week 3 - ML From Scratch (WNCC, IIT Bombay)

Part 1: PCA implemented from scratch using eigen-decomposition of the
        covariance matrix.
Part 2: Apply it to face images to build "eigenfaces" and use them to
        reconstruct / recognize faces with far fewer dimensions than
        the original pixel space.
"""

import numpy as np
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None  # eigenvectors (principal axes)

    def fit(self, X):
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean

        # Covariance matrix
        cov = np.cov(X_centered, rowvar=False)

        # Eigen-decomposition (covariance matrix is symmetric -> eigh)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Sort eigenvectors by descending eigenvalue
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        self.explained_variance_ = eigvals[: self.n_components]
        self.components = eigvecs[:, : self.n_components]
        return self

    def transform(self, X):
        return (X - self.mean) @ self.components

    def inverse_transform(self, Z):
        return Z @ self.components.T + self.mean

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def demo_face_recognition():
    """
    Generates a small set of synthetic "face-like" images (no internet
    download needed) to demonstrate eigenfaces: each face is a base pattern
    plus per-face variation, similar in spirit to real face datasets where
    PCA captures the dominant modes of variation across faces.
    """
    rng = np.random.default_rng(7)
    img_size = 32
    n_faces = 40

    # Build a base "face" pattern (simple radial gradient) + structured noise
    yy, xx = np.mgrid[0:img_size, 0:img_size]
    center = img_size / 2
    base_face = -((xx - center) ** 2 + (yy - center) ** 2) / (img_size ** 2)

    faces = []
    for _ in range(n_faces):
        variation = rng.normal(0, 0.05, size=(img_size, img_size))
        # random shift to simulate different "identities"
        shift = rng.normal(0, 0.1)
        face = base_face + variation + shift
        faces.append(face.flatten())
    X = np.array(faces)

    n_components = 10
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(Z)

    print(f"Reduced {X.shape[1]} pixels -> {n_components} principal components")
    total_var = np.var(X, axis=0).sum()
    print(f"Variance captured by top {n_components} components: "
          f"{pca.explained_variance_.sum() / total_var:.2%}")

    # Plot original vs reconstructed for a few sample faces
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes[0, i].imshow(X[i].reshape(img_size, img_size), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(X_reconstructed[i].reshape(img_size, img_size), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original", loc="left")
    axes[1, 0].set_title(f"Reconstructed ({n_components} PCs)", loc="left")
    plt.tight_layout()
    plt.savefig("pca_face_reconstruction.png", dpi=150)
    print("Saved pca_face_reconstruction.png")


if __name__ == "__main__":
    # Simple 2D sanity-check demo of PCA itself
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2)) @ np.array([[3, 1], [1, 0.5]])  # correlated 2D data

    pca = PCA(n_components=1)
    Z = pca.fit_transform(X)
    print("Top principal component direction:", pca.components.flatten())
    print("Explained variance:", pca.explained_variance_)

    # Face recognition mid-term demo (downloads a small dataset if available)
    demo_face_recognition()
