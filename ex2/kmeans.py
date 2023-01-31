# An implementation of lloyds algorithm
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x))


def get_init_centroids(X: np.ndarray, k: int) -> list[np.ndarray]:
    """
    k-means++ recursive implementation for finding initial centroids.

    For more info: https://en.wikipedia.org/wiki/K-means%2B%2B

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        The data to partition, where each row of X
        represents a vector/observation to classify.
        Note: We assume X has no missing values.
    k : int
        Number of clusters.

    Returns
    -------
    list[np.ndarray]
        List of initial centroids.
    """
    N_0 = X.shape[0]

    def get_centroids(
        X: np.ndarray, p: np.ndarray, centroids: list[np.ndarray] = []
    ) -> list[np.ndarray]:
        N = X.shape[0]
        if N == N_0 - k:
            return centroids
        else:
            centroid_row_idx = np.random.choice(N, size=1, p=p)
            centroid = X[centroid_row_idx, :]
            X_without_centroid = np.delete(X, centroid_row_idx, axis=0)
            ssd = np.sqrt(np.sum((X_without_centroid - centroid) ** 2, axis=1))
            ssd = (ssd - np.min(ssd)) / (np.max(ssd) - np.min(ssd))  # Normalize to avoid overflow errors
            p = softmax(ssd**2)
            centroids.append(centroid)
            return get_centroids(X_without_centroid, p, centroids)

    p0 = np.ones(N_0) / N_0  # Initial uniform distribution
    return get_centroids(X, p=p0)


def k_means(X: np.ndarray, k: int, max_iters: int = 100) -> np.ndarray:
    """Perform the k-means clustering algorithm.

    Uses k-means++ algorithm to initialize centroids
    then Lloyds' algorithm to partition N data
    observations into k samples.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        The data to partition, where each row of X
        represents a vector/observation to classify.
        Note: We drop any rows of X that have missing values.
    k : int
        Number of clusters.

    Returns
    -------
    np.ndarray, shape (N,)
        A 1D array of classification labels, where the ith
        element is an integer label, classifying which cluster
        the ith observation, (i.e ith row of X) belongs to.
    """
    # TODO: Play around with scaling/standard scaling rows of X.
    if len(X.shape) == 1:  # 1 Dimensional array
        X = X.reshape((-1, 1))  # Add second dimension

    N_0, d = X.shape
    X = X[~np.isnan(X).any(axis=1), :]  # Remove rows with missing values
    centroids = get_init_centroids(X, k)  # Get initial centroids using k-means++
    for i in range(max_iters):  # TODO: Add convergence metric
        ssd = (X - np.array(centroids)) ** 2
        # assert k == ssd.shape[0]
        # assert (N_0, d) == ssd[0].shape
        ssd = ssd.reshape((k * N_0, d))
        ssd = np.sqrt(np.sum(ssd, axis=1))
        ssd = ssd.reshape((k, N_0)).T
        labels = np.argmin(ssd, axis=1)
        new_centroids = []
        for idx, centroid in enumerate(list(centroids)):
            mask = np.asarray(labels == idx)
            X_in_cluster = X[mask, :]
            centroid = np.mean(X_in_cluster, axis=0).reshape((1, -1))
            new_centroids.append(centroid)

        centroids = new_centroids

    return labels


def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pca

    df = pd.read_csv("../Data/iris.csv", skiprows=1, names=["feat1", "feat2", "class"])
    df = df.reset_index()
    true_labels = np.array(df["class"])
    df = df.drop(columns=["class"])
    X = np.array(df)
    Z = pca.get_pc(X, 2)
    labels = k_means(Z, k=3, max_iters=10)
    z1 = Z.T[0]
    z2 = Z.T[1]
    fig, ax = plt.subplots(nrows=2, ncols=1)
    g1 = sns.scatterplot(x=z1, y=z2, hue=labels, ax=ax[0])
    g2 = sns.scatterplot(x=z1, y=z2, hue=true_labels, ax=ax[1])
    plt.show()


if __name__ == "__main__":
    main()
