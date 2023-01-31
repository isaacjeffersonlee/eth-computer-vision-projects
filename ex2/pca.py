import numpy as np


def get_pc(X: np.ndarray, n_components: int = None) -> np.ndarray:
    """Get the first k principal components of the array X.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Data, where the rows of the data represent observations and
        the columns represent features/variables.
        Note: For this implementation, we will drop any rows
        of X that have missing values.
    n_components : Optional[int]
        The number of principal components to return.
        So e.g if n_components=5,
        then the first 5 (ordered by magnitude of eigenvalues),
        principal components would be returned.
        If n_components is None, i.e not specified,
        then all principal components will be returned.

    Returns
    -------
    np.ndarray, shape (N, n_components)
        The array of principal component, with the ith column being the ith
        principal component.
    """
    X = X[~np.isnan(X).any(axis=1), :]  # Remove rows with missing values
    X = X - X.mean(axis=0)
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    return X @ Vh.T[:, :n_components]


def main():
    # | time | feat_1 | feat_2 | feat_3 |
    # Note: the rows of our array will represent an observation,
    # and the columns represent a variable/feature
    X = np.array([[15, 12, 16, 18], [28, 23, 22, 21], [39, 30, 30, 32]])
    print(get_pc(X, 2))


if __name__ == "__main__":
    main()
