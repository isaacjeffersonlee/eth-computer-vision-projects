import numpy as np
import random
import math

# Import maxflow (PyMaxflow library) which can be used to solve min-cut problem
import maxflow

# Set seeds for random generators to get reproducible results
random.seed(0)
np.random.seed(0)


def perform_min_cut(
    unary_potential_foreground, unary_potential_background, pairwise_potential
):
    """
    We provide a simple fuction to perform min cut using PyMaxFlow library. You
    may use this function to implement your algorithm if you wish. Feel free to
    modify this function as desired, or implement your own function to perform
    min cut.

    args:
        unary_potential_foreground - A single channel NumPy array specifying the
            source (foreground) unary potentials for each pixel in the image
        unary_potential_background - A single channel NumPy array specifying the
            sink (background) unary potentials for each pixel in the image
        pairwise_potential - A single channel NumPy array specifying the pairwise
            potentials. We assume a graph where each pixel in the image is
            connected to its four neighbors (left, right, top, and bottom).
            Furthermore, we assume that the pairwise potential for all these 4
            edges are same, and set to the value of pairwise_potential at that
            pixel location
    """

    # create graph
    maxflow_graph = maxflow.Graph[float]()

    # add a node for each pixel in the image
    nodeids = maxflow_graph.add_grid_nodes(unary_potential_foreground.shape[:2])

    # Add edges for pairwise potentials. We use 4 connectivety, i.e. each pixel
    # is connected to its 4 neighbors (up, down, left, right). Also we assume
    # that pairwise potential for all these 4 edges are same
    # Feel free to change this if you wish
    maxflow_graph.add_grid_edges(nodeids, pairwise_potential)

    # Add unary potentials
    maxflow_graph.add_grid_tedges(
        nodeids, unary_potential_foreground, unary_potential_background
    )

    maxflow_graph.maxflow()

    # Get the segments of the nodes in the grid.
    mask_bg = maxflow_graph.get_grid_segments(nodeids)
    mask_fg = (1 - mask_bg.astype(np.uint8)) * 255

    return mask_fg


def pad_image(img, kernel_size):
    pad_vert = kernel_size[0] // 2
    pad_horiz = kernel_size[1] // 2
    return np.pad(
        array=img,
        pad_width=(pad_vert, pad_horiz),
        mode="constant",
        constant_values=(0, 0),
    )


def convolve(X, f, window_shape):
    windows = np.lib.stride_tricks.sliding_window_view(X, window_shape)
    n1, n2, n3, n4 = windows.shape
    windows = windows.reshape((n1 * n2, n3, n4))
    output = np.array([f(w) for w in windows])
    return output.reshape((n1, n2))


def get_mean_of_neighbors(X):
    M = X.copy()
    mask = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    M[np.where(mask == 0)] = 0
    return np.sum(M) / 4


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def get_init_centroids(X, k):
    """k-means++ recursive implementation for finding initial centroids.

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

    def get_centroids(X, p, centroids=[]):
        N = X.shape[0]
        if N == N_0 - k:
            return centroids
        else:
            centroid_row_idx = np.random.choice(N, size=1, p=p)
            centroid = X[centroid_row_idx, :]
            X_without_centroid = np.delete(X, centroid_row_idx, axis=0)
            ssd = np.sqrt(np.sum((X_without_centroid - centroid) ** 2, axis=1))
            ssd = (ssd - np.min(ssd)) / (
                np.max(ssd) - np.min(ssd)
            )  # Normalize to avoid overflow errors
            p = softmax(ssd**2)
            centroids.append(centroid)
            return get_centroids(X_without_centroid, p, centroids)

    p0 = np.ones(N_0) / N_0  # Initial uniform distribution
    return get_centroids(X, p=p0)


def k_means(X, k, max_iters=100):
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


def split_channels(X):
    X = np.array(X)
    channels = []
    if len(X.shape) == 4:
        # Split into channels
        for n, color_str in enumerate(["r", "g", "b"]):
            X_i = X.flat[n::3].reshape((X.shape[0], X.shape[1], X.shape[2]))
            channels.append(
                np.transpose(X_i.reshape((X_i.shape[0], X_i.shape[1] * X_i.shape[2])))
            )
    elif len(X.shape) == 3:
        # Split into channels
        for n, color_str in enumerate(["r", "g", "b"]):
            channels.append(X.flat[n::3].reshape((X.shape[0], X.shape[1])))
    else:
        raise NotImplementedError(
            f"get_2D_channel does not support matrices of shape: {X.shape}"
        )

    return channels


def intersect_or_union(im, im_box, union=False):
    non_zero_idx = im_box.nonzero()
    i_min = non_zero_idx[0].min()
    i_max = non_zero_idx[0].max()
    j_min = non_zero_idx[1].min()
    j_max = non_zero_idx[1].max()
    if union:
        template = im_box.copy()
        template[i_min : i_max + 1, j_min : j_max + 1] = im
        return template
    else:
        return im[i_min : i_max + 1, j_min : j_max + 1]


class ImageSegmenter:
    def __init__(self):
        pass

    def segment_image(self, im_rgb, im_aux, im_box):
        im_mask = im_aux.copy()
        num_iters = 1
        for i in range(num_iters):
            print(f"Graphcut iteration: {i} / {num_iters}")
            im_mask = intersect_or_union(im_mask, im_box)
            rgb_cutout = intersect_or_union(im_rgb, im_box)
            rgb_cutout[np.where(im_mask == 0)] = 0
            N = rgb_cutout.shape[0] * rgb_cutout.shape[1]  # Number of observations = Number of pixels
            rgb_cutout_2D = rgb_cutout.reshape((N, 3))
            labels = k_means(rgb_cutout_2D, k=2, max_iters=100)
            # Note: k_means finds a partition, but it doesn't know
            # what the partition represents, therefore it could be that
            # our 1's should actually be 0's and our 0's could actually be 1's
            # We want the labelling to be such that the 1's represent foreground and 0's represent background.
            # The easiest way to do this will be to look at the pixel with highest total intensity/ sum of (R, G, B) values
            # and check whether they have been correctly labeled.
            if labels[np.sum(rgb_cutout_2D, axis=1).argmax()] != 1:
                labels = -1 * (labels - 1)

            assert labels[np.sum(rgb_cutout_2D, axis=1).argmax()] == 1

            kmeans_mask = np.reshape(labels, im_mask.shape[:2]) * 255

            # For now pairwise potentials are independent of intensity, so don't need to
            # be calculated for each channel.
            pairwise_potential = convolve(kmeans_mask, get_mean_of_neighbors, (3, 3)) / 255
            pairwise_potential = pad_image(pairwise_potential, (3, 3))
            # pairwise_potential = np.ones_like(kmeans_mask)

            # rgb_cutout = intersect_or_union(im_rgb, im_box)
            channels = split_channels(rgb_cutout)
            channel_im_masks = []
            for channel in channels:
                # Calculate foreground unary potentials
                channel_fg = channel[np.where(kmeans_mask == 255)]
                channel_fg_hist, channel_fg_bin_edges = np.histogram(
                    channel_fg, density=True, bins=100
                )
                channel_fg_hist = np.concatenate(([0], channel_fg_hist, [0]))
                channel_prob_fg = channel_fg_hist[
                    np.searchsorted(channel_fg_bin_edges, channel, side="left")
                ]
                channel_unary_potential_foreground = -np.log(channel_prob_fg)
                # Calculate backgound unary potentials
                channel_bg = channel[np.where(kmeans_mask == 0)]
                channel_bg_hist, channel_bg_bin_edges = np.histogram(
                    channel_bg, density=True, bins=100
                )
                channel_bg_hist = np.concatenate(([0], channel_bg_hist, [0]))
                channel_prob_bg = channel_bg_hist[
                    np.searchsorted(channel_bg_bin_edges, channel)
                ]
                channel_unary_potential_background = -np.log(channel_prob_bg)
                # Perform min cut to get segmentation mask
                channel_im_mask = perform_min_cut(
                    channel_unary_potential_foreground,
                    channel_unary_potential_background,
                    pairwise_potential,
                )
                channel_im_mask = np.abs(channel_im_mask-255) * 255
                channel_im_mask = intersect_or_union(channel_im_mask, im_box, union=True)
                channel_im_masks.append(channel_im_mask)

            # Now we want to take the intersection of each channels mask,
            # along with the kmeans mask.
            im_mask = intersect_or_union(kmeans_mask, im_box, union=True).astype(np.int32)
            # for channel_im_mask in channel_im_masks:
            #     im_mask += channel_im_mask.astype(np.int32)
            # im_mask = im_mask / 4
            # im_mask[np.where(im_mask != 255)] = 0
            # im_mask = im_mask.astype(np.uint8)

        return im_mask


if __name__ == "__main__":
    X = np.array(
        [[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34], [41, 42, 43, 44]]
    )

    print(X)
    c = convolve(X, get_mean_of_neighbors, (3, 3))
    print(c)
