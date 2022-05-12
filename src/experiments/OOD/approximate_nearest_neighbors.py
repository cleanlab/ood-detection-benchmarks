import numpy as np
from annoy import AnnoyIndex
from typing import Tuple


class ApproxNearestNeighbors:
    """

    Approximate nearest neighbor search is done with Annoy using binary trees. This is blazing fast.

    See link below for details on the implementation:
    https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html

    Parameters
    ----------
    features: Array of shape (n_samples, n_features)
        Features for each observation.

    labels: Array of shape (n_samples,)
        Given class label indices.

    features_key: Array of shape (n_samples,)
        Optional unique key for each observation (e.g. image paths for images)

    """

    def __init__(
        self, features: np.array, labels: np.array = None, features_key: np.array = None
    ):

        self.ann_index = None  # init empty approximate nearest neighbor index
        self.features = features
        self.labels = labels
        self.features_key = features_key
        self.num_labels = np.unique(labels).shape[0]  # number of unique labels
        self.num_obs = features.shape[0]  # number of observations

        # init nearest neighbors
        self.neighbors_idx = None
        self.neighbors_dist = None
        self.neighbors_labels = None

        # init predicted probabilities
        self.pred_probs = None

    def build_index(self, metric: str = "angular", n_trees: int = 10):
        """
        Build approximate nearest neighbors index.

        Parameters
        ----------
        metric: {"angular", "euclidean", "manhattan", "hamming", "dot"}, default="angular"
            Distance metric for approximate nearest neighbor search.

        n_trees: int, default=10
            Number of trees to use for building the index. More trees will give more accurate results but larger indexes.
            Annoy uses binary trees. See Annoy documentation for more details.
            https://github.com/spotify/annoy

        """

        # dimension of feature space
        dim = self.features.shape[1]

        # build approximate nearest neighbor index
        print("Building nearest neighbors index")
        self.ann_index = AnnoyIndex(dim, metric)
        for i, x in enumerate(self.features):
            self.ann_index.add_item(i, x)
        self.ann_index.build(n_trees)

        return self.ann_index

    def get_k_nearest_neighbors(self, k: int = 25) -> Tuple[np.array]:
        """
        Get the k nearest neighbors for each observation.

        Parameters
        ----------
        k: int, default=25
            Number of nearest neighbors to search for.

        Returns
        -------
        k_nearest_neighbors: Tuple of arrays of shape (n_samples, n_neighbors)
            First array contains the indices for the nearest neighbors.
            Second array contains the distances to the nearest neighbors.
            Third array contains the labels for the nearest neighbors.
            Fourth array contains the weights for the nearest neighbors.

        """

        # get k nearest neighbors for each observation
        neighbors_idx = []
        neighbors_dist = []
        for i in range(self.features.shape[0]):

            # note: we need to do k + 1 because the first neighbor for each observation is itself
            idx, dist = self.ann_index.get_nns_by_item(i, k + 1, include_distances=True)

            # save neighbors to list
            # note: we exclude the first neighbor because the first neighbor for each observation is itself
            neighbors_idx.append(idx[1:])
            neighbors_dist.append(dist[1:])

        # convert to numpy array
        self.neighbors_idx = np.array(neighbors_idx)
        self.neighbors_dist = np.array(neighbors_dist)

        # get labels for neighbors
        self.neighbors_labels = self.__neighbors_idx_to_label(self.neighbors_idx)

        return (
            self.neighbors_idx,
            self.neighbors_dist,
            self.neighbors_labels,
        )

    def __neighbors_idx_to_label(self, neighbors_idx: np.array) -> np.array:
        """
        Map indices of neighbors to labels.

        Parameters
        ----------
        neighbors_idx: Array of shape (n_samples, n_neighbors)
            Indices of neighbors for each observation.

        Returns
        -------
        neighbors_labels: Array of shape (n_samples, n_neighbors)
            Labels of neighbors for each observation.

        """

        return np.vectorize(lambda idx: self.labels[idx])(neighbors_idx)

    def save_ann_index(self, file_path: str = "./index.ann"):
        """Save approximate nearest neighbor index"""
        self.ann_index.save(file_path)

    def load_ann_index(
        self, dim: int, metric: str = "angular", file_path: str = "./index.ann"
    ):
        """Load approximate nearest neighbor index"""
        self.ann_index = AnnoyIndex(dim, metric)
        self.ann_index.load(file_path)
