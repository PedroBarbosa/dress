import abc
from typing import List
from typing import Iterable, Union
import pandas as pd
from dress.datasetevaluation.plotting.plot_utils import (
    silhouette_plot,
    y_density_plot,
)

from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerativeClustering
from scipy.spatial.distance import hamming
import numpy as np
from dress.datasetevaluation.off_the_shelf.dimensionality_reduction import (
    DimensionalityReduction,
)
from matplotlib import pyplot as plt


class Clustering(abc.ABC):
    def __init__(self, n_clusters: Union[int, Iterable[int]] = 5) -> None:
        self.n_clusters = n_clusters

    def __call__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        dim_reduce: DimensionalityReduction | None,
        use_reduced_data: bool = False,
    ) -> Union["Clustering", None]:
        """Fit a Clustering model for each cluster

        Args:
            data (Union[pd.DataFrame, np.ndarray]): Data to cluster
            dim_reduce (DimensionalityReduction, optional): Dimensionality reduction model to use when visualizing the \
                clusters in 2D. Defaults to PCA(dimensions=2).
            use_reduced_data (bool, optional): Use the lower dimensional data to perform the clustering. Defaults to \
                False.
        """
        self.X = (
            data.iloc[:, :-1].values if isinstance(data, pd.DataFrame) else data[:, :-1]
        )
        self.y = (
            data.iloc[:, -1].values if isinstance(data, pd.DataFrame) else data[:, -1]
        )
        self.data = self.X

        if dim_reduce is not None:
            self.reduced_data = dim_reduce.reduced_data

            if use_reduced_data:
                self.data = self.reduced_data

        self.fit(self.data)
        return self

    def fit(self, data: np.ndarray) -> None:
        """Fit an estimator (or list of estimators) to the data

        Args:
            data (Union[List, np.ndarray]): Data to fit the estimator to
        """
        if not hasattr(self, "estimator"):
            raise ValueError("The 'estimator' attribute should be defined.")

        if isinstance(self.estimator, List):
            for estimator in self.estimator:
                if (
                    isinstance(estimator, SklearnAgglomerativeClustering)
                    and estimator.metric == "precomputed"
                ):
                    data = self.custom_distance_matrix(data)
                estimator.fit(data)
        else:
            if isinstance(self.estimator, SklearnAgglomerativeClustering) and self.metric == "precomputed":  # type: ignore
                data = self.custom_distance_matrix(data)
            self.estimator.fit(data)

    def custom_distance_matrix(
        self, X: np.ndarray, metric: str = "hamming"
    ) -> np.ndarray:
        """Compute distance matrix to give as intput to the fit method when
        metric is set to "precomputed"

        Args:
            X (np.ndarray): Data array (with original or reduced dimensions)
            metric (str): Custom distance metric to use. Defaults to "hamming".

        Returns:
            np.ndarray: Pairwise distance matrix
        """
        if metric != "hamming":
            raise ValueError("Only hamming distance is supported for now")

        distance_matrix = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(i + 1, X.shape[0]):
                distance_matrix[i, j] = hamming(X[i], X[j])
                distance_matrix[j, i] = distance_matrix[i, j]
        return distance_matrix

    def plot(self) -> plt.Figure:
        """ """

        def _plot(subfigs, data, y, estimator, n_clusters):
            subfigs[0].suptitle(
                f"Silhouette plot for {estimator.__class__.__name__}(k={n_clusters})"
            )
            axleft = subfigs[0].subplots(1, 1)
            ax1, avg_silhouette = silhouette_plot(axleft, data, estimator, n_clusters)  # type: ignore

            subfigs[1].suptitle(f"Scores per cluster (k={n_clusters})")
            axsRight = subfigs[1].subplots(n_clusters, 1, sharex=True)
            for i, cluster_id in enumerate(range(n_clusters - 1, -1, -1)):
                mask = estimator.labels_ == cluster_id
                _y = y[mask]

                y_density_plot(
                    axsRight[i], y=_y, cluster_id=cluster_id, n_clusters=n_clusters
                )

        assert hasattr(self, "estimator")
        if isinstance(self.n_clusters, int):
            fig = plt.figure(layout="constrained", figsize=(7, 3.5))
            subfigs = fig.subfigures(1, 2, wspace=0.05)
            _plot(subfigs, self.data, self.y, self.estimator, self.n_clusters)
        else:
            n_rows = len(self.n_clusters)  # type: ignore
            fig = plt.figure(layout="constrained", figsize=(10, 8 * n_rows // 2))
            subfigs = fig.subfigures(n_rows, 2, wspace=0.05)

            for i, (_n_clusters, _estimator) in enumerate(
                zip(self.n_clusters, self.estimator)
            ):
                _plot(subfigs[i], self.data, self.y, _estimator, _n_clusters)

        return fig


class KMeans(Clustering):
    def __init__(self, n_clusters: Union[int, Iterable[int]] = 5, **kwargs) -> None:
        """Create a kmeans model

        Args:
            n_clusters (Union[int, Iterable[int]], optional): Number of clusters, can be an iterable. Defaults to 5.
        """
        super().__init__(n_clusters)

        if isinstance(self.n_clusters, int):
            self.estimator = SklearnKMeans(n_clusters=self.n_clusters, **kwargs)
        else:
            self.estimator = [
                SklearnKMeans(n_clusters=k, **kwargs) for k in self.n_clusters  # type: ignore
            ]


class AgglomerativeClustering(Clustering):
    def __init__(
        self,
        n_clusters: Union[int, Iterable[int]] = 5,
        metric: str = "euclidean",
        **kwargs,
    ) -> None:
        """Create a AgglomerativeClustering model

        Args:
            n_clusters (Union[int, Iterable[int]], optional): Number of clusters, can be an iterable. Defaults to 5.
            metric (str, optional): Metric to use. If "precomputed", it will compute a distance matrix using the \
                hamming distance metric.
        """
        super().__init__(n_clusters)
        self.metric = metric

        if isinstance(self.n_clusters, int):
            self.estimator = SklearnAgglomerativeClustering(
                n_clusters=self.n_clusters, metric=metric, **kwargs
            )
        else:
            self.estimator = [
                SklearnAgglomerativeClustering(n_clusters=k, metric=metric, **kwargs) for k in self.n_clusters  # type: ignore
            ]
