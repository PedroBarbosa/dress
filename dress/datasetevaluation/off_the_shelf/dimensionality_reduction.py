import abc
from typing import Union
import pandas as pd
from dress.datasetevaluation.plotting.plot_utils import scatterplot

from sklearn.decomposition import PCA as SklearnPCA
from sklearn.decomposition import TruncatedSVD as SklearnTruncatedSVD
from sklearn.manifold import TSNE as SklearnTSNE
import numpy as np
import matplotlib.pyplot as plt


class DimensionalityReduction(abc.ABC):
    def __init__(self, n_components: int) -> None:
        assert n_components >= 3, "n_components must be >= 3"
        self.n_components = n_components

    def __call__(
        self, data: Union[pd.DataFrame, np.ndarray]
    ) -> Union["DimensionalityReduction", None]:
        """Fit and transform a Dimensionality Reduction model

        Args:
            data (Union[pd.DataFrame, np.ndarray]): Dataset to reduce dimensionality.
        It expects the last column to be the labels.
        """
        self.y = (
            data.iloc[:, -1].values if isinstance(data, pd.DataFrame) else data[:, -1]
        )
        data = (
            data.iloc[:, :-1].values if isinstance(data, pd.DataFrame) else data[:, :-1]
        )

        self.reduced_data = self.estimator.fit_transform(data)  # type: ignore
        return self

    def plot(self, single: bool = False) -> plt.Figure:
        assert hasattr(self, "estimator")
        if single:
            fig, axes = plt.subplots(1, 1, figsize=(4, 3.25))
            ax1 = axes
        else:
            fig, axes = plt.subplots(
                1, 2, figsize=(8, 3.25), gridspec_kw={"width_ratios": [1.25, 1]}
            )
            ax1 = axes[0]

        ax1.set_xlabel("1st component")
        ax1.set_ylabel("2nd component")

        scatterplot(ax1, self.reduced_data[:, 0:2], self.y, True)

        if single is False:
            ax2 = axes[1]
            ax2.set_xlabel("2st component")
            ax2.set_ylabel("3nd component")

            scatterplot(ax2, self.reduced_data[:, 1:3], self.y, False)
            plt.subplots_adjust(wspace=0.3)
            plt.suptitle(
                f"{self.estimator.__class__.__name__} (n_components={self.n_components})"
            )

        else:
            plt.title(
                f"{self.estimator.__class__.__name__} (n_components={self.n_components})"
            )
        plt.tight_layout()
        return fig


class PCA(DimensionalityReduction):
    def __init__(self, n_components: int = 50, **kwargs) -> None:
        """Create a PCA model

        Args:
            dimensions (int, optional): Number of dimensions to reduce to. Defaults to 50.
            **kwargs: Arguments for the PCA model
        """
        super().__init__(n_components=n_components)
        self.estimator = SklearnPCA(n_components=self.n_components, **kwargs)


class TruncatedSVD(DimensionalityReduction):
    def __init__(self, n_components: int = 50, **kwargs) -> None:
        """Create a TruncatedSVD model

        Args:
            dimensions (int, optional): Number of dimensions to reduce to. Defaults to 50.
            **kwargs: Arguments for the TruncatedSVD model
        """
        super().__init__(n_components=n_components)
        self.estimator = SklearnTruncatedSVD(n_components=self.n_components, **kwargs)


class TSNE(DimensionalityReduction):
    def __init__(self, n_components: int = 3, **kwargs) -> None:
        """Create a TSNE model

        Args:
            dimensions (int, optional): Number of dimensions to reduce to. Defaults to 3.
            **kwargs: Arguments for the TSNE model
        """
        super().__init__(n_components=n_components)
        self.estimator = SklearnTSNE(n_components=self.n_components, **kwargs)
