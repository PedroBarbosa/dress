from abc import ABC, abstractmethod
import os
from typing import List, Union

from dress.datasetgeneration.dataset import Dataset, PairedDataset
from dress.datasetevaluation.off_the_shelf.classification import (
    Classification,
    DecisionTreeClassifier,
)
from dress.datasetevaluation.off_the_shelf.clustering import (
    Clustering,
    KMeans,
)
from dress.datasetevaluation.off_the_shelf.dimensionality_reduction import (
    DimensionalityReduction,
    PCA,
)


from dress.datasetevaluation.representation.sequences.repr import (
    SequenceRepresentation,
)

from dress.datasetevaluation.representation.phenotypes.repr import (
    PhenotypeRepresentation,
)

from dress.datasetevaluation.representation.motifs.repr import (
    MotifRepresentation,
)

from dress.datasetevaluation.plotting.plot_utils import buffered_ax
from dress.datasetgeneration.logger import setup_logger
from dress.datasetgeneration.os_utils import assign_proper_basename


class Evaluator(ABC):
    """Abstract class for the different evaluator schemes"""

    def __init__(
        self, dataset: Union[Dataset, PairedDataset], save_plots: bool = True, **kwargs
    ):  
        
        if "logger" in kwargs:
            self.logger = kwargs["logger"]
        else:
            self.logger = setup_logger(level=0)

        self.outdir = kwargs.get("outdir", None)
        if self.outdir:
            os.makedirs(self.outdir, exist_ok=True)
        self.outbasename = assign_proper_basename(
            kwargs.get("outbasename", "evaluation")
        )

        self.dataset = dataset
        self.save_plots = save_plots
        self.plots: dict = {}

    @abstractmethod
    def __call__(self):
        ...

    def _evaluate_via_off_the_shelf(
        self,
        data: List,
        repr: Union[
            SequenceRepresentation, PhenotypeRepresentation, MotifRepresentation
        ],
        classification: Classification,
        clustering: Clustering,
        dim_reduce: DimensionalityReduction,
    ):
        _dim_reduced_out = []
        if dim_reduce is not None:
            assert hasattr(dim_reduce, "estimator")
            self.logger.info(
                f"Dimensionality reduction method: {dim_reduce.__class__.__name__}"
            )
            self.logger.debug(
                f"Dimensionality reduction parameters: {dim_reduce.estimator.get_params()}"
            )

            for i, m in enumerate(data):
                g = repr.group if i == 0 else repr.group2
                dr = dim_reduce(m)
                fig = dr.plot()  # type: ignore

                if self.save_plots:
                    self.plots.update(
                        buffered_ax(
                            filename=f"{dim_reduce.__class__.__name__}_{g}.pdf", ax=fig
                        )
                    )

                _dim_reduced_out.append(dr)

        if clustering is not None:
            assert hasattr(clustering, "estimator")

            if isinstance(clustering.estimator, List):
                _params = clustering.estimator[0].get_params()
            else:
                _params = clustering.estimator.get_params()

            self.logger.info(f"Clustering method: {clustering.__class__.__name__}")
            self.logger.debug(f"Clustering parameters: {_params}")

            for i, m in enumerate(data):
                g = repr.group if i == 0 else repr.group2
                self.logger.debug(f"Clustering of dataset {g}")

                if dim_reduce:
                    _dim_reduced = _dim_reduced_out[i]
                else:
                    _dim_reduced = None
                clust = clustering(
                    data=m, dim_reduce=_dim_reduced, use_reduced_data=True
                )
                fig = clust.plot()  # type: ignore

                if self.save_plots:
                    self.plots.update(
                        buffered_ax(
                            filename=f"{clustering.__class__.__name__}_{g}.pdf", ax=fig
                        )
                    )

    def save_plots_in_pdf(self):
        """Save all plots to disk"""
        for _fn, buffer in self.plots.items():
            fn = os.path.join(
                self.outdir,
                f"{self.__class__.__name__}_{self.outbasename}{_fn}",
            )
            with open(fn, "wb") as f:
                f.write(buffer.getbuffer())
