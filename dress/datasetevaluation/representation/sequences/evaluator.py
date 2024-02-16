from typing import Union
from dress.datasetgeneration.dataset import Dataset, PairedDataset
from dress.datasetevaluation.evaluation import Evaluator
from dress.datasetevaluation.off_the_shelf.classification import (
    Classification,
    DecisionTreeClassifier,
)
from dress.datasetevaluation.off_the_shelf.clustering import Clustering, KMeans
from dress.datasetevaluation.off_the_shelf.dimensionality_reduction import (
    DimensionalityReduction,
    PCA,
)
from dress.datasetevaluation.representation.sequences.repr import SequenceRepresentation


class SequenceEvaluator(Evaluator):
    """
    Evaluate the quality of evolved dataset(s) based on the sequence level
    """

    def __init__(
        self,
        dataset: Union[Dataset, PairedDataset],
        save_plots: bool = True,
        **kwargs,
    ):
        super().__init__(dataset, save_plots, **kwargs)

        SequenceRepresentation.set_save_plots(self.save_plots)
        self.repr = SequenceRepresentation(dataset=self.dataset, **kwargs)

    def __call__(
        self,
        classification: Classification | None = DecisionTreeClassifier(),
        clustering: Clustering
        | None = KMeans(n_clusters=[4, 5], random_state=0, n_init=10),
        dim_reduce: DimensionalityReduction | None = PCA(n_components=50),
    ):
        k_mer_counts = self.repr.kmer_count_matrix(k=6, reduce_dims=True)

        self._evaluate_via_off_the_shelf(
            k_mer_counts, self.repr, classification, clustering, dim_reduce
        )
        if self.save_plots:
            self.save_plots_in_pdf()
