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
from dress.datasetevaluation.representation.phenotypes.repr import (
    PhenotypeRepresentation,
)


class PhenotypeEvaluator(Evaluator):
    """
    Evaluate the quality of evolved dataset(s) based on the individual phenotype
    """

    def __init__(
        self,
        dataset: Union[Dataset, PairedDataset],
        save_plots: bool = False,
        **kwargs,
    ):
        super().__init__(dataset, save_plots, **kwargs)

        PhenotypeRepresentation.set_save_plots(self.save_plots)
        self.repr = PhenotypeRepresentation(dataset=self.dataset, **kwargs)

    def __call__(
        self,
        classification: Classification | None = DecisionTreeClassifier(),
        clustering: Clustering
        | None = KMeans(n_clusters=[4, 5], random_state=0, n_init=10),
        dim_reduce: DimensionalityReduction | None = PCA(n_components=50),
    ):
        """
        Function that calls any evaluation method availabe

        Args:
            classification (Union[Classification, None], optional): Defaults to DecisionTreeClassifier().
            clustering (Union[Clustering, None], optional): Defaults to KMeans(n_clusters=[4,5]).
            dim_reduce (Union[DimensionalityReduction, None], optional): Defaults to PCA(n_components=50).
        """

        matrices = self.repr.binary_matrix()

        self._evaluate_via_off_the_shelf(
            matrices, self.repr, classification, clustering, dim_reduce
        )

        if self.save_plots:
            self.save_plots_in_pdf()
