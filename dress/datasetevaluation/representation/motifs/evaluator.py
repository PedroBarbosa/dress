from typing import Union, Literal
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
from dress.datasetevaluation.representation.motifs.repr import MotifRepresentation
from dress.datasetevaluation.representation.motifs.search import (
    BiopythonSearch,
    FimoSearch,
    PlainSearch,
)
from dress.datasetevaluation.representation.motifs.enrichment import (
    StremeEnrichment,
)

MOTIF_COUNTS_TO_USE = {"gene": 0, "motif": 1}


MOTIF_SEARCH_OPTIONS = {
    "fimo": FimoSearch,
    "plain": PlainSearch,
    "biopython": BiopythonSearch,
}

MOTIF_ENRICHMENT_OPTIONS = {
    "streme": StremeEnrichment,
}


class MotifEvaluator(Evaluator):
    """
    Evaluate the quality of evolved dataset(s) based on motif occurrences
    """

    def __init__(
        self,
        dataset: Union[Dataset, PairedDataset],
        save_plots: bool = True,
        disable_motif_representation: bool = False,
        **kwargs,
    ):
        super().__init__(dataset, save_plots, **kwargs)
        data = self.dataset.data
        self.counts_to_use = kwargs.get("motif_counts", "gene")

        motif_searcher = MOTIF_SEARCH_OPTIONS.get(kwargs.get("motif_search"))
        self.motif_search = motif_searcher(dataset=data, skip_location_mapping=True, **kwargs)
        self.motif_counts = self.motif_search.tabulate_occurrences(write_output=True)

        motif_enricher = MOTIF_ENRICHMENT_OPTIONS.get(kwargs.get("motif_enrichment"))
        self.motif_enrichment = motif_enricher(data, **kwargs)

        if not disable_motif_representation:
            MotifRepresentation.set_save_plots(self.save_plots)
            self.repr = MotifRepresentation(
                dataset=self.dataset, motif_search_obj=self.motif_search, **kwargs
            )

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
        it uses counts per RBP. If 'rbp_motif', it uses counts per each motif of each RBP. Defaults to "rbp_name".
        """
        if not hasattr(self, "repr"):
            self.logger.error(
                "Motif representation must be enabled to perform off-the-shelf evaluation"
            )
            exit(1)

        _data = self.motif_counts[MOTIF_COUNTS_TO_USE[self.counts_to_use]]
        if (
            "group" in self.rbp_counts.columns
            and len(self.rbp_counts.group.unique()) == 2
        ):
            counts_grp = _data.groupby("group")
            _data = [group for _, group in counts_grp]

        else:
            _data = [_data]

        [df.drop(columns=["Seq_id", "Delta_score"], inplace=True) for df in _data]
        self._evaluate_via_off_the_shelf(
            _data, self.repr, classification, clustering, dim_reduce
        )

        if self.save_plots:
            self.save_plots_in_pdf()
