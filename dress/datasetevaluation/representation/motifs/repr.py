from typing import List, Union
from matplotlib import pyplot as plt

import pandas as pd

from dress.datasetgeneration.dataset import Dataset, PairedDataset
from dress.datasetevaluation.representation.motifs.single_sequence_motifs import (
    SingleSequenceMotifs,
)

from dress.datasetevaluation.representation.api import Representation
from dress.datasetevaluation.representation.motifs.search import MotifSearch
from pandarallel import pandarallel


class MotifRepresentation(Representation):
    def __init__(
        self,
        dataset: Union[Dataset, PairedDataset],
        motif_search_obj: MotifSearch,
        **kwargs
    ):
        """
        Structured representation of motifs occurences in a dataset.

        Args:
            dataset (Union[Dataset, PairedDataset]): Dataset to extract motifs from
            motif_search_obj: MotifSearch: MotifSearch object containing the results of the motif search
        """
        super().__init__(dataset, **kwargs)
        self.motif_search_obj = motif_search_obj
        self.logger.info("Creating per-instance motifs representation")
        self.instances = self.create_representation()

    def create_representation(self) -> List:
        """
        Create structured representation of motifs
        """
        
        _motifs = self.motif_search_obj.motif_results.copy()
        cols_to_keep = [
            "Seq_id",
            "RBP_name",
            "Start",
            "End",
        ]  #  RBP_motif, RBP_name_motif
        _motifs = _motifs[cols_to_keep]

        def _individual_repr(
            row: pd.Series,
        ) -> SingleSequenceMotifs:
            if row.group == self.group:
                original_seq = self.wt_sequence
                original_ss_idx = self.wt_splice_sites

            elif row.group == self.group2:
                original_seq = self.wt_sequence2
                original_ss_idx = self.wt_splice_sites2

            else:
                raise ValueError("Group not found")

            return SingleSequenceMotifs(
                _motifs[_motifs.Seq_id == row.Seq_id],
                row.id,
                row.Score,
                row.Splice_site_positions,
                row.Seed,
                row.group,
                original_seq=original_seq,
                original_ss_idx=original_ss_idx,
            )
        
        pandarallel.initialize(progress_bar=True, verbose=0)
        print()
        return list(self.dataset.data.parallel_apply(_individual_repr, axis=1))


    def visualize(
        self,
    ) -> Union[dict, plt.Figure]:
        """
        Visualize motif ocurrences across each dataset

        Args:

        Returns:
            Union[dict, plt.Figure]: Dictionary with a buffer object containing the plot
        to be later written to disk
        """

        ...
