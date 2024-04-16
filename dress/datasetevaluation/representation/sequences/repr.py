from collections import defaultdict
import itertools
from multiprocessing import Pool
from typing import List, Union
import numpy as np

import pandas as pd
import tqdm

from dress.datasetgeneration.dataset import Dataset, PairedDataset

from dress.datasetevaluation.representation.api import Representation
from dress.datasetevaluation.representation.sequences.single_sequence import (
    SingleSequence,
)


class SequenceRepresentation(Representation):
    def __init__(self, dataset: Union[Dataset, PairedDataset], **kwargs):
        """
        Structured representation of sequences

        Args:
            dataset (Union[Dataset, PairedDataset]): Dataset to extract sequences from
        """

        super().__init__(dataset, **kwargs)
        self.instances = self.create_representation()

    def create_representation(self) -> List:
        """
        Create structured representation of sequences
        """

        def _individual_repr(
            row: pd.Series,
        ) -> SingleSequence:
            return SingleSequence(
                row.Sequence,
                row.id,
                row.Score,
                row.Splice_site_positions,
                row.Seed,
                row.group,
            )

        return list(self.dataset.data.apply(_individual_repr, axis=1))

    def kmer_count_matrix(
        self, k: int = 6, reduce_dims: bool = True
    ) -> List[pd.DataFrame]:
        """
        Create a matrix with k-mer counts.
        Last position of the matrix will refer to the SpliceAI score

        Args:
            k (int): K-mer size. Defaults to 6.
            reduce_dims (bool): Whether to reduce the dimensions of the matrix by removing k-mers with the same counts in all sequences rows. Defaults to True.
        Returns:
            List[pd.DataFrame]: K-mer count matrix, one per Dataset group
        """
        assert k < 9, "Max k-mer size allowed is 8"
        grouped_sequences = defaultdict(list)
        out = []
        for seq in self.repr:
            group = seq.group
            grouped_sequences[group].append(seq)

        possible_kmers = ["".join(x) for x in itertools.product("ATCG", repeat=k)]
        pool = Pool()

        for group, seqs_per_group in grouped_sequences.items():
            results = pool.starmap(
                self._count_kmers,
                tqdm.tqdm(
                    zip(seqs_per_group, itertools.repeat(possible_kmers)),
                    total=len(seqs_per_group),
                ),
            )

            _df = pd.DataFrame(np.vstack(results), columns=possible_kmers + ["Score"])
            if reduce_dims:
                unique_counts = _df.nunique()
                _to_drop = unique_counts[unique_counts == 1].index
                _df = _df.drop(columns=_to_drop)
            out.append(_df)

        return out

    def _count_kmers(self, seq, possible_kmers):
        n_kmers = len(possible_kmers)
        counts = np.zeros((1, n_kmers + 1))
        for j, kmer in enumerate(possible_kmers):
            counts[0, j] = seq.sequence.count(kmer)
            counts[0, -1] = seq.score
        return counts

    def visualize(self):
        """
        Visualize a multiple sequence alignment across the sequences of the dataset
        """
        ...
