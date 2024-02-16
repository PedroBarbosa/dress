from typing import List, Union
from matplotlib import pyplot as plt

import pandas as pd

from dress.datasetgeneration.dataset import Dataset, PairedDataset
from dress.datasetevaluation.plotting.plot_utils import (
    buffered_ax,
    create_gridSpec,
    draw_gene_structure,
    draw_position_dist,
)
import numpy as np

from dress.datasetevaluation.representation.api import Representation
from dress.datasetevaluation.representation.phenotypes.single_phenotype import (
    SinglePhenotype,
)


class PhenotypeRepresentation(Representation):
    def __init__(self, dataset: Union[Dataset, PairedDataset], **kwargs):
        """
        Structured representation of phenotypes

        Args:
            dataset (Union[Dataset, PairedDataset]): Dataset to extract phenotypes from
        """
        super().__init__(dataset, **kwargs)
        self.instances = self.create_representation()

    def create_representation(self) -> List:
        """
        Create structured representation of phenotypes
        """

        def _individual_repr(
            row: pd.Series,
        ) -> SinglePhenotype:
            if row.group == self.group:
                original_seq = self.sequence
                original_ss_idx = self.splice_sites

            elif row.group == self.group2:
                original_seq = self.sequence2
                original_ss_idx = self.splice_sites2

            else:
                raise ValueError("Group not found")

            return SinglePhenotype(
                row.Phenotype,
                row.id,
                row.Score,
                row.Splice_site_positions,
                row.Seed,
                row.group,
                original_seq=original_seq,
                original_ss_idx=original_ss_idx,
            )

        return list(self.dataset.data.apply(_individual_repr, axis=1))

    def phenotypes2df(self) -> pd.DataFrame:
        """
        Convert List of IndividualPhenotypes into a single dataframe
        """

        _data = [
            {
                "id": ind.id,
                "score": ind.score,
                "seed": ind.seed,
                "group": ind.group,
                "positions": ind.positions,
                "n_perturbations": len(ind.perturbations),
                "splice_sites": ind.splice_sites,
            }
            for ind in self.repr
        ]

        return pd.DataFrame(_data)

    def binary_matrix(self) -> List[pd.DataFrame]:
        """
        Create a binary matrix where positions with perturbations are filled with 1s.
        Last position of the matrix will refer to the model score

        Returns:
            List[pd.DataFrame]: Binary matrix of positions with perturbations, one per Dataset group
        """

        output = []
        if self.is_paired:
            # + 2 to account for an insertion after the last position and the score
            m = np.zeros((len(self.dataset.dataset1), self.sequence_size + 1 + 1))  # type: ignore
            m2 = np.zeros((len(self.dataset.dataset2), self.sequence_size2 + 1 + 1))  # type: ignore
        else:
            m = np.zeros((len(self.dataset), self.sequence_size + 1 + 1))
            m2 = None

        for ind in self.repr:
            if self.is_paired:
                if ind.group == self.group:
                    m[ind.id, ind.positions] = 1
                    m[ind.id, -1] = ind.score

                elif ind.group == self.group2:
                    m2[ind.id, ind.positions] = 1  # type: ignore
                    m2[ind.id, -1] = ind.score  # type: ignore
                else:
                    raise ValueError("Unexpected group")
            else:
                m[ind.id, ind.positions] = 1
                m[ind.id, -1] = ind.score

        for _m in [m, m2]:
            if _m is not None:
                _df = pd.DataFrame(m)
                _df.rename(columns={_df.columns[-1]: "score"}, inplace=True)
                output.append(_df.loc[:, (_df != 0).any(axis=0)])
        return output

    def visualize(
        self,
        split_effects: bool = False,
        split_seeds: bool = False,
        zoom_in: bool = False,
    ) -> Union[dict, plt.Figure]:
        """
        Visualize distribution of mutated positions across each dataset

        Args:
            split_effects (bool, optional): Whether to split the density plots
        so that individuals with positive and negative change in splicing are
        plotted separately. Defaults to False.
            split_seeds (bool, optional): Whether to split the density plots so
        that sequences generated from different seeds are plotted separately.
        Defaults to True.
            zoom_in (bool, optional): Whether to zoom in around the cassetet exon
        (-50bp, exon +50bp).

        Returns:
            Union[dict, plt.Figure]: Dictionary with a buffer object containing the plot
        to be later written to disk
        """

        _different_seq = False
        if self.dataset.ngroups == 2:
            assert hasattr(self.dataset, "dataset1")
            assert hasattr(self.dataset, "dataset2")

            if self.dataset.dataset1.id != self.dataset.dataset2.id:
                _different_seq = True

        fig, axes = create_gridSpec(
            n_groups=self.dataset.ngroups, is_different_original_seq=_different_seq
        )
        df = self.phenotypes2df()
        datasets = (
            [self.dataset.dataset1, self.dataset.dataset2]  # type: ignore
            if self.dataset.ngroups == 2
            else [self.dataset]
        )

        for i, dataset in enumerate(datasets):
            _ax = (i * 2) + 1 if _different_seq else 1
            _ax_to_draw = i * 2
            if i == 0 or (_different_seq and i == 1):
                _, _ = draw_gene_structure(
                    ax=axes[_ax],
                    ss_idx=dataset.splice_sites_list,
                    seq_len=dataset.sequence_size,
                    score=dataset.score,
                    zoom_in=zoom_in,
                )

            _df = df[df.group == dataset.group]
            draw_legend = True if i == 0 else False

            draw_position_dist(
                ax=axes[_ax_to_draw],
                df=_df,
                group_name=dataset.group,  # type: ignore
                seq_id=dataset.id,
                seq_len=dataset.sequence_size,
                ss_idx=dataset.splice_sites_list,
                score=dataset.score,
                split_effects=split_effects,
                split_seeds=split_seeds,
                draw_exon_boundaries=True,
                draw_legend=draw_legend,
                zoom_in=zoom_in,
            )

        if self.save_plots:
            filename = f"phenotypes_g{self.group}"
            if self.dataset.ngroups == 2:
                filename += f"_vs_g{self.group2}"
            if split_effects:
                filename += "_split_effects"
            if split_seeds:
                filename += "_split_seeds"
            if zoom_in:
                filename += "_zoom_in"

            filename += "_positions.pdf"
            return buffered_ax(filename=filename, ax=axes)

        else:
            return fig
