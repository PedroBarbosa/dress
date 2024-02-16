import re
from typing import List, Tuple, Union
from matplotlib import pyplot as plt


from dress.datasetevaluation.plotting.plot_utils import (
    buffered_ax,
    draw_gene_structure,
)

from dress.datasetevaluation.representation.api import SingleInstance


class SinglePhenotype(SingleInstance):
    def __init__(
        self,
        phenotype: str,
        id: str,
        score: float,
        splice_site_positions: str,
        seed: int,
        group: str,
        original_seq: str,
        original_ss_idx: List[list],
    ):
        """
        Structured representation of a single individual's phenotype

        Args:
            phenotype (str): Phenotype of a single individual
            id (str): ID of a single individual
            score (float): Model score of a single individual
            splice_site_positions (str): Splice site positions of mutated sequence
            seed (int): Seed used in an evolution procedure to generate the individual
            group (str): Group of the Dataset to which the individual belong
            original_seq (str): Original sequence which was mutated to generate the individual
            original_ss_idx (List[list]): List of splice site positions of the original sequence
        """

        super().__init__(
            id,
            score,
            splice_site_positions,
            seed,
            group,
            original_seq,
            original_ss_idx,
        )

        self.phenotype = phenotype
        self.positions, self.perturbations = self.parse_phenotypes(
            phenotype, original_seq
        )

    def __repr__(self) -> str:
        return self.phenotype

    def parse_phenotypes(
        self,
        phenotype_str: str,
        original_seq: str,
    ) -> Tuple[List[int], List[Tuple[str, int, str]]]:
        """
        Returns a list of mutated positions from a single phenotype
        by taking into account the type of DiffUnit represented

        Args:
            phenotype_str (str): Phenotype of a single individual
            original_seq (str): Original sequence which was mutated to generate the individual
        Returns:
            List[int]: List of mutated positions
        """
        positions, with_type = [], []
        for diff_unit in phenotype_str.split("|"):
            numbers = [int(x) for x in re.findall(r"\d+", diff_unit)]

            if diff_unit.startswith("SNV") or diff_unit.startswith("RandomInsertion"):
                positions.append(numbers[0])
                nucleotide = diff_unit.split(",")[1][:-1]
                with_type.append((diff_unit.split("[")[0], numbers[0], nucleotide))

            elif diff_unit.startswith("RandomDeletion"):
                assert len(numbers) == 2
                positions.extend(list(range(numbers[0], numbers[1] + 1)))
                deleted_nucleotides = original_seq[numbers[0] : numbers[1] + 1]
                with_type.append(("RandomDeletion", numbers[0], deleted_nucleotides))

            else:
                raise ValueError(f"Unknown DiffUnit: {diff_unit}")

        return positions, with_type  # type: ignore

    def visualize(self) -> Union[dict, plt.Figure]:
        """
        Visualize the phenotype of a single individual

        Returns:
            Union[dict, plt.Figure]: Dictionary with a buffer object containing the plot
        to be written to disk or a matplotlib Figure object
        """
        _, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax, _ = draw_gene_structure(
            ss_idx=self.original_splice_sites,
            seq_len=self.original_seq_size,
            perturbations=self.perturbations,
            zoom_in=False,
        )

        plt.title(f"ID: {self.id} | Score: {self.score}")
        plt.tight_layout()

        if self.save_plots:
            filename = f"phenotype_g{self.group}_seed_{self.seed}_ind{self.id}.pdf"
            return buffered_ax(filename=filename, ax=ax)

        else:
            return ax
