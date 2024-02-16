from typing import List, Union
from matplotlib import pyplot as plt


from dress.datasetevaluation.representation.api import SingleInstance


class SingleSequenceMotifs(SingleInstance):
    def __init__(
        self,
        singe_seq_motifs,
        id: str,
        score: float,
        splice_site_positions: str,
        seed: int,
        group: str,
        original_seq: str,
        original_ss_idx: List[list],
    ):
        """
        Structured representation of a single individual's motif ocurrences

        Args:
            singe_seq_motifs (str): Phenotype of a single individual
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
 
        self.motifs = (
            singe_seq_motifs.groupby("RBP_name")
            .agg({"Start": list, "End": list})
            .reset_index()
            .set_index("RBP_name")
            .to_dict(orient="index")
        )

    def __repr__(self) -> str:
        return f"Sequence ID {self.id} with hits for {len(self.motifs)} different RBPs"

    def visualize(self) -> Union[dict, plt.Figure]:
        """
        Visualize the motif ocurrences of a single individual

        Returns:
            Union[dict, plt.Figure]: Dictionary with a buffer object containing the plot
        to be written to disk or a matplotlib Figure object
        """
        ...
