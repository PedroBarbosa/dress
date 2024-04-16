from typing import List
from dress.datasetevaluation.representation.api import SingleInstance


class SingleSequence(SingleInstance):
    def __init__(
        self,
        sequence: str,
        id: str,
        score: float,
        splice_site_positions: str,
        seed: int,
        group: str,
        original_seq: str | None = None,
        original_ss_idx: List[list] | None = None,
    ):
        """
        Structured representation of a single individual's sequence

        Args:
            sequence (str): Sequence of a single individual
            id (str): ID of a single individual
            score (float): Model score of a single individual
            splice_site_positions (str): Splice site positions of mutated sequence
            seed (int): Seed used in an evolution procedure to generate the individual
            group (str): Group of the Dataset to which the individual belong
            original_seq (str): Original sequence which was mutated to generate the individual
            original_ss_idx (List[list]): List of splice site positions of the original sequence
        """
        super().__init__(
            id=id,
            score=score,
            splice_site_positions=splice_site_positions,
            seed=seed,
            group=group,
            original_seq=original_seq,
            original_ss_idx=original_ss_idx,
        )

        self.sequence = sequence

    def __repr__(self) -> str:
        return f"Sequence with {len(self.sequence)} nucleotides"

    # def visualize(self) -> dict:
    #     """
    #     Visualize the sequence of a single individual
    #     """
    #     pass
