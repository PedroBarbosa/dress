from typing import List, Union

import pandas as pd
from dress.datasetgeneration.dataset import Dataset, PairedDataset
import abc

from dress.datasetgeneration.logger import setup_logger


class Representation(object):
    save_plots = True

    def __init__(self, dataset: Union[Dataset, PairedDataset], **kwargs):
        """
        Base class for the different dataset representation schemes
        """

        self.dataset = dataset
        if "logger" in kwargs:
            self.logger = kwargs["logger"]
        else:
            self.logger = setup_logger(level=0)

        if isinstance(self.dataset, PairedDataset):
            self.is_paired = True
            self.sequence = self.dataset.dataset1.sequence
            self.sequence_size = self.dataset.dataset1.sequence_size
            self.splice_sites = self.dataset.dataset1.splice_sites_list
            self.group = self.dataset.dataset1.group

            self.sequence2 = self.dataset.dataset2.sequence
            self.sequence_size2 = self.dataset.dataset2.sequence_size
            self.group2 = self.dataset.dataset2.group
            self.splice_sites2 = self.dataset.dataset2.splice_sites_list

        else:
            self.is_paired = False
            self.sequence = self.dataset.sequence
            self.sequence_size = self.dataset.sequence_size
            self.group = self.dataset.group
            self.splice_sites = self.dataset.splice_sites_list

    @abc.abstractmethod
    def create_representation(self) -> List:
        ...

    @classmethod
    def set_save_plots(cls, value: bool):
        cls.save_plots = value


class SingleInstance(object):
    save_plots = True

    def __init__(
        self,
        id: str,
        score: float,
        splice_site_positions: str,
        seed: int,
        group: str,
        original_seq: str | None = None,
        original_ss_idx: List[list] | None = None,
    ):
        """
        Base class for representing a single instance in a dataset

        Args:
            id (str): ID of a single individual
            score (float): Model score of a single individual
            splice_site_positions (str): Splice site positions of mutated sequence
            seed (int): Seed used in an evolution procedure to generate the individual
            group (str): Group of the Dataset to which the individual belong
            original_seq (str, optional): Original sequence which was mutated to generate the individual
            original_ss_idx (List[list], optional): List of splice site positions of the original sequence
        """
        self.id = id
        self.score = score
        self.seed = seed
        self.group = group
        self.splice_sites = self.parse_splice_sites(splice_site_positions)
        self.original_seq_size = len(original_seq) if original_seq is not None else None
        self.original_splice_sites = (
            original_ss_idx if original_ss_idx is not None else None
        )

    def parse_splice_sites(self, ss_idx_string: str) -> List[list]:
        """
        Extract splice sites in the sequence

        Args:
            ss_idx_string (str): Flat string of splice site indexes

        Returns:
            List[list]: Position of splice sites
        """
        _l = [int(x) if x != "<NA>" else pd.NA for x in ss_idx_string.split(";")]
        return [_l[0:2], _l[2:4], _l[4:6]]

    @classmethod
    def set_save_plots(cls, value: bool):
        cls.save_plots = value
