from typing import List, Literal, Union

import pandas as pd
from dress.datasetgeneration.dataset import Dataset, PairedDataset
from dress.datasetgeneration.archive import Archive
from dress.datasetgeneration.logger import setup_logger
from dress.datasetgeneration.os_utils import assign_proper_basename


class ArchiveFilter:
    def __init__(
        self,
        dataset: Union[Dataset, PairedDataset],
        target_psi: float = None,
        target_dpsi: float = None,
        allowed_variability: float = 0.05,
        tag: Literal["higher", "lower", "equal"] = None,
        delta_score: float = 0.1,
        stack_datasets: bool = False,
        **kwargs,
    ):
        if isinstance(dataset, PairedDataset):
            self.info = [
                (dataset.dataset1.score, dataset.dataset1.group),
                (dataset.dataset2.score, dataset.dataset2.group),
            ]
        else:
            self.info = [(dataset.score, dataset.group)]

        self.dataset = dataset
        self.target_psi = target_psi
        self.target_dpsi = target_dpsi
        self.allowed_variability = allowed_variability
        self.tag = tag
        self.delta_score = delta_score
        self.stack_datasets = stack_datasets
        self.outdir = kwargs.get("outdir", "filter_outdir")
        self.outbasename = assign_proper_basename(kwargs.get("outbasename"))
        if "logger" in kwargs:
            self.logger = kwargs["logger"]
        else:
            self.logger = setup_logger(level=0)

    def filter(self) -> List[pd.DataFrame]:
        """Apply dataset filtering based on the provided options.

        Returns:
            List[pd.DataFrame]: Return a list of filtered dataset(s).
        """
        _data = self.dataset.data.copy()
        lists = []
        filtered = []
        if isinstance(self.dataset, PairedDataset):
            g1 = _data[_data.group == self.dataset.dataset1.group]
            g2 = _data[_data.group == self.dataset.dataset2.group]
            lists = [g1, g2]
        else:
            lists = [_data]

        for i, _df in enumerate(lists):
            self.logger.info(
                f"Filtering dataset {self.info[i][1]} with {len(_df)} sequences"
            )

            if self.target_psi:
                _filtered = self._by_psi(data=_df)
                if i == 0:
                    self.outbasename += f"psi{self.target_psi}"
            elif self.target_dpsi:
                if i == 0:
                    self.outbasename += f"dpsi{self.target_dpsi}"
                _filtered = self._by_dpsi(data=_df, original_score=self.info[i][0])
            elif self.tag:
                if i == 0:
                    self.outbasename += f"{self.tag}"
                _filtered = self._by_tag(data=_df, original_score=self.info[i][0])

            else:
                self.logger.error(
                    "No filter enabled. Either --target_psi, --target_dpsi or --tag must be provided."
                )

            if _filtered.empty:
                self.logger.warning(
                    f"No sequences found for dataset {self.info[i][1]} with the provided filtering options."
                )
            else:
                self.logger.info(
                    f"Filtered dataset has {len(_filtered)} sequences (min score={_filtered.Score.min()}, max score={_filtered.Score.max()})"
                )

                filtered.append(Dataset(_filtered, group=self.info[i][1]))

        if len(filtered) == 2:
            if self.stack_datasets:
                self.outbasename += "_stacked"
                self.filtered = PairedDataset(
                    dataset1=filtered[0], dataset2=filtered[1]
                )

            else:
                self.filtered = filtered
        else:
            self.filtered = filtered[0]

    def _by_psi(self, data):
        archive = Archive(dataset=data)
        _slice = archive[
            self.target_psi
            - self.allowed_variability : self.target_psi
            + self.allowed_variability
        ]
        return data[data.Phenotype.isin(_slice.ids)]

    def _by_dpsi(self, data, original_score):
        if (
            original_score + self.target_dpsi > 1
            or original_score + self.target_dpsi < 0
        ):
            raise ValueError(
                f"Target dPSI value {self.target_dpsi} is out of range for original score {original_score}."
            )

        archive = Archive(dataset=data)

        _slice = archive[
            original_score
            + self.target_dpsi
            - self.allowed_variability : original_score
            + self.target_dpsi
            + self.allowed_variability
        ]
        return data[data.Phenotype.isin(_slice.ids)]

    def _by_tag(self, data, original_score):
        archive = Archive(dataset=data)
        if self.tag == "lower":
            _slice = archive[: original_score - self.delta_score]

        elif self.tag == "higher":
            _slice = archive[original_score + self.delta_score :]

        elif self.tag == "equal":
            _slice = archive[
                original_score - self.delta_score : original_score + self.delta_score
            ]

        return data[data.Phenotype.isin(_slice.ids)]

    def write_output(self):
        """
        Writes filtered datasets to disk.

        If filering is applied on two datasets and 'stack_datasets' is 'False',
        the filtered datasets are written in separate files.
        """
        if not hasattr(self, "filtered"):
            raise ValueError("No filtered dataset available. Run filter() first.")

        # If two datasets to be written in in separate files
        if isinstance(self.filtered, list):
            for i, _df in enumerate(self.filtered):
                outfile = (
                    f"{self.outdir}/{self.outbasename}_group_{_df.group}_dataset.csv.gz"
                )
                _df.data.drop(columns=["id", "group"]).to_csv(
                    outfile,
                    index=False,
                    compression="gzip",
                )
        # If two datasets with 'stack_datasets' enabled
        elif isinstance(self.filtered, PairedDataset):

            g1 = self.filtered.dataset1.group
            g2 = self.filtered.dataset2.group
            outfile = f"{self.outdir}/{self.outbasename}_groups_{g1}_and_{g2}_dataset.csv.gz"
            self.filtered.data.drop(columns=["id"]).to_csv(
                outfile,
                index=False,
                compression="gzip",
            )

        # If a single dataset
        elif isinstance(self.filtered, Dataset):
     
            outfile = f"{self.outdir}/{self.outbasename}_group_{self.filtered.group}_dataset.csv.gz"
            self.filtered.data.drop(columns=["id", "group"]).to_csv(
                outfile,
                index=False,
                compression="gzip",
            )
