from itertools import chain
import itertools
from multiprocessing import Pool
import os
from pathlib import Path
import re
import subprocess
from dress.datasetgeneration.os_utils import assign_proper_basename
import numpy as np
import pandas as pd
from typing import Tuple, Union
from collections import defaultdict
from dress.datasetgeneration.logger import setup_logger
from tqdm import tqdm
from Bio import motifs
from Bio.Seq import Seq
from pandarallel import pandarallel
from dress.datasetevaluation.representation.motifs.rbp_lists import RBP_SUBSETS

from dress.datasetevaluation.representation.motifs.utils import (
    _get_loc_of_motif,
    _pwm_to_ambiguous,
    _pwm_to_unambiguous,
    _get_unique_pwms,
    _dataset_to_fasta,
    _process_subset_argument,
    _read_meme,
    _redundancy_and_density_analysis,
)


class MotifSearch:
    def __init__(self, dataset, **kwargs):
        """ """

        self.dataset = dataset
        if "logger" in kwargs:
            self.logger = kwargs["logger"]
        else:
            self.logger = setup_logger(level=0)

        self.motif_search = kwargs.get("motif_search", "fimo")
        self.motif_db = kwargs.get("motif_db", "ATtRACT")
        self.subset_rbps = _process_subset_argument(kwargs.get("subset_rbps", "encode"))
        self.min_motif_length = kwargs.get("min_motif_length", 5)
        self.min_nucleotide_probability = kwargs.get("min_nucleotide_probability", 0.15)
        self.skip_raw_motifs_filtering = kwargs.get("skip_raw_motifs_filtering", False)
        self.skip_location_mapping = kwargs.get("skip_location_mapping", False)
        self.outdir = os.path.join(kwargs["outdir"], "motifs")
        os.makedirs(self.outdir, exist_ok=True)
        self.outbasename = assign_proper_basename(kwargs.get("outbasename"))

        to_flat = False if self.motif_search == "biopython" else True
        if self.motif_db in ["rosina2017", "encode2020_RBNS"]:
            self.motifs = self._read_rosina()

        else:
            (
                self.motifs,
                self.pwm_ids_per_rbp,
                self.motif_db_file,
            ) = self._read_PWMs(to_flat=to_flat)

        self.subset_RBPs_in_motif_database()

    def scan(): ...

    def filter_raw_output(self, raw_hits: pd.DataFrame) -> pd.DataFrame:
        """
        Filters motif results

        Args:
            raw_hits(pd.DataFrame): Results from motif scanning

        Returns:
            pd.DataFrame: Filtered df with additional information
        """
        if raw_hits.empty:
            raise ValueError("No motifs found given this experimental setup.")

        self.logger.info("Filtering motif results")

        if len(raw_hits.Seq_id.unique()) > 50:
            groups = raw_hits.groupby("Seq_id")

            with Pool() as pool:
                results = pool.starmap(
                    _redundancy_and_density_analysis,
                    tqdm(
                        [(group, self.motif_search, False) for _, group in groups],
                        total=len(groups),
                    ),
                )

                df = pd.concat(results)

        else:
            df = _redundancy_and_density_analysis(raw_hits, self.motif_search, log=True)

        self.logger.info("Done. {} hits kept".format(df.shape[0]))
        return df

    def add_motif_location(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds additional annotations about the motif location
        in respect to the exon triplet given as input

        Args:
            df(pd.DataFrame): Results from motif scanning (raw filtered or not)

        Returns:
            pd.DataFrame: Df with additional information about location and distances
            to splice sites
        """
        self.logger.info("Mapping location of motifs")

        def _process_single_sequence(group: pd.DataFrame, dataset: pd.DataFrame):
            seq_id = group.iloc[0].Seq_id
            single_seq = dataset[dataset.Seq_id == seq_id]
            return _get_loc_of_motif(group, single_seq)

        if len(df.Seq_id.unique()) > 10:
            pandarallel.initialize(progress_bar=True, verbose=0)
            df = (
                df.groupby("Seq_id")
                .parallel_apply(_process_single_sequence, dataset=self.dataset)
                .reset_index(drop=True)
            )
        else:
            df = (
                df.groupby("Seq_id")
                .apply(_process_single_sequence, dataset=self.dataset)
                .reset_index(drop=True)
            )

        return df

    def tabulate_occurrences(self, write_output: bool = True) -> Tuple[pd.DataFrame]:
        """
        Generate counts of each motif in tabular format as well
        as a summary of motif gains/losses compared to the original sequence.


        If 'group' column exist in the main dataset, it is
        added to the output so that further comparisons
        can be performed downstream.

        Returns:
            Tuple(pd.DataFrame): Tuple with four dataframes:
                -motif counts per gene (RBP)
                -motif counts for each possible motif within each gene (RBP)
                -summary of gains/losses for each RBP
                -summary of gains/losses for each motif of each RBP
        """

        def _counts_per_rbp(col_to_use: str, all_values: list) -> pd.DataFrame:
            counts = self.motif_results.pivot_table(
                index="Seq_id", columns=col_to_use, aggfunc="size", fill_value=0
            )
            counts = counts.reindex(columns=all_values, fill_value=0)
            cols = ["Seq_id", "Score", "Delta_score"]
            if "group" in self.dataset.columns and self.dataset["group"].nunique() > 1:
                cols.append("group")

            return pd.merge(
                counts, self.dataset[cols], how="left", on="Seq_id"
            )


        def _get_gain_loss_summary(df: pd.DataFrame) -> pd.DataFrame:
                
            def _get_rbp_count(row: pd.Series): 
                pos, neg = [], []

                for rbp, c in row[row != 0].items():
                    if c == 1:
                        pos.append(f"{rbp}")
                    elif c == -1:
                        neg.append(f"{rbp}")
                    elif c > 1:
                        pos.append(f"{rbp}_x{c}")
                    elif c < -1:
                        neg.append(f"{rbp}_x{abs(c)}")
                return ';'.join(pos), ';'.join(neg)
            
            df_counts = df.copy()
            df_counts.set_index("Seq_id", inplace=True)
            rbp_cols = [
                c
                for c in df_counts.columns
                if c not in ["Score", "Delta_score", "group"]
            ]
            reference_row = df_counts.iloc[0]

            differences = df_counts[rbp_cols].sub(reference_row[rbp_cols], axis=1).astype(int)
            n_df = pd.concat([differences[differences > 0].sum(axis=1), 
                               differences[differences < 0].sum(axis=1).abs()], axis=1).astype(int)
            n_df.columns = ['N_gained', 'N_lost']

            differences[['RBP_gained', 'RBP_lost']] = differences.apply(_get_rbp_count, axis=1, result_type='expand')
            res_df = n_df.merge(differences[['RBP_gained', 'RBP_lost']], on='Seq_id')

            merge_cols = ["Score", "Delta_score"]
            if "group" in df_counts.columns:
                merge_cols.append("group")
            return res_df.merge(
                df_counts[merge_cols], left_index=True, right_index=True
            ).reset_index()

        # List all possible rbps
        all_rbps = list(self.motifs.keys())
        all_rbps_detailed = []

        if self.motif_search == "biopython":
            for rbp, uniq_pwms in self.motifs_uniq.items():
                for pwm in uniq_pwms:
                    all_rbps_detailed.append(
                        rbp
                        + "_"
                        + _pwm_to_ambiguous(
                            pwm=pwm, min_probability=self.min_nucleotide_probability
                        )
                    )
        else:
            for rbp, uniq_flat_motifs in self.motifs.items():
                for motif in uniq_flat_motifs:
                    all_rbps_detailed.append(rbp + "_" + motif)

        # Count occurrences per RBP and per individual RBP motif
        rbp_counts_list = []
        for col, all_values in zip(
            ["RBP_name", "RBP_name_motif"], [all_rbps, all_rbps_detailed]
        ):
            rbp_counts_list.append(_counts_per_rbp(col, all_values))
     
        # Generate summary of motif gains/losses
        summary_list = []
        import time
        start = time.time()
        for counts_df in rbp_counts_list:
            summary_list.append(_get_gain_loss_summary(counts_df))

        if write_output:
            self.motif_results.to_csv(
                self.outdir + "/{}MOTIF_MATCHES.tsv.gz".format(self.outbasename),
                compression="gzip",
                sep="\t",
                index=False,
            )
            # Motif counts
            for key, df in {
                "RBP_COUNTS": rbp_counts_list[0],
                "MOTIF_COUNTS": rbp_counts_list[1],
                "RBP_SUMMARY": summary_list[0],
                "MOTIF_SUMMARY": summary_list[1],
            }.items():
     
                df.to_csv(
                    self.outdir + f"/{self.outbasename}{key}.tsv.gz",
                    compression="gzip",
                    sep="\t",
                    index=False,
                )

        return rbp_counts_list[0], rbp_counts_list[1], summary_list[0], summary_list[1]

    def _read_rosina(self) -> dict:
        """
        Reads additional file 1 from rosina et al 2017
        paper and returns a dictionary with all the non-ambiguous
        motifs for each RBP

        :return dict:
        """
        if self.motif_db == "rosina2017":
            file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "db/")
            motifs = open(file_path + "rosina2017_motifs.txt", "r")

        elif self.motif_df == "encode2020_RBNS":
            file_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "db/RBNS_Encode/"
            )
            motifs = open(file_path + "encode2020_RBNS_motifs.txt", "r")

        out = {}
        too_short = []
        for line in motifs:
            line = line.rstrip()
            if line.startswith(">"):
                rbp_name = line[1:].split("|")[0]
                _too_short = set()
                flat_good = set()

            elif line.startswith("*"):
                if rbp_name in out.keys():
                    raise ValueError("Repeated RBP name in file ({}).".format(rbp_name))

                for m in line[1:].split("|"):
                    if len(m) >= self.min_motif_length:
                        flat_good.update({m})

                    else:
                        _too_short.update({})

                out[rbp_name] = list(flat_good)
                too_short.extend(list(_too_short))

            if line.startswith("MOUSE"):
                break

        self.logger.debug(
            "Number of motifs removed due to short size (< {}): {}".format(
                self.min_motif_length, len(too_short)
            )
        )

        return out

    def _read_PWMs(self, to_flat: bool = True, file_format: str = "meme") -> tuple:
        """
        Reads a database of PWMs and possible
        decomposes each PWM into all unambiguous
        sequences, if `to_flat` is set to `True`

        Args:
            file_format (str): Format of the PWM file
            to_flat (bool): Convert PWM matrices into flat,
        non-redundant motifs
            file_format (str): Format of the motif database.

        Returns:
            dict: Dict with all the valid non-ambiguous
        sequences for each RBP
            dict: Dict with all the PWM IDs for each RBP
            str: Path to the PWM file based on the motif database set
        """
        final = defaultdict(list)

        if self.motif_db == "oRNAment":
            db = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "db/oRNAment/oRNAment_PWMs_database.txt",
            )

        elif self.motif_db == "ATtRACT":
            db = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "db/ATtRACT/ATtRACT_PWMs_database.txt",
            )

        elif self.motif_db == "cisBP_RNA":
            db = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "db/cisBP_RNA/cisBP_RNA_PWMs_database.txt",
            )

        elif os.path.isfile(self.motif_db):
            self.logger.info("Custom PWM file given. It must be in MEME format")
            db = self.motif_db

        # Get PWMs per RBP represented as np.arrays
        if file_format == "meme":
            self.logger.log(
                "INFO",
                "Loading and processing PWM file from {} source".format(self.motif_db),
            )
            motifs = _read_meme(db)

        else:
            raise NotImplementedError("Only MEME db format is allowed for now.")

        if to_flat:
            self.logger.log(
                "DEBUG",
                "Generating unambiguous sequences using {} "
                "as the minimum nucleotide probability".format(
                    self.min_nucleotide_probability
                ),
            )

            # Convert PWMs to unambiguous sequences
            too_short = []

            for rbp_name, _motifs in motifs.items():
                per_rbp_motifs, _too_short = set(), set()

                for pwm in _motifs:
                    _flat = _pwm_to_unambiguous(pwm, self.min_nucleotide_probability)
                    flat_good = set()
                    for x in _flat:
                        if len(x) >= self.min_motif_length:
                            flat_good.add(x)
                        else:
                            _too_short.add(x)
                    per_rbp_motifs.update(flat_good)

                too_short.extend(list(_too_short))
                final[rbp_name] = list(per_rbp_motifs)

            self.logger.log(
                "DEBUG",
                "Number of motifs removed due to short size (< {}): {}".format(
                    self.min_motif_length, len(too_short)
                ),
            )

        else:
            final = motifs

        pwd_ids_per_RBP = {
            rbp_name: set().union(*(pwm.keys() for pwm in pwms))
            for rbp_name, pwms in motifs.items()
        }

        return final, pwd_ids_per_RBP, db

    def subset_RBPs_in_motif_database(self) -> None:
        """
        Subset motifs to scan based on gene names in `subset_rbps` variable.
        """

        if self.subset_rbps:
            if isinstance(self.subset_rbps, str):

                if self.subset_rbps in RBP_SUBSETS.keys():
                    self.subset_rbps = RBP_SUBSETS[self.subset_rbps]

                else:
                    if self.subset_rbps not in self.motifs.keys():
                        raise ValueError(
                            "RBP '{}' not found in the {} database.".format(
                                self.subset_rbps, self.motif_db
                            )
                        )
                    self.subset_rbps = [self.subset_rbps]

            absent = [x for x in self.subset_rbps if x not in self.motifs.keys()]

            if absent:
                if len(absent) == len(self.subset_rbps):
                    raise ValueError(
                        'None of the RBPs provided in the "--subset_rbps" argument is present in the {} database.'.format(
                            self.motif_db
                        )
                    )

                self.logger.log(
                    "WARNING",
                    "Some RBPs provided are not in the {} database (N={}):'{}'.".format(
                        self.motif_db, len(absent), ",".join(absent)
                    ),
                )

            if self.motif_db not in ["rosina2017", "encode2020_RBNS"]:
                self.motifs, self.pwm_ids_per_rbp = (
                    dict((k, v) for k, v in d.items() if k in self.subset_rbps)
                    for d in (self.motifs, self.pwm_ids_per_rbp)
                )
            else:
                self.motifs = {
                    k: v for k, v in self.motifs.items() if k in self.subset_rbps
                }


class PlainSearch(MotifSearch):
    """
    Motif search using plain string matching
    """

    def __init__(self, dataset: pd.DataFrame, subset_rbps: Union[str, list], **kwargs):
        super().__init__(dataset=dataset, subset_rbps=subset_rbps, **kwargs)
        assert self.motif_search == "plain"
        self.motif_results = self.scan()

    def scan(self):
        """
        Scan motif occurrences by blind
        substring search in fasta sequence.

        Additionally, if ref_ss_idx is provided and the number
        of different motifs to scan is higher than 20,
        sequences to be spanned will be shortened to the max
        length of the model resolution

        :param dict seqs: Input sequences to be scanned
        :param dict motifs: Motifs to scan
        :param dict ref_ss_idx: Splice site indexes. Used to
        restrict motif scanning space within sequences
        :return pd.DataFrame: Df with positions in the
        sequences where each motif was found with 0-based coordinates
        :return pd.DataFrame: Df with the counts of each
        RBP on each sequence
        :return pd.DataFrame: Df with the counts of each
        motif of each RBP on each sequence
        """
        self.logger.log(
            "INFO",
            f"Searching motifs in {self.dataset.shape[0]} sequences using a plain search",
        )

        def _scan(row: pd.Series, motifs: dict):
            res = []
            # for each RBP
            for rbp_name, _motifs in motifs.items():
                # matches are 0-based
                matches = [[_ for _ in re.finditer(m, row.Sequence)] for m in _motifs]

                # for each motif match
                for motif_seq, positions in zip(_motifs, matches):
                    if positions:
                        # Explode matches:
                        [
                            res.append(
                                [
                                    row.Seq_id,
                                    rbp_name,
                                    motif_seq,
                                    p.start(),
                                    p.end(),
                                    rbp_name + "_" + motif_seq,
                                ]
                            )
                            for p in positions
                        ]
            return res

        if self.dataset.shape[0] > 10:
            pandarallel.initialize(progress_bar=True, verbose=0)
            res = self.dataset.parallel_apply(_scan, motifs=self.motifs, axis=1)
            print()
        else:
            res = self.dataset.apply(_scan, motifs=self.motifs, axis=1)

        res = list(itertools.chain(*res))
        df = pd.DataFrame.from_records(
            res,
            columns=[
                "Seq_id",
                "RBP_name",
                "RBP_motif",
                "Start",
                "End",
                "RBP_name_motif",
            ],
        )

        self.logger.log("INFO", "Done. {} hits found".format(df.shape[0]))
        if not self.skip_raw_motifs_filtering:
            df = self.filter_raw_output(df)

        if not self.skip_location_mapping:
            df = self.add_motif_location(df)

        return df


class FimoSearch(MotifSearch):
    """
    Motif search using FIMO software
    """

    def __init__(self, dataset: pd.DataFrame, subset_rbps: Union[str, list], **kwargs):
        """
        Set up a FIMO search using FIMO.

        Args:
            **kwargs (dict): Additional arguments, including:
                - pvalue_threshold (float): only keep motif matches below p-value threshold. Default: `0.00005`
                - qvalue_threshold (float): only keep motif matches below q-value threshold. Default: `None`
        """
        super().__init__(dataset, subset_rbps=subset_rbps, **kwargs)
        assert self.motif_search == "fimo"
        assert self.motif_db not in [
            "rosina2017",
            "encode2020_RBNS",
        ], f"{self.motif_db} motif database allowed only when '--motif_search is 'plain'"

        self.pvalue_threshold = kwargs.get("pvalue_threshold", 0.0001)
        self.qvalue_threshold = kwargs.get("qvalue_threshold", None)
        self.motif_results = self.scan()

    def scan(self):
        """
        Scan for motif ocurrences using FIMO.

        If subset of RBPs is provided,
        intermediary PWM files will be
        created by selecting only
        the PWMs that belong to the subset
        of RBPs provided.

        :return pd.DataFrame: 0-based motif ocurrences
        """

        base_cmd = [
            "fimo",
            "--norc",
            "--thresh",
            str(self.pvalue_threshold),
            "--bfile",
            "--motif--",
        ]
        if self.qvalue_threshold is None:
            base_cmd.append("--no-qvalue")

        # If subset by at least one RBP, update the arg list
        # of FIMO to just use the PWM belonging to those RBPs.
        if self.subset_rbps:
            subset_RB = list(
                chain(
                    *[self.pwm_ids_per_rbp[rbp_name] for rbp_name in self.motifs.keys()]
                )
            )

            _aux = ["--motif"] * len(subset_RB)
            add_cmd = list(chain(*zip(_aux, subset_RB)))
            base_cmd.extend(add_cmd)

        def _scan(row: pd.Series, outdir: str, motif_db_file: str):
            fasta = row.fasta

            fimo_outdir = os.path.join(os.path.join(outdir, "fimo"), Path(fasta).stem)

            os.makedirs(fimo_outdir, exist_ok=True)
            fimo_cmd = base_cmd + ["--oc", fimo_outdir, motif_db_file, fasta]

            _p = subprocess.run(
                fimo_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            try:
                df_fimo_out = pd.read_csv(
                    os.path.join(fimo_outdir, "fimo.tsv"),
                    comment="#",
                    sep="\t",
                    low_memory=False,
                )
                rename_cols = {
                    "sequence_name": "Seq_id",
                    "matched_sequence": "RBP_motif",
                    "motif_alt_id": "RBP_name",
                    "start": "Start",
                    "stop": "End",
                }

                ordered_cols = [
                    "Seq_id",
                    "RBP_name",
                    "RBP_motif",
                    "Start",
                    "End",
                    "p-value",
                    "q-value",
                ]
                df_fimo_out.rename(columns=rename_cols, inplace=True)

                df = df_fimo_out[ordered_cols].drop_duplicates(
                    ["Seq_id", "RBP_name", "RBP_motif", "Start", "End"]
                )

                return df

            except pd.errors.EmptyDataError:
                return

        self.logger.debug("Base command to run FIMO: {}".format(" ".join(base_cmd))
        )

        fimo_outdir = self.outdir + "/fimo"
        os.makedirs(fimo_outdir, exist_ok=True)

        if self.dataset.shape[0] > 10:
            self.logger.info(
                "Writing sequences to multiple files to parallelize FIMO search"
            )
            fasta_files = _dataset_to_fasta(
                self.dataset, outdir=fimo_outdir, chunk_size=50
            )
            fasta_df = pd.DataFrame(fasta_files, columns=["fasta"])
            self.logger.info("Running..")
            pandarallel.initialize(progress_bar=True, use_memory_fs=False, verbose=0)
            res = fasta_df.parallel_apply(
                _scan, outdir=self.outdir, motif_db_file=self.motif_db_file, axis=1
            )
            print()
        else:
            fasta_files = _dataset_to_fasta(self.dataset, fimo_outdir)
            fasta_df = pd.DataFrame([fasta_files], columns=["fasta"])
            res = fasta_df.apply(
                _scan, outdir=self.outdir, motif_db_file=self.motif_db_file, axis=1
            )

        res = res.dropna()
        if res.empty:
            self.logger.warning(
                "No motif hits found using FIMO against {} database".format(
                    self.motif_db
                ),
            )
            raise ValueError

        df = pd.concat(list(res))
        df["RBP_name_motif"] = df.RBP_name + "_" + df.RBP_motif

        # Remove matches that include positions with
        # nucleotides with a frequency below the min_prob
        all_possible_features = []
        [
            all_possible_features.extend([k + "_" + x for x in v])
            for k, v in self.motifs.items()
        ]

        _n = df.shape[0]
        df = df[df.RBP_name_motif.isin(all_possible_features)]

        self.logger.debug(
            "Number of hits removed due to the minimum nucleotide probability "
            "threshold set ({}): {}".format(
                self.min_nucleotide_probability, _n - df.shape[0]
            ),
        )

        # Remove matches not passing the q-value threshold
        _n = df.shape[0]
        if self.qvalue_threshold is not None:

            df = df[df["q-value"] <= self.qvalue_threshold]
            self.logger.debug(
                "Number of hits removed due to the q-value "
                "threshold set ({}): {}".format(
                    self.qvalue_threshold, _n - df.shape[0]
                ),
            )

            if df.empty:
                self.logger.warning(
                    'No remaining FIMO hits after applying "qvalue_threshold"',
                )
                raise ValueError
            
        df.Start -= 1

        if isinstance(fasta_files, list):
            [os.remove(f) for f in fasta_files]
        else:
            os.remove(fasta_files)

        df = df.drop_duplicates(["Seq_id", "RBP_name", "Start", "End"], keep="first")
        self.logger.log("INFO", "Done. {} hits found".format(df.shape[0]))

        if not self.skip_raw_motifs_filtering:
            df = self.filter_raw_output(df)

        if not self.skip_location_mapping:
            df = self.add_motif_location(df)

        return df


class BiopythonSearch(MotifSearch):
    """
    Motif search using biopython Bio.motifs package with
    position-specific scoring matrices (PSSM)
    """

    def __init__(self, dataset: pd.DataFrame, subset_rbps: Union[str, list], **kwargs):
        """
        Set up a Biopython search.

        Args:
            **kwargs (dict): Additional arguments, including:
                - pssm_threshold (float): threshold to consider a sequence position
            with high log-odds scores against a motif. Default: `3`
        """
        super().__init__(dataset, subset_rbps=subset_rbps, **kwargs)
        assert self.motif_search == "biopython"
        self.motifs_uniq = _get_unique_pwms(self.motifs)
        self.pssm_threshold = kwargs.get("pssm_threshold", 3)
        self.estimate_best_thr = kwargs.get("just_estimate_pssm_threshold", False)

        if self.estimate_best_thr:
            if not isinstance(self.subset_rbps, str):
                raise ValueError(
                    '"--just_estimate_pssm_threshold" is available when scanning PSSMs of a single RBP.'
                )

        self.motif_results = self.scan()

    def scan(self):
        """
        Scans for motif ocurrences using PSSM scoring along
        the sequences using biopython library

        For each PWM of each gene, it constructs biopython
        motifs, adds pseudocounts and a background distribution
        to calculate the PWM and PSSM, and scores the given PSSM
        over the input sequence.

        A score threshold is selected based on the distribution
        of the PSSM. Hits in the input sequence are selected
        based on the positions above the threshold
        """

        self.logger.log("INFO", "Scanning motifs using biopython")

        def _scan(row: pd.Series, **kwargs) -> list:
            motifs_uniq = kwargs["motifs_uniq"]
            just_estimate_thrsh = kwargs["just_estimate_thrsh"]
            min_nuc_prob = kwargs["min_nucleotide_probability"]
            pssm_threshold = kwargs["pssm_threshold"]
            res = []
            id = row.Seq_id
            seq = row.Sequence
            _len = len(seq)
            background_dist = {
                "A": seq.count("A") / _len,
                "C": seq.count("C") / _len,
                "G": seq.count("G") / _len,
                "T": seq.count("T") / _len,
            }

            for rbp_name, pwms in motifs_uniq.items():
                for pwm in pwms:
                    pwm_as_str = _pwm_to_ambiguous(pwm, min_probability=min_nuc_prob)

                    mt = _pwm_to_unambiguous(pwm, min_probabibility=min_nuc_prob)

                    biopy_m = motifs.create([Seq(m) for m in mt])

                    biopy_m.pseudocounts = 0.01
                    biopy_m.background = background_dist
                    # Background based on GC content
                    # biopy_m.background = 0.4

                    pssm = biopy_m.pssm

                    if just_estimate_thrsh:
                        distribution = pssm.distribution(
                            background=background_dist, precision=10 * 4
                        )
                        res.append(distribution.threshold_patser())
                        continue

                    if pssm_threshold < 0:
                        distribution = pssm.distribution(
                            background=background_dist, precision=10 * 3
                        )
                        pssm_threshold = round(distribution.threshold_patser(), 5)

                    scored = pssm.calculate(seq)
                    positions = np.argwhere(scored > pssm_threshold).flatten()
                    scores = scored[positions]

                    if positions.any():
                        for _pos, _score in zip(positions, scores):
                            s = seq[_pos : _pos + len(biopy_m)]
                            res.append(
                                [
                                    id,
                                    rbp_name,
                                    s,
                                    _pos,
                                    _pos + len(biopy_m),
                                    rbp_name + "_" + pwm_as_str,
                                    _score,
                                    pssm_threshold,
                                ]
                            )

            return res

        kwargs = {
            "motifs_uniq": self.motifs_uniq,
            "min_nucleotide_probability": self.min_nucleotide_probability,
            "just_estimate_thrsh": self.estimate_best_thr,
            "pssm_threshold": self.pssm_threshold,
        }

        if self.estimate_best_thr:
            res = self.dataset.head(1).apply(_scan, **kwargs, axis=1)
            res = list(res.explode())

            self.logger.log(
                "INFO",
                "Median best threshold obtained for {} PSSMs "
                "available at {} db for {} gene: {}".format(
                    len(res), self.motif_db, self.subset_rbps, round(np.median(res), 5)
                ),
            )
            raise ValueError

        if self.dataset.shape[0] > 10:
            pandarallel.initialize(progress_bar=True, use_memory_fs=False, verbose=0)
            res = self.dataset.parallel_apply(_scan, **kwargs, axis=1)
            print()
        else:
            res = self.dataset.apply(_scan, **kwargs, axis=1)

        self.logger.log("INFO", "Aggregating results")

        res = list(itertools.chain(*res))
        df = pd.DataFrame.from_records(
            res,
            columns=[
                "Seq_id",
                "RBP_name",
                "RBP_motif",
                "Start",
                "End",
                "RBP_name_motif",
                "PSSM_score",
                "PSSM_threshold",
            ],
        )

        #  self.logger.log('INFO', "Selecting highest score for repeated motif matches")
        # If match to different PWMs at the same positions, keep highest score
        # idx = df.groupby(['Seq_id', 'RBP_name', 'Start', 'End'])['PSSM_score'].idxmax()
        # df = df.loc[idx]
        df = df.drop_duplicates(["Seq_id", "RBP_name", "Start", "End"], keep="first")
        df.round({"PSSM_score": 4})
        self.logger.log("INFO", "Done. {} hits found".format(df.shape[0]))

        if not self.skip_raw_motifs_filtering:
            df = self.filter_raw_output(df)

        if not self.skip_location_mapping:
            df = self.add_motif_location(df)

        return df
