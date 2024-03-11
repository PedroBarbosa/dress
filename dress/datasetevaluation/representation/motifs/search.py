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
        self.motif_db = kwargs.get("motif_db", "cisBP_RNA")
        self.subset_rbps = _process_subset_argument(kwargs.get("subset_rbps", "encode"))
        self.min_motif_length = kwargs.get("min_motif_length", 5)
        self.min_nucleotide_probability = kwargs["min_nucleotide_probability"]
        self.skip_raw_motifs_filtering = kwargs["skip_raw_motifs_filtering"]
        self.outdir = os.path.join(kwargs["outdir"], "motifs")
        os.makedirs(self.outdir, exist_ok=True)
        self.outbasename = assign_proper_basename(kwargs.get("outbasename"))

        to_flat = False if kwargs["motif_search"] == "biopython" else True
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

    def filter_output(self, raw_hits: pd.DataFrame):
        """
        Filters motif results and adds additional information
        if splice sites information is available

        Args:
            raw_hits(pd.DataFrame): Results from motif scanning

        Returns:
            pd.DataFrame: Filtered df with additional information
        """
        if raw_hits.empty:
            raise ValueError("No motifs found given this experimental setup.")

        self.logger.log("INFO", "Filtering motif results")

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
        self.logger.log("INFO", "Mapping location of motifs")
        def _process_single_sequence(group: pd.DataFrame, dataset:pd.DataFrame):
            seq_id = group.iloc[0].Seq_id
            single_seq = dataset[dataset.Seq_id == seq_id]
            return _get_loc_of_motif(group, single_seq)

        df = df.groupby('Seq_id').parallel_apply(_process_single_sequence, dataset=self.dataset).reset_index(drop=True)
        return df

    def tabulate_occurrences(self, write_output: bool = True) -> Tuple[pd.DataFrame]:
        """
        Generate counts of each motif in tabular format.
        Rows are sequence IDs, columns are counts for
        specific RBP motifs.

        If 'group' column exist in the main dataset, it is
        added to the output so that further comparisons
        can be performed downstream.

        Returns:
            Tuple(pd.DataFrame): Tuple with two dataframes:
        one with motif counts per gene (RBP), other with counts
        for each possible motif within each gene (RBP)
        """
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

        # Count ocurrences per RBP and per individual RBP motif
        rbp_counts = (
            self.motif_results.groupby(["Seq_id", "RBP_name"])
            .size()
            .unstack(fill_value=0)
        )
        rbp_counts_detailed = (
            self.motif_results.groupby(["Seq_id", "RBP_name_motif"])
            .size()
            .unstack(fill_value=0)
        )

        # Fill motif df with absent occurrences
        absent_rbps_hits = [x for x in all_rbps if x not in list(rbp_counts)]
        absent_rbps_detailed_hits = [
            x for x in all_rbps_detailed if x not in list(rbp_counts_detailed)
        ]

        d_rbps1 = pd.concat(
            [pd.DataFrame(dict.fromkeys(absent_rbps_hits, 0), index=[0])]
            * rbp_counts.shape[0],
            ignore_index=True,
        ).set_index(rbp_counts.index)

        d_rbps2 = pd.concat(
            [pd.DataFrame(dict.fromkeys(absent_rbps_detailed_hits, 0), index=[0])]
            * rbp_counts_detailed.shape[0],
            ignore_index=True,
        ).set_index(rbp_counts_detailed.index)

        rbp_counts = pd.concat([rbp_counts, d_rbps1], axis=1).sort_index(axis=1)
        rbp_counts_detailed = pd.concat(
            [rbp_counts_detailed, d_rbps2], axis=1
        ).sort_index(axis=1)

        cols = ["Seq_id", "Score", "Delta_score"]
        # If PairedDataset, last column is the group
        if "group" in self.dataset.columns and self.dataset.group.nunique() > 1:
            cols.append("group")

        rbp_counts = pd.merge(rbp_counts, self.dataset[cols], how="left", on="Seq_id")

        rbp_counts_detailed = pd.merge(
            rbp_counts_detailed,
            self.dataset[cols],
            how="left",
            on="Seq_id",
        )

        if write_output:
            self.motif_results.to_csv(
                self.outdir + "/{}MOTIF_MATCHES.tsv.gz".format(self.outbasename),
                compression="gzip",
                sep="\t",
                index=False,
            )
            rbp_counts.to_csv(
                self.outdir + "/{}RBP_COUNTS.tsv.gz".format(self.outbasename),
                compression="gzip",
                sep="\t",
                index=False,
            )
            rbp_counts_detailed.to_csv(
                self.outdir + "/{}MOTIF_COUNTS.tsv.gz".format(self.outbasename),
                compression="gzip",
                sep="\t",
                index=False,
            )

        return rbp_counts, rbp_counts_detailed

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

                elif self.subset_rbps not in self.motifs.keys():
                    raise ValueError(
                        "RBP '{}' not found in the {} database.".format(
                            self.subset_rbps, self.motif_db
                        )
                    )

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

        def _scan(row: pd.Series):
            res = []
            # for each RBP
            for rbp_name, _motifs in self.motifs.items():
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
            res = self.dataset.parallel_apply(_scan, axis=1)
            print()
        else:
            res = self.dataset.apply(_scan, axis=1)

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
        if self.skip_raw_motifs_filtering:
            return df

        return self.filter_output(df)


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

        assert self.motif_db not in [
            "rosina2017",
            "encode2020_RBNS",
        ], f"{self.motif_db} motif database allowed only when '--motif_search is 'plain'"

        self.pvalue_threshold = kwargs.get(
            "pvalue_threshold",
        )
        self.qvalue_threshold = kwargs.get("qvalue_threshold")
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

        def _scan(row: pd.Series):
            fasta = row.fasta

            fimo_outdir = os.path.join(
                os.path.join(self.outdir, "fimo"), Path(fasta).stem
            )

            os.makedirs(fimo_outdir, exist_ok=True)
            fimo_cmd = base_cmd + ["--oc", fimo_outdir, self.motif_db_file, fasta]

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

        self.logger.log(
            "INFO", "Base command to run FIMO: {}".format(" ".join(base_cmd))
        )

        fimo_outdir = self.outdir + "/fimo"
        os.makedirs(fimo_outdir, exist_ok=True)

        if self.dataset.shape[0] > 10:
            self.logger.log(
                "INFO", "Writing sequences to multiple files to parallelize FIMO search"
            )
            fasta_files = _dataset_to_fasta(
                self.dataset, outdir=fimo_outdir, chunk_size=100
            )
            fasta_df = pd.DataFrame(fasta_files, columns=["fasta"])
            self.logger.log("INFO", "Running..")
            pandarallel.initialize(progress_bar=True, use_memory_fs=False, verbose=0)
            res = fasta_df.parallel_apply(_scan, axis=1)
            print()
        else:
            fasta_files = _dataset_to_fasta(self.dataset, fimo_outdir)
            fasta_df = pd.DataFrame([fasta_files], columns=["fasta"])
            res = fasta_df.apply(_scan, axis=1)

        res = res.dropna()
        if res.empty:
            self.logger.log(
                "INFO",
                "No motif hits found using FIMO against {} database".format(
                    self.motif_db
                ),
            )
            exit(1)

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

        self.logger.log(
            "DEBUG",
            "Number of hits removed due to the minimum nucleotide probability "
            "threshold set ({}): {}".format(
                self.min_nucleotide_probability, _n - df.shape[0]
            ),
        )

        # Remove matches not passing the q-value threshold
        _n = df.shape[0]
        if self.qvalue_threshold is not None:

            df = df[df["q-value"] <= self.qvalue_threshold]
            self.logger.log(
                "DEBUG",
                "Number of hits removed due to the q-value "
                "threshold set ({}): {}".format(
                    self.qvalue_threshold, _n - df.shape[0]
                ),
            )

            if df.empty:
                self.logger.log(
                    "INFO",
                    'No remaining FIMO hits after applying "qvalue_threshold"',
                )
                exit(1)
        df.Start -= 1

        [os.remove(f) for f in fasta_files]

        df = df.drop_duplicates(["Seq_id", "RBP_name", "Start", "End"], keep="first")
        self.logger.log("INFO", "Done. {} hits found".format(df.shape[0]))

        if self.skip_raw_motifs_filtering:
            return df

        return self.filter_output(df)


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

        self.motifs_uniq = _get_unique_pwms(self.motifs)
        self.pssm_threshold = kwargs["pssm_threshold"]
        self.estimate_best_thr = kwargs["just_estimate_pssm_threshold"]

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
        based on the position above the threshold
        """

        self.logger.log("INFO", "Scanning motifs using biopython")

        def _scan(row: pd.Series, just_estimate_thrsh: bool = False) -> list:
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

            for rbp_name, pwms in self.motifs_uniq.items():
                for pwm in pwms:
                    pwm_as_str = _pwm_to_ambiguous(
                        pwm, min_probability=self.min_nucleotide_probability
                    )

                    mt = _pwm_to_unambiguous(
                        pwm, min_probabibility=self.min_nucleotide_probability
                    )

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

                    if self.pssm_threshold < 0:
                        distribution = pssm.distribution(
                            background=background_dist, precision=10 * 3
                        )
                        self.pssm_threshold = round(distribution.threshold_patser(), 5)

                    scored = pssm.calculate(seq)
                    positions = np.argwhere(scored > self.pssm_threshold).flatten()
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
                                    self.pssm_threshold,
                                ]
                            )

            return res

        if self.estimate_best_thr:
            res = self.dataset.head(1).apply(_scan, just_estimate_thrsh=True, axis=1)
            res = list(res.explode())

            self.logger.log(
                "INFO",
                "Median best threshold obtained for {} PSSMs "
                "available at {} db for {} gene: {}".format(
                    len(res), self.motif_db, self.subset_rbps, round(np.median(res), 5)
                ),
            )
            exit(1)

        if self.dataset.shape[0] > 10:
            pandarallel.initialize(progress_bar=True, use_memory_fs=False, verbose=0)
            res = self.dataset.parallel_apply(_scan, axis=1)
            print()
        else:
            res = self.dataset.apply(_scan, axis=1)

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

        if self.skip_raw_motifs_filtering:
            return df

        return self.filter_output(df)
