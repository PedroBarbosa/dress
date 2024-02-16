from collections import defaultdict
import itertools
import os
import pandas as pd
from typing import Union

from dress.datasetevaluation.representation.motifs.rbp_lists import RBP_SUBSETS
import numpy as np
import pyranges as pr
from loguru import logger


def _process_subset_argument(subset: list) -> Union[None, list, str]:
    """
    Process subset RBPs argument to return a
    list (RBPs) or str (group of RBPs in a set)
    to be searched in the motif scanning step

    :param list subset: Input argument

    :return Union[None, List, str]:
    """
    if not subset:
        return None

    # RBPs found in a file, one per line
    if len(subset) == 1 and os.path.isfile(subset[0]):
        return [rbp.rstrip() for rbp in open(subset[0], "r")]

    # RBPs present in the set provided
    elif len(subset) == 1 and subset[0] in list(RBP_SUBSETS.keys()):
        return subset[0]

    # RPBs provided in input argument (1 or more)
    return subset


def _dataset_to_fasta(
    dataset: pd.DataFrame,
    outdir: str,
    outfile: str = "seqs.fa",
    groupby_col: Union[list, str] = None,
    chunk_size: int = None,
) -> str:
    """
    Write dataset to fasta file based on `seq_id` and `sequence` columns.

    If `groupby_col` is set, dataset is grouped by col(s) provided,
    and one file per group is generated

    Args:
    dataset (pd.DataFrame): Dataset containing the sequences to write in fasta
    outfile (str): Output file
    groupby_col (Union[str, list]): Column(s) to group by dataset
    chunk_size (int): Generate multiple fasta file each one with `chunk_size` sequences

    Returns:
    Union[str,list]: Path(s) to the written fasta file(s)
    """

    assert all(x in dataset.columns for x in ["Seq_id", "Sequence"])
    assert any(
        x is None for x in [groupby_col, chunk_size]
    ), "groupby_col and chunk_size cant be set simultaneously"

    dataset = dataset.copy()
    dataset.loc[:, "Seq_id"] = ">" + dataset.Seq_id

    outfiles = []
    if groupby_col:
        for name, group in dataset.groupby(groupby_col):
            if isinstance(groupby_col, list):
                name = "_".join([str(x) for x in name])

            out = outfile.replace(".fa", "_" + name + ".fa")
            out = os.path.join(outdir, out)
            group[["Seq_id", "Sequence"]].to_csv(
                out, sep="\n", index=False, header=False, quoting=3
            )
            outfiles.append(out)

        return outfiles

    elif chunk_size:
        assert chunk_size > 10, "Chunk size should be higher than 10"
        chunks = np.array_split(dataset, len(dataset) // chunk_size + 1)
        for i, _chunk in enumerate(chunks):
            out = outfile.replace(".fa", "_" + str(i) + ".fa")
            out = os.path.join(outdir, out)
            _chunk[["Seq_id", "Sequence"]].to_csv(
                out, sep="\n", index=False, header=False, quoting=3
            )

            outfiles.append(out)
        return outfiles
    else:
        out = os.path.join(outdir, outfile)
        dataset[["Seq_id", "Sequence"]].to_csv(
            out, sep="\n", index=False, header=False, quoting=3
        )

        return out


def _read_meme(db: str) -> dict:
    """Reads PWMs from a database in MEME format

    Args:
        db (str): Path to the database file

    Returns:
        dict: Dict of dicts mapping genes to
    the PWM matrices as numpy arrays
    """
    motifs = defaultdict(list)
    motif_id, rbp_name, pwm = "", "", ""
    records = open(db, "r")
    save_pwm = False
    for line in records:
        line = line.rstrip()

        if line.startswith("MOTIF"):
            motif_id = line.split()[1]
            rbp_name = line.split()[2]
            pwm = []

        if line.startswith("letter-probability"):
            save_pwm = True

        elif line and line[0].isdigit():
            pwm.append([float(x) for x in line.split()])

        elif save_pwm:
            motifs[rbp_name].append({motif_id: np.array(pwm)})
            save_pwm = False

    return motifs


def _get_unique_pwms(all_pwms: dict) -> dict:
    """Reads per-gene PWMs stored in numpy arrays, compares them
    and returns unique matrices.

    Args:
        pwms (dict): Dictionary with PWMs per gene

    Returns:
        dict: Dictionary with unique PWMs per gene
    """
    motifs_unique = {}
    for gene, pwms in all_pwms.items():
        list_pwms = []
        for v in pwms:
            for pwm in v.values():
                list_pwms.append(pwm)

        hashable_list = [hash(str(arr)) for arr in list_pwms]
        hashable_set = set(hashable_list)
        unique_arr_list = [list_pwms[hashable_list.index(h)] for h in hashable_set]
        motifs_unique[gene] = unique_arr_list

    return motifs_unique


def _pwm_to_ambiguous(pwm: Union[dict, np.array], min_probability: float) -> str:
    """
    Reads PWMs as numpy arrays and returns a string
    representing the PWM, possibly with ambiguous
    characters.

    Args:
        pwm (Union[dict, np.array]): PWM as a numpy array or dictionary
        min_probabibility (float): Minimum per-position nucleotide probability to consider

    Returns:
        str: String representing the PWM
    """
    if isinstance(pwm, dict):
        for _, _pwm in pwm.items():
            pwm = _pwm

    if isinstance(pwm, np.ndarray):
        assert (
            pwm.shape[1] == 4
        ), "PWM should only have 4 columns, 1 for each nucleotide."

    nucleotides = ["A", "C", "G", "T"]
    ambiguous = {
        "AC": "M",
        "AG": "R",
        "AT": "W",
        "CG": "S",
        "CT": "Y",
        "GT": "K",
        "ACG": "V",
        "ACT": "H",
        "AGT": "D",
        "CGT": "B",
        "ACGT": "N",
    }

    flat_sequence = ""
    for position in range(pwm.shape[0]):
        idx = np.argwhere(pwm[position] > min_probability).flatten()
        nuc = "".join([nucleotides[i] for i in idx])
        flat_sequence += ambiguous.get(nuc, nuc)

    return flat_sequence


def _pwm_to_unambiguous(pwm: Union[dict, np.array], min_probabibility: float) -> list:
    """
    Reads PWMs as numpy arrays and returns a list
    of unambiguous motif sequences for which all
    their positions have a probability higher than
    a given threshold.

    Args:
        pwm (Union[dict, np.array]): PWM as a numpy array or dictionary
        min_probabibility (float): Minimum per-position nucleotide probability to consider

    Returns:
        list: List of all possible sequences for the given PWM
    """
    if isinstance(pwm, dict):
        for _, _pwm in pwm.items():
            pwm = _pwm

    if isinstance(pwm, np.ndarray):
        assert (
            pwm.shape[1] == 4
        ), "PWM should only have 4 columns, 1 for each nucleotide."
        unambiguous_motifs = []
        num_positions = pwm.shape[0]

        for pos in range(num_positions):
            pos_probs = pwm[pos, :]
            # which nucleotides have probabilities above the threshold
            pos_above_thresh = np.where(pos_probs >= min_probabibility)[0]
            # If no nucleotides above the threshold, skip to the next position
            if len(pos_above_thresh) == 0:
                continue
            # Otherwise, create a list of all possible unambiguous nucleotide combinations
            # for the current position
            pos_motifs = []
            for nuc in pos_above_thresh:
                nuc_char = "ACGT"[nuc]
                pos_motifs.append(nuc_char)

            unambiguous_motifs.append(pos_motifs)

        # Generate all possible combinations of motifs for each position
        motif_combinations = [
            "".join(c) for c in itertools.product(*unambiguous_motifs)
        ]

        return motif_combinations


def _redundancy_and_density_analysis(
    df: pd.DataFrame, scan_method: str, log: bool = False
) -> pd.DataFrame:
    """
    Finds overlaps in motif occurrences and do
    several operations, mostly using pyranges library.

    Specifically:
        - Removes self contained motifs of the
    same RBP so that those occurrences are counted
    once.
        - Flags partially overlapped motifs of
    the same RBP together with motifs in close
    proximity (up to 5bp apart) as high-density
    region for that RBP.
        - Aggregates duplicate hits where
    multiple RBPs share the exact same motif [NOT DONE NOW]

    Args:
        df (pd.DataFrame): Df with raw motif hits
        scan_method (str): Strategy used to scan motifs
        log (bool, optional): Whether to log progress. Defaults to False.
    Returns:
        pd.DataFrame: Subset of original df with additional info
    """
    rename_pr = {"Seq_id": "Chromosome"}

    df = df.reset_index(drop=True)
    _df = df.copy()
    _df = _df.rename(columns=rename_pr)

    if log:
        logger.info("Self contained hits analysis..")
    _df = _remove_self_contained(pr.PyRanges(_df), scan_method)

    if log:
        logger.info("Nearby hits analysis")
    _df = _tag_high_density(_df)

    # logger.info("Aggregating duplicate motifs across multiple RBPs")
    #_df = _remove_duplicate_hits(_df)
    if isinstance(_df, pr.PyRanges):
        _df = _df.as_df()

    return _df.rename(columns={"Chromosome": "Seq_id"})


def _remove_self_contained(gr: pr.PyRanges, scan_method: str) -> pr.PyRanges:
    """
    Flags motif ocurrences that are fully contained
    within other motif (whether it is the same RBP or not)

    Removes those motif ocurrences of the same RBP
    that are self contained.

    Args:
        gr (Union[pr.PyRanges, pd.DataFrame]): Motifs
        scan_method (str): Strategy used to scan motifs
    """

    _gr = gr.cluster(by="RBP_name", slack=-4)
    _gr.Length = _gr.lengths()
    df = _gr.df

    longest = df.groupby("Cluster").Length.idxmax()
    l = pr.PyRanges(df.reindex(longest))
    j = gr.join(l)

    to_drop_cols = [
        "Chromosome",
        "Start",
        "End",
        "RBP_name",
        "RBP_motif",
        "RBP_name_motif",
    ]

    if scan_method == "biopython":
        to_drop_cols.extend(["PSSM_score", "PSSM_threshold"])
    elif scan_method == "fimo":
        to_drop_cols.extend(["p-value", "q-value"])

    to_clean_cols = ["Cluster", "Length", "_merge"]
    df.drop(columns=to_clean_cols[:-1], inplace=True)

    contained = j[
        ((j.Start >= j.Start_b) & (j.End < j.End_b))
        | ((j.Start > j.Start_b) & (j.End <= j.End_b))
    ]

    if contained.empty:
        df["Has_self_submotif"] = False
        df["Has_other_submotif"] = False
        # logger.debug(".. no self containments found ..")

    else:
        #####################
        # Per RBP contained #
        #####################
        # logger.debug(".. self contained hits ..")

        contained_same_rbp = contained[contained.RBP_name == contained.RBP_name_b]

        if contained_same_rbp.empty:
            df["Has_self_submotif"] = False
            logger.debug(".. no self contained hits found for the same RBP ..")

        else:
            contained_same_rbp.Has_self_submotif = True
            contained_same_rbp = contained_same_rbp.df

            # Remove self.contained rows per RBP
            df = pd.merge(
                df,
                contained_same_rbp[to_drop_cols],
                on=to_drop_cols,
                how="left",
                indicator=True,
            )

            df = df.loc[df._merge == "left_only"]
            logger.debug(".. {} hits removed ..".format(contained_same_rbp.shape[0]))

            # Add self.contained.tag
            contained_same_rbp.drop(columns=to_drop_cols[1:], inplace=True)
            contained_same_rbp.columns = contained_same_rbp.columns.str.rstrip("_b")
            contained_same_rbp = contained_same_rbp.drop_duplicates()

            df = pd.merge(df, contained_same_rbp, how="left", on=to_drop_cols).drop(
                columns=to_clean_cols
            )
            df.Has_self_submotif.fillna(False, inplace=True)

        #######################
        # Other RBP contained #
        #######################
        # logger.debug(".. other contained hits ..")
        contained_other_rbp = contained[contained.RBP_name != contained.RBP_name_b]
        if contained_other_rbp.empty:
            df["Has_other_submotif"] = False
            # logger.debug(".. no self contained found hits for other RBPs ..")

        else:
            contained_other_rbp.Has_other_submotif = True
            contained_other_rbp = contained_other_rbp.df

            # Add other.contained.tag
            contained_other_rbp.drop(columns=to_drop_cols[1:], inplace=True)
            contained_other_rbp.columns = contained_other_rbp.columns.str.rstrip("_b")
            contained_other_rbp = contained_other_rbp.drop_duplicates()
            df = pd.merge(df, contained_other_rbp, how="left", on=to_drop_cols).drop(
                columns=to_clean_cols[:-1]
            )
            df.Has_other_submotif.fillna(False, inplace=True)
            # logger.debug(".. {} hits flagged ..".format(contained_other_rbp.shape[0]))

    return pr.PyRanges(df)


def _tag_high_density(gr: pr.PyRanges) -> pr.PyRanges:
    """
    Tags motifs of the same RBP and other RBP located
    in close proximity (motifs that either overlap,
    or are up to 5bp apart)

    Args:
        gr (pr.PyRanges): Motifs

    Returns:
        :pr.PyRanges
    """
    # logger.debug(".. clustering proximal ..")
    proximal_hits = gr.cluster(slack=5).df

    proximal_hits["Size"] = proximal_hits.groupby("Cluster")["Start"].transform(np.size)

    no_high_density = proximal_hits[proximal_hits.Size == 1].copy()
    no_high_density["Is_high_density_region"] = False
    no_high_density["N_at_density_block"] = 1

    high_density = proximal_hits[proximal_hits.Size > 1].copy()
    high_density["Is_high_density_region"] = True
    high_density["N_at_density_block"] = high_density.Size

    return pr.PyRanges(
        pd.concat([no_high_density, high_density]).drop(columns=["Cluster", "Size"])
    )


def _remove_duplicate_hits(gr: pr.PyRanges) -> pr.PyRanges:
    """
    Aggregate duplicate motifs of different
    RBPs into a single dataframe row

    Args:
        gr (pr.PyRanges): Motifs

    Returns
        :pr.PyRanges
    """

    exact_cols = ["Chromosome", "Start", "End"]

    df = gr.df
    g = df.groupby(exact_cols)

    no_dup = df[g["Start"].transform("size") == 1]
    dup = df[g["Start"].transform("size") > 1]

    if dup.empty:
        return pr.PyRanges(no_dup).sort()

    else:
        to_aggregate = [
            "RBP_name",
            "RBP_motif",
            "RBP_name_motif",
            "Has_self_submotif",
            "Has_other_submotif",
            "Is_high_density_region",
            "N_at_density_block",
        ]

        # g = dup.groupby(exact_cols)
        # x = g.agg("first")
        # x.update(g.agg({"rbp_name": ";".join, "rbp_motif": ";".join}))
        # x = x.reset_index()

        # out = dup.drop_duplicates(subset=exact_cols)
        # out = dup.groupby(exact_cols)['rbp_name'].agg(list)
        out = dup.groupby(exact_cols)[to_aggregate].agg(";".join).reset_index().dropna()

        if not no_dup.empty:
            out = pd.concat([out, no_dup])

        return pr.PyRanges(out).sort()
