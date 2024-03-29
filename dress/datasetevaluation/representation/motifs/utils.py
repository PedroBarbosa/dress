from collections import defaultdict
import itertools
import os
import re
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

    logger.debug("Self contained hits analysis..")
    _df = _remove_self_contained(pr.PyRanges(_df), scan_method)

    logger.debug("Nearby hits analysis")
    _df = _tag_high_density(_df)

    # logger.info("Aggregating duplicate motifs across multiple RBPs")
    # _df = _remove_duplicate_hits(_df)
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


def _get_loc_of_motif(_info: pd.DataFrame, _dataset: pd.DataFrame):
    """
    Generate spatial information of where
    motif features occur in the sequences.

    :param pd.DataFrame info: Dataframe with motif occurrences
    :param pd.DataFrame dataset: Input dataset

    :return pd.DataFrame: Motif ocurrences with additional
    spatial resolution: location, and distances to known
    splice sites
    """

    def _generate_intervals(ss_idx: pd.DataFrame):
        """
        Generates pyranges of the exon/intron
        intervals for each input sequence with
        splice site information

        :return pr.PyRanges: Exon intervals
        :return pr.PyRanges: Intron intervals
        """
        # Pyranges of the reference seqs
        chrom_e, start_e, end_e, label_e = [], [], [], []
        chrom_i, start_i, end_i, label_i = [], [], [], []
        tags_e = ["Exon_upstream", "Exon_cassette", "Exon_downstream"]
        _ss_idx_list = ss_idx["Splice_site_positions"].apply(
            lambda x: [int(y) if y != "<NA>" else y for y in x.split(";")]
        )
        _ss_idx_list = [[x[0:2], x[2:4], x[4:6]] for x in _ss_idx_list]
        ss_idx["Splice_site_positions"] = _ss_idx_list

        for _, row in ss_idx.iterrows():

            seq_id = row["Seq_id"]
            ss_idx = row["Splice_site_positions"]

            ref_coords = re.split(r"[(?:\()-]", seq_id)
            interval_len = int(ref_coords[2]) - int(ref_coords[1])

            # exons
            _ups = ss_idx[0]
            _cass = ss_idx[1]
            _down = ss_idx[2]

            _ups = [x if isinstance(x, int) else 0 for x in _ups]
            _down = [x if isinstance(x, int) else interval_len for x in _down]

            zipped = list(zip(*[_ups, _cass, _down]))

            chrom_e.extend([seq_id] * 3)
            start_e.extend(zipped[0])
            end_e.extend(zipped[1])
            label_e.extend(tags_e)

            # introns
            if _ups[0] != 0:
                chrom_i.append(seq_id)
                start_i.append(0)
                end_i.append(_ups[0])
                label_i.append("Intron_upstream_2")

            i_ups = [_ups[1], _cass[0]]
            i_down = [_cass[1], _down[0]]
            zipped = list(zip(*[i_ups, i_down]))
            chrom_i.extend([seq_id] * 2)
            start_i.extend(zipped[0])
            end_i.extend(zipped[1])
            label_i.extend(["Intron_upstream", "Intron_downstream"])

            if _down[1] != ref_coords[1]:
                chrom_i.append(seq_id)
                start_i.append(_down[1])
                end_i.append(interval_len)
                label_i.append("Intron_downstream_2")

        d_e = {
            "Chromosome": chrom_e,
            "Start": start_e,
            "End": end_e,
            "Strand": ["+"] * len(chrom_e),
            "Name": label_e,
        }

        exons = pr.from_dict(d_e)
        exons_cassette = exons[exons.Name == "Exon_cassette"]

        d_i = {
            "Chromosome": chrom_i,
            "Start": start_i,
            "End": end_i,
            "Strand": ["+"] * len(chrom_i),
            "Name": label_i,
        }

        introns = pr.from_dict(d_i)
        return exons, exons_cassette, introns

    def _distance_to_cassette(motifs: pr.PyRanges, exons: pr.PyRanges):
        """
        Generates motif distances to cassette exons splice site indexes

        :param pr.PyRanges motifs: All motifs ocurrences in the dataset
        :param pr.PyRanges exons: Intervals where cassette exons are located

        :return pr.PyRanges: motifs with 2 additional columns representing the
        distances to cassette splice sites
        """
        to_drop_cols = ["Start_b", "End_b", "Strand_b", "Name", "Distance"]
        cass = motifs.nearest(exons, strandedness="same")

        # Motifs that overlap cassette
        cass_overlap = cass[cass.Distance == 0]
        if not cass_overlap.empty:
            cass_overlap.distance_to_cassette_acceptor = (
                cass_overlap.Start - cass_overlap.Start_b
            )
            cass_overlap.distance_to_cassette_donor = (
                cass_overlap.End_b - cass_overlap.End
            )

        # Motifs that locate upstream
        cass_upstream = cass[cass.End <= cass.Start_b]
        if not cass_upstream.empty:
            cass_upstream.distance_to_cassette_acceptor = cass_upstream.Distance
            cass_upstream.distance_to_cassette_donor = (
                cass_upstream.End_b - cass_upstream.End
            )

        # Motifs that locate downstream
        cass_downstream = cass[cass.Start >= cass.End_b]
        if not cass_downstream.empty:
            cass_downstream.distance_to_cassette_acceptor = (
                cass_downstream.Start - cass_downstream.Start_b
            )
            cass_downstream.distance_to_cassette_donor = cass_downstream.Distance

        motifs = pr.concat([cass_upstream, cass_overlap, cass_downstream])
        motifs.distance_to_cassette_acceptor = (
            motifs.distance_to_cassette_acceptor.clip(0)
        )
        motifs.distance_to_cassette_donor = motifs.distance_to_cassette_donor.clip(0)
        return motifs.drop(to_drop_cols).sort()

    def _map_exonic_motifs(motifs: pr.PyRanges, exons: pr.PyRanges):
        """
        Maps location of motifs that overlap with exonic intervals

        :param pr.PyRanges motifs: All motifs ocurrences in the dataset
        :param pr.PyRanges exons: Intervals where exons are located

        :return pr.PyRanges: motifs with 3 additional columns representing the
        discrete location as well as the distance to the splice sites of the
        exon where the motif was found
        """
        final_to_concat = []
        to_drop_cols = ["Start_b", "End_b", "Strand_b", "Name", "Overlap"]

        _exon_match = motifs.join(exons, report_overlap=True, nb_cpu=1)

        if _exon_match.__len__() > 0:
            _exon_match.is_in_exon = True

            # Subtract first/last nucleotide of exon
            # So that they can latter be assigned to ss region
            fully = _exon_match[
                (_exon_match.Overlap == _exon_match.End - _exon_match.Start)
                & (_exon_match.Start - _exon_match.Start_b > 1)
                & (_exon_match.End_b - _exon_match.End > 1)
            ]

            # There are fully contained
            if fully.__len__() > 0:
                fully.location = fully.Name
                final_to_concat.append(fully)

                # There are some partial
                if _exon_match.__len__() != fully.__len__():
                    _p = pd.merge(
                        _exon_match.df,
                        fully.df,
                        on=list(_exon_match.df),
                        how="left",
                        indicator=True,
                    )

                    partial = pr.PyRanges(
                        _p[_p._merge == "left_only"].drop("_merge", axis=1)
                    )
                else:
                    partial = pr.PyRanges()

            # All are partial
            else:
                partial = _exon_match.copy()

            # PARTIAL
            if partial.__len__() > 0:

                # SHORT EXONS FULLY SPANNED BY MOTIF
                full_span = partial[
                    (partial.Start <= partial.Start_b) & (partial.End >= partial.End_b)
                ]

                if full_span.__len__() > 0:

                    full_span.location = full_span.Name + "_full_span"
                    final_to_concat.append(full_span)

                    # There are some partial that are not full span
                    if partial.__len__() != full_span.__len__():
                        _p = pd.merge(
                            partial.df,
                            full_span.df,
                            on=list(partial.df),
                            how="left",
                            indicator=True,
                        )

                        partial = pr.PyRanges(
                            _p[_p._merge == "left_only"].drop("_merge", axis=1)
                        )

                # MOTIFS NEAR/SPANNING ACCEPTORS
                acceptor_region = partial[
                    (partial.Start < partial.Start_b)
                    | (partial.Start - partial.Start_b < 2)
                ]
     
                if acceptor_region.__len__() > 0:
                    acceptor_region.location = acceptor_region.Name + "_acceptor_region"
                    final_to_concat.append(acceptor_region)

                # MOTIFS NEAR/SPANNING DONORS
                donor_region = partial[
                    (partial.End > partial.End_b) | (partial.End_b - partial.End < 2)
                ]
                if donor_region.__len__() > 0:
                    donor_region.location = donor_region.Name + "_donor_region"
                    final_to_concat.append(donor_region)

            exonic_motifs = pr.concat(final_to_concat)
            exonic_motifs.distance_to_acceptor = (
                exonic_motifs.Start - exonic_motifs.Start_b
            )
            exonic_motifs.distance_to_donor = exonic_motifs.End_b - exonic_motifs.End
            exonic_motifs = exonic_motifs.drop(to_drop_cols)

        else:
            return pr.PyRanges()

        return exonic_motifs

    def _map_intronic_motifs(motifs: pr.PyRanges, introns: pr.PyRanges):
        """
        Maps location of motifs that exclusively overlap with intronic intervals

        :param pr.PyRanges motifs: All motifs ocurrences in the dataset
        :param pr.PyRanges introns: Intervals where introns are located

        :return pr.PyRanges: motifs with 3 additional columns representing the
        discrete location as well as the distance to the splice sites of the
        intron where the motif was found
        """
        to_drop_cols = ["Start_b", "End_b", "Strand_b", "Name", "Overlap"]
        _intron_match = motifs.join(
            introns, report_overlap=True, nb_cpu=1, apply_strand_suffix=False
        )

        if _intron_match.__len__() > 0:
            i_motifs = _intron_match[
                _intron_match.Overlap == _intron_match.End - _intron_match.Start
            ]

            if i_motifs.__len__() > 0:
                i_motifs.is_in_exon = False
                i_motifs.location = i_motifs.Name
                i_motifs.distance_to_acceptor = i_motifs.End_b - i_motifs.End
                i_motifs.distance_to_donor = i_motifs.Start - i_motifs.Start_b
                i_motifs = i_motifs.drop(to_drop_cols)
                i_motifs = i_motifs.df
                i_motifs.loc[
                    i_motifs.location == "Intron_upstream_2", "distance_to_donor"
                ] = pd.NA
                i_motifs.loc[
                    i_motifs.location == "Intron_downstream_2", "distance_to_acceptor"
                ] = pd.NA
                i_motifs = pr.PyRanges(i_motifs)
            else:
                return pr.PyRanges()
        else:
            return pr.PyRanges()

        return i_motifs

    ############################
    #### Generate intervals ####
    ############################
    info = _info.copy()
    dataset = _dataset.copy()
    logger.debug(".. generating intervals fromm ss idx ..")
    exons, exons_cassette, introns = _generate_intervals(dataset)
    motifs = pr.PyRanges(
        info.rename(columns={"Seq_id": "Chromosome", "start": "Start", "end": "End"})
    )
    motifs.Strand = "+"

    #############################
    ### Dist to cassette exon ###
    #############################
    logger.debug(".. calculating distances to cassette ss ..")
    motifs = _distance_to_cassette(motifs, exons_cassette)
    
    ############################
    ### Loc of exonic motifs ###
    ############################
    logger.debug(".. mapping location of exonic motifs ..")
    exonic_motifs = _map_exonic_motifs(motifs, exons)

    ############################
    ## Loc of intronic motifs ##
    ############################
    logger.debug(".. mapping location of intronic motifs ..")
    intronic_motifs = _map_intronic_motifs(motifs, introns)

    out = pr.concat([exonic_motifs, intronic_motifs])
    out.distance_to_acceptor = out.distance_to_acceptor.clip(0)
    out.distance_to_donor = out.distance_to_donor.clip(0)
    return out.sort().as_df().rename(columns={"Chromosome": "Seq_id"}).drop(columns='Strand')
