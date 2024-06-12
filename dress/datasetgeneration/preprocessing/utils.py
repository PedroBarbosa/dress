import os
from typing import Iterable, Literal, Optional, Tuple, Union
import pandas as pd
from itertools import chain
from pyfaidx import Fasta
import pyranges as pr
from loguru import logger


def process_ss_idx(
    seqs: Union[dict, pd.DataFrame], ss_info: str = None, return_seqs: bool = False
) -> Tuple[dict, dict]:
    """
    Assign splice site indexes to a set of sequences

    Args:
        seqs (Union[dict, pd.DataFrame]): Data with sequences to process
        ss_info (str, optional): File with splice site indexes of the
    upstream, cassette and downstream exons
        return_seqs (bool, optional): Whether to return the sequences,
    in addition to the splice sites

    Returns:
        dict: Dictionary with exon indexes for each input sequence
        dict: Dictionary with sequences, if `return_seqs` is set to `True`
    """
    if ss_info:
        assert isinstance(seqs, dict), "seqs must be a dict"
    else:
        assert isinstance(seqs, pd.DataFrame), "seqs must be a pandas DataFrame"

    if isinstance(ss_info, str):
        if os.path.exists(ss_info):
            ss = pd.read_csv(ss_info, sep="\t")
        else:
            raise FileNotFoundError(
                "Splice site idx file not found at the same directory of the fata input file."
            )
    else:
        ss = seqs.copy()
        seqs = dict(zip(ss.header_long, ss.seq))

    ss_idx = {}

    for i, seq_id in enumerate(seqs.keys()):

        seq_header = seq_id.split("_")[0]
        tx_id = "_".join(seq_id.replace("_REF_seq", "").split("_")[1:])

        if seq_header not in list(ss.header):
            raise ValueError(
                "{} seq ID not in splice site idx file.".format(seq_header)
            )

        _ss = ss.loc[(ss.header == seq_header) & (ss.tx_id == tx_id)]

        assert (
            _ss.shape[0] == 1
        ), "More than one row with splice site indexes for {} seq.".format(seq_id)

        idx = list(
            chain.from_iterable(
                zip(
                    _ss.iloc[0].acceptor_idx.split(";"),
                    _ss.iloc[0].donor_idx.split(";"),
                )
            )
        )

        idx = [x if x == "<NA>" else int(x) for x in idx]

        upstream = idx[0:2]
        cassette = idx[2:4]
        downstream = idx[4:]

        ss_idx[seq_id] = [upstream, cassette, downstream]

    return ss_idx if not return_seqs else seqs, ss_idx


def _get_flat_ss(
    info: pd.Series, _level: str, start: int, end: int, extract_full_triplet: bool
):
    """
    Extracts flat splice site indexes for sequences
    with surrounding features up to a given level,
    associated with a given transcript id.

    Args:
        info (pd.Series): Information regarding the
    coordinates of surrounding features.
        _level (str): Max level for which there
    are surrounding features available.
        start (int): Start coordinate (0-based index)
    of the flat sequence, after accounting for the
    extensions.
        end (int): End coordinate of the flat sequence,
    after accounting for the extensions.
        extract_full_triplet (bool, optional): Whether `start` and `end`
    coordinates represent the true start and end of
    the sequence at a given surrounding level. If `False`,
    they represent start and end positions up to the
    limit of the model resolution.

    Returns:
        str: Flat acceptor indexes
        str: Flat donor indexes
    """
    _info = info.copy()

    # Check out of scope indexes
    if extract_full_triplet is False:
        if _level != 0:
            cols = [
                "Start_upstream" + _level,
                "End_upstream" + _level,
                "Start",
                "End",
                "Start_downstream" + _level,
                "End_downstream" + _level,
            ]

            _info[cols] = _info[cols].apply(
                lambda x: int(x) if start <= x <= end else pd.NA
            )

    # if target is exon, we know
    # what's upstream and downstream
    if _info.Feature == "exon" and _level == "_2":
        if _info.Strand == "+":
            ups_donor = _info["End_upstream" + _level] - start
            target_donor = _info.End - start
            down_donor = _info["End_downstream" + _level] - start
            ups_accept = _info["Start_upstream" + _level] - start
            target_accept = _info.Start - start
            down_accept = _info["Start_downstream" + _level] - start

        else:
            ups_donor = end - _info["Start_upstream" + _level]
            target_donor = end - _info.Start
            down_donor = end - _info["Start_downstream" + _level]
            ups_accept = end - _info["End_upstream" + _level]
            target_accept = end - _info.End
            down_accept = end - _info["End_downstream" + _level]

        # Substract additional 1 from donor idx to represent idx for last exon position, just like spliceai
        donors = [ups_donor - 1, target_donor - 1, down_donor - 1]
        acceptors = [ups_accept, target_accept, down_accept]

    elif _info.Feature == "exon":
        if _info.Strand == "+":
            target_donor = _info.End - start
            target_accept = _info.Start - start

        else:
            target_donor = end - _info.Start
            target_accept = end - _info.End

        donors = [pd.NA, target_donor - 1, pd.NA]
        acceptors = [pd.NA, target_accept, pd.NA]

    else:
        raise NotImplementedError("There is only support for exons as target features")

    acceptor_idx = ";".join(str(i) for i in acceptors)
    donor_idx = ";".join(str(i) for i in donors)

    return acceptor_idx, donor_idx


def get_fasta_sequences(
    x: pd.Series,
    fasta: Fasta,
    slack: Optional[int] = None,
    slack_upstream: Optional[int] = None,
    slack_downstream: Optional[int] = None,
    is_one_based: bool = False,
    extend: str = None,
    chrom_sizes: dict = None,
    start_col: str = "Start",
    end_col: str = "End",
):
    """
    Retrieve Fasta sequence from a set of genomic coordinates of a
    given feature using an indexed reference genome

    Args:
        x (pd.Series): Single row of a Feature dataframe (e.g, Transcript, exon)
        fasta (pyfaidx.Fasta): Indexed reference genome
        slack (int, optional): How many additional bp to extract
    from each interval side. Defaults to None. Can't be used together with `slack_upstream` and `slack_downstream`.
        slack_upstream (int, optional): How many additional bp to extract
    upstream of the interval. Defaults to None. Can't be used together with `slack`.
        slack_downstream (int, optional): How many additional bp to extract
    downstream of the interval. Defaults to None. Can't be used together with `slack`.
        is_one_based (bool, optional): Whether start coordinates of
    `x` are 1-based. Defaults to False.
        extend (str, optional): If slack is provided, and genomic
    sequence of each interval is already present in the
    column provided in this argument, it extends the sequence
    without changing the existing sequence (e.g. useful if
    artificial mutations were inserted within the intervals). Defaults to None.
        chrom_sizes (dict, optional): Chromosome sizes of the genome
        start_col (str, optional): Colname in `x` where start coordinate is. Defaults to "Start".
        end_col (str, optional): Colname in `x` where end coordinate is. Defaults to "End".

    Returns:
        pd.Series: Additional column with the fasta sequence for the requested interval
    """

    if slack is not None and any(
        x is not None for x in [slack_upstream, slack_downstream]
    ):
        raise ValueError(
            "Can't use `slack` and `slack_upstream`/`slack_downstream` together."
        )

    if chrom_sizes and x.Chromosome not in chrom_sizes.keys():
        logger.info(
            "Chrom of interval not found in chromosome "
            "sizes dict. Not possible to extract fasta seq."
        )
        return

    if extend is not None:
        assert slack is not None, (
            "Set the slack argument when " "'extend' argument is provided."
        )

        # assert extend in x.index, "{} is not in the columns names.".format(extend)

    # missing assert that checks whether
    # intervals comply with chrom sizes

    try:
        if is_one_based:
            x[start_col] - 1
        else:
            start = x[start_col]

        end = x[end_col]

        if slack is not None:
            start -= slack
            end += slack

        if slack_upstream is not None:
            if x.Strand == "+":
                start -= slack_upstream
            else:
                end += slack_upstream

        if slack_downstream is not None:
            if x.Strand == "+":
                end += slack_downstream
            else:
                start -= slack_downstream

        int_id = x.Chromosome + "_" + str(start) + "_" + str(end)
        if chrom_sizes and end > chrom_sizes[x.Chromosome]:
            logger.info(
                "Interval {} exceeds chromosome "
                "boundaries ({})".format(int_id, chrom_sizes[x.Chromosome])
            )
            return

        if extend is None or extend not in x.index:
            # start attributes in pyFaidx are 0-based
            if x["Strand"] == "-":
                out_seq = fasta[x["Chromosome"]][start:end].reverse.complement.seq
            else:
                out_seq = fasta[x["Chromosome"]][start:end].seq

        else:
            if x["Strand"] == "-":
                left = fasta[x["Chromosome"]][
                    start : start + slack
                ].reverse.complement.seq
                right = fasta[x["Chromosome"]][end - slack : end].reverse.complement.seq
                out_seq = right + x[extend] + left
            else:
                left = fasta[x["Chromosome"]][start : start + slack].seq
                right = fasta[x["Chromosome"]][end - slack : end].seq
                out_seq = left + x[extend] + right

        if len(out_seq) == end - start:
            # If seq is all composed by Ns
            if not out_seq.replace("N", ""):
                _s = chrom_sizes[x.Chromosome] if chrom_sizes else None
                logger.info(
                    "Sequence for the interval {} "
                    "is just composed by Ns. Chromosome "
                    "len in Fasta obj: {}. Chromosome "
                    "len in the chrom dict, if provided: {}".format(
                        int_id, len(fasta[x.Chromosome]), _s
                    )
                )
                return
            return out_seq

        else:
            logger.warning(
                "Problem extracting Fasta sequence for "
                "the given interval: {}. Length of "
                "chromosome in Fasta where interval is "
                "located: {}".format(int_id, len(fasta[x.Chromosome]))
            )
            return

    except KeyError:
        raise KeyError(
            "{} chromosome is not in Fasta header. Additionally, "
            "make sure that the apply function is performed rowwise "
            "(axis=1), when extracting fasta sequences.".format(x.Chromosome)
        )


def bed_file_to_genomics_df(data: str):
    """
    Reads a bed file (0-based) and creates a genomics df for downstream
    interval operations.

    Args:
        data: str: Input bed file
    """

    try:
        for l in open(data).readlines():
            start = l.split()[1]
            end = l.split()[2]
            int(start)
            int(end)

    except IndexError:
        raise IndexError(f"{data} does not seem to be a valid bed file.")

    except ValueError:

        raise ValueError(
            f"{data}' file does not seem to have a proper bed format. "
            "Does it have an header? If so, remove it"
        )

    c_map = {0: "Chromosome", 1: "Start", 2: "End", 3: "Name", 4: "Score", 5: "Strand"}
    df = pd.read_csv(data, sep="\t", header=None)
    if len(df.columns) >= 6:
        if not df.iloc[:, 5].isin(["+", "-"]).all():
            raise ValueError(
                f"Strand column (6th) must contain only '+' or '-' values."
            )
    cols = []
    for i in range(len(list(df))):
        c = c_map.get(i, "Col")
        c = c + "_{}".format(i) if c == "Col" else c
        cols.append(c)
    df.columns = cols
    return pr.PyRanges(df)


def tabular_file_to_genomics_df(
    data: str,
    col_index: int = 0,
    is_0_based: bool = True,
    header: int = None,
    c_map: dict = None,
    subset_columns: Iterable = None,
    remove_str_from_col_names: str = None,
) -> pr.PyRanges:
    """
    Reads a tab-delimited file where
    one column contains genomic
    intervals (chr:start-end) and creates
    a genomics df for downstream interval
    operations.

    Args:
        data (str): Input file
        col_index (int): Column index where
    coordinates are located. Default: `=0`
        is_0_based (bool): Whether column of
    the genomic coordinates have 0-based intervals.
    Default: `False`, coordinates are 1-based.
    Coordinates will be returned in 0-based half
    open intervals.
        header (int): Row index where header
    is located. Default: `None`, no header
        c_map (dict): Dictionary mapping col
    indexes to col names when `header` is `None`.
        subset_columns (Iterable): Columns to subset.
    All other arguments will be in respect to the
    remaining columns
        remove_str_from_col_names (str): Remove
    specific string from column names. only applies
    when header is not None

    Returns:
        pr.PyRanges: Input data as a PyRanges df
    """

    if isinstance(data, str):
        df = pd.read_csv(data, sep="\t", header=header)
    else:
        raise ValueError("Wrong input")

    df = df.iloc[:, subset_columns] if subset_columns else df

    if remove_str_from_col_names and header is not None:
        df.columns = [x.replace(remove_str_from_col_names, "") for x in df.columns]

    df_intervals = df.iloc[:, col_index].str.split(":|-", expand=True)

    df_intervals.columns = ["Chromosome", "Start", "End"]
    df_intervals[["Start", "End"]] = df_intervals[["Start", "End"]].apply(pd.to_numeric)

    if header is None:
        cols = []
        if not c_map:
            c_map = {col_index: "Name", 1: "Score", 2: "Strand"}

        for i in range(len(list(df))):
            c = c_map.get(i, "Col")
            c = c + "_{}".format(i) if c == "Col" else c
            cols.append(c)
        df.columns = cols
        df_intervals.columns = ["Chromosome", "Start", "End"]

    c_map = {"strand": "Strand", df.columns[col_index]: "Name"}
    df = df.rename(columns=c_map)

    if "Strand" in df.columns:
        assert all(x in ["+", "-"] for x in df.Strand), (
            "Strand column must contain only '+' or '-' values. "
            "Please set c_map argument to manually assign column names "
            "to specific column indexes"
        )
    if not is_0_based:
        df_intervals["Start"] -= 1

    return pr.PyRanges(pd.concat([df_intervals, df], axis=1))


def generate_pipeline_input(
    df: pd.DataFrame,
    fasta: str,
    extend_borders: int = 0,
    extract_full_triplet: bool = False,
    extract_dynamically: bool = False,
    model: str = "spliceai",
):
    """
    Generates model input sequences from a dataframe
    of target exons with upstream and downstream
    intervals.

    It will generate the splice site indexes of the exons and
    introns surrounding the target exon of interest, for the
    associated transcript ID.

    Sequence extraction can be done in three ways:
    - `extract_full_triplet` is `True`: The complete sequence
    from the start of the exon upstream until the end of the
    exon downstream is extracted, regardless of the size of
    the resulting sequence. This can result in very large
    sequences if introns are very long.
    - `extract_dynamically` is `True`: The sequence will be extracted
    such that: if the full exon triplet is smaller than the model resolution,
    the full exon triplet is extracted, and then at inference time the sequence
    is padded to the model resolution. If the exon triplet is larger than the
    model resolution, the sequence is trimmed at the model resolution, which may not
    include upstream and/or downstream exons.
  
    - `extract_dynamically` is `False` and `extract_full_triplet` is `False`:
    If neither of the above is set (default), the sequence is
    extracted up to the maximum resolution of the model. For example,
    for SpliceAI, the extracted sequence will be the cassette exon plus
    5000bp upstream of the acceptor and 5000bp downstream of the donor.

    If `extend_borders` > 0, the sequence will be extended on both sides.
    This is useful for cases where the acceptor of the uptream exon 
    or the donor of the downstream exon represent the start or end of the 
    sequence, respectively. Because the splice site score of such positions
    may not be properly captured (because the full context is not present),
    extending the sequence can help to provide a more realistic prediction 
    as if those positions were in the middle of the sequence. This extension
    is applied when `extract_full_triplet` or `extract_dynamically`is `True`,
    the latter only if the upstream/downstream exon is within the model resolution.

    Returns:
        Tuples with the following information:
            - List with the main output sequences and splice site info
            - If available, a list with dPSI information for the given exons
            - A dataframe with the exons excluded due to having NAs, if level == 2
    """

    assert any(
        x is False for x in [extract_full_triplet, extract_dynamically]
    ), "Can't set both `extract_full_triplet` and `extract_dynamically` to `True`."

    if list(df.filter(regex="stream")):
        # Extract level so that we know the
        # borders of the genomic sequence to extract
        try:
            level = max(
                [int(x.split("_")[-1]) for x in list(df.filter(regex="stream"))]
            )
            _level = "_" + str(level)
        except ValueError:
            # level 1
            level, _level = "", ""

        # Rows with NAs in the intervals upstream or downstream are removed,
        # meaning that only exons that are not the first and last are kept
        # (available to be considered as cassette)
        _with_NAs = df[df.filter(regex="^Start.*stream", axis=1).isna().any(axis=1)]
        df = df[~df.filter(regex="^Start.*stream", axis=1).isna().any(axis=1)]

    else:
        level, _level = 0, 0
        _with_NAs = pd.DataFrame(
            columns=["Chromosome", "Start", "End", "Score", "Strand"]
        )

    out, out_dpsi = [], []

    for _, seq_record in df.iterrows():

        if extract_full_triplet:
            if seq_record.Strand == "+":
                start = "Start_upstream" + _level if level != 0 else "Start"
                end = "End_downstream" + _level if level != 0 else "End"

            else:
                start = "Start_downstream" + _level if level != 0 else "Start"
                end = "End_upstream" + _level if level != 0 else "End"

            seq = get_fasta_sequences(
                seq_record,
                fasta=fasta,
                start_col=start,
                end_col=end,
                slack=extend_borders,
                is_one_based=False,
            )

            spanning_coords = (
                str(seq_record[start] + 1 - extend_borders)
                + "-"
                + str(seq_record[end] + extend_borders)
            )

            header = (
                seq_record.Chromosome
                + ":"
                + spanning_coords
                + "({})".format(seq_record.Strand)
            )
            header_long = header + "_" + seq_record.transcript_id

            acceptor_idx, donor_idx = _get_flat_ss(
                seq_record,
                _level,
                start=seq_record[start] - extend_borders,
                end=seq_record[end] + extend_borders,
                extract_full_triplet=extract_full_triplet,
            )

            out.append(
                [
                    header,
                    header_long,
                    seq,
                    acceptor_idx,
                    donor_idx,
                    seq_record.transcript_id,
                    seq_record.Name,
                ]
            )

        else:

            slack_upst = _get_slack(
                seq_record,
                region="upstream",
                _level=_level,
                extend_borders=extend_borders,
                no_model_resolution=extract_dynamically,
                model=model,
            )

            slack_downst = _get_slack(
                seq_record,
                region="downstream",
                _level=_level,
                extend_borders=extend_borders,
                no_model_resolution=extract_dynamically,
                model=model,
            )

            seq = get_fasta_sequences(
                seq_record,
                fasta=fasta,
                start_col="Start",
                end_col="End",
                slack_upstream=slack_upst,
                slack_downstream=slack_downst,
                is_one_based=False,
            )

            if seq_record.Strand == "+":
                left = seq_record.Start - slack_upst
                right = seq_record.End + slack_downst

            else:
                left = seq_record.Start - slack_downst
                right = seq_record.End + slack_upst

            header = (
                seq_record.Chromosome
                + ":"
                + str(left + 1)
                + "-"
                + str(right)
                + "({})".format(seq_record.Strand)
            )

            header_long = header + "_" + seq_record.transcript_id
            acceptor_idx, donor_idx = _get_flat_ss(
                seq_record,
                _level,
                start=min(left, right),
                end=max(left, right),
                extract_full_triplet=extract_full_triplet,
            )

            out.append(
                [
                    header,
                    header_long,
                    seq,
                    acceptor_idx,
                    donor_idx,
                    seq_record.transcript_id,
                    f"{seq_record.Chromosome}:{seq_record.Start + 1}-{seq_record.End}",
                ]
            )

        try:
            out_dpsi.append(
                [header_long, seq_record.Name, seq_record.gene_name, seq_record.dPSI]
            )
        except AttributeError:
            pass

    out = pd.DataFrame.from_records(
        out,
        columns=[
            "header",
            "header_long",
            "seq",
            "acceptor_idx",
            "donor_idx",
            "tx_id",
            "exon",
        ],
    ).drop_duplicates()

    if out_dpsi:
        out_dpsi = pd.DataFrame.from_records(
            out_dpsi, columns=["header_long", "exon", "gene_name", "dPSI"]
        ).drop_duplicates()

    return out, out_dpsi, _with_NAs

def _get_slack(
    seq_record: pd.Series,
    region: Literal["upstream", "downstream"],
    _level: str,
    extend_borders: int,
    no_model_resolution: bool,
    model: str
):
    """
    Returns the number of base pairs to extend
    coordinates surrounding the central exon.
     
    If `no_model_resolution` is `True` returns the number
    of base pairs up to location of the  upstream or 
    downstream exons, if their distance is lower
    than the model resolution. If the distance is higher, 
    returns the distance to the model resolution. If `False`,
    returns the number of base pairs corresponding to the model
    resolution (e.g., 5000bp on each side for SpliceAI).

    Args:
        seq_record (pd.Series): A row from the input dataframe
        region (Literal): Either 'upstream' or 'downstream'
        _level (str): The level to be considered
        extend_borders (int): The number of base pairs to extend the coordinates,
    regardless of the region
        no_model_resolution (bool): Whether to avoid using the model resolution
    to extract distances
        model (str): The model to be used

    Returns:
        int: The number of base pairs to extend the coordinates
    """
    models = {"spliceai": 5000, "pangolin": 5000}
    model_res = models[model]
    if no_model_resolution is False:
        return model_res
    
    if region == "upstream":

        if seq_record.Strand == "+":
            if seq_record.Start - seq_record["End_upstream{}".format(_level)] >= model_res:
                return model_res

            else:
                return (
                    seq_record.Start
                    - seq_record["Start_upstream{}".format(_level)]
                    + extend_borders
                )

        else:
            if seq_record["Start_upstream{}".format(_level)] - seq_record.End >= model_res:

                return model_res
            else:
                return (
                    seq_record["End_upstream{}".format(_level)]
                    - seq_record.End
                    + extend_borders
                )

    else:
        if seq_record.Strand == "+":
            if seq_record["Start_downstream{}".format(_level)] - seq_record.End >= model_res:
                return model_res
            else:
                return (
                    seq_record["End_downstream{}".format(_level)]
                    - seq_record.End
                    + extend_borders
                )
        else:

            if seq_record.Start - seq_record["End_downstream{}".format(_level)] >= model_res:
                return model_res
            else:
                return (
                    seq_record.Start
                    - seq_record["Start_downstream{}".format(_level)]
                    + extend_borders
                )
