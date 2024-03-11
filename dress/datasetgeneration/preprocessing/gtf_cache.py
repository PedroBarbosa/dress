import os
from dress.datasetgeneration.os_utils import (
    open_fasta,
    write_fasta_sequences,
    write_bed_file,
)
from loguru import logger
import pandas as pd
from typing import Tuple
import pyranges as pr

from dress.datasetgeneration.preprocessing.utils import (
    generate_pipeline_input,
    process_ss_idx,
)


def extractGeneStructure(
    data: pd.DataFrame, cache_dir: str, genome: str, level: int = 2
) -> Tuple[pd.DataFrame]:
    """
    Extract genomic context for the exons presented in input data, up to a given level
    of resolution. If level is 0, it extracts features about the input exonss.
    If level is 1, it extracts information about the surrounding introns. If
    the level is 2 (default), it extracts information about the surrounding
    introns and exons.
    """

    logger.info("Reading cached exons to extract genomic context.")
    _f = os.path.join(cache_dir, "Exons_level_{}.tsv.gz".format(level))

    # Read cache
    exons = pd.read_csv(_f, sep="\t")

    int_cols = [
        x
        for x in exons.columns
        if any(j in x for j in ["upstream", "downstream"])
        and all(j not in x for j in ["GC", "Feature"])
    ]
    exons[int_cols] = exons[int_cols].astype("Int64", errors="ignore")
    exons.drop(list(exons.filter(regex="Length_|GC_")), axis=1, inplace=True)

    # Subset cache if gene_name|gene_id|transcript_id exist in the input data
    subset_by = ["transcript_id", "gene_name", "gene_id"]
    filt_col_found = False

    for filter_by in subset_by:
        if filt_col_found is False and filter_by in data.columns:
            filt_col_found = True
            logger.info(
                f"Column {filter_by} found in the input data. Filtering cache by those IDs."
            )

            ids = data[filter_by].unique()
            exons = exons[exons[filter_by].isin(ids)]

            if exons.shape[0] == 0:
                raise ValueError(
                    f"No exons found in the cache for the IDs present in {filter_by} column of input"
                )

    # Merging
    cols = ["Chromosome", "Start", "End"]
    for c in ["Strand", "gene_name", "gene_id", "transcript_id", "transcript_type"]:
        if c in data.columns:
            cols.append(c)

    df = pd.merge(
        data, exons, how="left", on=cols, suffixes=("_repeat", ""), indicator=True
    )

    df = df[[c for c in df.columns if not c.endswith("_repeat")]]

    # Split by known/unknown
    known = df[df._merge == "both"].drop(columns="_merge")
    absent_in_cache = df[df._merge == "left_only"].drop(columns="_merge")[data.columns]

    if absent_in_cache.shape[0] > 0:
        logger.warning(f"{absent_in_cache.shape[0]} exons were not found in the cache.")

    # If there are known exons
    if known.shape[0] > 0:
        # Select the top ranked transcript, if more than one exists
        output = (
            known.groupby(cols, group_keys=False)
            .apply(lambda x: x.nlargest(1, "rank_score"))
            .reset_index(drop=True)
        )

    else:
        absent_in_cache = data
        output = []

    return output, absent_in_cache


def preprocessing(data: pr.PyRanges, **kwargs):
    """
    Extract genomic context for the exons presented in input data,
    using a cache previously generated from a GTF annotation file,
    and transforms the data into a suitable structure for the evolutionary algorithm.

    Args:
        data (pd.DataFrame): Input data with exons coordinates.
        kwargs (dict): Additional arguments:
            cache_dir (str): Path to the directory where the cache is stored.
            genome (str): Path to the genome fasta file.
            level (int): Level of genomic context to extract.
    """

    cache_dir = kwargs["cache_dir"]
    genome = open_fasta(kwargs["genome"], kwargs["cache_dir"])
    level = kwargs.get("level", 2)

    extracted, absent_in_gtf = extractGeneStructure(
        data.df, cache_dir=cache_dir, genome=genome, level=level
    )

    data, dpsi_info, na_exons = generate_pipeline_input(
        df=extracted,
        fasta=genome,
        extend_borders=100,
        use_full_seqs=kwargs["use_full_sequence"],
    )

    if os.path.isdir(kwargs["outdir"]):
        write_output(
            extracted=extracted,
            absent_in_gtf=absent_in_gtf,
            extracted_with_seqs=data,
            dpsi_info=dpsi_info,
            with_NAs=na_exons,
            level=level,
            **kwargs,
        )

    return process_ss_idx(data, return_seqs=True)


def write_output(
    extracted: pd.DataFrame,
    absent_in_gtf: pd.DataFrame,
    extracted_with_seqs: pd.DataFrame,
    with_NAs: pd.DataFrame,
    dpsi_info: pd.DataFrame,
    level: int,
    **kwargs,
):
    """
    Write preprocessing output to disk, so that this step can be skipped in the future.

    Args:
        extracted (pd.DataFrame): Exons from the input that were found in the cache.
        absent_in_gtf (pd.DataFrame): Exons from the input that were not found in the cache.
        Additional arguments in **kwargs:
            outdir (str): Output directory.
            outbasename (str): Output basename.
            use_full_sequence (bool): Whether to use the full sequence when running the black box model.
    """

    to_write = {
        "extracted_from_cache.bed": extracted,
        "absent_in_cache.bed": absent_in_gtf,
        "first_or_last_exon.tsv": with_NAs,
    }

    outpath = os.path.join(kwargs["outdir"], "preprocessing")

    for name, _df in to_write.items():
        if isinstance(_df, pd.DataFrame) and _df.shape[0] > 0:
            if name == "first_or_last_exon.tsv" and level != 0:
                logger.warning(
                    f"{with_NAs.shape[0]} exon(s) are first or last within the associated transcript. "
                    "These exons were ignored, since it was not possible to extract surrounding context."
                )
            bed6 = True if "Strand" in _df.columns else False
            write_bed_file(
                _df,
                name=outpath + "_" + name,
                bed6=bed6,
                additional_fields=[
                    x
                    for x in list(_df)
                    if "Col_" in x
                    or x
                    in [
                        "gene_name",
                        "gene_id",
                        "transcript_id",
                        "transcript_type",
                        "dPSI",
                    ]
                ],
            )

    out_flag = "" if kwargs["use_full_sequence"] else "_trimmed_at_5000bp"
    if len(extracted_with_seqs) > 0:
        extracted_with_seqs[
            ["header", "acceptor_idx", "donor_idx", "tx_id", "exon"]
        ].to_csv(
            outpath + "_sequences{}_ss_idx.tsv".format(out_flag),
            sep="\t",
            index=False,
        )

        write_fasta_sequences(
            extracted_with_seqs,
            outname=outpath + "_sequences{}.fa".format(out_flag),
            seq_col="seq",
            header_col="header_long",
        )

    if len(dpsi_info) > 0:
        dpsi_info.to_csv(outpath + "_dPSI.tsv", sep="\t", index=False)
