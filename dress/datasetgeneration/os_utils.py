import itertools
import os
from typing import List, Optional, Union

import yaml
from Bio import SeqIO
import pandas as pd
from pyfaidx import Fasta
import pyranges as pr

from dress.datasetgeneration.archive import Archive


def write_input_seq(input: dict, outfile: str):
    ss_idx_str = ";".join([str(ss) for exon in input["ss_idx"] for ss in exon])
    pd.DataFrame(
        {
            "Seq_id": [input["seq_id"]],
            "Sequence": [input["seq"]],
            "Splice_site_positions": [ss_idx_str],
            "Score": [input["score"]],
        }
    ).to_csv(outfile, index=False)


def write_dataset(
    dataset: list, outfile: str, outbasename: str, seq_id: str, seed: int
):
    out_generated = []

    if not dataset:
        out_generated.append(
            [
                outbasename,
                seed,
                seq_id,
                None,
                None,
                None,
                None,
                None,
            ]
        )
    else:
        for ind in dataset:
            phenotype, seq, _ss_idx, pred, pred_diff = (
                ind[0],
                ind[1],
                ind[2],
                ind[3],
                ind[4],
            )
            out_generated.append(
                [
                    outbasename,
                    seed,
                    seq_id,
                    phenotype,
                    seq,
                    _ss_idx,
                    pred,
                    pred_diff,
                ]
            )

        header = [
            "Run_id",
            "Seed",
            "Seq_id",
            "Phenotype",
            "Sequence",
            "Splice_site_positions",
            "Score",
            "Delta_score",
        ]

        pd.DataFrame(out_generated).to_csv(
            outfile,
            index=False,
            header=header,
            compression="gzip",
            lineterminator="\n",
        )


def return_dataset(input_seq: dict, archive: Archive) -> List[list]:
    """
    Return the final dataset

    Args:
        input_seq (dict): Info about the original sequence
        archive (Archive): Archive at the end of evolution

    Returns:
        List[list]: Final dataset
    """
    out_data = []
    seq = input_seq["seq"]
    ss_idx = input_seq["ss_idx"]
    original_pred = input_seq["score"]

    individuals = archive.instances
    for ind in individuals:
        phenotype = ind.get_phenotype()
        _seq, _ss_idx = phenotype.apply_diff(seq, ss_idx)
        _ss_idx = ";".join([str(ss) for ss in itertools.chain(*_ss_idx)])
        out_data.append(
            [str(ind), _seq, _ss_idx, ind.pred, round(ind.pred - original_pred, 4)]
        )

    out_data.sort(key=lambda x: x[1], reverse=True)
    return out_data


def assign_proper_basename(outbasename: str) -> str:
    """Assign a proper basename to the output files.

    Args:
        outbasename (str): Output basename passed to the main execution file

    Returns:
        str: Proper basename to be assigned to the output files.
    """
    output_bn = outbasename or ""
    if output_bn != "" and not output_bn.endswith("_"):
        output_bn += "_"
    return output_bn


def open_fasta(fasta: Union[str, Fasta], cache_dir: str) -> Fasta:
    """
    Opens a fasta file for fast random access.

    :param str fasta: Original Fasta file
    """

    if isinstance(fasta, Fasta):
        return fasta

    if fasta is not None:
        try:
            open(fasta).readline()
        except FileNotFoundError:
            raise FileNotFoundError(f"File {fasta} not found")

        except UnicodeDecodeError:
            raise ValueError("Make sure you provide an uncompressed fasta for pyfaidx")

        return Fasta(fasta)
    
    else:
        default_path = os.path.join(cache_dir, "GRCh38.primary_assembly.genome.fa")
        try:
            open(default_path).readline()
        except FileNotFoundError:
            raise FileNotFoundError("Fasta file was not originally provided. "
                                    "Tried to look instead into the cache directory for the default filename "
                                    f"{default_path}, but it was not found. "
                                    "Please provide a valid fasta file via the --genome option.")
        return Fasta(default_path)
    return None


def fasta_to_dict(fasta: str) -> dict:
    """
    Reads a fasta file into a dictionary

    :param str fasta: Input fasta

    :return dict: Processed dict where headers are the keys,
    and the values correspond to the sequences
    """
    return {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta, "fasta")}


def read_features_file(file: str):
    """
    Reads a file with feature IDs (exon, transcripts, etc)

    :param file: File with one feature ID per line
    :return list: List with IDs
    """
    try:
        with open(file, "r") as f:
            feature_ids = [line.rstrip() for line in f]
        f.close()
        return feature_ids
    except FileNotFoundError:
        raise ValueError("{} is not a valid file".format(file))


def write_fasta_sequences(
    df: Union[pr.PyRanges, pd.DataFrame],
    outname: str,
    seq_col: str = "Fasta",
    header_col: Optional[str] = None,
):
    """
    Write fasta file from a dataframe of features

    Args:
        df (Union[pr.PyRanges, pd.DataFrame]): Df with a set of sequences
        outname (str): Filename to write output
        seq_col (str, optional): Column where Fasta is located. Defaults to 'Fasta'.
        header_col (Optional[str], optional): Column to serve as fasta header. Default: Df index.
    """

    assert (
        seq_col in df.columns
    ), "Dataframe with genomic features must " "contain a {} column".format(seq_col)
    assert not os.path.isdir(outname), "A directory was provided as the output file"

    if isinstance(df, pr.PyRanges):
        df = df.as_df()

    out = open(outname, "w")
    if header_col:
        assert header_col in df.columns, "{} is not in the list of columns".format(
            header_col
        )
        _df = df.copy()

    else:
        header_col = df.index.name
        _df = df.reset_index()

    _df[header_col] = _df[header_col].apply(lambda x: "{}{}".format(">", x))
    for record in list(zip(_df[header_col], _df[seq_col])):
        out.write("\n".join(str(s) for s in record) + "\n")


def write_bed_file(
    data: pd.DataFrame,
    name: str,
    bed6: bool = False,
    compression: str = None,
    is_1_based: bool = False,
    use_as_name_or_score: Optional[dict] = None,
    additional_fields: Optional[list] = None,
):
    """
    Write bed from pandas Dataframe

    Args:
        data (pd.DataFrame): Data to write (Must contain Chromosome,
     Start and End) columns
        name (str): Filename to write output
        bed6 (bool, optional): Whether 6 column bed file should be written. Defaults to False.
        compression (str, optional): How to compress file. Default: None
        is_1_based (bool, optional): Whether input `data` owns 1-based coordinates. Defaults to False.
        use_as_name_or_score (dict, optional): Mapping of columns from `data` to be used as the
    name and/or score column when `bed6` is set to `True`. E.g. {'gene_name':'Name', 'exon_number:'Score'}
        additional_fields (list, optional): Add additional columns to the output.

    """

    assert all(
        x in data.columns for x in ["Chromosome", "Start", "End"]
    ), "Dataframe must contain required columns"

    if any(v is not None for v in [use_as_name_or_score, use_as_name_or_score]):
        assert (
            bed6 is True
        ), "'bed6' should be enabled to use 'use_as_name_or_score' or 'additional_fields' args"

    if additional_fields:
        assert isinstance(
            additional_fields, list
        ), "additional_fields argument requires a list"
        assert all(
            x in data.columns for x in additional_fields
        ), "Dataframe must contain all additional columns passed"

    _data = data.copy()

    if is_1_based is False:
        _data["Start"] -= 1

    if bed6:
        cols = ["Chromosome", "Start", "End", "Name", "Score", "Strand"]
        assert (
            "Strand" in _data.columns
        ), 'When "bed6" is set to True, "Strand" must exist'

        if isinstance(use_as_name_or_score, dict):
            assert all(
                x in ["Name", "Score"] for x in use_as_name_or_score.values()
            ), "'use_as_name_or_score' only accepts 'Name' and 'Score' as values."
            assert all(
                x in _data.columns for x in use_as_name_or_score.keys()
            ), "Columns set to be used as 'Name' and/or 'Score' do not exist in the data."

            # if Score and Name column already existed
            if "Score" in use_as_name_or_score.values() and "Score" in _data.columns:
                _data.drop("Score", axis=1, inplace=True)
            if "Name" in use_as_name_or_score.values() and "Name" in _data.columns:
                _data.drop("Name", axis=1, inplace=True)
            _data = _data.rename(columns=use_as_name_or_score)

        if additional_fields is not None:
            cols = cols + additional_fields

        try:
            _data = _data[cols]

        except KeyError:
            if "Name" not in _data.columns:
                _data["Name"] = "."
            if "Score" not in _data.columns:
                _data["Score"] = "."

            _data = _data[cols]

    else:
        _data = data[["Chromosome", "Start", "End"]]

    _data.to_csv(name, compression=compression, sep="\t", index=False, header=False)


def dump_yaml(filename, **kwargs):
    with open(filename, "w") as file:
        yaml.dump(kwargs, file, sort_keys=False)
