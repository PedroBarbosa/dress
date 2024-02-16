
import rich_click as click
import os

from dress.datasetgeneration.dataset import structure_dataset
from dress.datasetevaluation.off_the_shelf.dimensionality_reduction import PCA, TSNE

from dress.datasetgeneration.logger import setup_logger
from dress.datasetevaluation.representation.motifs.evaluator import MotifEvaluator
from dress.datasetevaluation.representation.phenotypes.evaluator import (
    PhenotypeEvaluator,
)
from dress.datasetevaluation.representation.sequences.evaluator import SequenceEvaluator

from dress.datasetgeneration.run import OptionEatAll
from dress.datasetevaluation.validate_args import check_args


@click.command(name="explain")
@click.option(
    "-i",
    "--input_seq",
    type=click.File(),
    help="File with information about original sequence. If not provided, we will try to automatically extract from the dataset.",
)
@click.option(
    "-d",
    "--dataset",
    metavar="e,g. file1 file2 ...",
    type=tuple,
    cls=OptionEatAll,
    required=True,
    help="File(s) referring to the dataset generated.",
)
@click.option(
    "-ai",
    "--another_input_seq",
    type=click.File(),
    help="File with information about original sequences in a second dataset. If not provided, we will try to automatically extract from the second dataset.",
)
@click.option(
    "-ad",
    "--another_dataset",
    metavar="e,g. file1 file2 ...",
    type=tuple,
    cls=OptionEatAll,
    help="File(s) referring to a second dataset generated.",
)
@click.option(
    "-od",
    "--outdir",
    default="output",
    help="Path to where output files will be written.",
)
@click.option(
    "-ob",
    "--outbasename",
    default="eval",
    help="Basename to include in the output files.",
)
@click.option(
    "-vb",
    "--verbosity",
    default=0,
    type=click.IntRange(0, 1),
    help="Verbosity level of the logger. Default: 0. If '1', debug "
    "messages will be printed.",
)
@click.option(
    "-g",
    "--groups",
    metavar="e,g. group1 group2",
    type=tuple,
    cls=OptionEatAll,
    help="Group name(s) for dataset(s) argument.",
)
@click.option(
    "-l",
    "--list",
    is_flag=True,
    help="If '--dataset' and '--another_dataset' represent a list of files, one per line.",
)
def explain(**args):
    """
    (ALPHA) Explain synthetic dataset(s) produced by <dress generate> or <dress filter>.
    """
    args = check_args(args)
    args["logger"] = setup_logger(level=int(args["verbosity"]))

    os.makedirs(args["outdir"], exist_ok=True)

    dataset_obj = structure_dataset(
        generated_datasets=[args["dataset"], args["another_dataset"]],
        original_seqs=[args["input_seq"], args["another_input_seq"]],
        **args,
    )

    args.pop("dataset")
