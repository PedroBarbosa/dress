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

def input_options(fun):
    fun = click.option(
    "-od",
    "--outdir",
    default="output",
    help="Path to where output files will be written.",
    )(fun)
    
    fun = click.option(
        "-ob",
        "--outbasename",
        help="Basename to include in the output files.",
    )(fun)
    
    fun = click.option(
    "-vb",
    "--verbosity",
    default=0,
    type=click.IntRange(0, 1),
    help="Verbosity level of the logger. Default: 0. If '1', debug "
    "messages will be printed.",
    )(fun)
    
    fun = click.option(
    "-i",
    "--input_seq",
    type=click.File(),
    help="File with information about original sequence. If not provided, we will try to automatically extract from the dataset.",
    )(fun)
    
    fun =click.option(
    "-d",
    "--dataset",
    metavar="e,g. file1 file2 ...",
    type=tuple,
    cls=OptionEatAll,
    required=True,
    help="File(s) referring to the dataset generated.",
    )(fun)
    
    fun = click.option(
    "-ai",
    "--another_input_seq",
    type=click.File(),
    help="File with information about original sequences in a second dataset. If not provided, we will try to automatically extract from the second dataset.",
    )(fun)
    
    fun = click.option(
    "-ad",
    "--another_dataset",
    metavar="e,g. file1 file2 ...",
    type=tuple,
    cls=OptionEatAll,
    help="File(s) referring to a second dataset generated.",
    )(fun)
    
    fun = click.option(
    "-g",
    "--groups",
    metavar="e,g. group1 group2",
    type=tuple,
    cls=OptionEatAll,
    help="Group name(s) for dataset(s) argument.",
    )(fun)
    
    fun = click.option(
    "-l",
    "--list",
    is_flag=True,
    help="If '--dataset' and '--another_dataset' represent a list of files, one per line.",
    )(fun)
    
    return fun

def motif_core_options(fun):
    fun = click.option(
    "-mdb",
    "--motif_db",
    type=click.Choice(
        ["encode2020_RBNS", "rosina2017", "oRNAment", "ATtRACT", "cisBP_RNA"]
    ),
    default="ATtRACT",
    help="Motif database, either in meme PWM. Default: 'ATtRACT'. Custom PWM databases "
    "can be used by providing the path to the PWM file. This file must be in meme format.",
    )(fun)
    
    fun = click.option(
    "-ms",
    "--motif_search",
    type=click.Choice(["plain", "fimo", "biopython"]),
    default="fimo",
    help="How to search motifs in sequences. Default: 'fimo'",
    )(fun)
    
    fun = click.option(
    "-sr",
    "--subset_rbps",
    metavar="e,g. rbp1 rbp2 ...",
    type=tuple,
    cls=OptionEatAll,
    default=["encode"],
    help="Subset motif scanning for the given list of RNA Binding Proteins (RBPs), or "
    "to the RBPs belonging to a specific set. Default: ['encode'], list of "
    "splicing-associated RBPs identified in the context of the ENCODE project.",
    )(fun)

    fun = click.option(
    "-mnp",
    "--min_nucleotide_probability",
    type=float,
    default=0.15,
    help="Minimum probability for a nucleotide in a given position of the PWM "
    "for it to be considered as relevant. Default: 0.15",
    )(fun)

    fun = click.option(
    "-mml",
    "--min_motif_length",
    type=int,
    default=5,
    help="Minimum length of a sequence motif to search in the sequences. Default: 5",
    )(fun)

    fun = click.option(
    "-pt",
    "--pvalue_threshold",
    type=float,
    default=0.0001,
    help="Maximum p-value threshold from FIMO output to consider a motif occurrence as valid. Default: 0.0001",
    )(fun)
    
    fun = click.option(
    "-qt",
    "--qvalue_threshold",
    type=float,
    default=None,
    help="Apply additional q-value threshold (control for multiple testing) when filtering FIMO output to consider a motif occurrence as valid. Default: None. If not set, it will not be used. Default: None",
    )(fun)
    
    fun = click.option(
    "-pt",
    "--pssm_threshold",
    type=float,
    default=3,
    help="Log-odds threshold to consider a match against a PSSM score as valid when '--motif_search' is set to 'biopython'. Default: 3",
    )(fun)
      
    return fun

def motif_extra_options(fun):
    fun = click.option(
    "--just_estimate_pssm_threshold",
    is_flag=True,
    help="If set, it does not run the motif search. Rather, it estimates what is a "
    "reasonably good log-odds threshold to consider for the PSSMs of a single gene/RBP. "
    "Only used when '--motif_search' is set to 'biopython'.",
    )(fun)
    
    fun = click.option(
    "-srmf",
    "--skip_raw_motifs_filtering",
    is_flag=True,
    help="Disable the filtering of raw motif hits. By default, raw motif hits are filtered "
    "such self-contained hits of the same RBP are removed and high-density regions are identified.",
    )(fun)
    
    return fun

@click.command(name="evaluate")
@click.argument("evaluator", type=click.Choice(["phenotypes", "sequences", "motifs"]))
@input_options
@motif_core_options
@motif_extra_options
@click.option(
    "-cf", "--config", help="YAML config file with values for all hyperparameters. If set, "
    "it overrides all other non-mandatory arguments. Default: None. A working "
    "config file is presented in 'dress/configs/evaluate.yaml'.",
)
def evaluate(**args):
    """
    (ALPHA) Evaluate synthetic dataset(s) produced by <dress generate> or <dress filter>.

    EVALUATOR: Type of evaluation to perform.

    If 'phenotypes', it will look at the positions and genotypes of the perturbations applied to each sequence.\n
    If 'sequences', it will look at the sequences themselves.\n
    If 'motifs' it will look at the motifs ocurrences found in the sequences.
    """
    args = check_args(args)
    args["logger"] = setup_logger(level=int(args["verbosity"]))

    os.makedirs(args["outdir"], exist_ok=True)

    dataset_obj = structure_dataset(
        generated_datasets=[args["dataset"], args["another_dataset"]],
        original_seqs=[args["input_seq"], args["another_input_seq"]],
        **args,
    )

    # dump_yaml(os.path.join(args["outdir"], "args_used.yaml"), **args)
    args.pop("dataset")
    if args["evaluator"] == "phenotypes":
        evaluator = PhenotypeEvaluator(dataset=dataset_obj, save_plots=True, **args)
        evaluator.plots.update(
            evaluator.repr.visualize(split_effects=True, zoom_in=False)
        )
        evaluator.plots.update(
            evaluator.repr.visualize(split_effects=False, zoom_in=True)
        )
        evaluator.plots.update(evaluator.repr.visualize(split_seeds=True))
        evaluator.plots.update(evaluator.repr.phenotypes[0].visualize())
        evaluator(clustering=None)

    elif args["evaluator"] == "sequences":
        evaluator = SequenceEvaluator(dataset=dataset_obj, save_plots=True, **args)
        evaluator(classification=None, dim_reduce=TSNE(n_components=3))

    elif args["evaluator"] == "motifs":
        evaluator = MotifEvaluator(
            dataset=dataset_obj,
            save_plots=True,
            disable_motif_representation=True,
            **args,
        )
        evaluator.motif_enrichment.visualize()
        evaluator(classification=None, dim_reduce=PCA(n_components=10))