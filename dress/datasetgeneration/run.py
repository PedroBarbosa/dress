import rich_click as click
import os
import re
import random
from pathlib import Path


from dress.datasetgeneration.os_utils import (
    fasta_to_dict,
    return_dataset,
    write_dataset,
    write_input_seq,
)
from dress.datasetgeneration.evolution import (
    do_evolution,
    get_score_of_input_sequence,
)
from dress.datasetgeneration.preprocessing.utils import (
    process_ss_idx,
    tabular_file_to_genomics_df,
)

from dress.datasetgeneration.validate_args import check_args
from dress.datasetgeneration.preprocessing.gtf_cache import preprocessing
from dress.datasetgeneration.logger import setup_logger
import tensorflow as tf


DATA_PATH = Path(__file__).parents[2] / "data"


class OptionEatAll(click.Option):
    """
    This class is a workaround for the fact that click does not support nargs='*'
    Copied from https://stackoverflow.com/questions/48391777/nargs-equivalent-for-options-in-click
    """

    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop("save_other_options", True)
        nargs = kwargs.pop("nargs", -1)
        assert nargs == -1, "nargs, if set, must be -1 not {}".format(nargs)
        super(OptionEatAll, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):
        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break

        return retval


@click.command(
    name="generate",
)
@click.argument(
    "input",
    type=click.Path(exists=True, resolve_path=True),
)
@click.option(
    "-od",
    "--outdir",
    default="output",
    help="Output directory. Default: 'output'.",
)
@click.option(
    "-ob",
    "--outbasename",
    help="Output basename for the run when input contains one single sequence "
    "to evolve. Default: extracted from seq ID.",
)
@click.option(
    "-cd",
    "--cache_dir",
    type=click.Path(exists=True, resolve_path=True),
    default=f"{DATA_PATH}/cache/",
    help="Directory where exon cache is located. Required when 'input' is 'bed' or 'tabular'. "
    "Default: 'data/cache/'.",
)
@click.option(
    "-gn",
    "--genome",
    type=click.Path(exists=True, resolve_path=True),
    default=f"{DATA_PATH}/cache/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
    help="Genome in fasta format. Only used when 'input' is 'bed' or 'tabular'.",
)
@click.option(
    "-cf",
    "--config",
    type=click.Path(exists=True, resolve_path=True),
    help="YAML config file with values for all hyperparameters. If set, "
    "it overrides all other non-mandatory arguments. Default: None. A working "
    "config file is presented in 'dress/configs/generate.yaml' file.",
)
@click.option(
    "-ufs",
    "--use_full_sequence",
    is_flag=True,
    help="Whether to extract and predict the full sequence "
    "from the start coordinate of the upstream exon to the end coordinate of the "
    "downstream exon. Default: 'False': Use restricted sequence regions up to the resolution limit "
    "of the selected model (e.g, for SpliceaAI, 5000bp on each side of the target exon). "
    "Only used when input is 'bed' or 'tabular'. Use this option with caution, it can easily lead to "
    "memory exhaustion if the sequence triplet size is too large.",
)
@click.option(
    "-dr",
    "--dry_run",
    is_flag=True,
    help="Do a test run without making black box model inferences",
)
@click.option(
    "-dgpu",
    "--disable_gpu",
    is_flag=True,
    help="Disable running model inferences in a GPU. If set, it will critically slow down the evolutionary algorithm.",
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
    "-sd",
    "--seed",
    type=int,
    default=0,
    help="Seed to use in the evolution. Default: 0",
)
@click.option(
    "-md",
    "--model",
    type=click.Choice(["spliceai", "pangolin"]),
    default="spliceai",
    help="Deep learning model to use as the reference for the evolutionary algorithm. Default: spliceai",
)
@click.option(
    "-msm",
    "--model_scoring_metric",
    type=click.Choice(["mean", "max", "min"]),
    default="mean",
    help="Aggregation function to label an input sequence. If 'mean', the mean of the acceptor and donor scores is used. "
    "If 'max' or 'min', the max or min score between the acceptor and donor is used, respectively. Default: mean",
)
@click.option(
    "-pm",
    "--pangolin_mode",
    type=click.Choice(["ss_usage", "ss_probability"]),
    default="ss_usage",
    help="Which type of predictions to consider when '--model' is 'pangolin'. By default, it uses splice site usage, but splice site probabilities (like SpliceAI) can be used.")

@click.option(
    "-pm",
    "--pangolin_tissue",
    type=click.Choice(["heart", "liver", "brain", "testis"]),
    help="Use tissue specific predictions to generate the dataset when '--model' is 'pangolin'. Default: average predictions across all tissues.")

@click.option(
    "-mf",
    "--minimize_fitness",
    is_flag=True,
    help="Evolve sequences that minimize a given fitness function. Default: 'False', "
    "evolution maximizes the given fitness function.",
)
@click.option(
    "-ff",
    "--fitness_function",
    type=click.Choice(
        ["bin_filler", "increase_archive_diversity"], case_sensitive=True
    ),
    default="bin_filler",
    help="Fitness function to use to score an individual sequence. Default: 'bin_filler', "
    "a sequence is added to the archive if the black box prediction falls in a score bucket "
    "that is not filled.",
)
@click.option(
    "-ft",
    "--fitness_threshold",
    type=float,
    default=0.0,
    help="Fitness threshold value to add sequences to the archive. Default: 0.0",
)
@click.option(
    "-as",
    "--archive_size",
    type=int,
    default=5000,
    help="Number of desired sequences in archive to generate a per-bin target size when "
    "the '--fitness_function' is 'bin_filler.",
)
@click.option(
    "-adm",
    "--archive_diversity_metric",
    default="normalized_shannon",
    type=click.Choice(["normalized_shannon"]),
    help="Metric to measure diversity of the archive in a given evolution step. "
    "Default: 'normalized_shannon'.",
)
@click.option(
    "-ps",
    "--population_size",
    type=int,
    default=1000,
    help="Size of the population. Default: 1000",
)
@click.option(
    "-sc",
    "--stopping_criterium",
    cls=OptionEatAll,
    default=["archive_size", "time"],
    type=tuple,
    # type=click.Choice(
    #     ["n_evaluations", "n_generations", "time", "archive_quality", "archive_size", "archive_diversity"],
    #     case_sensitive=True,
    # ),
    metavar=f"STRING + ... e.g. -sc n_evaluations n_generations. ({'|'.join(['n_evaluations', 'n_generations', 'time', 'archive_quality', 'archive_size', 'archive_diversity'])})",
    help="Criteria to stop evolution. If multiple criteria are given evolution will "
    "end when any or all the criteria are met, according to the '--stop_when_all' arg. "
    "Default: ['archive_size', 'time'], evolution finishes when one of the criterium "
    "is met, according to the values provided in '--stop_at_value' arg. ",
)
@click.option(
    "-sat",
    "--stop_at_value",
    cls=OptionEatAll,
    type=tuple,
    default=[5000, 30],
    metavar="INTEGER|FLOAT + ... e.g. -sat 10000 50.",
    help="Value to stop evolution based on the '--stopping_criterium'. Default: [5000, 30], "
    "considering that default '--stopping_criterium' is '['archive_size', 'time']'.",
)
@click.option(
    "-swa",
    "--stop_when_all",
    is_flag=True,
    help="Stop evolution when all '--stopping_criterium' provided are met. Default: 'False'. "
    "Stop evolution when any of the '--stopping_criterium' is met.",
)
@click.option(
    "-dt",
    "--disable_tracking",
    is_flag=True,
    help="Whether any tracking during evolution (population and archive-wide) should be disabled. ",
)
@click.option(
    "-tfp",
    "--track_full_population",
    is_flag=True,
    help="Whether several properties of the whole population should be tracked during evolution. "
    "Default: 'False', only the best individual (with highest fitness) will be recorded.",
)
@click.option(
    "-tfa",
    "--track_full_archive",
    is_flag=True,
    help="Whether all individuals in the archive should be tracked during evolution. "
    "Default: 'False', only some overall metrics of the archive will be recorded.",
)
@click.option(
    "-pai",
    "--prune_archive_individuals",
    is_flag=True,
    help="Simplify individual genotypes so that irrelevant perturbations "
    "(that do not change sequence score) are removed. Default: 'False'. "
    "If set, it will prune the archive individuals at the end of the evolution.",
)
@click.option(
    "-pag",
    "--prune_at_generations",
    metavar="INTEGER + ... e.g. -pag 10 20",
    type=tuple,
    cls=OptionEatAll,
    help="At which generation(s) (besides the end of evolution) pruning "
    "of archive individuals should be performed when '--prune_archive_individuals' "
    "is set.",
)
@click.option(
    "-sm",
    "--selection_method",
    type=click.Choice(["tournament", "lexicase"], case_sensitive=True),
    default="tournament",
    help="Selection method to use. Default: 'tournament'.",
)
@click.option(
    "-ts",
    "--tournament_size",
    type=int,
    default=5,
    help="Number of individuals to be randomly selected from the population to do "
    "a tournament when '--selection_method' is 'tournament'. Default: 5",
)
@click.option(
    "-ow",
    "--operators_weight",
    metavar="FLOAT + ... e.g. -ow 0.6 0.2",
    cls=OptionEatAll,
    type=tuple,
    default=[0.6],
    help="Weight(s) given to genetic operators when doing selection. Default: 0.6, 60 percent of "
    "the individuals in the population will be subjected to selection_method|mutation|crossover operators. "
    "If multiple values are given, the weight will be updated at the given generation(s). "
    "(see '--update_weights_at_generation'). The first value always refers to the initial weight, at "
    "generation 0.",
)
@click.option(
    "-ew",
    "--elitism_weight",
    metavar="FLOAT + ... e.g. -ew 0.05 0.4",
    cls=OptionEatAll,
    type=tuple,
    default=[0.05],
    help="Weight(s) given to elitism when doing selection. Default: 0.05, the top 5 percent of "
    "the population will be selected for the next generation. If multiple values are given, "
    "the elitism weight will be updated at the given generation(s) (see '--update_weights_at_generation'). "
    "The first value always refers to the initial weight, at generation 0.",
)
@click.option(
    "-nw",
    "--novelty_weight",
    metavar="FLOAT + ... e.g. -nw 0.35 0.4",
    cls=OptionEatAll,
    type=tuple,
    default=[0.35],
    help="Weight(s) given to novelty when doing selection. Default: 0.05, 5 percent of "
    "individuals at the next generation will be novel. If multiple values are given, "
    "the novelty weight will be updated at the given generation(s) (see '--update_weights_at_generation'). "
    "The first value always refers to the initial weight, at generation 0.",
)
@click.option(
    "-uw",
    "--update_weights_at_generation",
    metavar="INTEGER... e.g. -uw 10 20",
    cls=OptionEatAll,
    type=tuple,
    help="At which generation(s) selection weights should be updated when dynamic selection "
    "rates are desired (e.g., when multiple values are given in '--operators_weight').",
)
@click.option(
    "-cmo",
    "--custom_mutation_operator",
    is_flag=True,
    help="Use a custom mutation operator that mutates individuals by selecting positions in "
    "the sequence that are close to existing perturbed positions in the same individual. "
    "(e.g, to foster the combination of perturbations affecting a single binding "
    "motif). Default: 'False'",
)
@click.option(
    "-cmow",
    "--custom_mutation_operator_weight",
    type=float,
    default=0.9,
    help="When '--custom_mutation_operator' is set, what is the weight (probability) of that operator "
    "to be used in respect to the default (random) mutation operator. This does not affect the "
    "'--mutation_probability' (probability that individuals exposed to a MutationStep will be "
    "actually mutated). Default: '0.9', 90 percent using the custom mutation operator, 10 percent "
    "using the default random operator.",
)
@click.option(
    "-mp",
    "--mutation_probability",
    type=float,
    default=0.9,
    help="Probability of an individual to be mutated when passing through a MutationStep. Default: 0.9",
)
@click.option(
    "-cr",
    "--crossover_probability",
    type=float,
    default=0.01,
    help="Probability of an individual to be subjected to crossover operator when passing through a "
    "CrossoverStep. Default: 0.01",
)
@click.option(
    "-ir",
    "--individual_representation",
    type=click.Choice(["tree_based"], case_sensitive=True),
    default="tree_based",
    help="Representation of an individual of the population. Default: 'tree_based'.",
)
@click.option(
    "-mdu",
    "--max_diff_units",
    type=int,
    default=6,
    help="Max diffUnits (perturbations in a single sequence) allowed for each individual. Default: 6",
)
@click.option(
    "-snvw",
    "--snv_weight",
    type=float,
    default=0.33,
    help="Probability to generate a grammar node of type SNV. Default: 0.33",
)
@click.option(
    "-insw",
    "--insertion_weight",
    type=float,
    default=0.33,
    help="Probability to generate a grammar node of type RandomInsertion. Default: 0.33",
)
@click.option(
    "-delw",
    "--deletion_weight",
    type=float,
    default=0.33,
    help="Probability to generate a grammar node of type RandomDeletion. Default: 0.33",
)
@click.option(
    "-mis",
    "--max_insertion_size",
    type=int,
    default=5,
    help="Max size of a random insertion allowed in the grammar. Default: 5",
)
@click.option(
    "-mds",
    "--max_deletion_size",
    type=int,
    default=5,
    help="Max size of a random deletion allowed in the grammar. Default: 5",
)
@click.option(
    "-aur",
    "--acceptor_untouched_range",
    metavar="INTEGER... e.g. -aur -10 2",
    nargs=2,
    default=[-10, 2],
    help="How many basepairs should stay untouched in the surroundings of "
    "each splicing acceptor. Default: [-10, 2], last 10bp of the upstream "
    "intron and 2 first bp of the the exon. Disable restriction at the acceptor region with '-aur 0 0'.",
)
@click.option(
    "-dur",
    "--donor_untouched_range",
    metavar="INTEGER ... e.g. -dur -3 6",
    nargs=2,
    default=[-3, 6],
    help="How many basepairs should stay untouched in the surroundings of "
    "each splicing donor. Default: [-3, 6], last 3bp of the exon and first 6bp "
    "of the intron downstream. Disable restriction at the donor region with '-dur 0 0'",
)
@click.option(
    "-ur",
    "--untouched_regions",
    cls=OptionEatAll,
    type=tuple,
    # type=click.Choice(
    #    ['exon_upstream', 'intron_upstream', 'target_exon', 'intron_downstream', 'exon_downstream'],
    #     case_sensitive=True,
    # ),
    metavar=f"STRING + ... e.g. -sc exon_upstream exon_downstream. ({'|'.join(['exon_upstream', 'intron_upstream', 'target_exon', 'intron_downstream', 'exon_downstream'])})",
    help="Region(s) within the exon triplet that should stay untouched in the "
    "evolutionary search.",
)
def generate(**args):
    """
    Generate a local synthetic dataset from a single input sequence.

    INPUT: Path to the input data. Type:[fasta|bed|txt|tab|tsv]. If fasta (.fa|.fasta),
    it requires that a previous preprocessing step was already performed (splice site indexes extracted).
    If any of the tabular option is provided (txt|tab|tsv), it expects a header line with informative column names.)
    """
    args = check_args(args)
    logger = setup_logger(level=int(args["verbosity"]))
    os.makedirs(args["outdir"], exist_ok=True)

    if any(args["input"].endswith(ext) for ext in ["fa", "fasta"]):
        ext = os.path.splitext(args["input"])[1]
        seqs = fasta_to_dict(args["input"])
        ss_idx, _ = process_ss_idx(seqs, args["input"].replace(ext, "_ss_idx.tsv"))

    elif args["input"].endswith("bed"):
        raise NotImplementedError("Bed files are not supported yet.")

    elif any(args["input"].endswith(ext) for ext in ["tsv", "txt", "tab"]):
        df = tabular_file_to_genomics_df(
            args["input"],
            col_index=0,
            is_0_based=False,
            header=0,
        )

        seqs, ss_idx = preprocessing(df, **args)

    else:
        raise NotImplementedError(
            "Input file format not recognized [only *fa, *fasta, *bed, *tsv *txt, *tab allowed]."
        )

    if args["outbasename"] is not None and len(seqs) > 1:
        raise ValueError(
            "Do not use --outbasename argument when more than 1 sequence\
 is provided as input. Let the program automatically assign basenames based on unique seq IDs."
        )

    if args["disable_gpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.set_visible_devices([], "GPU")
    else:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:    
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                gpu = random.choice(gpus)
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu.name.split(":")[-1]
            except RuntimeError as e:
                print(e)
        
    for seq_id, SEQ in seqs.items():
        if args["outbasename"]:
            outbasename = args["outbasename"]
        else:
            _seq_id = seq_id.replace("(+)", "").replace("(-)", "")
            outbasename = re.sub(r"[:-]", "_", _seq_id)
            args["outbasename"] = outbasename

        logger.info(f"Starting seq {seq_id}")
        _input = {
            "seq_id": seq_id,
            "seq": SEQ,
            "ss_idx": ss_idx[seq_id],
            "dry_run": args["dry_run"],
        }

        outoriginalfn = f"{args['outdir']}/{outbasename}_original_seq.csv"
        outdatasetfn = (
            f"{args['outdir']}/{outbasename}_seed_{args['seed']}_dataset.csv.gz"
        )
        logger.info("Calculating original score")
        _input = get_score_of_input_sequence(
            _input, **args)
        

        write_input_seq(_input, outoriginalfn)
        archive = do_evolution(_input, **args)
        dataset = return_dataset(input_seq=_input, archive=archive)

        write_dataset(
            dataset,
            outdatasetfn,
            outbasename,
            seq_id,
            args["seed"],
        )

        args["outbasename"] = None
