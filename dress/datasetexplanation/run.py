import rich_click as click
import os
from dress.datasetevaluation.representation.motifs.evaluator import MOTIF_SEARCH_OPTIONS

from dress.datasetgeneration.run import OptionEatAll, evo_alg_options_core_options
from dress.datasetevaluation.run import input_options, motif_core_options
from dress.datasetexplanation.evolution import do_evolution
from dress.datasetexplanation.motif_db import create_db
from dress.datasetexplanation.validate_args import check_args
from dress.datasetgeneration.dataset import structure_dataset
from dress.datasetgeneration.logger import setup_logger


def test_query_db(c):
    from pypika import Query, Table, Case, Field
    from pypika import functions as fn
    import time

    print("\nQuerying database")

    motifs = Table("motifs")
    rbp = "SRSF3"

    for i in range(3):
        start_time = time.time()
        # q = (
        #     Query.from_(motifs)
        #     .select("Seq_id")
        #     .where((motifs.RBP_name == rbp)& (motifs.Start.between(750, 780)))
        # )
        q = Query.from_(motifs).select(motifs.Seq_id, fn.Count('*').as_('C')).where(
        (motifs.distance_to_cassette_donor.between(300, 500)) &
        (motifs.location == 'Intron_downstream')
    ).groupby(motifs.Seq_id)
        
        print(str(q))
        rows = c.execute(str(q)).fetchall()
        print(rows)
        print(time.time() - start_time, "seconds")
        print("\n")

    #c.close()
    # True sqlite syntax
    # c.execute("SELECT seq_id, COUNT(*) as C FROM motifs WHERE distance_to_donor BETWEEN 1000 AND 1010 AND rbp_name='SRSF1' AND location='Intron_upstream' GROUP BY seq_id")
    q = c.execute("SELECT * FROM motifs WHERE rbp_name=? AND location=? AND distance_to_donor >=? AND distance_to_donor <=?", ["HNRNPK", "Intron_upstream_2", "200", "500"])
    print(q.fetchall())

def evo_alg_options_explain(fun):
    fun = click.option(
        "-ff",
        "--fitness_function",
        type=click.Choice(
            ["r2", "rmse"], case_sensitive=True
        ),
        default="r2",
        help="Fitness function to score an individual (explanation). Default: 'r2', "
        "how well the features generated predict the dependent variable, the oracle score.",
    )(fun)
    
    fun = click.option(
        "-sc",
        "--stopping_criterium",
        cls=OptionEatAll,
        default=["r2", "time"],
        type=tuple,
        metavar=f"STRING + ... e.g. -sc r2 n_generations. ({'|'.join(['n_evaluations', 'n_generations', 'time', 'rmse', 'r2'])})",
        help="Criteria to stop evolution. If multiple criteria are given evolution will "
        "end when any or all the criteria are met, according to the '--stop_when_all' arg. "
        "Default: ['r2', 'n_generations'], evolution finishes when one of the criterium "
        "is met. ",
    )(fun)
    
    fun = click.option(
        "-sat",
        "--stop_at_value",
        cls=OptionEatAll,
        type=tuple,
        default=[0.9, 30],
        metavar="INTEGER|FLOAT + ... e.g. -sat 0.9 50.",
        help="Value to stop evolution based on the '--stopping_criterium'. Default: [0.9, 30], "
        "considering that default '--stopping_criterium' is '['r2', 'n_generations']'.",
    )(fun)
    
    fun = click.option(
        "-ow",
        "--operators_weight",
        type=float,
        default=0.6,
        help="Weight given to genetic operators when doing selection. Default: 0.6, 60 percent of "
        "the individuals in the population will be subjected to selection_method|mutation|crossover operators.",
    )(fun)
    
    fun = click.option(
        "-ew",
        "--elitism_weight",
        type=float,
        default=0.05,
        help="Weight given to elitism when doing selection. Default: 0.05, the top 5 percent of "
        "the population will be selected for the next generation.",
    )(fun)
    
    fun = click.option(
        "-nw",
        "--novelty_weight",
        type=float,
        default=0.35,
        help="Weight given to novelty when doing selection. Default: 0.35, 35 percent of "
        "individuals at the next generation will be novel.",
    )(fun)
    
    fun = click.option(
        "-se",
        "--simplify_explanation",
        is_flag=True,
        help="Simplify explanations to deal with bloat of individual trees. Default: 'False'. "
        "If set, it will prune GP trees at the end of the evolution.",
    )(fun)
    
    fun = click.option(
        "-sat",
        "--simplify_at_generations",
        metavar="INTEGER + ... e.g. -sat 10 20",
        type=tuple,
        cls=OptionEatAll,
        help="At which generation(s) (besides the end of evolution) pruning "
        "of the best explanation tree should be performed when '--simplify_explanation' "
        "is set.",
    )(fun)
    return fun 

@click.command(name="explain")
@input_options
@evo_alg_options_core_options
@evo_alg_options_explain
@motif_core_options
@click.option(
    "-mr",
    "--motif_results",
    help="Motif results (tabular, or sqlite) from a previous motif search for the same dataset.",
)
@click.option(
    "-cf", "--config", help="YAML config file with values for all hyperparameters. If set, "
    "it overrides all other non-mandatory arguments. Default: None. A working "
    "config file is presented in 'dress/configs/explain.yaml'.",
)
def explain(**kwargs):
    """
    (ALPHA) Explain synthetic dataset(s) produced by <dress generate> or <dress filter>.
    """
    logger = setup_logger(level=int(kwargs["verbosity"]))
    kwargs["logger"] = logger
    kwargs = check_args(kwargs)

    os.makedirs(kwargs["outdir"], exist_ok=True)

    dataset_obj = structure_dataset(
        generated_datasets=[kwargs["dataset"], kwargs["another_dataset"]],
        original_seqs=[kwargs["input_seq"], kwargs["another_input_seq"]],
        **kwargs,
    )

    if kwargs["motif_results"] is None:
        motif_searcher = MOTIF_SEARCH_OPTIONS.get(kwargs.get("motif_search"))
        motif_search = motif_searcher(dataset=dataset_obj.data, **kwargs)
        motif_search.tabulate_occurrences(write_output=True)
        motif_hits = motif_search.motif_results
        _db = create_db(motif_hits, **kwargs)
    else:
        _db = kwargs["db_to_query"]
        kwargs.pop("db_to_query")

    #test_query_db(kwargs["motif_results"])
    do_evolution(dataset_obj, _db, **kwargs)
