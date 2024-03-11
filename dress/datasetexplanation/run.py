import sqlite3
import pandas as pd
import rich_click as click
import os
from dress.datasetevaluation.representation.motifs.evaluator import MOTIF_SEARCH_OPTIONS

from dress.datasetevaluation.run import input_options, motif_options
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

@click.command(name="explain")
@input_options
@motif_options
@click.option(
    "-mr",
    "--motif_results",
    help="Motif results (tabular, or sqlite) from a previous motif search for the same dataset.",
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

    kwargs.pop("dataset")
    if kwargs["motif_results"] is None:
        motif_searcher = MOTIF_SEARCH_OPTIONS.get(kwargs.get("motif_search"))
        motif_search = motif_searcher(dataset=dataset_obj.data, **kwargs)
        motif_search.tabulate_occurrences(write_output=True)
        motif_hits = motif_search.motif_results
        kwargs["motif_results"] = create_db(motif_hits, **kwargs)

    test_query_db(kwargs["motif_results"])
