import sqlite3
import pandas as pd
import rich_click as click
from dress.datasetevaluation.validate_args import check_input_args, check_motif_args
from dress.datasetexplanation.motif_db import create_db

EXPLAIN_GROUP_OPTIONS = {
    "dress explain": [
        {
            "name": "Input / Output options",
            "options": [
                "--dataset",
                "--input_seq",
                "--another_dataset",
                "--another_input_seq",
                "--list",
                "--groups",
                "--outdir",
                "--outbasename",
            ],
        },
        {
            "name": "Motifs-related options",
            "options": [
                "--motif_db",
                "--motif_search",
                "--subset_rbps",
                "--min_nucleotide_probability",
                "--min_motif_length",
                "--pssm_threshold",
                "--pvalue_threshold",
                "--qvalue_threshold",
                "--skip_raw_motifs_filtering",
                "--just_estimate_pssm_threshold",
                "--motif_enrichment",
            ],
        },
        {
            "name": "Other options",
            "options": ["--verbosity", "--help"],
        },
    ]
}


def check_args(args) -> dict:
    """
    Checks if the combination of given arguments is valid.
    """
    check_input_args(args)
    check_motif_args(args)
    
    if args['motif_results'] is not None:
        if args['motif_results'].endswith('MOTIF_MATCHES.tsv.gz'):
                motif_hits = pd.read_csv(args['motif_results'], sep='\t', compression='gzip')
                cursor = create_db(motif_hits, **args)
    
        else:
            # Check if it is a sqlite file
            try:
                conn = sqlite3.connect(args['motif_results'])
                cursor = conn.cursor()
            except sqlite3.DatabaseError as e:
                raise click.UsageError(f"'{args['motif_results']}' is not a valid SQLite database: {e}")
            
        args['motif_results'] = cursor
    return args