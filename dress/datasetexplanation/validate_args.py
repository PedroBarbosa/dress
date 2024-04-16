import sqlite3
import pandas as pd
import rich_click as click
import yaml
from dress.datasetevaluation.validate_args import check_input_args, check_motif_args
from dress.datasetexplanation.motif_db import create_db
from jsonschema import ValidationError, validate
from dress.datasetexplanation.json_schema import schema
from dress.datasetgeneration.json_schema import flatten_dict

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
                "--config",
                "--outdir",
                "--outbasename",
            ],
        },
           {
            "name": "Evolutionary algorithm",
            "options": [
                "--seed",
                "--minimize_fitness",
                "--fitness_function",
                "--population_size",
                "--stopping_criterium",
                "--stop_at_value",
                "--stop_when_all",
                "--selection_method",
                "--tournament_size",
                "--operators_weight",
                "--elitism_weight",
                "--novelty_weight",
                "--mutation_probability",
                "--crossover_probability",
                "--simplify_explanation",
                "--simplify_at_generations",
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
                "--motif_results",
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
    if args["config"]:
        mandatory = {k: v for k, v in args.items() if k in ["dataset"]}
        with open(args["config"], "r") as f:
            config = yaml.safe_load(f)

        try:
            validate(instance=config, schema=schema)
            _args = flatten_dict(config["evaluate"])
            args = {**mandatory, **_args}

        except ValidationError as e:
            print("YAML file validation failed:")
            print(e)
            exit(1)
        
        args["config"] = None
        return check_args(args)
    
    else:
        check_input_args(args)
        check_motif_args(args)
    
        if args['minimize_fitness'] and args['fitness_function'] == 'r2':
            raise click.UsageError("Do not set '--minimize_fitness' when fitness function is 'r2'.")
        
        elif args['fitness_function'] == 'rmse' and not args['minimize_fitness']:
            raise click.UsageError("Set '--minimize_fitness' when fitness function is 'rmse'.")
        
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
            args['db_to_query'] = cursor

        return args