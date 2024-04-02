import rich_click as click
import yaml
from jsonschema import ValidationError, validate
from dress.datasetevaluation.json_schema import schema
from dress.datasetgeneration.json_schema import flatten_dict

EVALUATE_GROUP_OPTIONS = {
    "dress evaluate": [
        {
            "name": "Input / Output options",
            "options": [
                "--dataset",
                "--input_seq",
                "--another_dataset",
                "--another_input_seq",
                "--groups",
                "--list",
                "--config",
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

def check_input_args(args) -> None:
    
    if args["list"]:
        for dataset in [args["dataset"], args["another_dataset"]]:
            if dataset and len(dataset) > 1:
                raise click.UsageError(
                        "If '--list' is provided, '--dataset' and '--another_dataset' "
                        "must be a single file listing one dataset per line."
                )
        args["dataset"] = [line.rstrip("\n") for line in open(args["dataset"][0])]

        if args["another_dataset"]:
            groups = args["groups"] or ["1", "2"]
            if len(groups) != 2:
                raise click.UsageError(
                    f"When '--another_dataset' is provided, '--groups' must have two elements.  Groups given: {args['groups']}"
                )
            args["another_dataset"] = [line.rstrip("\n") for line in open(args["another_dataset"][0])]

        if args["groups"] and len(args["groups"]) != len(
            [d for d in [args["dataset"], args["another_dataset"]] if d]
        ):
            raise click.UsageError(
                f"The number of '--groups' must be equal to the number of datasets provided. Groups given: {args['groups']}"
            )
            
def check_motif_args(args):

    if args["motif_db"] in ["rosina2017", "encode2020_RBNS"]:
                if args["motif_search"] != "plain":
                    raise click.UsageError(
                        "When '--motif_db' is 'rosina2017' or 'encode2020_RBNS', "
                        "'--motif_search' must be 'plain'."
                    )

    if len(args["subset_rbps"]) == 1:
        args["subset_rbps"] = args["subset_rbps"][0]

    if ("just_estimate_pssm_threshold" in args.keys()
        and args["just_estimate_pssm_threshold"]
        and args["motif_search"] != "biopython"
    ):
        raise click.UsageError(
            "'--just_estimate_pssm_threshold' flag must be disabled when '--motif_search' is not 'biopython'."
        )
        
def check_args(args) -> dict:
    """
    Checks if the combination of given arguments is valid.
    """
    if args["config"]:
        mandatory = {k: v for k, v in args.items() if k in ["evaluator", "dataset"]}

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
        if args["evaluator"] == "motifs":
            check_motif_args(args)
        return args
