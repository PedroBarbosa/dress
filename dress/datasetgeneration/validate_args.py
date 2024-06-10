import os
import rich_click as click
import yaml
from jsonschema import ValidationError, validate
from dress.datasetgeneration.json_schema import schema, flatten_dict

GENERATE_GROUP_OPTIONS = {
    "dress generate": [
        {
            "name": "Input / Output options",
            "options": [
                "--shuffle_input",
                "--outdir",
                "--outbasename",
                "--cache_dir",
                "--genome",
                "--config",
                "--use_model_resolution",
                "--use_full_triplet"
            ],
        },
        {
            "name": "Grammar options and restriction of the search space",
            "options": [
                "--which_grammar",
                "--max_diff_units",
                "--snv_weight",
                "--insertion_weight",
                "--deletion_weight",
                "--motif_ablation_weight",
                "--motif_substitution_weight",
                "--acceptor_untouched_range",
                "--donor_untouched_range",
                "--untouched_regions",
                "--max_insertion_size",
                "--max_deletion_size",
                "--motif_db",
                "--motif_search",
                "--subset_rbps",
                "--min_nucleotide_probability",
                "--min_motif_length",
                "--pvalue_threshold"
            ],
        },
        {
            "name": "Motif based grammar options",
            "options": [

            ]
        },
        {
            "name": "Evolutionary algorithm",
            "options": [
                "--seed",
                "--model",
                "--batch_size",
                "--model_scoring_metric",
                "--pangolin_mode",
                "--pangolin_tissue",
                "--minimize_fitness",
                "--fitness_function",
                "--fitness_threshold",
                "--archive_size",
                "--archive_diversity_metric",
                "--population_size",
                "--stopping_criterium",
                "--stop_at_value",
                "--stop_when_all",
                "--selection_method",
                "--tournament_size",
                "--operators_weight",
                "--elitism_weight",
                "--novelty_weight",
                "--update_weights_at_generation",
                "--mutation_probability",
                "--crossover_probability",
                "--custom_mutation_operator",
                "--custom_mutation_operator_weight",
                "--prune_archive_individuals",
                "--prune_at_generations",
            ],
        },
        {
            "name": "Tracking of evolutionary algorithm",
            "options": [
                "--disable_tracking",
                "--track_full_archive",
                "--track_full_population",
            ],
        },
        {
            "name": "Other options",
            "options": [
                "--dry_run",
                "--disable_gpu",
                "--verbosity",
                "--help",
            ],
        },
    ],
}


def check_args(args) -> dict:
    """
    Checks if the combination of given arguments is valid.
    """
    # Yaml config provided
    if args["config"]:
        mandatory = {k: v for k, v in args.items() if k in ["input"]}

        with open(args["config"], "r") as f:
            config = yaml.safe_load(f)

        try:
            validate(instance=config, schema=schema)
            _args = flatten_dict(config["generate"])
            args = {**mandatory, **_args}

        except ValidationError as e:
            print("YAML file validation failed:")
            print(e)
            exit(1)

        args["config"] = None
        return check_args(args)

    # Cache dir
    if not os.path.exists(args["cache_dir"]):
        raise click.UsageError(f"Cache directory {args['cache_dir']} does not exist.")
    
    # Selection operators
    if len(args["operators_weight"]) != len(args["elitism_weight"]) or len(
        args["elitism_weight"]
    ) != len(args["novelty_weight"]):
        raise click.UsageError(
            "Number of weights for each selection step must be the same (-ow, -ew, -nw)."
        )

    for key in ["operators_weight", "elitism_weight", "novelty_weight"]:
        args[key] = list(map(float, args[key]))

    for i, (ow, ew, nw) in enumerate(
        zip(args["operators_weight"], args["elitism_weight"], args["novelty_weight"])
    ):
        if ow < 0 or ew < 0 or nw < 0:
            raise click.UsageError(
                "Weights for each selection step must be greater than 0 (-ow, -ew, -nw). "
                f"Check values at {i} index."
            )

        if ow + ew + nw > 1:
            raise click.UsageError(
                "Sum of weights for each index of the selection steps must be less than or "
                f"equal to 1 (-ow, -ew, -nw). Observed at index {i}: -ow: {ow}, -ew: {ew}, -nw: {nw}."
            )

    if args["update_weights_at_generation"] and len(args["operators_weight"]) == 1:
        raise click.UsageError(
            "Number of weights for each selection step (-ow, -ew, -nw) must be greater than 1 when "
            "'--update_weights_at_generation' is given."
        )

    cond1 = (
        len(args["operators_weight"]) > 1 and not args["update_weights_at_generation"]
    )
    cond2 = (
        len(args["operators_weight"]) > 1
        and args["update_weights_at_generation"]
        and len(args["update_weights_at_generation"])
        != len(args["operators_weight"]) - 1
    )

    if cond1 or cond2:
        raise click.UsageError(
            "Generations to update selection weights must be equal to number of weights for each "
            "selection step - 1 because the first weight is used at generation 0"
        )

    # Archive pruning
    if args["prune_at_generations"]:
        args["prune_at_generations"] = list(map(int, args["prune_at_generations"]))
        raise click.UsageError(
            "--prune_at_generations requires --prune_archive_individuals to be set."
        )

    # Stopping criterium
    if len(args["stopping_criterium"]) != len(args["stop_at_value"]):
        raise click.UsageError(
            "A stop value must be given for each stopping criterium provided."
        )
    args["stop_at_value"] = list(map(float, args["stop_at_value"]))

    # Grammar weights
    weight_sums = (
        args["snv_weight"] + args["insertion_weight"] + args["deletion_weight"]
    )
    if weight_sums > 1:
        raise click.UsageError(
            f"Sum of weights for SNV, Insertions and Deletions can't exceed 1 (observed: {weight_sums})"
        )

    if args["which_grammar"] == "motif_based":
        if len(args["subset_rbps"]) == 1:
            args["subset_rbps"] = args["subset_rbps"][0]

    return args
