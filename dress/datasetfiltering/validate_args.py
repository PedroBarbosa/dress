import rich_click as click

FILTER_GROUP_OPTIONS = {
    "dress filter": [
        {
            "name": "Input / Output options",
            "options": [
                "--dataset",
                "--another_dataset",          
                "--outdir",
                "--outbasename",
                "--groups",
                "--list",
                "--stack_datasets",
            ],
        },
        {
            "name": "Prediction-based filtering options",
            "options": [
                "--target_psi",
                "--target_dpsi",
                "--allowed_variability",
            ],
        },
        {
            "name": "Tag-based filtering options",
            "options": [
                "--tag",
                "--delta_score",
            ],
        },
        {
            "name": "Other options",
            "options": ["--help"],
        },
    ]
}


def check_args(args) -> dict:
    """
    Checks if the combination of given arguments is valid.
    """

    if args["list"]:
        for dataset in [args["dataset"], args["another_dataset"]]:
            if dataset and len(dataset) > 1:
                raise click.UsageError(
                    "If '--list' is provided, '--dataset' and '--another_dataset' "
                    "must be a single file listing one dataset per line."
                )

            args["dataset"] = [line.rstrip("\n") for line in open(dataset[0])]

    if args["another_dataset"]:
        groups = args["groups"] or ["1", "2"]
        if len(groups) != 2:
            raise click.UsageError(
                "When '--another_dataset' is provided, '--groups' must have two elements."
            )

    if args["groups"] and len(args["groups"]) != len(
        [d for d in [args["dataset"], args["another_dataset"]] if d]
    ):
        raise click.UsageError(
            "The number of '--groups' must be equal to the number of datasets provided."
        )

    if not any(args[key] for key in ["target_psi", "target_dpsi", "tag"]):
        raise click.UsageError(
            "One of the filtering options '--target_psi', '--target_dpsi', '--tag' must be provided."
        )

    if sum(args[key] is not None for key in ["target_psi", "target_dpsi", "tag"]) > 1:
        raise click.UsageError(
            "Only one of '--target_psi', '--target_dpsi', '--tag' can be provided."
        )

    return args
