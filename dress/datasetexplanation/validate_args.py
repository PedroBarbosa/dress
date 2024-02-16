import rich_click as click
import yaml
from jsonschema import ValidationError, validate
from dress.datasetevaluation.validate_args import check_input_args

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
    return args
