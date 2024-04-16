import rich_click as click
import importlib.metadata

__version__ = importlib.metadata.version("dress")

from dress.datasetgeneration.validate_args import GENERATE_GROUP_OPTIONS
from dress.datasetfiltering.validate_args import FILTER_GROUP_OPTIONS
from dress.datasetevaluation.validate_args import EVALUATE_GROUP_OPTIONS
from dress.datasetexplanation.validate_args import EXPLAIN_GROUP_OPTIONS
from dress.datasetgeneration import run as generate
from dress.datasetfiltering import run as filtration
from dress.datasetevaluation import run as evaluate
from dress.datasetexplanation import run as explain

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

click.rich_click.SHOW_METAVARS_COLUMN = False

click.rich_click.APPEND_METAVARS_HELP = True
click.rich_click.OPTION_GROUPS = {
    **GENERATE_GROUP_OPTIONS,
    **FILTER_GROUP_OPTIONS,
    **EVALUATE_GROUP_OPTIONS,
    **EXPLAIN_GROUP_OPTIONS
}
click.rich_click.COMMAND_GROUPS = {
    "dress": [
        {
            "name": "Commands",
            "commands": ["generate", "filter", "evaluate", "explain"],
        },
    ]
}

@click.group(
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.version_option(__version__, prog_name="DRESS")
def cli():
    """
    DRESS: Deep learning based Resource for Exploring Splicing Signatures

    A toolkit for generating synthetic datasets related to RNA splicing.
    """
    pass


cli.add_command(generate.generate, "generate")
cli.add_command(filtration.filter, "filter")
cli.add_command(evaluate.evaluate, "evaluate")
cli.add_command(explain.explain, "explain")

if __name__ == "__main__":
    cli()
