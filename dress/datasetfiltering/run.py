import rich_click as click
import os

from dress.datasetfiltering.validate_args import check_args
from dress.datasetgeneration.logger import setup_logger
from dress.datasetgeneration.run import OptionEatAll
from dress.datasetgeneration.dataset import structure_dataset
from dress.datasetfiltering.filtering import ArchiveFilter


@click.command(name="filter")
@click.option(
    "-od",
    "--outdir",
    default="filter_output",
    help="Path to where output files will be written. Default: 'filter_output",
)
@click.option(
    "-ob",
    "--outbasename",
    default="filter",
    help="Basename to include in the output files. Default: 'filter'. Additional info "
    "will be added based on '--target_psi', '--target_dpsi' or '--tag' arguments.",
)
@click.option(
    "-d",
    "--dataset",
    metavar="e,g. file1 file2 ...",
    type=tuple,
    cls=OptionEatAll,
    required=True,
    help="File(s) referring to the dataset to be filtered.",
)
@click.option(
    "-ad",
    "--another_dataset",
    metavar="e,g. file1 file2 ...",
    type=tuple,
    cls=OptionEatAll,
    help="File(s) referring to a second dataset to be filtered.",
)
@click.option(
    "-g",
    "--groups",
    metavar="e,g. group1 group2",
    type=tuple,
    cls=OptionEatAll,
    help="Group name(s) for dataset(s) argument.",
)
@click.option(
    "-l",
    "--list",
    is_flag=True,
    help="If '--dataset' and '--another_dataset' represent a list of files, one per line.",
)
@click.option(
    "-sd",
    "--stack_datasets",
    is_flag=True,
    help="When '--another_dataset' is provided, combines the filtered outputs of each dataset into a single result. "
    "Adds a column indicating the original group each sequence belongs.",
)
@click.option(
    "-tpsi",
    "--target_psi",
    type=click.FloatRange(0, 1, clamp=False),
    help="Target black box model predictions (PSI values or splice site probabilities). "
    "Sequences centered at this value plus the '--allowed_variability' on each "
    "direction are kept.",
)
@click.option(
    "-tdpsi",
    "--target_dpsi",
    type=click.FloatRange(-1, 1, clamp=False),
    help="Target black box model delta score predictions, representing differences from the original sequence. "
    "It can be a negative value. The dataset is filtered to keep sequences centered at 'original score + target_dpsi'. "
    "Similar to '--target_psi', the filtering outcome depends on '--allowed_variability'.",
)
@click.option(
    "-av",
    "--allowed_variability",
    type=click.FloatRange(0, 1, clamp=False),
    default=0.05,
    help="Allowed variability in the target PSI or dPSI value. Default: 0.05. For a '--target_psi 0.5' and "
    "'--allowed_variability 0.05', the dataset will be filtered to include sequences with a prediction value "
    "between 0.45 and 0.55. For a '--target_dpsi - 0.2' and '--allowed_variability 0.05', the filtering returns "
    "sequences centered at the (original score - 0.2) +- 0.05.",
)

@click.option(
    "-t",
    "--tag",
    type=click.Choice(["lower", "higher", "equal"]),
    help="Tag to filter the dataset. Choose 'lower' to include sequences with scores lower than the original, "
    "'higher' for sequences with scores higher than the original, and 'equal' for sequences with scores similar "
    "to the original.",
)

@click.option(
    "-ds",
    "--delta_score",
    type=click.FloatRange(0, 1, clamp=False),
    default=0.1,
    help="Absolute delta score difference to the original score for tag-based filtering. "
    "For example, with an original sequence score of 0.7, and the options '--tag lower' and "
    "'--delta_score 0.1', the dataset will be filtered to include sequences with scores less than 0.6. "
    "When '--tag equal' is used, this option represents the maximum delta score allowed. For example, "
    "with an original sequence score of 0.3, and the options '--tag equal' and '--delta_score 0.1', the "
    "dataset will be filtered to include sequences with scores between 0.2 and 0.4."
)
def filter(**args):
    """
    Filter synthetic dataset(s) produced by <dress generate> into desired PSI or dPSI intervals.

    If '--target_psi' is provided, the dataset(s) will be filtered to include sequences predicted at this particular level of inclusion.\n
    If '--target_dPSI' is provided, the dataset(s) will be filtered to include sequences that mimic a particular level of delta PSI in \
model predictions relative to the original sequence."
    """
    args = check_args(args)
    args["logger"] = setup_logger(level=0)
    os.makedirs(args["outdir"], exist_ok=True)

    dataset_obj = structure_dataset(
        generated_datasets=[args["dataset"], args["another_dataset"]],
        **args,
    )

    args.pop("dataset", "another_dataset")
    arch_filt = ArchiveFilter(dataset_obj, **args)
    arch_filt.filter()
    arch_filt.write_output()