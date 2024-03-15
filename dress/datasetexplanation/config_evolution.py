import os

from dress.datasetexplanation.custom_fitness import (
    FitnessEvaluator,
    fitness_function_placeholder,
)
from dress.datasetexplanation.custom_callbacks import ExplainCSVCallback, SimplifyExplanationCallback
from dress.datasetgeneration.custom_stopping import (
    AnyOfStoppingCriterium,
    AllOfStoppingCriterium
)

from dress.datasetexplanation.custom_stopping import (
    ReachedRegressionRMSE,
    TimeStoppingCriterium,
    GenerationStoppingCriterium,
    EvaluationLimitCriterium,
    ReachedRegressionRSquared
)

from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import (
    TournamentSelection,
)

from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import SingleObjectiveProblem
from typing import List, Union
from geneticengine.prelude import RandomSource
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation

from dress.datasetgeneration.dataset import Dataset, PairedDataset
from dress.datasetgeneration.os_utils import dump_yaml


def configureEvolution(
    dataset_obj: Union[Dataset, PairedDataset],
    grammar: Grammar,
    **kwargs,
) -> GP:
    """
    Configures the evolutionary algorithm

    Args:
        grammar (Grammar): Grammar object that defines the language of the
        program.

    Returns:
        GP: A Genetic Programming object to be evolved
    """

    if kwargs["outbasename"] is None:
        kwargs["outbasename"] = "dress_explain"
        
    random_source = RandomSource(kwargs["seed"])
    parent_selection = TournamentSelection(5)
    representation = TreeBasedRepresentation(grammar, max_depth=10)
    problem = SingleObjectiveProblem(
            minimize=kwargs["minimize_fitness"],
            fitness_function=fitness_function_placeholder,
    )

    # Mutation and Crossover steps
    mutation_step = GenericMutationStep(kwargs["mutation_probability"])
    crossover_step = GenericCrossoverStep(kwargs["crossover_probability"])

    seq_fitness = FitnessEvaluator(
        name=kwargs["fitness_function"]
    )

    custom_step = SequenceStep(
        ParallelStep(
            [
                ElitismStep(),
                NoveltyStep(),
                SequenceStep(
                    parent_selection,
                    crossover_step,
                    mutation_step,
                ),
            ],
            weights=[
                kwargs["elitism_weight"],
                kwargs["novelty_weight"],
                kwargs["operators_weight"],
            ],
        ),
    )

    # Stopping criterium
    all_stopping_criteria = []
    for i, _stopping_criterium in enumerate(kwargs["stopping_criterium"]):
        if _stopping_criterium == "n_evaluations":
            n_evaluations = int(kwargs["stop_at_value"][i])
            stopping_criterium = EvaluationLimitCriterium(
                max_evaluations=n_evaluations
            )
        elif _stopping_criterium == "n_generations":
            n_generations = int(kwargs["stop_at_value"][i])
            stopping_criterium = GenerationStoppingCriterium(
                max_generations=n_generations
            )
        elif _stopping_criterium == "time":
            n_minutes = kwargs["stop_at_value"][i]
            stopping_criterium = TimeStoppingCriterium(n_minutes * 60)

        elif _stopping_criterium == "r2":
            target_r2 = kwargs["stop_at_value"][i]
            stopping_criterium = ReachedRegressionRSquared(
                    target_r2=target_r2
                )
        elif kwargs["fitness_function"] == "rmse":
            target_rmse = kwargs["stop_at_value"][i]
            stopping_criterium = ReachedRegressionRMSE(
                target_rmse=target_rmse
                )
        else:
            raise ValueError(f"Stopping criterium {_stopping_criterium} not recognized")

        all_stopping_criteria.append(stopping_criterium)

    if kwargs["stop_when_all"]:
        stopping_criterium = AllOfStoppingCriterium(all_stopping_criteria)
    else:
        stopping_criterium = AnyOfStoppingCriterium(all_stopping_criteria)

    # Callbacks
    callbacks: List[Callback] = []
    callbacks.append(
                ExplainCSVCallback(
                    filename=os.path.join(
                        kwargs["outdir"],
                        kwargs["outbasename"]
                        + "_seed_"
                        + str(kwargs["seed"])
                        + "_archive_logger.csv",
                    ),
                    run_id=kwargs["outbasename"],
                    seed=kwargs["seed"],
                ),
        )
    
    if kwargs["simplify_explanation"]:
        if kwargs["simplify_at_generations"]:
            simplify_at = sorted(kwargs["simplify_at_generations"])
        else:
            simplify_at = None

        callbacks.append(
            SimplifyExplanationCallback(simplify_at_generations=simplify_at)
        )

    if os.path.isdir(kwargs["outdir"]):
        kwargs.pop('logger')
        dump_yaml(os.path.join(kwargs["outdir"], "args_used.yaml"), **kwargs)

    return GP(
            random_source=random_source,
            representation=representation,
            problem=problem,
            population_size=kwargs["population_size"],
            stopping_criterium=stopping_criterium,
            step=custom_step,
            callbacks=callbacks,
        )

