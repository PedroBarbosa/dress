import os

from dress.datasetgeneration.archive import (
    Archive,
    UpdateArchive,
)
from dress.datasetgeneration.custom_steps import custom_mutation_operator
from dress.datasetgeneration.custom_callbacks import (
    ArchiveCSVCallback,
    DynamicGeneticStepCallback,
    PrintBestCallbackWithGeneration,
    PruneArchiveCallback,
)
from dress.datasetgeneration.custom_fitness import (
    FitnessEvaluator,
    fitness_function_placeholder,
)
from dress.datasetgeneration.custom_stopping import (
    AnyOfStoppingCriterium,
    AllOfStoppingCriterium,
    TimeStoppingCriterium,
    GenerationStoppingCriterium,
    EvaluationLimitCriterium,
    ReachedNumberOfArchivedSeqs,
    ReachedArchiveDiversity,
    ReachedArchiveQuality,
)
from dress.datasetgeneration.custom_population_evaluators import (
    EvaluateAllSequencesInParallel,
    PopulationInitializerWithFitness,
)

from dress.datasetgeneration.black_box.model import (
    DeepLearningModel,
    Pangolin,
    SpliceAI,
)
from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.callbacks.csv_callback import CSVCallback
from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import (
    LexicaseSelection,
    TournamentSelection,
)

from geneticengine.core.grammar import Grammar
from geneticengine.core.problems import MultiObjectiveProblem, SingleObjectiveProblem
from typing import Callable, List, Tuple, Union
from geneticengine.core.representations.grammatical_evolution.ge import (
    GrammaticalEvolutionRepresentation,
)
from geneticengine.prelude import RandomSource
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation

from dress.datasetgeneration.os_utils import dump_yaml


def _is_valid_individual(ind: Individual, regions: List[range]) -> bool:
    """
    Checks if an individual still holds a valid genotype
    after removing overlaps with forbidden regions
    """
    _genotype = ind.get_phenotype().exclude_forbidden_regions(regions)  # type: ignore
    return False if _genotype is None else True


def correct_phenotypes(
    population: list[Individual],
    input_seq: dict,
    excluded_regions: List[range],
    grammar: Grammar,
    rep: Union[TreeBasedRepresentation, GrammaticalEvolutionRepresentation],
    random_source: RandomSource,
) -> list[Individual]:
    """
    Checks generated individuals and corrects some incongruences that
    the grammar does not takes into account.

    Args:
        population (list[Individual]): A list of `Individual` objects representing the individuals to fix.
        input_seq (dict): Dictionary with several attributes about the original sequence.
        excluded_regions (List[range]): Restricted intervals that will not be mutated

    Returns:
        A list of fixed individuals
    """
    original_seq = input_seq["seq"]

    new_pop = []
    for ind in population:
        if _is_valid_individual(ind, excluded_regions):
            new_pop.append(ind)

        else:
            is_valid = False
            while is_valid is False:
                _ind = Individual(
                    genotype=rep.create_individual(r=random_source, g=grammar),
                    genotype_to_phenotype=rep.genotype_to_phenotype,  # type: ignore
                )
                is_valid = (
                    True if _is_valid_individual(_ind, excluded_regions) else False
                )

            new_pop.append(_ind)  # type: ignore

    return new_pop


def evaluate_population(
    population: list[Individual],
    input_seq: dict,
    model: DeepLearningModel,
    seq_fitness: FitnessEvaluator,
) -> Union[List[float], List[List[Union[float, int]]]]:
    """
    Evaluate the fitness of a population of individuals.

    Args:
        population (list[Individual]): A list of `Individual` objects representing the solutions to evaluate.
        input_seq (dict): Dictionary with several attributes about the original sequence.
        model (DeepLearningModel): Black box model to use as reference
        seq_fitness (FitnessEvaluator): Fitness evaluator object

    Returns:
        Union[List[float], List[List[Union[float, int]]]]: A list of fitness values for each individual in the population.
    """
    original_seq = input_seq["seq"]
    ss_idx = input_seq["ss_idx"]
    dry_run = input_seq["dry_run"]

    for ind in population:
        assert len(ind.get_phenotype().diffs) > 0, "Individual must have at least 1 diff unit"  # type: ignore

    seqs, new_ss_positions = map(list, zip(*[ind.get_phenotype().apply_diff(original_seq, ss_idx) for ind in population]))  # type: ignore

    if dry_run:
        black_box_preds = [i / len(population) for i, _ in enumerate(population)]
    else:
        _raw_preds = model.run(seqs)
        new_scores = model.get_exon_score(_raw_preds, ss_idx=new_ss_positions)  # type: ignore
        black_box_preds = [*new_scores.values()]  # type: ignore

    for ind, seq, _ss_idx, pred in zip(
        population, seqs, new_ss_positions, black_box_preds
    ):
        ind.pred = pred
        ind.seq = seq
        ind.ss_idx = _ss_idx

    fitnesses = seq_fitness.eval_fitness(black_box_preds, population)
    return fitnesses


def configPhenotypeCorrector(
    correct_phenotypes: Callable[..., List[Individual]],
    input_seq: dict,
    excluded_regions: List[range],
    grammar: Grammar,
    representation: Union[GrammaticalEvolutionRepresentation, TreeBasedRepresentation],
    random_source: RandomSource,
):
    """
    Configs the function that serves as the phenotype corrector
    """

    def corrector(population: list[Individual]) -> list[Individual]:
        return correct_phenotypes(
            population,
            input_seq,
            excluded_regions,
            grammar,
            representation,
            random_source,
        )

    return corrector


def configPopulationEvaluator(
    evaluate_population: Callable[
        ..., Union[List[float], List[List[Union[float, int]]]]
    ],
    input_seq: dict,
    oracle_model: DeepLearningModel,
    seq_fitness: FitnessEvaluator,
):
    """
    Configs the function that serves as the population evaluator
    by running a deep learning model in batch mode

    Returns:
        mapper: function that takes a list of individuals and returns their fitness
    """

    def mapper(
        population: list[Individual],
    ) -> Union[List[float], List[List[Union[float, int]]]]:
        return evaluate_population(population, input_seq, oracle_model, seq_fitness)

    return mapper


def configureEvolution(
    input_seq: dict,
    grammar: Grammar,
    excluded_regions: List[range],
    **kwargs,
) -> Tuple[GP, Archive]:
    """
    Configures the evolutionary algorithm

    Args:
        input_seq (dict): Dictionary with several attributes about the
        the original sequence.
        grammar (Grammar): Grammar object that defines the language of the
        program.
        excluded_regions (List[range]): Restricted intervals that will not be perturbed

    Returns:
        GP: A Genetic Programming object to be evolved
        Archive: An archive object with the archive to be filled
    """
    if "rs" not in kwargs:
        random_source = RandomSource(kwargs.get("seed", 0))
    else:
        random_source = kwargs["rs"]

    if kwargs["selection_method"] == "tournament":
        parent_selection = TournamentSelection(kwargs["tournament_size"])
        problem = SingleObjectiveProblem(
            minimize=True if kwargs["minimize_fitness"] else False,
            fitness_function=fitness_function_placeholder,
        )

    elif kwargs["selection_method"] == "lexicase":
        # For lexicase selection, we define the problem as multiobjective,
        # where the additional objective refers to minimization of the number
        # of mutated positions in an individual
        parent_selection = LexicaseSelection()

        # First objective: are we trying to maximize or minimize the fitness?
        first_obj = True if kwargs["minimize_fitness"] else False

        # Second objective: minimize the number of mutated positions
        second_obj = True

        problem = MultiObjectiveProblem(
            minimize=[first_obj, second_obj],
            fitness_function=fitness_function_placeholder,
        )

    else:
        raise NotImplementedError(
            f"Representation {kwargs['representation']} not implemented"
        )

    # MutationStep
    mut_prob = kwargs["mutation_probability"]
    if (
        grammar._type == "random"
        and kwargs["custom_mutation_operator"]
        and kwargs["custom_mutation_operator_weight"] > 0.0
    ):

        mutation_operator = custom_mutation_operator(grammar, input_seq, **kwargs)

        if kwargs["custom_mutation_operator_weight"] == 1.0:
            mutation_step = GenericMutationStep(mut_prob, operator=mutation_operator)

        else:
            custom_weight = kwargs["custom_mutation_operator_weight"]
            default_weight = 1 - custom_weight
            with_default_operator = GenericMutationStep(mut_prob)
            with_custom_operator = GenericMutationStep(
                mut_prob, operator=mutation_operator
            )
            mutation_step = ParallelStep(
                [with_custom_operator, with_default_operator],
                weights=[custom_weight, default_weight],
            )

    else:
        mutation_step = GenericMutationStep(mut_prob)

    # CrossoverStep
    crossover_step = GenericCrossoverStep(kwargs["crossover_probability"])

    representation = TreeBasedRepresentation(grammar, max_depth=3)
    phenotypeCorrector = configPhenotypeCorrector(
        correct_phenotypes=correct_phenotypes,
        input_seq=input_seq,
        excluded_regions=excluded_regions,
        grammar=grammar,
        representation=representation,
        random_source=random_source,
    )

    # Deep learning model
    scoring_metric = kwargs["model_scoring_metric"]
    if kwargs["model"] == "spliceai":
        oracle_model = SpliceAI(
            batch_size=kwargs["batch_size"], scoring_metric=scoring_metric
        )
    elif kwargs["model"] == "pangolin":
        oracle_model = Pangolin(
            batch_size=kwargs["batch_size"],
            scoring_metric=scoring_metric,
            mode=kwargs["pangolin_mode"],
            tissue=kwargs["pangolin_tissue"],
        )
    else:
        raise ValueError(f"Model {kwargs['model']} not recognized")

    archive = Archive(
        target_size=kwargs["archive_size"],
        diversity_metric=kwargs["archive_diversity_metric"],
        bin_width=0.025,
    )

    archiveUpdater = UpdateArchive(
        input_seq=input_seq,
        fitness_threshold=kwargs["fitness_threshold"],
        minimize_fitness=kwargs["minimize_fitness"],
        archive=archive,
    )

    seq_fitness = FitnessEvaluator(
        name=kwargs["fitness_function"],
        original_score=input_seq["score"],
        archive=archive,
        is_lexicase_selection=(
            True if kwargs["selection_method"] == "lexicase" else False
        ),
    )

    mapper = configPopulationEvaluator(
        evaluate_population=evaluate_population,
        input_seq=input_seq,
        oracle_model=oracle_model,
        seq_fitness=seq_fitness,
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
                kwargs["elitism_weight"][0],
                kwargs["novelty_weight"][0],
                kwargs["operators_weight"][0],
            ],
        ),
        EvaluateAllSequencesInParallel(mapper=mapper, corrector=phenotypeCorrector),
        archiveUpdater,
    )

    # Stopping criterium
    all_stopping_criteria = []
    for i, _stopping_criterium in enumerate(kwargs["stopping_criterium"]):
        if _stopping_criterium == "n_evaluations":
            n_evaluations = int(kwargs["stop_at_value"][i])
            stopping_criterium = EvaluationLimitCriterium(
                max_evaluations=n_evaluations, archive=archive
            )
        elif _stopping_criterium == "n_generations":
            n_generations = int(kwargs["stop_at_value"][i])
            stopping_criterium = GenerationStoppingCriterium(
                max_generations=n_generations, archive=archive
            )
        elif _stopping_criterium == "time":
            n_minutes = kwargs["stop_at_value"][i]
            stopping_criterium = TimeStoppingCriterium(n_minutes * 60, archive=archive)
        elif _stopping_criterium == "archive_size":
            n_in_archive = int(kwargs["stop_at_value"][i])
            stopping_criterium = ReachedNumberOfArchivedSeqs(
                target_number=n_in_archive, archive=archive
            )
        elif _stopping_criterium == "archive_diversity":
            archive_diversity = kwargs["stop_at_value"][i]
            stopping_criterium = ReachedArchiveDiversity(
                target_diversity=archive_diversity, archive=archive
            )
        elif _stopping_criterium == "archive_quality":
            archive_quality = kwargs["stop_at_value"][i]
            stopping_criterium = ReachedArchiveQuality(
                target_quality=archive_quality, archive=archive
            )
        else:
            raise ValueError(f"Stopping criterium {_stopping_criterium} not recognized")

        all_stopping_criteria.append(stopping_criterium)

    if kwargs["stop_when_all"]:
        stopping_criterium = AllOfStoppingCriterium(all_stopping_criteria)
    else:
        stopping_criterium = AnyOfStoppingCriterium(all_stopping_criteria)

    # Callbacks
    callbacks: List[Callback] = []  # [PrintBestCallbackWithGeneration()]

    if len(kwargs["operators_weight"]) > 1:
        update_at = sorted(kwargs["update_weights_at_generation"])
        weights = [
            (el, nov, gen)
            for el, nov, gen in zip(
                kwargs["elitism_weight"][1:],
                kwargs["novelty_weight"][1:],
                kwargs["operators_weight"][1:],
            )
        ]
        callbacks.append(
            DynamicGeneticStepCallback(
                update_at_generations=update_at,
                weights=weights,  # type: ignore
                mapper=mapper,
                phenotypeCorrector=phenotypeCorrector,
                archiveUpdater=archiveUpdater,
                parentSelectionStep=parent_selection,
                mutationStep=mutation_step,
                crossoverStep=crossover_step,
            )
        )

    if kwargs["prune_archive_individuals"]:
        if kwargs["prune_at_generations"]:
            prune_at = sorted(kwargs["prune_at_generations"])
        else:
            prune_at = None

        callbacks.append(
            PruneArchiveCallback(
                archive=archive, mapper=mapper, prune_at_generations=prune_at
            )
        )

    if not kwargs["disable_tracking"]:
        callbacks.extend(
            [
                CSVCallback(
                    filename=os.path.join(
                        kwargs["outdir"],
                        kwargs["outbasename"]
                        + "_seed_"
                        + str(kwargs["seed"])
                        + "_evolution_logger.csv",
                    ),
                    only_record_best_ind=not kwargs["track_full_population"],
                    extra_columns={
                        "Phenotype": lambda a, b, c, d, ind: ind.phenotype,
                        "Prediction": lambda a, b, c, d, ind: ind.pred,
                        "Evaluations": lambda a, b, c, gp, d: gp.evaluator.get_count(),
                    },
                ),
                ArchiveCSVCallback(
                    filename=os.path.join(
                        kwargs["outdir"],
                        kwargs["outbasename"]
                        + "_seed_"
                        + str(kwargs["seed"])
                        + "_archive_logger.csv",
                    ),
                    archive=archive,
                    run_id=kwargs["outbasename"],
                    seed=kwargs["seed"],
                    input_seq=input_seq,
                    only_record_metrics=not kwargs["track_full_archive"],
                ),
            ]
        )

    if os.path.isdir(kwargs["outdir"]):
        kwargs.pop("logger")
        dump_yaml(os.path.join(kwargs["outdir"], "args_used.yaml"), **kwargs)

    return (
        GP(
            random_source=random_source,
            representation=representation,
            problem=problem,
            population_size=kwargs["population_size"],
            stopping_criterium=stopping_criterium,
            step=custom_step,
            initializer=PopulationInitializerWithFitness(
                mapper=mapper,
                archiveUpdater=archiveUpdater,
                corrector=phenotypeCorrector,
            ),
            callbacks=callbacks,
        ),
        archive,
    )
