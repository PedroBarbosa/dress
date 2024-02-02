import abc
from dress.datasetgeneration.archive import Archive
from geneticengine.algorithms.heuristics import Individual
from geneticengine.core.evaluators import Evaluator
from geneticengine.core.problems import Problem
from typing import List
from loguru import logger


class StoppingCriterium(abc.ABC):
    """TerminationCondition provides information when to terminate
    evolution."""

    def __init__(self, archive: Archive):
        self.archive = archive

    @abc.abstractmethod
    def is_ended(
        self,
        problem: Problem,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
        evaluator: Evaluator,
    ) -> bool:
        ...

    def log_archive(self, generation: int):
        logger.info(
            "End of generation {}. Archive quality|Archive size: {}|{}".format(
                generation, round(self.archive.quality, 4), len(self.archive)
            )
        )

        _arr = "|".join(
            [str(v) for v in list(self.archive.binarize().values())]
        ).strip()
        logger.info(f"Number of sequences per score bin:\n{_arr}")


class AnyOfStoppingCriterium(StoppingCriterium):
    """Stops the evolution when any of the two (or more) stopping criteria are reached."""

    def __init__(self, stopping_criteria: List[StoppingCriterium]):
        self.stopping_criteria = stopping_criteria

    def is_ended(
        self,
        problem: Problem,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
        evaluator: Evaluator,
    ) -> bool:
        for i, stopping_criterium in enumerate(self.stopping_criteria):
            if stopping_criterium.is_ended(
                problem=problem,
                population=population,
                generation=generation,
                elapsed_time=elapsed_time,
                evaluator=evaluator,
                which_criteria=i,
            ):
                return True

        return False


class AllOfStoppingCriterium(StoppingCriterium):
    """Stops the evolution when all the stopping criteria are reached."""

    def __init__(self, stopping_criteria: List[StoppingCriterium]):
        self.stopping_criteria = stopping_criteria

    def is_ended(
        self,
        problem: Problem,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
        evaluator: Evaluator,
    ) -> bool:
        has_finished = False
        for i, stopping_criterium in enumerate(self.stopping_criteria):
            if stopping_criterium.is_ended(
                problem=problem,
                population=population,
                generation=generation,
                elapsed_time=elapsed_time,
                evaluator=evaluator,
                which_criteria=i,
            ):
                has_finished = True

            if has_finished is False:
                return False

        return True


class GenerationStoppingCriterium(StoppingCriterium):
    """**StoppingCriterium adapted from geneticEngine core

    Runs the evolution during a number of generations.
    """

    def __init__(self, max_generations: int, archive: Archive):
        """Creates a limit for the evolution, based on the number of
        generations.

        Arguments:
            max_generations (int): Number of generations to execute
        """
        self.max_generations = max_generations
        super().__init__(archive)

    def is_ended(
        self,
        problem: Problem,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
        evaluator: Evaluator,
        which_criteria: int = 0,
    ) -> bool:
        if which_criteria == 0:
            self.log_archive(generation)

        return generation >= self.max_generations


class TimeStoppingCriterium(StoppingCriterium):
    """**StoppingCriterium adapted from geneticEngine core

    Runs the evolution during a given amount of time.

    Note that termination is not pre-emptive. If the fitness function is
    slow, this might take more than the pre-specified time.
    """

    def __init__(self, max_time: int, archive: Archive):
        """Creates a limit for the evolution, based on the execution time.

        Arguments:
            max_time (int): Maximum time in seconds to run the evolution
        """
        self.max_time = max_time
        super().__init__(archive)

    def is_ended(
        self,
        problem: Problem,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
        evaluator: Evaluator,
        which_criteria: int = 0,
    ) -> bool:
        if which_criteria == 0:
            self.log_archive(generation)
        return elapsed_time >= self.max_time


class EvaluationLimitCriterium(StoppingCriterium):
    """**StoppingCriterium adapted from geneticEngine core

    Runs the evolution with a fixed budget for evaluations."""

    def __init__(self, max_evaluations: int, archive: Archive):
        """StoppingCriterium adapted from geneticEngine core

        Creates a limit for the evolution, based on the budget for
        evaluation.

        Arguments:
            max_evaluations (int): Maximum number of evaluations
        """
        self.max_evaluations = max_evaluations
        super().__init__(archive)

    def is_ended(
        self,
        problem: Problem,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
        evaluator: Evaluator,
        which_criteria: int = 0,
    ) -> bool:
        if which_criteria == 0:
            self.log_archive(generation)

        return evaluator.get_count() >= self.max_evaluations


class ReachedNumberOfArchivedSeqs(StoppingCriterium):
    """
    Runs the evolution until a given number of seqs in archive is achieved
    """

    def __init__(self, target_number: int, archive: Archive):
        """Creates a limit for the evolution, based on the number of generated sequences
        with proper fitness.

        Arguments:
            target_number (int): Desired number of sequences in archive
        """
        self.target_number = target_number
        super().__init__(archive)

    def is_ended(
        self,
        problem: Problem,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
        evaluator: Evaluator,
        which_criteria: int = 0,
    ) -> bool:
        if which_criteria == 0:
            self.log_archive(generation)

        return len(self.archive) >= self.target_number


class ReachedArchiveDiversity(StoppingCriterium):
    """
    Runs the evolution until the diversity of the archive desired is achieved
    """

    def __init__(self, target_diversity: float, archive: Archive):
        """Creates a limit for the evolution, based on the diversity of the
        arquive

        Arguments:
            target_diversity (float): Desired diversity of the archive
        """
        self.target_diversity = target_diversity
        super().__init__(archive)

    def is_ended(
        self,
        problem: Problem,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
        evaluator: Evaluator,
        which_criteria: int = 0,
    ) -> bool:
        if which_criteria == 0:
            self.log_archive(generation)

        return self.archive.diversity() >= self.target_diversity


class ReachedArchiveQuality(StoppingCriterium):
    """
    Runs the evolution until the quality of the archive desired is achieved
    """

    def __init__(self, target_quality: float, archive: Archive):
        """Creates a limit for the evolution, based on the quality of the
        arquive

        Arguments:
            target_quality (float): Desired diversity of the archive
        """
        self.target_quality = target_quality
        super().__init__(archive)

    def is_ended(
        self,
        problem: Problem,
        population: list[Individual],
        generation: int,
        elapsed_time: float,
        evaluator: Evaluator,
        which_criteria: int = 0,
    ) -> bool:
        if which_criteria == 0:
            self.log_archive(generation)

        return self.archive.quality >= self.target_quality
