import abc
from dress.datasetgeneration.custom_stopping import StoppingCriterium
from geneticengine.algorithms.heuristics import Individual
from geneticengine.core.evaluators import Evaluator
from geneticengine.core.problems import Problem
from typing import List
from loguru import logger


class StoppingCriterium(abc.ABC):
    """TerminationCondition provides information when to terminate
    evolution."""

    def __init__(self):
        pass
    
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
            f"End of generation {generation}."
        )


class GenerationStoppingCriterium(StoppingCriterium):
    """**StoppingCriterium adapted from geneticEngine core

    Runs the evolution during a number of generations.
    """

    def __init__(self, max_generations: int):
        """Creates a limit for the evolution, based on the number of
        generations.

        Arguments:
            max_generations (int): Number of generations to execute
        """
        self.max_generations = max_generations
        super().__init__()

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

    def __init__(self, max_time: int):
        """Creates a limit for the evolution, based on the execution time.

        Arguments:
            max_time (int): Maximum time in seconds to run the evolution
        """
        self.max_time = max_time
        super().__init__()

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

    def __init__(self, max_evaluations: int):
        """StoppingCriterium adapted from geneticEngine core

        Creates a limit for the evolution, based on the budget for
        evaluation.

        Arguments:
            max_evaluations (int): Maximum number of evaluations
        """
        self.max_evaluations = max_evaluations
        super().__init__()

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


class ReachedRegressionRSquared(StoppingCriterium):
    """**StoppingCriterium adapted from geneticEngine core

    Stops the evolution when the R-squared of the regression model
    is higher than a certain value.
    """

    def __init__(self, target_r2: float):
        """Creates a limit for the evolution, based on the R-squared of the
        regression model.

        Arguments:
            target_r2 (float): Target R-squared value
        """
        self.target_r2 = target_r2
        super().__init__()

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

        return evaluator.get_r2() >= self.target_r2
    
class ReachedRegressionRMSE(StoppingCriterium):
    """**StoppingCriterium adapted from geneticEngine core

    Stops the evolution when the RMSE of the regression model
    is below a certain value.
    """

    def __init__(self, target_rmse: float):
        """Creates a limit for the evolution, based on the RMSE of the
        regression model.

        Arguments:
            target_rmse (float): Target RMSE value
        """
        self.target_rmse = target_rmse
        super().__init__()

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

        return evaluator.get_rmse() <= self.target_rmse