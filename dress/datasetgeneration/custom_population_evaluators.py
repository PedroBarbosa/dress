from typing import Any, Callable, List, Tuple, Union
from dress.datasetgeneration.archive import UpdateArchive
from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.structure import PopulationInitializer
from geneticengine.algorithms.hill_climbing import StandardInitializer
from geneticengine.core.problems import Fitness, Problem
from geneticengine.core.random.sources import Source
from geneticengine.core.representations.api import Representation
from geneticengine.core.evaluators import Evaluator


class PopulationInitializerWithFitness(PopulationInitializer):
    def __init__(
        self,
        mapper: Callable[[list[Any]], Tuple[List[float], List[float]]],
        archiveUpdater: UpdateArchive,
        corrector: Union[Callable[[list[Any]], list[Individual]], None] = None,
    ):
        self.mapper = mapper
        self.archiveUpdater = archiveUpdater
        self.corrector = corrector

    def initialize(
        self,
        problem: Problem,
        representation: Representation,
        random_source: Source,
        target_size: int,
    ) -> list[Individual]:
        pi = StandardInitializer()
        population = pi.initialize(problem, representation, random_source, target_size)

        if self.corrector is not None:
            population = self.corrector(population)

        fitnesses = self.mapper(population)

        for ind, _fitness in zip(population, fitnesses):
            if not isinstance(_fitness, list):
                _fitness = [_fitness]

            mf = -_fitness[0] if problem.minimize[0] else _fitness
            ind.set_fitness(problem, Fitness(mf, _fitness))

        self.archiveUpdater.iterate(
            problem=problem,
            evaluator=None,
            representation=representation,
            random_source=random_source,
            population=population,
            target_size=target_size,
            generation=0,
        )

        return population


class EvaluateAllSequencesInParallel(GeneticStep):
    def __init__(
        self,
        mapper: Callable[[list[Any]], Tuple[List[float], List[float]]],
        corrector: Union[Callable[[list[Any]], list[Individual]], None] = None,
    ):
        self.mapper = mapper
        self.corrector = corrector

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random_source: Source,
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> list[Individual]:
        if self.corrector is not None:
            population = self.corrector(population)

        fitnesses = self.mapper(population)

        for ind, _fitness in zip(population, fitnesses):
            if not isinstance(_fitness, list):
                _fitness = [_fitness]

            mf = -_fitness[0] if problem.minimize[0] else _fitness
            ind.set_fitness(problem, Fitness(mf, _fitness))

        evaluator.count += len(population)
        return population
