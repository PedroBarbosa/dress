import copy
from typing import List, Union
from dress.datasetgeneration.archive import Archive
from geneticengine.algorithms.gp.individual import Individual


class FitnessEvaluator(object):
    def __init__(
        self,
        name: str,
        original_score: float,
        archive: Union[Archive, None],
        is_lexicase_selection: bool = False,
    ):
        """Define a fitness function to use when evaluating an individual

        Args:
            name (str): Tag describing the fitness function
            original_score (float): Model score of the original sequence
            archive (Union[Archive, None]): Current Archive object
            is_lexicase_selection (bool, optional): Whether the selection method is lexicase, which
            makes the problem multiobjective. Defaults to False.
        """
        self.name = name
        self.archive = archive
        self.original_score = original_score
        self.is_lexicase_selection = is_lexicase_selection

    def eval_fitness(
        self, preds: List, population: List[Individual]
    ) -> Union[List[float], List[List[Union[float, int]]]]:
        fitness_functions = {
            "raw_diff": self.raw_diff,
            "increase_archive_diversity": self.increase_archive_diversity,
            "bin_filler": self.bin_filler,
        }

        _func = fitness_functions[self.name]

        fitnesses = _func(preds)
        if self.is_lexicase_selection:
            # We need to access the individual genotype to count the number of positions mutated
            n_affect_nuc = self.count_nucleotides_affected(population)
            return [[fitness, n_nuc] for fitness, n_nuc in zip(fitnesses, n_affect_nuc)]

        return fitnesses

    def raw_diff(self, preds: list) -> List[float]:
        """Fitness is the raw difference between
        the SpliceAI score of original and mutated sequences

        Args:
            preds (list): List of population black box predictions

        Returns:
            List[float]: List of individual fitnesses in the population
        """
        return [round(pred_score - self.original_score, 4) for pred_score in preds]

    def increase_archive_diversity(self, preds: list) -> List[float]:
        """Fitness is the the difference in the archive
        diversity after including the individual in the
        archive

        Args:
            preds (list): List of population black box predictions

        Returns:
            List[float]: List of individual fitnesses in the population
        """
        assert self.archive is not None
        fitnesses = []
        diversity = self.archive.diversity()

        for pred in preds:
            self.archive.predictions.append(pred)
            _diversity = self.archive.diversity()
            fitnesses.append(_diversity - diversity)
            self.archive.predictions.pop()

        return fitnesses

    def bin_filler(self, preds: list) -> List[float]:
        """Fitness is the relative number of instances
        to be added in the archive bin where the individual
        prediction falls. In other words, the greater the
        scarcity of sequences in the bin, the greater its fitness
        level.

        Args:
            preds (list): List of population black box predictions

        Returns:
            List[float]: List of individual fitnesses in the population
        """
        assert self.archive is not None
        fitnesses = []
        target_n_seqs_per_bin = self.archive.target_size // self.archive.n_bins
        bin_counts = self.archive.binarize()
        _control_bin_counts = copy.deepcopy(bin_counts)
        for pred in preds:
            bin = self.archive.get_bin_of_prediction(pred, bin_counts)
            _control_bin_counts[bin] += 1

            # We let a bin to exceed 5% of its capacity. If it exceeds, we penalize the fitness
            if _control_bin_counts[bin] > target_n_seqs_per_bin + (
                target_n_seqs_per_bin * 0.05
            ):
                fitnesses.append(-1.0)
                continue

            fitness = 1 - bin_counts[bin] / target_n_seqs_per_bin
            fitnesses.append(fitness)

        return fitnesses

    def count_nucleotides_affected(self, population: List[Individual]) -> List[int]:
        """Count the number of nucleotides affected by all diffUnits of each individual

        Args:
            population (List[Individual]): List of individuals in the population

        Returns:
            List[int]: List with the number of nucleotides affected by all diffUnits of each individual
        """
        n_nuc_affected = []
        for ind in population:
            n_nuc_affected.append(
                sum([diff.get_size() for diff in ind.get_phenotype().diffs])
            )
        return n_nuc_affected


def fitness_function_placeholder(diffs) -> float:
    """
    This fitness function is just a placeholder. It should never be called
    in an evolution where individuals of the population are evaluated in
    a single batch.
    """
    raise Exception("The program should not reach here")
