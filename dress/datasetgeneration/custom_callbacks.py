import csv
import gzip
import copy
import itertools
import os
from typing import Any, Callable, List, Tuple, Union
from geneticengine.algorithms.gp.structure import GeneticStep
from dress.datasetgeneration.archive import Archive, UpdateArchive
from dress.datasetgeneration.custom_population_evaluators import (
    EvaluateAllSequencesInParallel,
)
from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.elitism import ElitismStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection


class ArchiveCSVCallback(Callback):
    def __init__(
        self,
        filename: str,
        archive: Archive,
        run_id: str,
        seed: int,
        input_seq: dict,
        only_record_metrics: bool = True,
    ):
        self.filename = filename
        self.archive = archive
        self.time = 0.0
        self.run_id = run_id
        self.seed = seed
        self.input_seq = input_seq
        self.only_record_metrics = only_record_metrics
        self.has_printed_header = False

    def end_evolution(self):
        if self.has_printed_header:
            self.outfile.close()

            gzip_fn = f"{self.filename}.gz"
            with open(self.filename, "rb") as csv_file:
                with gzip.open(gzip_fn, "wb") as gzip_file:
                    gzip_file.writelines(csv_file)

            if os.path.exists(self.filename):
                os.remove(self.filename)

    def write_header(self):
        self.outfile = open(f"{self.filename}", "w", newline="")
        self.writer = csv.writer(self.outfile)

        row = [
            "Run_id",
            "Seed",
            "Seq_id",
            "Generation",
            "Execution_time",
            "Archive_quality",
            "Archive_size",
            "Archive_diversity",
            "Archive_avg_diversity_per_bin",
            "Archive_empty_bin_ratio",
            "Archive_low_count_bin_ratio",
            "Archive_avg_number_diff_units",
            "Archive_avg_edit_distance",
        ]

        if not self.only_record_metrics:
            row.extend(
                [
                    "Phenotype",
                    "Sequence",
                    "Splice_site_positions",
                    "Score",
                    "Delta_score",
                ]
            )

        self.writer.writerow(row)

    def process_iteration(
        self, generation: int, population: list[Individual], time: float, gp
    ):
        if not self.has_printed_header:
            self.write_header()
            self.has_printed_header = True

        self.time = time
        _metrics = self.archive.metrics

        row = [
            self.run_id,
            gp.random_source.seed,
            self.input_seq["seq_id"],
            generation,
            round(time, 4),
            round(self.archive.quality, 4),
            len(self.archive),
            _metrics["Diversity"],
            _metrics["Avg_Diversity_per_bin"],
            _metrics["Empty_bin_ratio"],
            _metrics["Low_count_bin_ratio"],
            _metrics["Avg_number_diff_units"],
            _metrics["Avg_edit_distance"],
        ]

        if not self.only_record_metrics:
            for ind in self.archive.instances:
                _row = row.copy()
                score = ind.pred
                delta_score = round(score - self.input_seq["score"], 4)
                _row.extend(
                    [
                        ind.get_phenotype(),
                        ind.seq,
                        ";".join([str(ss) for ss in itertools.chain(*ind.ss_idx)]),
                        score,
                        delta_score,
                    ]
                )
                self.writer.writerow([str(x) for x in _row])
        else:
            self.writer.writerow([str(x) for x in row])

        self.outfile.flush()


class PrintBestCallbackWithGeneration(Callback):
    """
    Prints the number and the time spent at the end of each generation.
    """

    def process_iteration(self, generation: int, population, time: float, gp):
        best_individual: Individual = gp.get_best_individual(gp.problem, population)
        best_fitness = best_individual.get_fitness(gp.problem)  # type: ignore

        print(f"[{self.__class__.__name__}] Generation {generation}. Time {time:.2f}")
        print(f"[{self.__class__.__name__}] Best fitness: {best_fitness}")
        print(f"[{self.__class__.__name__}] Best genotype: {best_individual}")


class DynamicGeneticStepCallback(Callback):
    """
    Updates the custom step to use different ratios for multiple genetic steps (Elitism, Novelty, Genetic).

    Requires a list with generations to apply the update, as well as a list with the weights
    to update.
    """

    def __init__(
        self,
        update_at_generations: list,
        weights: list[Tuple],
        archiveUpdater: UpdateArchive,
        mapper: Callable[[list[Any]], list[float]],
        phenotypeCorrector: Union[Callable[[list[Any]], list[Individual]], None] = None,
        parentSelectionStep: GeneticStep = TournamentSelection(5),
        mutationStep: GeneticStep = GenericMutationStep(0.9),
        crossoverStep: GeneticStep = GenericCrossoverStep(0.01),
    ) -> None:
        """
        Args:
            update_at_generations (list): Update the weights at these generations
            weights (list[Tuple]): List of tuples with the weights to update
            archiveUpdater (UpdateArchive): Archive updater
            mapper (Callable[[list[Any]], list[float]]): Function to evaluate the population
            phenotypeCorrector (Union[Callable[[list[Any]], list[Individual]], None]): Function to correct the
        phenotypes
            parentSelectionStep GeneticStep: Parent selection step. Default: TournamentSelection(5)
            mutationStep GeneticStep: Mutation step. Default: GenericMutationStep(0.9)
            crossoverStep GeneticStep: Crossover step. Default: GenericCrossoverStep(0.01)
        """

        super().__init__()

        assert len(update_at_generations) == len(weights)
        self.update_at_generations = update_at_generations
        self.weights = weights
        self.archiveUpdater = archiveUpdater
        self.mapper = mapper
        self.phenotypeCorrector = phenotypeCorrector

        self.parentSelectionStep = parentSelectionStep
        self.mutationStep = mutationStep
        self.crossoverStep = crossoverStep

    def process_iteration(self, generation: int, population, time: float, gp):
        try:
            index = self.update_at_generations.index(generation)
            weights_at_idx = self.weights[index]

            updated_step = SequenceStep(
                ParallelStep(
                    [
                        ElitismStep(),
                        NoveltyStep(),
                        SequenceStep(
                            self.parentSelectionStep,
                            self.crossoverStep,
                            self.mutationStep,
                        ),
                    ],
                    weights=weights_at_idx,
                ),  # type: ignore
                EvaluateAllSequencesInParallel(
                    mapper=self.mapper, corrector=self.phenotypeCorrector
                ),
                self.archiveUpdater,
            )

            gp.step = updated_step

        except ValueError:
            pass


class PruneArchiveCallback(Callback):
    """
    Simplify individuals of the archive by inspecting irrelevant diffUnits.
    """

    def __init__(
        self,
        archive: Archive,
        mapper: Callable[[list[Any]], list[float]],
        prune_at_generations: Union[list, None] = None,
    ) -> None:
        """
        Args:
            archive (Archive): Archive object
            mapper (Callable[[list[Any]], list[float]]): Function to evaluate the population
            prune_at_generations (list): Prune the archive at these generations
            evaluated_individuals (list): List of evaluated individuals to avoid pruning them again
        """

        super().__init__()

        self.archive = archive
        self.mapper = mapper
        self.prune_at_generations = prune_at_generations
        self.n_pruned = 0
        self.evaluated_individuals: List[int] = []

    def end_evolution(self):
        """
        Prune the archive at the end of the evolution
        """
        self.simplify()

    def process_iteration(self, generation: int, population, time: float, gp):
        if self.prune_at_generations and generation in self.prune_at_generations:
            self.simplify()

        else:
            pass

    def simplify(self):
        """
        Simplify the individuals in the archive by testing the effect of individual diff units
        """
        from loguru import logger

        logger.info("Pruning archive")
        expanded_pop, map_preds = self._prepare_perturbations()
        n_pruned = 0

        if len(expanded_pop) > 0:
            self.mapper(expanded_pop)

            for arq_idx, _map in map_preds.items():
                ref_pred = _map[0]
                _preds = expanded_pop[_map[1][0] : _map[1][1] + 1]

                # Get perturbations with the same score
                filtered_preds = [p for p in _preds if p.pred == ref_pred]

                # All perturbations have an impact in the score, original individual is kept
                if len(filtered_preds) == 0:
                    self.evaluated_individuals.append(arq_idx)

                # Only one perturbation with the same score, let's replace it
                elif len(filtered_preds) == 1:
                    self.archive.instances[arq_idx] = filtered_preds[0]
                    self.evaluated_individuals.append(arq_idx)
                    n_pruned += 1

                # Several perturbations with the same score, select the first one.
                # Alternatively, we could test additional pruning by removing
                # combinations of diff units and see if the score is still the same,
                # or whether perturbations compensate each other
                else:
                    self.archive.instances[arq_idx] = filtered_preds[0]
                    self.evaluated_individuals.append(arq_idx)
                    n_pruned += 1

        self.n_pruned += n_pruned
        _arq_size = self.archive.size
        self.archive.drop_duplicates()

        logger.info(
            f"[{self.__class__.__name__}] Individuals tested(> 1 diffUnit)|Individuals pruned|Duplicates after pruning: {len(map_preds)}|{n_pruned}|{_arq_size - self.archive.size}"
        )

    def _prepare_perturbations(self) -> Tuple[list[Individual], dict]:
        """
        Prepare the perturbations to test by removing one diff unit at a time

        Returns:
            Tuple[list[Individual], dict]: List of individuals with one diff unit removed and a dictionary
        mapping perturbations to single individuals
        """
        map_preds = {}
        expanded_pop = []
        seq_counter = 0
        for idx, (ind, pred) in enumerate(
            zip(self.archive.instances, self.archive.predictions)
        ):
            if idx in self.evaluated_individuals:
                continue

            perturb = ind.get_phenotype().diffs
            if len(perturb) == 1:
                self.evaluated_individuals.append(idx)
                continue

            new_lists = [copy.deepcopy(ind) for _ in range(len(perturb))]
            [_ind.get_phenotype().diffs.pop(i) for i, _ind in enumerate(new_lists)]
            map_preds[idx] = [pred, (seq_counter, seq_counter + len(perturb) - 1)]
            expanded_pop.extend(new_lists)
            seq_counter += len(perturb)

        return expanded_pop, map_preds
