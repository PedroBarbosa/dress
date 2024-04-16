from geneticengine.algorithms.gp.structure import GeneticStep
from geneticengine.algorithms.heuristics import Individual, Representation
from geneticengine.core.random.sources import Source
from geneticengine.core.evaluators import Evaluator
from geneticengine.core.problems import Problem
import operator
from typing import List, Dict, Union
import pandas as pd
import numpy as np
import re


class Archive(object):
    def __init__(
        self,
        target_size: int = 5000,
        diversity_metric: str = "normalized_shannon",
        bin_width: float = 0.025,
        dataset: pd.DataFrame = None,
    ):
        """Representation of an archive

        Args:
            target_size (int): Target size of the archive. Defaults to 5000.
            diversity_metric (str): Metric to evaluate the quality of the archive. Defaults to "normalized_shannon".
            bin_width (float, optional): Width of the bins for the binarization of Model scores. Defaults to 0.025.
            dataset (pd.DataFrame, optional): A dataset that represents a final archive generated with a previous evolutionary run. Defaults to None, an
            empty archive is instantiated.
        """
        self.target_size = target_size
        self.diversity_metric = diversity_metric

        if dataset is not None:
            self.is_from_dataset = True
            self.ids = dataset["Phenotype"].tolist()
            self.seqs = dataset["Sequence"].tolist()
            self.instances = None
            self.predictions = dataset["Score"].tolist()
        else:
            self.is_from_dataset = False
            self.ids = []
            self.seqs = []
            self.instances = []
            self.predictions = []

        self.bin_width = bin_width
        self.n_bins = len(np.arange(0, 1, self.bin_width))

        assert (
            self.n_bins > 1
        ), "Number of bins must be larger than 1. Decrease 'bin_width'"

    def __len__(self) -> int:
        return len(self.predictions)

    def __str__(self) -> str:
        return f"Archive with {len(self.predictions)} individuals"

    def __repr__(self) -> str:
        return ";".join(self.ids)

    @property
    def size(self) -> int:
        return self.__len__()

    def __getattr__(self, __name: str) -> Union[float, Dict[str, float]]:
        """
        Calculates several properties considering
        the current status of the archive
        """
        counts_per_bin = self.binarize().values()
        diversity_per_bin = self.diversity_per_bin().values()
        avg_diversity_per_bin = np.mean(
            list(filter(lambda x: x != "--", diversity_per_bin))
        )

        low_count_bins = sum(1 for x in counts_per_bin if x < 10) / len(counts_per_bin)

        if __name == "quality":
            size = 1 if len(self) >= self.target_size else (len(self) / 5000)
            return (
                size * 0.3
                + self.diversity() * 0.3
                + avg_diversity_per_bin * 0.2
                + (1 - low_count_bins) * 0.2
            )

        elif __name == "metrics":
            to_use = self.ids if self.is_from_dataset else self.instances
            avg_edit_distance = np.mean([self.get_edit_distance(x) for x in to_use])

            return {
                "Size": len(self),
                "Diversity": round(self.diversity(), 4),
                "Avg_Diversity_per_bin": round(avg_diversity_per_bin, 4),
                "Empty_bin_ratio": round(
                    sum(1 for x in counts_per_bin if x == 0) / len(diversity_per_bin),
                    4,
                ),
                "Low_count_bin_ratio": round(low_count_bins, 4),
                "Avg_number_diff_units": round(
                    np.mean([len(x.split("|")) for x in self.ids]), 4
                ),
                "Avg_edit_distance": round(avg_edit_distance, 4),
            }

        else:
            raise AttributeError(f"Invalid attribute: {__name}")

    def drop_duplicates(self) -> None:
        """Drop duplicates in the archive. They will not exist originally,
        but duplicates may arise if applying the `TreePruningCallback`"""

        uniq_ids, uniq_indvs, uniq_seqs, uniq_preds = [], [], set(), []

        for i, seq in enumerate(self.seqs):
            if seq not in uniq_seqs:
                uniq_ids.append(self.ids[i])
                uniq_seqs.add(seq)
                uniq_indvs.append(self.instances[i])
                uniq_preds.append(self.predictions[i])

        self.ids = uniq_ids
        self.seqs = list(uniq_seqs)
        self.instances = uniq_indvs
        self.predictions = uniq_preds

    def diversity(self, **kwargs) -> float:
        diversity_functions = {
            "normalized_shannon": self.normalized_shannon_index,  # type: ignore
            "shannon": self.shannon_index,
        }

        _func = diversity_functions[self.diversity_metric]
        if _func is None:
            raise ValueError(f"Invalid diversity metric: {self.diversity_metric}")
        return _func(**kwargs)

    def diversity_per_bin(self) -> Dict[float, float]:
        """Calculate the diversity per bin of the archive by
        creating 10 bins within each main bin

        Returns:
            Dict[float, float]: Dictionary with the diversity per bin
        """

        bin_diversity = {}
        bin_counts = self.binarize()
        for bin_id in bin_counts.keys():
            _min = bin_id
            _max = bin_id + self.bin_width
            _bin_width = _max - _min
            _preds = self.get_predictions_at_bin(bin_id, _bin_width)
            _bin_counts = self.binarize(
                min_v=_min, max_v=_max, bin_width=_bin_width / 10, predictions=_preds
            )

            bin_diversity[bin_id] = self.diversity(bin_counts=_bin_counts, n_bins=10)
        return bin_diversity

    def binarize(
        self,
        min_v: int | float = 0,
        max_v: int | float = 1,
        bin_width: float | None = None,
        predictions: List[float] | None = None,
    ) -> Dict[float, int]:
        """Initialize a counter for the number of sequences per bin

        Args:
            min_v (int | float, optional): Minimum value of the binarization. Defaults to 0.
            max_v (int | float, optional): Maximum value of the binarization. Defaults to 1.
            bin_width (float | None, optional): Width of the bins. Defaults to None.

        Returns:
            Dict[float, int]: Dictionary with the number of sequences
            per bin
        """
        if bin_width is None:
            bin_width = self.bin_width

        if predictions is None:
            predictions = self.predictions

        assert min_v < max_v, "Minimum value must be smaller than maximum value"
        assert bin_width > 0, "Bin width must be larger than 0"
        assert (
            max_v - min_v >= bin_width
        ), "Bin width must be equal or smaller than the range of values"
        assert (
            min_v >= 0 and max_v <= 1
        ), "Minimum and maximum values must be between 0 and 1"

        bin_edges = np.around(
            np.arange(min_v, max_v + bin_width, step=bin_width), decimals=3
        )
        bin_counts, _ = np.histogram(predictions, bins=bin_edges)
        return dict(zip(bin_edges, bin_counts))

    def get_bin_of_prediction(self, pred: float, bin_counts: dict[float, int]) -> float:
        """Return the bin where a given prediction falls

        Args:
            pred (float): A single prediction value
            bin_counts (dict[float, int]):  Dictionary with the number of
            sequences per bin in the archive

        Returns:
            int: Bin where the prediction belongs
        """
        for bin_id in bin_counts.keys():
            if pred >= bin_id and pred < bin_id + self.bin_width:
                return bin_id
        raise ValueError(f"Prediction {pred} does not fall in any bin")

    def get_predictions_at_bin(
        self, bin_id: float, bin_width: float | None = None
    ) -> List[float]:
        """Return the predictions of the sequences in a given bin

        Args:
            bin_id (float): Bin where the prediction belongs

        Returns:
            List[float]: List of predictions of the sequences in the bin
        """
        if bin_width is None:
            bin_width = self.bin_width

        bin_predictions = []
        for pred in self.predictions:
            if pred >= bin_id and pred < bin_id + bin_width:
                bin_predictions.append(pred)

        return bin_predictions

    def get_edit_distance(self, ind: Individual | str) -> int:
        """
        Returns the edit distance of an individual with respect to the original sequence

        This method depends on the representation of an individual, which is specified using
        the grammar.

        When an `Individual` is passed, it expects that each grammar node
        implements a `get_size` method.
        When a string is passed, it expects that the string is a phenotype
        string, where each diff unit is separated by "|". See `datasetgeneration.grammars.with_indels_grammar.py`
        for details.
        """
        if isinstance(ind, Individual):
            try:
                return sum([diff.get_size() for diff in ind.get_phenotype().diffs])
            except AttributeError:
                raise AttributeError(
                    f'Individual {ind} does not have a "get_size" method'
                )

        else:
            spanned = 0
            
            for diff_unit in ind.split("|"):
                # Random Grammar
                # Deletion[184,185,Intron_upstream,130]
                if diff_unit.startswith('Deletion'):
                    n = list(map(int, re.findall("\d+", diff_unit)))[0:2]
                    spanned += n[1] - n[0] + 1

                # Insertion[362,AG,Exon_cassette,14]
                elif diff_unit.startswith('Insertion'):
                    spanned += len(diff_unit.split(',')[1])
                elif diff_unit.startswith("SNV"):
                    spanned += 1
                
                # Motif-based Grammar
                elif diff_unit.startswith(('MotifDeletion', 'MotifInsertion', 'MotifSubstitution', 'MotifAblation')):
                    spanned += int(diff_unit.split(',')[2])
                elif diff_unit.startswith('MotifSNV'):
                    spanned += 1
                else:
                    raise ValueError(
                        f"Unable to parse individual string representation: {ind}"
                    )

            return spanned

    def shannon_index(self, bin_counts: dict | None = None, **kwargs) -> float:
        """Calculate the Shannon index of the archive

        H = -sum(p_i * log(p_i))

        Args:
            bin_counts (dict | None, optional): Dictionary with the counts per bin. Defaults to None.

        Returns:
            float: Shannon index of the archive
        """
        if len(self.predictions) == 0:
            return 0.0
        else:
            if bin_counts is None:
                bin_counts = self.binarize()
                n_preds = len(self.predictions)
            else:
                n_preds = sum(bin_counts.values())

            prob = np.array(list(bin_counts.values())) / n_preds if n_preds > 0 else 0

            # Ignore zero values
            prob_masked = np.ma.masked_equal(prob, 0)
            _shannon = -np.sum(prob_masked * np.ma.log(prob_masked))

            # Add low counts
            # epsilon = 1e-10
            # prob = np.maximum(prob, epsilon)
            # _shannon = -np.sum(prob * np.log(prob))

            return _shannon

    def normalized_shannon_index(
        self, bin_counts: dict | None = None, n_bins: int | None = None
    ) -> float:
        """Calculate the normalized Shannon index (evenness) by dividing Shannon index
        of the archive by the maximum possible Shannon value of a distribution with
        `n_bins` categories

        H_max = H / log(n_bins)

        Args:
            bin_counts (dict | None, optional): Dictionary with the counts per bin. Defaults to None.
            n_bins (int | None, optional): Number of bins. Defaults to None.
        Returns:
            float: Normalized Shannon index of the archive
        """
        if bin_counts is None:
            bin_counts = self.binarize()

        if n_bins is None:
            n_bins = self.n_bins

        _shannon = self.shannon_index(bin_counts=bin_counts)
        return _shannon / np.log(n_bins)

    def __getitem__(self, pred_range: slice) -> "Archive":
        """
        Return a slice of the archive with the individuals falling in the given prediction range

        Args:
            pred_range (slice): Slice with the prediction range

        Returns:
            Archive: Slice of the archive
        """
        if isinstance(pred_range, (float, int)):
            raise ValueError(
                "A slice must be used: e.g. archive[0.1:0.2], archive[:0.2], archive[0.8:]"
            )

        start = pred_range.start
        stop = pred_range.stop
        if pred_range.step is not None:
            raise ValueError("Slice step is not supported")

        if start is None:
            start = 0
        else:
            assert start >= 0
        if stop is None:
            stop = 1.001
        else:
            assert stop <= 1

        _preds = np.array(self.predictions)
        indices = np.where((start <= _preds) & (_preds < stop))[0]

        archiveSlice = Archive(
            target_size=self.target_size,
            diversity_metric=self.diversity_metric,
            bin_width=self.bin_width,
        )
        archiveSlice.ids = np.array(self.ids)[indices].tolist()
        archiveSlice.seqs = np.array(self.seqs)[indices].tolist()
        archiveSlice.predictions = np.array(self.predictions)[indices].tolist()
        archiveSlice.is_from_dataset = self.is_from_dataset
        if not self.is_from_dataset:
            archiveSlice.instances = np.array(self.instances)[indices].tolist()
        return archiveSlice


class UpdateArchive(GeneticStep):
    """
    Updates the archive with all individuals achieving an absolute fitness larger than a target value
    """

    def __init__(
        self,
        input_seq: dict,
        fitness_threshold: float,
        minimize_fitness: bool,
        archive: Archive,
    ):
        self.sequence = input_seq["seq"]
        self.ss_idx = input_seq["ss_idx"]
        self.original_score = input_seq["score"]
        self.fitness_threshold = fitness_threshold
        self.minimize_fitness = minimize_fitness
        self.archive = archive

    def iterate(
        self,
        problem: Problem,
        evaluator: Evaluator,
        representation: Representation,
        random_source: Source,
        population: list[Individual],
        target_size: int,
        generation: int,
    ) -> List[Individual]:
        op = operator.lt if self.minimize_fitness else operator.gt

        # Archive is empty. Add any individual with a relevant difference
        # compared to the original sequence. In addition, randomly sample
        # 5 individuals and add them so that the fitness of individuals
        # of the most common bin in the next generation will be lower
        if generation == 0:
            for i, ind in enumerate(population):
                if abs(ind.pred - self.original_score) >= 0.1:
                    self.add_individual(ind)

            ind_idx = random_source.random.sample(range(0, len(population)), 5)
            for i in ind_idx:
                self.add_individual(population[i])

        else:
            for i, ind in enumerate(population):
                if op(
                    ind.get_fitness(problem).fitness_components[0],
                    self.fitness_threshold,
                ):
                    self.add_individual(ind)

        return population

    def add_individual(self, ind: Individual):
        """Add individual to the archive

        Args:
            ind (Individual): Individual do add
        """
        if ind.seq in self.archive.seqs:
            return

        self.archive.ids.append(str(ind))
        self.archive.seqs.append(ind.seq)
        ind.seq = ""
        self.archive.instances.append(ind)
        self.archive.predictions.append(ind.pred)
