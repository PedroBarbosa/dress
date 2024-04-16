import csv
import gzip
import itertools
import os
from loguru import logger
from typing import Union
from geneticengine.algorithms.callbacks.callback import Callback
from geneticengine.algorithms.gp.individual import Individual


class ExplainCSVCallback(Callback):
    def __init__(
        self,
        filename: str,
        run_id: str,
        seed: int,
    ):
        self.filename = filename
        self.time = 0.0
        self.run_id = run_id
        self.seed = seed
        self.has_printed_header = False

    def end_evolution(self):
        pass
        # if self.has_printed_header:
        #     self.outfile.close()

        #     gzip_fn = f"{self.filename}.gz"
        #     with open(self.filename, "rb") as csv_file:
        #         with gzip.open(gzip_fn, "wb") as gzip_file:
        #             gzip_file.writelines(csv_file)

        #     if os.path.exists(self.filename):
        #         os.remove(self.filename)

    def write_header(self):
        pass
        # self.outfile = open(f"{self.filename}", "w", newline="")
        # self.writer = csv.writer(self.outfile)

        # row = [
        #     "Run_id",
        #     "Seed",
        #     "Seq_id",
        #     "Generation",
        #     "Execution_time",
        #     "Archive_quality",
        #     "Archive_size",
        #     "Archive_diversity",
        #     "Archive_avg_diversity_per_bin",
        #     "Archive_empty_bin_ratio",
        #     "Archive_low_count_bin_ratio",
        #     "Archive_avg_number_diff_units",
        #     "Archive_avg_edit_distance",
        # ]

        # if not self.only_record_metrics:
        #     row.extend(
        #         [
        #             "Phenotype",
        #             "Sequence",
        #             "Splice_site_positions",
        #             "Score",
        #             "Delta_score",
        #         ]
        #     )

        # self.writer.writerow(row)

    def process_iteration(
        self, generation: int, population: list[Individual], time: float, gp
    ):
        pass
        # if not self.has_printed_header:
        #     self.write_header()
        #     self.has_printed_header = True

        # self.time = time
        # _metrics = self.archive.metrics

        # row = [
        #     self.run_id,
        #     gp.random_source.seed,
        #     self.input_seq["seq_id"],
        #     generation,
        #     round(time, 4),
        #     round(self.archive.quality, 4),
        #     len(self.archive),
        #     _metrics["Diversity"],
        #     _metrics["Avg_Diversity_per_bin"],
        #     _metrics["Empty_bin_ratio"],
        #     _metrics["Low_count_bin_ratio"],
        #     _metrics["Avg_number_diff_units"],
        #     _metrics["Avg_edit_distance"],
        # ]

        # if not self.only_record_metrics:
        #     for ind in self.archive.instances:
        #         _row = row.copy()
        #         score = ind.pred
        #         delta_score = round(score - self.input_seq["score"], 4)
        #         _row.extend(
        #             [
        #                 ind.get_phenotype(),
        #                 ind.seq,
        #                 ";".join([str(ss) for ss in itertools.chain(*ind.ss_idx)]),
        #                 score,
        #                 delta_score,
        #             ]
        #         )
        #         self.writer.writerow([str(x) for x in _row])
        # else:
        #     self.writer.writerow([str(x) for x in row])

        # self.outfile.flush()
        
class SimplifyExplanationCallback(Callback):
    """
    Simplify individuals (explanations).
    """

    def __init__(
        self,
        simplify_at_generations: Union[list, None] = None,
    ) -> None:
        """
        Args:
            simplify_at_generations (list): Simplify the best explanation at these generations
        """

        super().__init__()

        self.simplify_at_generations = simplify_at_generations
        self.n_pruned = 0

    def end_evolution(self):
        """
        Simpligy the explanation at the end of the evolution
        """
        self.simplify()

    def process_iteration(self, generation: int, population, time: float, gp):
        if self.simplify_at_generations and generation in self.simplify_at_generations:
            self.simplify()

        else:
            pass

    def simplify(self):
        """
        Simplify individual trees by ....
        """
        logger.info("Pruning GP trees")
        ...