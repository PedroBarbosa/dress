from dress.datasetgeneration.archive import (
    Archive,
    UpdateArchive,
)
from dress.datasetgeneration.black_box.model import SpliceAI
from dress.datasetgeneration.config_evolution import (
    configPhenotypeCorrector,
    configPopulationEvaluator,
    correct_phenotypes,
    evaluate_population,
)
from dress.datasetgeneration.custom_callbacks import (
    PruneArchiveCallback,
    PrintBestCallbackWithGeneration,
)
from dress.datasetgeneration.custom_fitness import FitnessEvaluator
from dress.datasetgeneration.custom_population_evaluators import (
    EvaluateAllSequencesInParallel,
    PopulationInitializerWithFitness,
)
from dress.datasetgeneration.custom_stopping import (
    EvaluationLimitCriterium,
)
from dress.datasetgeneration.grammars.random_perturbation_grammar import (
    create_random_grammar,
)

from geneticengine.algorithms.gp.gp import GP
from geneticengine.algorithms.gp.operators.combinators import ParallelStep, SequenceStep
from geneticengine.algorithms.gp.operators.crossover import GenericCrossoverStep
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep
from geneticengine.algorithms.gp.operators.novelty import NoveltyStep
from geneticengine.algorithms.gp.operators.selection import TournamentSelection

from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.prelude import RandomSource
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation

SEQ = "CGCCGCGCCGTTGCCCCGCCCCTTGCAACCCCGCCCCGCGCCGGCCCCGCCCCTGCTCTCGCGCCGGCGTCGGCTGCGTCTCCGGCGTTTGAATTGCGCTTCCGCCATCTTTCCAGCCTCAGTCGGACGGGCGCGGAGACGCTTCTGGAAGGTATCGCGACCCGGCGGGCCCGGCACGGCCGGGCGGGGACAGGGGTGGCGGCGGCGGGATAGGGGTCGGAGCCGGGGCCAGGGCCGGGGCGGGTGGCGGACCCAGGGGCAGCGGGCGGCTGAGTAGGTGGGTGGTGCGGGCCCGGCCGGGCCGGGGCAGGAGACGGGCGTGGGGTCGGCGCTAGCCCCCGCGAACCCCCGTTTCATCCTCCGCTCTCATCCCCGTCCCGGTCCCAGTCCCGTTCCCATCCCTACACCTCCGGCCGCCGTTCCCCGGGCCCCGCCGCCCCGGATGCCGGCCCCGCCCGCCGCCTTCCCGCTCCCAGGCCTGGCCGCCATGGCGCCGCGGGCGGGAGGCCTTTGTGGGGCGGGCACGTGGGGCGCTGGGGGCGCGGGAGCGGGGCCGCCATGGGCTGCGGGGCCGCGCGAGCGCTCGCCTCCGTCCTCTGCCTCCGCAGGAACGCCGCGATGGCTGCGCAGGGAGAGCCCCAGGTCCAGTTCAAAGTAGGTAACCCTGCGGGGCGGGAGGCGGCCGAGCCCGACCGCGTGCGACTCGCGGGTCCCTCCTCCTGGGGCCACGATGGCTGTAATGGGGCCCCGCATCCACATTCTTTGTTTTAAGTGAGCCTGTGGTGGTTAAAGTTCCGTGACTCTGGGATCTTGAGAGGTGAAGTGTTTAGGGTTTACTTCCAAAATGTGTTTTTCAACAGCTTGTATTGGTTGGTGATGGTGGTACTGGAAAAACGACCTTCGTGAAACGTCATTTGACTGGTGAATTTGAGAAGAAGTATGTAGGTATGTGCTGGAAAACCTTGCTTGTGGAAATATGTGAGAAATGGGTAAGTTCATCCACTCAATCGCATCGTTTCCGTTTCAGCCACCTTGGGTGTTGAGG"
SS_IDX = [[100, 150], [608, 641], [860, 944]]
input_seq = {"seq": SEQ, "ss_idx": SS_IDX, "score": 0.3523, "dry_run": False}


def fitness_function(diffs) -> float:
    return 0.5


g, excluded_r = create_random_grammar(
    input_seq=input_seq,
    max_diff_units=6,
    snv_weight=0.33,
    insertion_weight=0.33,
    deletion_weight=0.33,
    max_insertion_size=5,
    max_deletion_size=5,
)

p = SingleObjectiveProblem(minimize=False, fitness_function=fitness_function)
repr = TreeBasedRepresentation(g, max_depth=3)


def create_setup():
    (
        model,
        mapper,
        phenotypeCorrector,
        seq_fitness,
        archiveUpdater,
        rs,
        archive,
        custom_step,
    ) = (None, None, None, None, None, None, None, None)
    rs = RandomSource(0)
    archive = Archive()

    archiveUpdater = UpdateArchive(
        input_seq=input_seq,
        fitness_threshold=0,
        minimize_fitness=False,
        archive=archive,
    )

    seq_fitness = FitnessEvaluator(
        name="bin_filler",
        original_score=input_seq["score"],  # type: ignore
        archive=archive,
    )

    model = SpliceAI(scoring_metric="mean")
    mapper = configPopulationEvaluator(
        evaluate_population=evaluate_population,
        input_seq=input_seq,
        oracle_model=model,
        seq_fitness=seq_fitness,
    )

    phenotypeCorrector = configPhenotypeCorrector(
        correct_phenotypes=correct_phenotypes,
        input_seq=input_seq,
        excluded_regions=excluded_r,
        grammar=g,
        representation=repr,
        random_source=rs,
    )

    custom_step = SequenceStep(
        ParallelStep(
            [
                NoveltyStep(),
                SequenceStep(
                    TournamentSelection(5),
                    GenericCrossoverStep(0.01),
                    GenericMutationStep(0.9),
                ),
            ],
            weights=[
                0.2,
                0.8,
            ],
        ),
        EvaluateAllSequencesInParallel(mapper=mapper, corrector=phenotypeCorrector),
        archiveUpdater,
    )

    return rs, archive, archiveUpdater, mapper, phenotypeCorrector, custom_step


class TestArchivePruning:
    def test_at_initialization(self):
        rs, archive, archiveUpdater, mapper, phenotypeCorrector, _ = create_setup()

        pi = PopulationInitializerWithFitness(
            mapper=mapper,
            archiveUpdater=archiveUpdater,
            corrector=phenotypeCorrector,
        )

        population = pi.initialize(p, repr, rs, 128)
        assert len(population) == 128

        # Because it's initializer (gen 0), archive updater
        # adds 5 random individuals to the archive + any
        # individual with a |diff| > 0.1 compared to
        # the original sequence, in this example 6 case
        assert archive.size == 11

        pruner = PruneArchiveCallback(archive=archive, mapper=mapper)

        n_diffs_before = sum(
            [len(ind.get_phenotype().diffs) for ind in archive.instances]
        )

        pruner.simplify()

        assert archive.size == 11
        assert len(pruner.evaluated_individuals) == 11
        n_diffs_after = sum(
            [len(ind.get_phenotype().diffs) for ind in archive.instances]
        )
        assert n_diffs_before == n_diffs_after + 2

    def test_at_end_of_evolution(self):
        (
            rs,
            archive,
            archiveUpdater,
            mapper,
            phenotypeCorrector,
            custom_step,
        ) = create_setup()

        stopping_criterium = EvaluationLimitCriterium(500, archive)

        alg = GP(
            random_source=rs,
            representation=repr,
            problem=p,
            population_size=100,
            stopping_criterium=stopping_criterium,
            step=custom_step,
            initializer=PopulationInitializerWithFitness(
                mapper=mapper,
                archiveUpdater=archiveUpdater,
                corrector=phenotypeCorrector,
            ),
            callbacks=[PrintBestCallbackWithGeneration()],
        )

        alg.evolve()
        assert archive.size == 258
        n_diffs_before = sum(
            [len(ind.get_phenotype().diffs) for ind in archive.instances]
        )

        pruner = PruneArchiveCallback(archive=archive, mapper=mapper)
        pruner.simplify()

        # All individuals should have been tested by now
        assert len(pruner.evaluated_individuals) == 258

        # Number of individuals pruned should be 66
        assert pruner.n_pruned == 66

        # Archive did not have any duplicate
        assert archive.size == 258

        n_diffs_after = sum(
            [len(ind.get_phenotype().diffs) for ind in archive.instances]
        )

        assert n_diffs_before == n_diffs_after + 66

        # Simplifying again should have not effect
        pruner.simplify()
        n_diffs_after2 = sum(
            [len(ind.get_phenotype().diffs) for ind in archive.instances]
        )
        assert n_diffs_after == n_diffs_after2
