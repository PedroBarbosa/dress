import os
from numpy.testing import assert_allclose
import pytest
from dress.datasetgeneration.grammars.random_perturbation_grammar import (
    create_random_grammar,
)
from dress.datasetgeneration.black_box.model import Pangolin, SpliceAI
from geneticengine.core.random.sources import RandomSource
from geneticengine.algorithms.gp.individual import Individual
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEQ = "CGCCGCGCCGTTGCCCCGCCCCTTGCAACCCCGCCCCGCGCCGGCCCCGCCCCTGCTCTCGCGCCGGCGTCGGCTGCGTCTCCGGCGTTTGAATTGCGCTTCCGCCATCTTTCCAGCCTCAGTCGGACGGGCGCGGAGACGCTTCTGGAAGGTATCGCGACCCGGCGGGCCCGGCACGGCCGGGCGGGGACAGGGGTGGCGGCGGCGGGATAGGGGTCGGAGCCGGGGCCAGGGCCGGGGCGGGTGGCGGACCCAGGGGCAGCGGGCGGCTGAGTAGGTGGGTGGTGCGGGCCCGGCCGGGCCGGGGCAGGAGACGGGCGTGGGGTCGGCGCTAGCCCCCGCGAACCCCCGTTTCATCCTCCGCTCTCATCCCCGTCCCGGTCCCAGTCCCGTTCCCATCCCTACACCTCCGGCCGCCGTTCCCCGGGCCCCGCCGCCCCGGATGCCGGCCCCGCCCGCCGCCTTCCCGCTCCCAGGCCTGGCCGCCATGGCGCCGCGGGCGGGAGGCCTTTGTGGGGCGGGCACGTGGGGCGCTGGGGGCGCGGGAGCGGGGCCGCCATGGGCTGCGGGGCCGCGCGAGCGCTCGCCTCCGTCCTCTGCCTCCGCAGGAACGCCGCGATGGCTGCGCAGGGAGAGCCCCAGGTCCAGTTCAAAGTAGGTAACCCTGCGGGGCGGGAGGCGGCCGAGCCCGACCGCGTGCGACTCGCGGGTCCCTCCTCCTGGGGCCACGATGGCTGTAATGGGGCCCCGCATCCACATTCTTTGTTTTAAGTGAGCCTGTGGTGGTTAAAGTTCCGTGACTCTGGGATCTTGAGAGGTGAAGTGTTTAGGGTTTACTTCCAAAATGTGTTTTTCAACAGCTTGTATTGGTTGGTGATGGTGGTACTGGAAAAACGACCTTCGTGAAACGTCATTTGACTGGTGAATTTGAGAAGAAGTATGTAGGTATGTGCTGGAAAACCTTGCTTGTGGAAATATGTGAGAAATGGGTAAGTTCATCCACTCAATCGCATCGTTTCCGTTTCAGCCACCTTGGGTGTTGAGG"
SEQ_ID = "test"
SS_IDX = [[100, 150], [608, 641], [860, 944]]

input_seq = {"seq_id": SEQ_ID, "seq": SEQ, "ss_idx": SS_IDX}

g, excluded_r = create_random_grammar(
    max_diff_units=6,
    snv_weight=0.33,
    insertion_weight=0.33,
    deletion_weight=0.33,
    max_insertion_size=5,
    max_deletion_size=5,
    input_seq=input_seq,
)

p = SingleObjectiveProblem(minimize=False, fitness_function=lambda x: [0.5] * x)
repr = TreeBasedRepresentation(g, max_depth=3)


def create_population(rs: RandomSource):

    return [
        Individual(
            genotype=repr.create_individual(r=rs, g=g),
            genotype_to_phenotype=repr.genotype_to_phenotype,
        )
        for _ in range(100)
    ]


def apply_diff_to_individuals(pop: list, rs: RandomSource):

    return zip(
        *[
            ind.get_phenotype()
            .remove_diffunit_overlaps(SEQ, rs)
            .apply_diff(input_seq["seq"], input_seq["ss_idx"])
            for ind in pop
        ]
    )


class TestPangolin:
    SCORE_BY_MEAN = 0.32
    SCORE_BY_MEAN_SS_PROB = 0.2598
    SCORE_BY_MAX = 0.6394
    SCORE_BY_MIN = 0.0006

    SCORE_HEART = 0.3566
    SCORE_LIVER = 0.3132
    SCORE_BRAIN = 0.3318
    SCORE_TESTIS = 0.2786
    SCORE_HEART_TESTIS = 0.3176

    @pytest.mark.parametrize(
        "scoring_metric, mode, expected_result",
        [
            ("mean", "ss_usage", SCORE_BY_MEAN),
            ("mean", "ss_probability", SCORE_BY_MEAN_SS_PROB),
            ("max", "ss_usage", SCORE_BY_MAX),
            ("min", "ss_usage", SCORE_BY_MIN),
        ],
    )
    def test_original_seq_pangolin(self, scoring_metric, mode, expected_result):
        model = Pangolin(scoring_metric=scoring_metric, mode=mode)
        raw_pred = model.run([SEQ], original_seq=True)
        score = model.get_exon_score({SEQ_ID: raw_pred}, ss_idx={SEQ_ID: SS_IDX})
        assert_allclose(score[SEQ_ID], expected_result, atol=1e-04)
        model = None

    @pytest.mark.parametrize(
        "tissue, expected_result",
        [
            ("heart", SCORE_HEART),
            ("liver", SCORE_LIVER),
            ("brain", SCORE_BRAIN),
            ("testis", SCORE_TESTIS),
            (["heart", "testis"], SCORE_HEART_TESTIS),
        ],
    )
    def test_tissue_specific_pangolin(self, tissue, expected_result):
        model = Pangolin(tissue=tissue)
        raw_pred = model.run([SEQ], original_seq=True)
        score = model.get_exon_score({SEQ_ID: raw_pred}, ss_idx={SEQ_ID: SS_IDX})
        assert_allclose(score[SEQ_ID], expected_result, atol=1e-04)
        model = None

    def test_generated_seqs_pangolin(self):
        model = Pangolin(scoring_metric="mean", mode="ss_usage")
        rs = RandomSource(0)
        pop = create_population(rs)
        seqs, new_ss_positions = map(list, apply_diff_to_individuals(pop, rs))

        raw_preds = model.run(seqs, original_seq=False)

        new_scores = model.get_exon_score(raw_preds, ss_idx=new_ss_positions)
        black_box_preds = [*new_scores.values()]

        assert len(raw_preds) == 100
        assert_allclose(
            sorted(black_box_preds, reverse=True)[0:5],
            [0.3916, 0.3665, 0.3603, 0.3555, 0.3546],
            atol=1e-04,
        )
        assert_allclose(
            sorted(black_box_preds)[0:5],
            [0.0916, 0.2219, 0.2275, 0.2345, 0.24],
            atol=1e-04,
        )


class TestSpliceAI:
    SCORE_BY_MEAN = 0.3523
    SCORE_BY_MAX = 0.6814
    SCORE_BY_MIN = 0.0231
    predict_batch = None

    @pytest.mark.parametrize(
        "scoring_metric,expected_result",
        [("mean", SCORE_BY_MEAN), ("max", SCORE_BY_MAX), ("min", SCORE_BY_MIN)],
    )
    def test_original_seq_spliceai(self, scoring_metric, expected_result):
        model = SpliceAI(scoring_metric=scoring_metric)
        raw_pred = model.run([SEQ], original_seq=True)
        score = model.get_exon_score({SEQ_ID: raw_pred}, ss_idx={SEQ_ID: SS_IDX})
        assert_allclose(score[SEQ_ID], expected_result, atol=1e-04)
        model = None

    def test_generated_seqs_spliceai(self):
        model = SpliceAI(scoring_metric="mean")
        rs = RandomSource(0)
        pop = create_population(rs)
        seqs, new_ss_positions = map(list, apply_diff_to_individuals(pop, rs))
        raw_preds = model.run(seqs, original_seq=False)
        new_scores = model.get_exon_score(raw_preds, ss_idx=new_ss_positions)
        black_box_preds = [*new_scores.values()]

        assert len(raw_preds) == 100

        assert_allclose(
            sorted(black_box_preds, reverse=True)[0:5],
            [0.4798, 0.4447, 0.4285, 0.3953, 0.387],
            atol=1e-04,
        )
        assert_allclose(
            sorted(black_box_preds)[0:5],
            [0.2234, 0.2302, 0.248, 0.2512, 0.2559],
            atol=1e-04,
        )
