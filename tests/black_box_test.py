import os
import numpy as np
import pytest
from dress.datasetgeneration.grammars.with_indels_grammar import create_grammar
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

g = create_grammar(
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


def apply_diff_to_individuals(pop):
    return zip(
        *[
            ind.get_phenotype()
            .clean(SEQ, None)
            .apply_diff(input_seq["seq"], input_seq["ss_idx"])
            for ind in pop
        ]
    )


class TestPangolin:
    SCORE_BY_MEAN = 0.1803
    SCORE_BY_MAX = 0.3603
    SCORE_BY_MIN = 0.0003

    @pytest.mark.parametrize(
        "scoring_metric, mode, expected_result",
        [
            ("mean", "ss_usage", SCORE_BY_MEAN),
            ("mean", "ss_probablity", SCORE_BY_MEAN),
            ("max", "ss_usage", SCORE_BY_MAX),
            ("min", "ss_usage", SCORE_BY_MIN),
        ],
    )
    def test_original_seq_pangolin(self, scoring_metric, mode, expected_result):
        model = Pangolin(scoring_metric=scoring_metric, mode=mode)
        raw_pred = model.run([SEQ], original_seq=True)
        score = model.get_exon_score({SEQ_ID: raw_pred}, ss_idx={SEQ_ID: SS_IDX})
        assert np.isclose(score[SEQ_ID], expected_result, atol=1e-05)

    def test_generated_seqs_pangolin(self):
        model = Pangolin(scoring_metric="mean", mode="ss_usage")
        pop = create_population(RandomSource(0))
        seqs, new_ss_positions = map(list, apply_diff_to_individuals(pop))

        raw_preds = model.run(seqs, original_seq=False)

        new_scores = model.get_exon_score(raw_preds, ss_idx=new_ss_positions)
        black_box_preds = [*new_scores.values()]

        assert len(raw_preds) == 100

        assert np.allclose(
            sorted(black_box_preds, reverse=True)[0:5],
            [0.2803, 0.2796, 0.2407, 0.2217, 0.2136],
            atol=1e-04,
        )

        assert np.allclose(
            sorted(black_box_preds)[0:5],
            [0.0001, 0.021, 0.0548, 0.0601, 0.085],
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
        assert np.isclose(score[SEQ_ID], expected_result, atol=1e-05)

    def test_generated_seqs_spliceai(self):
        model = SpliceAI(scoring_metric="mean")
        pop = create_population(RandomSource(0))
        seqs, new_ss_positions = map(list, apply_diff_to_individuals(pop))
        raw_preds = model.run(seqs, original_seq=False)
        new_scores = model.get_exon_score(raw_preds, ss_idx=new_ss_positions)
        black_box_preds = [*new_scores.values()]

        assert len(raw_preds) == 100

        assert np.allclose(
            sorted(black_box_preds, reverse=True)[0:5],
            [0.4735, 0.449, 0.4472, 0.3794, 0.3788],
            atol=1e-05,
        )
        assert np.allclose(
            sorted(black_box_preds)[0:5],
            [0.0165, 0.1504, 0.2439, 0.2446, 0.2537],
            atol=1e-05,
        )
