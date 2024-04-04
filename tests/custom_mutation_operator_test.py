from geneticengine.algorithms.gp.individual import Individual
from geneticengine.algorithms.gp.operators.mutation import GenericMutationStep

from geneticengine.core.evaluators import SequentialEvaluator
from geneticengine.core.problems import SingleObjectiveProblem
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import TreeBasedRepresentation
from dress.datasetgeneration.grammars.random_perturbation_grammar import create_random_grammar
from dress.datasetgeneration.custom_steps import custom_mutation_operator

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
rs = RandomSource(0)


class TestRandomGrammar:
    def test_snv_motif_replacement(self):
        g, _ = create_random_grammar(
            input_seq=input_seq
        )

        motif_mutation_operator = custom_mutation_operator(
            g, input_seq, **{"max_insertion_size": 5, "max_deletion_size": 5}
        )
        custom_mutation_step = GenericMutationStep(1, operator=motif_mutation_operator)

        population = [
            Individual(
                genotype=repr.create_individual(r=rs, g=g),
                genotype_to_phenotype=repr.genotype_to_phenotype,
            )
            for _ in range(100)
        ]

        new_population = custom_mutation_step.iterate(
            p, SequentialEvaluator(), repr, rs, population, 100, 0
        )

        for prev_ind, ind in zip(population, new_population):
            prev_ph = prev_ind.get_phenotype()
            ph = ind.get_phenotype()

            changed_diffunit = 0
            for prev_diffs, diffs in zip(prev_ph.diffs, ph.diffs):
                assert diffs.__class__ == prev_diffs.__class__
                if prev_diffs != diffs:

                    changed_diffunit += 1
                    # The most distant position observed is 12
                    assert abs(prev_diffs.position - diffs.position) <= 12

            # All individuals should have exactly 1 diff unit changed
            assert changed_diffunit == 1