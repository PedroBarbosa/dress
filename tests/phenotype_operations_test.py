from dress.datasetgeneration.grammars.with_indels_grammar import (
    DiffUnit,
    create_grammar,
)
from geneticengine.core.random.sources import RandomSource

# SEQ = "CGCCGCGCCGTTGCCCCGCCCCTTGCAACCCCGCCCCGCGCCGGCCCCGCCCCTGCTCTCGCGCCGGCGTCGGCTGCGTCTCCGGCGTTTGAATTGCGCTTCCGCCATCTTTCCAGCCTCAGTCGGACGGGCGCGGAGACGCTTCTGGAAGGTATCGCGACCCGGCGGGCCCGGCACGGCCGGGCGGGGACAGGGGTGGCGGCGGCGGGATAGGGGTCGGAGCCGGGGCCAGGGCCGGGGCGGGTGGCGGACCCAGGGGCAGCGGGCGGCTGAGTAGGTGGGTGGTGCGGGCCCGGCCGGGCCGGGGCAGGAGACGGGCGTGGGGTCGGCGCTAGCCCCCGCGAACCCCCGTTTCATCCTCCGCTCTCATCCCCGTCCCGGTCCCAGTCCCGTTCCCATCCCTACACCTCCGGCCGCCGTTCCCCGGGCCCCGCCGCCCCGGATGCCGGCCCCGCCCGCCGCCTTCCCGCTCCCAGGCCTGGCCGCCATGGCGCCGCGGGCGGGAGGCCTTTGTGGGGCGGGCACGTGGGGCGCTGGGGGCGCGGGAGCGGGGCCGCCATGGGCTGCGGGGCCGCGCGAGCGCTCGCCTCCGTCCTCTGCCTCCGCAGGAACGCCGCGATGGCTGCGCAGGGAGAGCCCCAGGTCCAGTTCAAAGTAGGTAACCCTGCGGGGCGGGAGGCGGCCGAGCCCGACCGCGTGCGACTCGCGGGTCCCTCCTCCTGGGGCCACGATGGCTGTAATGGGGCCCCGCATCCACATTCTTTGTTTTAAGTGAGCCTGTGGTGGTTAAAGTTCCGTGACTCTGGGATCTTGAGAGGTGAAGTGTTTAGGGTTTACTTCCAAAATGTGTTTTTCAACAGCTTGTATTGGTTGGTGATGGTGGTACTGGAAAAACGACCTTCGTGAAACGTCATTTGACTGGTGAATTTGAGAAGAAGTATGTAGGTATGTGCTGGAAAACCTTGCTTGTGGAAATATGTGAGAAATGGGTAAGTTCATCCACTCAATCGCATCGTTTCCGTTTCAGCCACCTTGGGTGTTGAGG"
# SS_IDX = [[100, 150], [608, 641], [860, 944]]

TOY_SEQ = "ACAGCAGGGGGGTTTTAGCCGTTACAGTCGATGC"
TOY_SS_IDX = [[3, 5], [10, 12], [15, 18]]
TOY_DANGER_ZONE = [range(2, 6), range(8, 13), range(15, 21)]

toy_grammar = create_grammar(
    max_diff_units=6,
    snv_weight=0.33,
    insertion_weight=0.33,
    deletion_weight=0.33,
    max_insertion_size=5,
    max_deletion_size=5,
    input_seq={"seq": TOY_SEQ, "ss_idx": TOY_DANGER_ZONE},
)

DiffSequence = toy_grammar.starting_symbol
SNV = toy_grammar.alternatives[DiffUnit][0]
RandomDeletion = toy_grammar.alternatives[DiffUnit][1]
RandomInsertion = toy_grammar.alternatives[DiffUnit][2]
r = RandomSource(0)


class TestPhenotypeCorrector:
    def test_exclude_forbidden_regions(self):
        """
        Tests if diffUnits in forbidden zones are properly excluded.
        Toy splice sites are [3, 5], [10, 12] and [15, 18], start and end included
        Toy forbidden ranges are [2,6), [8,13) and [13,21), start included, end excluded.
        """
        ind = DiffSequence(
            [
                SNV(position=12, nucleotide="A"),
                RandomDeletion(position=8, size=4),
                RandomInsertion(position=22, nucleotides="ATCGG"),
            ]
        )

        ind.exclude_forbidden_regions(TOY_DANGER_ZONE, r)
        assert len(ind.diffs) == 1
        assert ind.diffs[0].position == 22

        # Now, SNV is not in the forbidden zone because 12 is exclusive
        ind = DiffSequence(
            [
                SNV(position=13, nucleotide="A"),
                RandomDeletion(position=8, size=4),
                RandomInsertion(position=22, nucleotides="ATCGG"),
            ]
        )

        ind.exclude_forbidden_regions(TOY_DANGER_ZONE, r)
        assert len(ind.diffs) == 2

        # Two deletions, one in the forbidden zone
        ind = DiffSequence(
            [RandomDeletion(position=6, size=5), RandomDeletion(position=0, size=2)]
        )
        ind.exclude_forbidden_regions(TOY_DANGER_ZONE, r)
        assert len(ind.diffs) == 1
        assert ind.diffs[0].position == 0

        # Two deletions in forbidden zone, remove those
        ind = DiffSequence(
            [RandomDeletion(position=6, size=5), RandomDeletion(position=1, size=2)]
        )

        ind.exclude_forbidden_regions(TOY_DANGER_ZONE, r)
        assert not ind.diffs

    def test_clean_phenotype(self):
        """
        Tests if the overlaps in the phenotype are properly handled.
        """

        # Overlappping SNV, Deletion and Insertion
        ind = DiffSequence(
            [
                SNV(position=10, nucleotide="A"),
                RandomDeletion(position=8, size=4),
                RandomInsertion(position=9, nucleotides="ATG"),
            ]
        )

        ind.clean(TOY_SEQ, r)
        assert len(ind.diffs) == 1
        assert isinstance(ind.diffs[0], RandomDeletion)
        assert ind.diffs[0].position == 8

        # Overlappping RandomDeletions
        # Because range overlaps are clustered, only one deletion
        # will be returned here (the longest of the cluster)
        ind = DiffSequence(
            [
                RandomDeletion(position=7, size=4),
                RandomDeletion(position=8, size=3),
                RandomDeletion(position=9, size=5),
                RandomDeletion(position=6, size=2),
            ]
        )
        ind.clean(TOY_SEQ, r)
        assert len(ind.diffs) == 1
        assert ind.diffs[0].position == 9

        # Overlappping RandomDeletions
        # Two clusters here, indvidual should be composed by 2 DiffUnits
        ind = DiffSequence(
            [
                RandomDeletion(position=7, size=4),
                RandomDeletion(position=8, size=3),
                RandomDeletion(position=9, size=5),
                RandomDeletion(position=6, size=1),
            ]
        )

        ind.clean(TOY_SEQ, r)
        assert len(ind.diffs) == 2
        assert ind.diffs[0].position == 6
        assert ind.diffs[1].position == 9

        # Overlappping RandomDeletions of same size, first (position-wise) will be kept
        ind = DiffSequence(
            [
                RandomDeletion(position=8, size=4),
                RandomDeletion(position=7, size=4),
                RandomDeletion(position=15, size=4),
            ]
        )
        ind.clean(TOY_SEQ, r)
        assert len(ind.diffs) == 2
        assert ind.diffs[0].position == 7

        # Overlappping RandomInsertion and SNV
        ind = DiffSequence(
            [
                SNV(position=8, nucleotide="A"),
                RandomInsertion(position=8, nucleotides="ATCGG"),
                RandomInsertion(position=9, nucleotides="ATCGG"),
            ]
        )
        ind.clean(TOY_SEQ, r)
        assert len(ind.diffs) == 2
        assert all([isinstance(x, RandomInsertion) for x in ind.diffs])

        # Redundant SNV, empty phenotype
        ind = DiffSequence([SNV(position=8, nucleotide="G")])
        ind.clean(TOY_SEQ, r)
        assert not ind.diffs

        # 1 Redundant SNVs will be removed
        ind = DiffSequence(
            [SNV(position=8, nucleotide="G"), SNV(position=0, nucleotide="T")]
        )
        ind.clean(TOY_SEQ, r)
        assert len(ind.diffs) == 1
        assert ind.diffs[0].position == 0 and ind.diffs[0].nucleotide != "A"

        # Multiple elements with redundant SNVs
        ind = DiffSequence(
            [
                SNV(position=8, nucleotide="G"),
                RandomInsertion(position=8, nucleotides="ATCGG"),
                RandomInsertion(position=9, nucleotides="ATCGG"),
                RandomDeletion(position=2, size=3),
                SNV(position=0, nucleotide="A"),
            ]
        )

        ind.clean(TOY_SEQ, r)
        assert len(ind.diffs) == 3
        assert not all([isinstance(x, SNV) for x in ind.diffs])


class TestPhenotypeToSequence:
    def test_mutate(self):
        """
        Tests if the mutation operations work as expected

                                |         |         |
        Toy seq idxs: 0123456789012345678901234567890134
        Toy sequence: ACAGCAGGGGGGTTTTAGCCGTTACAGTCGATGC
        Toy splice sites are [3, 5], [10, 12] and [15, 18], start and end included
        Toy danger zone are [2,6), [8,13) and [15,21), start included, end excluded.
        """

        # General test
        ind = DiffSequence(
            [
                SNV(position=5, nucleotide="A"),
                SNV(position=6, nucleotide="G"),
                SNV(position=6, nucleotide="T"),
                RandomInsertion(position=15, nucleotides="CCGG"),
                RandomInsertion(position=21, nucleotides="AAAA"),
                SNV(position=2, nucleotide="A"),
            ]
        )

        ind.exclude_forbidden_regions(TOY_DANGER_ZONE, r)
        assert len(ind.diffs) == 3
        assert ind.diffs[0].position == 6
        assert ind.diffs[1].position == 6
        assert ind.diffs[2].position == 21

        ind.clean(TOY_SEQ, r)
        assert len(ind.diffs) == 2
        assert ind.diffs[0].nucleotide == "T"

        seq = TOY_SEQ
        ss_idx = TOY_SS_IDX
        new_seq, new_ss_idx = ind.apply_diff(seq, ss_idx)
        assert ss_idx == new_ss_idx
        assert len(new_seq) == len(seq) + 4
        assert new_seq[21:25] == "AAAA"

        # Mutate in the last position
        ind = DiffSequence(
            [RandomDeletion(position=6, size=2), SNV(position=33, nucleotide="A")]
        )

        seq = TOY_SEQ
        ss_idx = TOY_SS_IDX
        new_seq, new_ss_idx = ind.apply_diff(seq, ss_idx)
        assert len(ind.diffs) == 2
        assert len(new_seq) == len(seq) - 2
        assert new_seq[-1] == "A"
        assert new_ss_idx == [[3, 5], [8, 10], [13, 16]]

        ind = DiffSequence(
            [
                RandomDeletion(position=6, size=2),
                RandomInsertion(position=34, nucleotides="A"),
            ]
        )

        seq = TOY_SEQ
        ss_idx = TOY_SS_IDX
        new_seq, new_ss_idx = ind.apply_diff(seq, ss_idx)
        assert len(ind.diffs) == 2
        assert len(new_seq) == len(seq) - 1
        assert new_seq[-1] == "A"
        assert new_ss_idx == [[3, 5], [8, 10], [13, 16]]
