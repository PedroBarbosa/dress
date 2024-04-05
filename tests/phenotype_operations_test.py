from dress.datasetgeneration.grammars.random_perturbation_grammar import (
    DiffUnit,
    create_random_grammar,
)
from geneticengine.core.random.sources import RandomSource

# SEQ = "CGCCGCGCCGTTGCCCCGCCCCTTGCAACCCCGCCCCGCGCCGGCCCCGCCCCTGCTCTCGCGCCGGCGTCGGCTGCGTCTCCGGCGTTTGAATTGCGCTTCCGCCATCTTTCCAGCCTCAGTCGGACGGGCGCGGAGACGCTTCTGGAAGGTATCGCGACCCGGCGGGCCCGGCACGGCCGGGCGGGGACAGGGGTGGCGGCGGCGGGATAGGGGTCGGAGCCGGGGCCAGGGCCGGGGCGGGTGGCGGACCCAGGGGCAGCGGGCGGCTGAGTAGGTGGGTGGTGCGGGCCCGGCCGGGCCGGGGCAGGAGACGGGCGTGGGGTCGGCGCTAGCCCCCGCGAACCCCCGTTTCATCCTCCGCTCTCATCCCCGTCCCGGTCCCAGTCCCGTTCCCATCCCTACACCTCCGGCCGCCGTTCCCCGGGCCCCGCCGCCCCGGATGCCGGCCCCGCCCGCCGCCTTCCCGCTCCCAGGCCTGGCCGCCATGGCGCCGCGGGCGGGAGGCCTTTGTGGGGCGGGCACGTGGGGCGCTGGGGGCGCGGGAGCGGGGCCGCCATGGGCTGCGGGGCCGCGCGAGCGCTCGCCTCCGTCCTCTGCCTCCGCAGGAACGCCGCGATGGCTGCGCAGGGAGAGCCCCAGGTCCAGTTCAAAGTAGGTAACCCTGCGGGGCGGGAGGCGGCCGAGCCCGACCGCGTGCGACTCGCGGGTCCCTCCTCCTGGGGCCACGATGGCTGTAATGGGGCCCCGCATCCACATTCTTTGTTTTAAGTGAGCCTGTGGTGGTTAAAGTTCCGTGACTCTGGGATCTTGAGAGGTGAAGTGTTTAGGGTTTACTTCCAAAATGTGTTTTTCAACAGCTTGTATTGGTTGGTGATGGTGGTACTGGAAAAACGACCTTCGTGAAACGTCATTTGACTGGTGAATTTGAGAAGAAGTATGTAGGTATGTGCTGGAAAACCTTGCTTGTGGAAATATGTGAGAAATGGGTAAGTTCATCCACTCAATCGCATCGTTTCCGTTTCAGCCACCTTGGGTGTTGAGG"
# SS_IDX = [[100, 150], [608, 641], [860, 944]]

TOY_SEQ = "ACAGCAGGGGGGTTTTAGCCGTTACAGTCGATGC"
TOY_SS_IDX = [[3, 5], [10, 12], [15, 18]]
TOY_DANGER_ZONE = [range(2, 6), range(8, 13), range(15, 21)]

toy_grammar, _ = create_random_grammar(
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
Deletion = toy_grammar.alternatives[DiffUnit][1]
Insertion = toy_grammar.alternatives[DiffUnit][2]
rs = RandomSource(0)


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
                Deletion(position=8, size=4),
                Insertion(position=22, nucleotides="ATCGG"),
            ]
        )

        ind.exclude_forbidden_regions(TOY_DANGER_ZONE)
        assert len(ind.diffs) == 1
        assert ind.diffs[0].position == 22

        # Now, SNV is not in the forbidden zone because 12 is exclusive
        ind = DiffSequence(
            [
                SNV(position=13, nucleotide="A"),
                Deletion(position=8, size=4),
                Insertion(position=22, nucleotides="ATCGG"),
            ]
        )

        ind.exclude_forbidden_regions(TOY_DANGER_ZONE)
        assert len(ind.diffs) == 2

        # Two deletions, one in the forbidden zone
        ind = DiffSequence(
            [Deletion(position=6, size=5), Deletion(position=0, size=2)]
        )
        ind.exclude_forbidden_regions(TOY_DANGER_ZONE)
        assert len(ind.diffs) == 1
        assert ind.diffs[0].position == 0

        # Two deletions in forbidden zone, remove those
        ind = DiffSequence(
            [Deletion(position=6, size=5), Deletion(position=1, size=2)]
        )
      
        ind.exclude_forbidden_regions(TOY_DANGER_ZONE)
        assert not ind.diffs

    def remove_diffunit_overlaps(self):
        """
        Tests if the overlaps in the phenotype are properly handled.
        """

        # Overlappping SNV, Deletion and Insertion
        ind = DiffSequence(
            [
                SNV(position=10, nucleotide="A"),
                Deletion(position=8, size=4),
                Insertion(position=9, nucleotides="ATG"),
            ]
        )

        ind.remove_diffunit_overlaps(TOY_SEQ, rs)
        assert len(ind.diffs) == 1
        assert isinstance(ind.diffs[0], Deletion)
        assert ind.diffs[0].position == 8

        # Overlappping Deletions
        # Because range overlaps are clustered, only one deletion
        # will be returned here (the longest of the cluster)
        ind = DiffSequence(
            [
                Deletion(position=7, size=4),
                Deletion(position=8, size=3),
                Deletion(position=9, size=5),
                Deletion(position=6, size=2),
            ]
        )
        ind.remove_diffunit_overlaps(TOY_SEQ, rs)
        assert len(ind.diffs) == 1
        assert ind.diffs[0].position == 9

        # Overlappping RandomDeletions
        # Two clusters here, indvidual should be composed by 2 DiffUnits
        ind = DiffSequence(
            [
                Deletion(position=7, size=4),
                Deletion(position=8, size=3),
                Deletion(position=9, size=5),
                Deletion(position=6, size=1),
            ]
        )

        ind.remove_diffunit_overlaps(TOY_SEQ, rs)
        assert len(ind.diffs) == 2
        assert ind.diffs[0].position == 6
        assert ind.diffs[1].position == 9

        # Overlappping RandomDeletions of same size, first (position-wise) will be kept
        ind = DiffSequence(
            [
                Deletion(position=8, size=4),
                Deletion(position=7, size=4),
                Deletion(position=15, size=4),
            ]
        )
        ind.remove_diffunit_overlaps(TOY_SEQ, rs)
        assert len(ind.diffs) == 2
        assert ind.diffs[0].position == 7

        # Overlappping RandomInsertion and SNV
        ind = DiffSequence(
            [
                SNV(position=8, nucleotide="A"),
                Insertion(position=8, nucleotides="ATCGG"),
                Insertion(position=9, nucleotides="ATCGG"),
            ]
        )
        ind.remove_diffunit_overlaps(TOY_SEQ, rs)
        assert len(ind.diffs) == 2
        assert all([isinstance(x, Insertion) for x in ind.diffs])

        # Redundant SNV, replace by another nucleotide
        ind = DiffSequence([SNV(position=8, nucleotide="G")])
        ind.remove_diffunit_overlaps(TOY_SEQ, rs)
        assert ind.diffs[0].nucleotide != "G"

        # 1 Redundant SNVs will be removed
        ind = DiffSequence(
            [SNV(position=8, nucleotide="G"), SNV(position=0, nucleotide="T")]
        )
        ind.remove_diffunit_overlaps(TOY_SEQ, rs)
        assert len(ind.diffs) == 2
        assert ind.diffs[0].position == 8 and ind.diffs[0].nucleotide != "G"
        assert ind.diffs[1].position == 0 and ind.diffs[1].nucleotide == "T"

        # Redundant SNV with 1 overlap
        ind = DiffSequence(
            [
                SNV(position=8, nucleotide="G"),
                Insertion(position=8, nucleotides="ATCGG"),
                Insertion(position=9, nucleotides="ATCGG"),
                Deletion(position=2, size=3),
                SNV(position=0, nucleotide="A"),
            ]
        )

        ind.remove_diffunit_overlaps(TOY_SEQ, rs)
        assert len(ind.diffs) == 4
        sort_order = [SNV, Deletion, Insertion, Insertion]
        assert all([isinstance(x, y) for x, y in zip(ind.diffs, sort_order)])
        assert ind.diffs[0].position == 0 and ind.diffs[0].nucleotide != "A"


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
                Insertion(position=15, nucleotides="CCGG"),
                Insertion(position=21, nucleotides="AAAA"),
                SNV(position=2, nucleotide="A"),
            ]
        )

        ind.exclude_forbidden_regions(TOY_DANGER_ZONE)
        assert len(ind.diffs) == 3
        assert ind.diffs[0].position == 6
        assert ind.diffs[1].position == 6
        assert ind.diffs[2].position == 21


        ind.remove_diffunit_overlaps(TOY_SEQ, rs)
        assert len(ind.diffs) == 2
        assert ind.diffs[0].nucleotide == "C"

        seq = TOY_SEQ
        ss_idx = TOY_SS_IDX
        new_seq, new_ss_idx = ind.apply_diff(seq, ss_idx)
        assert ss_idx == new_ss_idx
        assert len(new_seq) == len(seq) + 4
        assert new_seq[21:25] == "AAAA"

        # Mutate in the last position
        ind = DiffSequence(
            [Deletion(position=6, size=2), SNV(position=33, nucleotide="A")]
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
                Deletion(position=6, size=2),
                Insertion(position=34, nucleotides="A"),
            ]
        )

        seq = TOY_SEQ
        ss_idx = TOY_SS_IDX
        new_seq, new_ss_idx = ind.apply_diff(seq, ss_idx)
        assert len(ind.diffs) == 2
        assert len(new_seq) == len(seq) - 1
        assert new_seq[-1] == "A"
        assert new_ss_idx == [[3, 5], [8, 10], [13, 16]]
