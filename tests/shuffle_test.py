from geneticengine.core.random.sources import RandomSource, Source
from dress.datasetgeneration.grammars.utils import shuffle, dinuc_shuffle
import numpy as np

rs = RandomSource(0)
nucs = ["A", "C", "G", "T"]


def dinuc_content(seq):

    counts = {}
    for i in range(len(seq) - 1):
        try:
            counts[seq[i : i + 2]] += 1
        except KeyError:
            counts[seq[i : i + 2]] = 1
    return counts


def nuc_content(seq):
    return dict(zip(*np.unique(list(seq), return_counts=True)))


class TestRandomNucleotides:

    def test_shuffle(self):
        
        for _ in range(100):
            random_seq = [rs.choice(nucs) for _ in range(2000)]
            shuffled_seq = shuffle(random_seq, rs=rs)

            assert random_seq != shuffled_seq

            orig_nuc_cont = nuc_content(random_seq)
            shuf_nuc_cont = nuc_content(shuffled_seq)
            for nuc in orig_nuc_cont.keys():
                assert orig_nuc_cont[nuc] == shuf_nuc_cont[nuc]

    def test_dinuc_shuffle(self):
        # Return one shuffle
        for _ in range(100):
            random_seq = "".join(rs.choice(nucs) for _ in range(2000))
            dinuc_shuffle_seq = dinuc_shuffle(random_seq, rs=rs)

            assert random_seq != dinuc_shuffle_seq

            orig_dinuc_cont = dinuc_content(random_seq)
            shuf_dinuc_conts = dinuc_content(dinuc_shuffle_seq)

            for dinuc in orig_dinuc_cont.keys():
                assert orig_dinuc_cont[dinuc] == shuf_dinuc_conts[dinuc]
        
        # Return multiple shuffles
        for _ in range(10):
            random_seq = "".join(rs.choice(nucs) for _ in range(2000))
            dinuc_shuffle_seqs = dinuc_shuffle(random_seq, rs=rs, num_shufs=10)
            assert len(dinuc_shuffle_seqs) == 10
            for shuff in dinuc_shuffle_seqs:
                assert random_seq != shuff
            orig_dinuc_cont = dinuc_content(random_seq)
            for dinuc in orig_dinuc_cont.keys():
                for seq in dinuc_shuffle_seqs:
                    assert len(seq) == len(random_seq)
                    _d = dinuc_content(seq)
                    assert orig_dinuc_cont[dinuc] == _d[dinuc]
        
