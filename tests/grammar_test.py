import pytest
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.utils import get_arguments

from dress.datasetgeneration.grammars.random_perturbation_grammar import (
    DiffUnit,
    create_random_grammar,
)

TOY_SEQ = "ACAGCAGGGGGGTTTTAGCCGTTACAGTCGATGC"
TOY_SS_IDX = [[3, 5], [10, 12], [15, 18]]
TOY_DANGER_ZONE = [range(2, 6), range(8, 13), range(15, 21)]

toy_grammar = create_random_grammar(
    max_diff_units=6,
    snv_weight=0.33,
    insertion_weight=0.33,
    deletion_weight=0.33,
    max_insertion_size=5,
    max_deletion_size=5,
    input_seq={"seq": TOY_SEQ, "ss_idx": TOY_SS_IDX},
)

DiffSequence = toy_grammar.starting_symbol
SNV = toy_grammar.alternatives[DiffUnit][0]
RandomDeletion = toy_grammar.alternatives[DiffUnit][1]
RandomInsertion = toy_grammar.alternatives[DiffUnit][2]


class Useless(DiffUnit):
    def __init__(self, a):
        pass  # a does not have a type


def contains_type(t, ty: type):
    if isinstance(t, ty):
        return True
    elif isinstance(t, list):
        for el in t:
            if contains_type(el, ty):
                return True
    else:
        for argn, argt in get_arguments(t):
            if contains_type(getattr(t, argn), ty):
                return True
    return False


class TestGrammar:
    def test_root(self):
        for i in range(10):
            r = RandomSource(seed=i)
            g = extract_grammar([SNV], DiffSequence)
            x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

            assert isinstance(x, DiffSequence)

    def test_leaf(self):
        for i in range(10):
            r = RandomSource(seed=i)
            g = extract_grammar([RandomDeletion], DiffSequence)
            x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

            assert isinstance(x, DiffSequence)
            assert contains_type(x, RandomDeletion)

    def test_depth(self):
        for i in range(10):
            r = RandomSource(seed=i)
            g = extract_grammar([SNV, RandomDeletion, RandomInsertion], DiffSequence)
            x = random_node(r, g, max_depth=10, starting_symbol=DiffSequence)

            assert x.gengy_distance_to_term == 2

    def test_n_ind_diffs(self):
        for i in range(10):
            r = RandomSource(seed=i)
            g = extract_grammar([SNV], DiffSequence)
            x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

            assert 1 <= len(x.diffs) <= 6
            assert contains_type(x, SNV)
            assert isinstance(x, DiffSequence)

    def test_invalid_node(self):
        with pytest.raises(Exception):
            extract_grammar([SNV, DiffUnit, Useless], DiffSequence)
