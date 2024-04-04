import os
import re
from typing import Union
import pandas as pd
import pytest
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.utils import get_arguments

from dress.datasetgeneration.grammars.pwm_perturbation_grammar import (
    DiffUnit,
    create_motif_grammar,
)
from dress.datasetgeneration.logger import setup_logger
from tests.random_grammar_test import _get_input_data

motif_matches = pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}/data/motif_matches.csv.gz", sep="\t")

def get_grammar(subset_rbps: Union[list, str] = None):
    toy_grammar, excluded_r = create_motif_grammar(
        input_seq=_get_input_data(),
        max_diff_units=6,
        snv_weight=0.33,
        insertion_weight=0.33,
        deletion_weight=0.33,
        outdir="",
        logger=setup_logger(),
        subset_rbps='encode' if subset_rbps is None else subset_rbps,
    )
    return toy_grammar, excluded_r

toy_grammar, excluded_r = get_grammar()
DiffSequence = toy_grammar.starting_symbol
MotifSNV = toy_grammar.alternatives[DiffUnit][0]
MotifDeletion = toy_grammar.alternatives[DiffUnit][1]
MotifInsertion = toy_grammar.alternatives[DiffUnit][2]

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
            g = extract_grammar([MotifSNV], DiffSequence)
            x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

            assert isinstance(x, DiffSequence)

    def test_leaf(self):
        for i in range(10):
            r = RandomSource(seed=i)
            g = extract_grammar([MotifDeletion], DiffSequence)
            x = random_node(
                r, g, max_depth=3, starting_symbol=DiffSequence
            )  # why does 2 not work?

            assert isinstance(x, DiffSequence)
            assert contains_type(x, MotifDeletion)
            assert x.gengy_distance_to_term == 2

    def test_depth(self):
        for i in range(10):
            r = RandomSource(seed=i)
            g = extract_grammar([MotifSNV, MotifDeletion, MotifInsertion], DiffSequence)
            x = random_node(r, g, max_depth=10, starting_symbol=DiffSequence)

            assert x.gengy_distance_to_term == 2

    def test_n_ind_diffs(self):
        for i in range(10):
            r = RandomSource(seed=i)
            g = extract_grammar([MotifSNV], DiffSequence)
            x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

            assert 1 <= len(x.diffs) <= 6
            assert contains_type(x, MotifSNV)
            assert isinstance(x, DiffSequence)

    def test_leaf_object(self):
        for i in range(10):
            r = RandomSource(seed=i)
            g = extract_grammar([MotifSNV], DiffSequence)
            x = random_node(r, g, max_depth=3, starting_symbol=DiffSequence)

            for diff in x.diffs:
                assert isinstance(diff.position, int)
                assert isinstance(diff.nucleotide, str)
                assert len(diff.nucleotide) == 1

            g = extract_grammar([MotifDeletion], DiffSequence)
            x = random_node(r, g, max_depth=3, starting_symbol=DiffSequence)

            for x in x.diffs:
                assert isinstance(x.position, tuple)
                assert len(x.position) == 3
                assert all(isinstance(i, int) for i in x.position)

            g = extract_grammar([MotifInsertion], DiffSequence)
            x = random_node(r, g, max_depth=3, starting_symbol=DiffSequence)

            for x in x.diffs:
                assert isinstance(x.position, int)
                assert isinstance(x.nucleotides, tuple)
                assert len(x.nucleotides) == 2
                assert all(isinstance(i, str) for i in x.nucleotides)

    def test_phenotype(self):
        for i in range(10):
            r = RandomSource(seed=i)
            g = extract_grammar([MotifSNV], DiffSequence)
            x = random_node(r, g, max_depth=3, starting_symbol=DiffSequence)

            for diffunit in x.diffs:
                rgex = re.search(f"\[(.*?)\]", str(diffunit))
                assert len(rgex.group(1).split(",")) == 6

        for i in range(10):
            r = RandomSource(seed=i)
            g = extract_grammar([MotifDeletion, MotifInsertion], DiffSequence)
            x = random_node(r, g, max_depth=3, starting_symbol=DiffSequence)

            for diffunit in x.diffs:
                rgex = re.search(f"\[(.*?)\]", str(diffunit))
                assert len(rgex.group(1).split(",")) == 5

    def test_invalid_node(self):
        with pytest.raises(Exception):
            extract_grammar([MotifSNV, DiffUnit, Useless], DiffSequence)

class TestSubsetRBPs:

    def test_subset_single(self):
        toy_grammar_celf1, _ = get_grammar(subset_rbps="CELF1")
        DiffSequence = toy_grammar_celf1.starting_symbol
        MotifSNV = toy_grammar_celf1.alternatives[DiffUnit][0]
        MotifDeletion = toy_grammar_celf1.alternatives[DiffUnit][1]
        MotifInsertion = toy_grammar_celf1.alternatives[DiffUnit][2]

        for i in range(100):

            r = RandomSource(seed=i)
            g = extract_grammar([MotifSNV, MotifDeletion, MotifInsertion], DiffSequence)
            x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

            for diffunit in x.diffs:
                rgex = re.search(f"\[(.*?)\]", str(diffunit))
                assert rgex.group(1).split(",")[1] == "CELF1"

    def test_subset_multiple(self):
        toy_grammar_mult, _ = get_grammar(subset_rbps=["SRSF1", "CELF1"])
        DiffSequence = toy_grammar_mult.starting_symbol
        MotifSNV = toy_grammar_mult.alternatives[DiffUnit][0]
        MotifDeletion = toy_grammar_mult.alternatives[DiffUnit][1]
        MotifInsertion = toy_grammar_mult.alternatives[DiffUnit][2]

        for i in range(100):

            r = RandomSource(seed=i)
            g = extract_grammar([MotifSNV, MotifDeletion, MotifInsertion], DiffSequence)
            x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

            for diffunit in x.diffs:
                rgex = re.search(f"\[(.*?)\]", str(diffunit))
                assert rgex.group(1).split(",")[1] in ["CELF1", "SRSF1"]

class TestGrammarMetaHandlers:
    for i in range(100):
        r = RandomSource(seed=i)
        g = extract_grammar([MotifSNV, MotifDeletion, MotifInsertion], DiffSequence)
        x = random_node(r, g, max_depth=3, starting_symbol=DiffSequence)

        for diff_unit in x.diffs:
            
            if isinstance(diff_unit, (MotifSNV, MotifInsertion)):
                assert isinstance(diff_unit.position, int)
                
                for range in excluded_r:
                    assert diff_unit.position not in range

            elif isinstance(diff_unit, MotifDeletion):
                assert isinstance(diff_unit.position, tuple)
                for range in excluded_r:

                    assert not (range.start < diff_unit.position[1] and range.stop > diff_unit.position[0])
