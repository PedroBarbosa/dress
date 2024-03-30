import re
import os
import pandas as pd
import pytest
from geneticengine.core.grammar import extract_grammar
from geneticengine.core.random.sources import RandomSource
from geneticengine.core.representations.tree.treebased import random_node
from geneticengine.core.utils import get_arguments

from dress.datasetgeneration.grammars.random_perturbation_grammar import (
    DiffUnit,
    create_random_grammar,
)

def _get_input_data() -> dict:
    TOY_SS_IDX = [[100, 161], [314, 376], [1560, 1642]]
    rename = {
        "Seq_id": "seq_id",
        "Sequence": "seq",
        "Splice_site_positions": "ss_idx",
        "Score": "score",
    }
    data = (
        pd.read_csv(f"{os.path.dirname(os.path.abspath(__file__))}/data/dataset_original_seq.csv")
        .rename(columns=rename)
        .iloc[0]
        .to_dict()
    )
    data["ss_idx"] = TOY_SS_IDX
    return data

toy_grammar, excluded_r = create_random_grammar(
    input_seq=_get_input_data(),
    max_diff_units=6,
    snv_weight=0.33,
    insertion_weight=0.33,
    deletion_weight=0.33,
    max_insertion_size=5,
    max_deletion_size=5,
)

DiffSequence = toy_grammar.starting_symbol
SNV = toy_grammar.alternatives[DiffUnit][0]
Deletion = toy_grammar.alternatives[DiffUnit][1]
Insertion = toy_grammar.alternatives[DiffUnit][2]


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


class TestGrammarStructure:
    def test_root(self):
        for i in range(10):
            r = RandomSource(seed=i)
            g = extract_grammar([SNV], DiffSequence)
            x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

            assert isinstance(x, DiffSequence)

    def test_leaf(self):
        for i in range(10):
            r = RandomSource(seed=i)
            g = extract_grammar([Deletion], DiffSequence)
            x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

            assert isinstance(x, DiffSequence)
            assert contains_type(x, Deletion)
            assert x.gengy_distance_to_term == 2

    def test_depth(self):
        for i in range(10):
            r = RandomSource(seed=i)
            g = extract_grammar([SNV, Deletion, Insertion], DiffSequence)
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

    def test_leaf_object(self):
        for i in range(10):
            r = RandomSource(seed=i)
            g = extract_grammar([SNV], DiffSequence)
            x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

            for diff in x.diffs:
                assert isinstance(diff.position, int)
                assert isinstance(diff.nucleotide, str)

            g = extract_grammar([Deletion], DiffSequence)
            x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

            for x in x.diffs:
                assert isinstance(x.position, int)
                assert isinstance(x.size, int)
                assert x.size <= 5

            g = extract_grammar([Insertion], DiffSequence)
            x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

            for x in x.diffs:
                assert isinstance(x.position, int)
                assert isinstance(x.nucleotides, str)
                assert len(x.nucleotides) <= 5

    def test_phenotype(self):
        for i in range(10):
            r = RandomSource(seed=i)
            g = extract_grammar([SNV, Deletion, Insertion], DiffSequence)
            x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

            for diffunit in x.diffs:
                rgex = re.search(f'\[(.*?)\]', str(diffunit))
                assert len(rgex.group(1).split(',')) == 4
   
    def test_invalid_node(self):
        with pytest.raises(Exception):
            extract_grammar([SNV, DiffUnit, Useless], DiffSequence)


class TestGrammarMetaHandlers:
    for i in range(100):
        r = RandomSource(seed=i)
        g = extract_grammar([SNV, Deletion, Insertion], DiffSequence)
        x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

        for diff_unit in x.diffs:
            assert isinstance(diff_unit.position, int)

            for range in excluded_r:
                assert diff_unit.position not in range