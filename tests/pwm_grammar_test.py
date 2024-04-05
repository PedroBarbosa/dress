import os
import re
from typing import Union
import pandas as pd
import pytest
import shutil
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

motif_matches = pd.read_csv(
    f"{os.path.dirname(os.path.abspath(__file__))}/data/motif_matches.csv.gz", sep="\t"
)


def get_grammar(input_seq: dict, subset_rbps: Union[list, str] = None):
    outdir = f"{os.path.dirname(os.path.abspath(__file__))}/pwm_test"
    toy_grammar, excluded_r = create_motif_grammar(
        input_seq=input_seq,
        max_diff_units=6,
        snv_weight=0.2,
        insertion_weight=0.2,
        deletion_weight=0.2,
        motif_substitution_weight=0.2,
        motif_ablation_weight=0.2,
        outdir=outdir,
        logger=setup_logger(),
        subset_rbps="encode" if subset_rbps is None else subset_rbps,
    )
    shutil.rmtree(outdir)
    return toy_grammar, excluded_r


input_seq = _get_input_data()
wt_seq = input_seq["seq"]
ss_idx = input_seq["ss_idx"]
toy_grammar, excluded_r = get_grammar(input_seq)

DiffSequence = toy_grammar.starting_symbol
MotifSNV = toy_grammar.alternatives[DiffUnit][0]
MotifDeletion = toy_grammar.alternatives[DiffUnit][1]
MotifInsertion = toy_grammar.alternatives[DiffUnit][2]
MotifAblation = toy_grammar.alternatives[DiffUnit][3]
MotifSubstitution = toy_grammar.alternatives[DiffUnit][4]


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
            g = extract_grammar(
                [
                    MotifSNV,
                    MotifDeletion,
                    MotifInsertion,
                    MotifAblation,
                    MotifSubstitution,
                ],
                DiffSequence,
            )
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

            # SNV
            g = extract_grammar([MotifSNV], DiffSequence)
            x = random_node(r, g, max_depth=3, starting_symbol=DiffSequence)

            for x in x.diffs:
                assert isinstance(x.position, int)
                assert isinstance(x.nucleotide, str)
                assert len(x.nucleotide) == 1
                assert len(x.perturb(wt_seq, 0)) == len(wt_seq)

            # Del
            g = extract_grammar([MotifDeletion], DiffSequence)
            x = random_node(r, g, max_depth=3, starting_symbol=DiffSequence)

            for x in x.diffs:
                assert isinstance(x.position, tuple)
                assert len(x.position) == 3
                assert all(isinstance(i, int) for i in x.position)
                new_seq = x.perturb(wt_seq, 0)
                assert (
                    len(new_seq) < len(wt_seq)
                    and len(wt_seq) - len(new_seq) == x.get_size()
                )

            # Ins
            g = extract_grammar([MotifInsertion], DiffSequence)
            x = random_node(r, g, max_depth=3, starting_symbol=DiffSequence)

            for x in x.diffs:
                assert isinstance(x.position, int)
                assert isinstance(x.nucleotides, tuple)
                assert len(x.nucleotides) == 2
                assert all(isinstance(i, str) for i in x.nucleotides)
                new_seq = x.perturb(wt_seq, 0)
                assert (
                    len(new_seq) > len(wt_seq)
                    and len(new_seq) - len(wt_seq) == x.get_size()
                )

            # Ablation
            g = extract_grammar([MotifAblation], DiffSequence)
            x = random_node(r, g, max_depth=3, starting_symbol=DiffSequence)

            for x in x.diffs:

                assert isinstance(x.position, tuple)
                assert len(x.position) == 4
                assert all(isinstance(i, int) for i in x.position[0:3])
                assert isinstance(x.position[3], str)
                assert len(x.perturb(wt_seq, 0)) == len(wt_seq)

            # Substitution
            g = extract_grammar([MotifSubstitution], DiffSequence)
            x = random_node(r, g, max_depth=3, starting_symbol=DiffSequence)

            for x in x.diffs:
                assert isinstance(x.position, int)
                assert isinstance(x.nucleotides, tuple)
                assert len(x.nucleotides) == 2
                assert all(isinstance(i, str) for i in x.nucleotides)
                assert len(x.perturb(wt_seq, 0)) == len(wt_seq)

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
            g = extract_grammar(
                [MotifDeletion, MotifInsertion, MotifAblation, MotifSubstitution],
                DiffSequence,
            )
            x = random_node(r, g, max_depth=3, starting_symbol=DiffSequence)

            for diffunit in x.diffs:
                rgex = re.search(f"\[(.*?)\]", str(diffunit))
                assert len(rgex.group(1).split(",")) == 5

    def test_invalid_node(self):
        with pytest.raises(Exception):
            extract_grammar([MotifSNV, DiffUnit, Useless], DiffSequence)


class TestSubsetRBPs:

    def test_subset_single(self):
        toy_grammar_celf1, _ = get_grammar(input_seq=input_seq, subset_rbps="CELF1")
        DiffSequence = toy_grammar_celf1.starting_symbol
        MotifSNV = toy_grammar_celf1.alternatives[DiffUnit][0]
        MotifDeletion = toy_grammar_celf1.alternatives[DiffUnit][1]
        MotifInsertion = toy_grammar_celf1.alternatives[DiffUnit][2]

        for i in range(100):

            r = RandomSource(seed=i)
            g = extract_grammar(
                [
                    MotifSNV,
                    MotifDeletion,
                    MotifInsertion,
                    MotifSubstitution,
                    MotifSubstitution,
                ],
                DiffSequence,
            )
            x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

            for diffunit in x.diffs:
                rgex = re.search(f"\[(.*?)\]", str(diffunit))
                assert rgex.group(1).split(",")[1] == "CELF1"

    def test_subset_multiple(self):
        toy_grammar_mult, _ = get_grammar(
            input_seq=input_seq, subset_rbps=["SRSF1", "CELF1"]
        )
        DiffSequence = toy_grammar_mult.starting_symbol
        MotifSNV = toy_grammar_mult.alternatives[DiffUnit][0]
        MotifDeletion = toy_grammar_mult.alternatives[DiffUnit][1]
        MotifInsertion = toy_grammar_mult.alternatives[DiffUnit][2]

        for i in range(100):

            r = RandomSource(seed=i)
            g = extract_grammar(
                [
                    MotifSNV,
                    MotifDeletion,
                    MotifInsertion,
                    MotifAblation,
                    MotifSubstitution,
                ],
                DiffSequence,
            )
            x = random_node(r, g, max_depth=2, starting_symbol=DiffSequence)

            for diffunit in x.diffs:
                rgex = re.search(f"\[(.*?)\]", str(diffunit))
                assert rgex.group(1).split(",")[1] in ["CELF1", "SRSF1"]


class TestGrammarMetaHandlers:
    for i in range(100):
        r = RandomSource(seed=i)
        g = extract_grammar(
            [MotifSNV, MotifDeletion, MotifInsertion, MotifAblation, MotifSubstitution],
            DiffSequence,
        )
        x = random_node(r, g, max_depth=3, starting_symbol=DiffSequence)

        for diff_unit in x.diffs:

            if isinstance(diff_unit, (MotifSNV, MotifInsertion)):
                assert isinstance(diff_unit.position, int)

                for range in excluded_r:
                    assert diff_unit.position not in range

            elif isinstance(diff_unit, (MotifAblation, MotifDeletion)):
                assert isinstance(diff_unit.position, tuple)
                for range in excluded_r:
                    assert not (
                        range.start < diff_unit.position[1]
                        and range.stop > diff_unit.position[0]
                    )

            # End position of insertion is handled elsewhere (exclude_forbidden_regions)
            elif isinstance(diff_unit, MotifSubstitution):
                assert isinstance(diff_unit.position, int)
                for range in excluded_r:
                    assert diff_unit.position not in range
