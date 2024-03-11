from abc import ABC
import abc
from dataclasses import dataclass
import itertools
import sqlite3
from typing import Annotated, Tuple, Union

from geneticengine.grammars.coding.classes import Condition
from geneticengine.grammars.coding.classes import Expr
from geneticengine.grammars.coding.classes import Number
from geneticengine.grammars.coding.conditions import Equals
from geneticengine.grammars.coding.conditions import GreaterThan
from geneticengine.grammars.coding.conditions import LessThan
from geneticengine.grammars.coding.logical_ops import And
from geneticengine.grammars.coding.logical_ops import Not
from geneticengine.grammars.coding.logical_ops import Or
from geneticengine.grammars.coding.numbers import Literal
from geneticengine.grammar.metahandlers.ints import IntRange
import numpy as np

import pandas as pd
from dress.datasetgeneration.metahandlers.strings import RandomNucleotides
from geneticengine.core.random.sources import RandomSource
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.metahandlers.vars import VarRange
from geneticengine.core.decorators import weight
from typing import List
from geneticengine.core.grammar import Grammar, extract_grammar
import pyranges as pr

from pypika import Query, Table

NUCLEOTIDES = ["A", "C", "G", "T"]
LOCATIONS = ["Exon_upstream", "Exon_cassette", "Exon_downstream", 
             "Intron_upstream", "Intron_downstream"]

LOCATIONS_MAP = {
    'Exon_upstream': ['Exon_upstream_donor_region', 'Exon_upstream_acceptor_region', 'Exon_upstream_fully_contained'],
    'Exon_cassette': ['Exon_cassette_donor_region', 'Exon_cassette_acceptor_region', 'Exon_cassette_fully_contained'],
    'Exon_downstream': ['Exon_downstream_donor_region', 'Exon_downstream_acceptor_region', 'Exon_downstream_fully_contained'],
    'Intron_upstream': ['Intron_upstream'],
    'Intron_downstream': ['Intron_downstream']
}
class MotifRule(ABC):
    pass

def create_grammar(
    db: sqlite3,
    rbp_list: list,
    max_n_rules: int,
    motif_presence_weight: float,
    motif_co_occurrence_weight: float,
    motif_inter_distance_weight: float,
    motif_ss_distance_weight: int,
) -> Grammar:
    """Creates a grammar to apply on the original sequence of length seqsize

    Args:
        rbp_list (list): List of RBPs to be used in the explanation
        max_n_rules (int): Maximum number of rules allowed per explanation
        motif_presence_weight (float): Probability to generate a MotifPresence rule from the grammar
        motif_co_occurrence_weight (float): Probability to generate a MotifCoOccurrence rule from the grammar
        motif_inter_distance_weight (float): Probability to generate a MotifDistancePair rule from the grammar
        motif_ss_distance_weight (float): Probability to generate a MotifDistanceToSS rule from the grammar

    Returns:
        Grammar: Returns the grammar that specifies the production rules of the explainer
    """

    @dataclass
    class DatasetExplanation(object):
        rules: Annotated[list[MotifRule], ListSizeBetween(1, max_n_rules)]
        
    @dataclass
    class MotifPresence(MotifRule):
        rbp: Annotated[int, VarRange(rbp_list)]
        location: Annotated[str, VarRange(LOCATIONS)]

        def evaluate(self, cursor: sqlite3, X: pd.DataFrame) -> str:
            
            motifs = Table('motifs')
            q = Query.from_(motifs).groupby(motifs.Seq_id).select(motifs.Seq_id).where((motifs.rbp_name == self.rbp) & (motifs.location.isin(LOCATIONS_MAP[self.location])))
            cursor.execute(str(q))
            res = [x[0] for x in cursor.fetchall()]
            return np.isin(X.iloc[:, 0], res)

        def __str__(self):
            return f"{self.rbp} motif at {self.location}]"

    
    return extract_grammar([MotifPresence], DatasetExplanation)