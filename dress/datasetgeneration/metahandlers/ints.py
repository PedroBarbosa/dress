from __future__ import annotations

from typing import Iterable, Union, TypeVar

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.metahandlers.base import MetaHandlerGenerator
from itertools import chain
min = TypeVar("min", covariant=True)
max = TypeVar("max", covariant=True)


class IntListExcludingSomeValues(MetaHandlerGenerator):
    """
    IntListExcludingSomeValues(a,b) chooses an element from a, 
    but making sure that elements in a present in b are not selected

    IntListExcludingSomeValues([1,2,3,4,5,20,30],[range(2,5),range(10,15), 20]) will generate
    an integer between [1,5,30]
    """

    def __init__(self, elements: Iterable[int], exclude: Union[int, Iterable[int], Iterable[range]]):

        if isinstance(exclude, int):
            self.exclude = set([exclude])
        else:           
            self.exclude = set(chain(*[x if isinstance(x, Iterable) else [x] for x in exclude]))

        self.elements = set(elements)

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        new_symbol,
        depth: int,     
        base_type,
        context: dict[str, str],
    ):
        possible_values = list(self.elements - self.exclude)
        assert possible_values, f"no possible values left after excluding {self.exclude}"
        val = r.choice(possible_values, prod=base_type)
        rec(val)

    def __class_getitem__(self, args):
        return IntListExcludingSomeValues(*args)

    def __repr__(self):
        return f"[{self.elements}] excluding {self.exclude}"
    
class IntRangeExcludingSomeValues(MetaHandlerGenerator):
    """
    IntRangeExcludingSomeValues(a,b,c) restricts ints to be between a and b, 
    but making sure that c (which can be an iterable) is not selected

    IntRangeExcludingSomeValues(0,20,[range(2,5),range(10,15), 18]) will generate
    an integer between 0 and 20, but excluding the values [2,3,4,10,11,12,13,14,18]
    """

    def __init__(self, min: int, max: int, exclude: Union[int, Iterable[int], Iterable[range]]):
        assert min <= max, "min must be <= max"

        if isinstance(exclude, int):
            self.exclude = set([exclude])
        else:           
            self.exclude = set(chain(*[x if isinstance(x, Iterable) else [x] for x in exclude]))

        self.min = min
        self.max = max

    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        new_symbol,
        depth: int,     
        base_type,
        context: dict[str, str],
    ):
        possible_values = set(range(self.min, self.max+1)) - self.exclude
        assert possible_values, f"no possible values left after excluding {self.exclude}"
        val = r.choice(list(possible_values), prod=base_type)
        rec(val)

    def __class_getitem__(self, args):
        return IntRangeExcludingSomeValues(*args)

    def __repr__(self):
        return f"[{self.min},..,{self.max}] excluding {self.exclude}"
    
class CustomIntListDeletions(MetaHandlerGenerator):
    """CustomIntListDeletions([a_1, .., a_n]) restricts ints to be an element from a list.

    This metahandler returns a tuple of integers from the element selected from the list.
    The tuple is of the form (del_start_position, del_end_position, element_index)
    """

    def __init__(self, elements):
        assert len(elements) > 0, f"no deletions to select from. Either redo motif scanning "
        "with a more permissive setup, increase the sequence space to search or set deletion_weight to 0"

        self.elements = elements
        
    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        new_symbol,
        depth: int,
        base_type,
        context: dict[str, str],
    ):
        i = r.randint(0, len(self.elements) - 1, "")
        _selected = self.elements[i]
        rec((_selected[0], _selected[1], i))

    def __class_getitem__(self, args):
        return CustomIntListDeletions(*args)

    def __repr__(self):
        return f"[{self.elements}]"


class CustomIntListInsertions(MetaHandlerGenerator):
    """CustomIntListInsertions([a_1, .., a_n]) restricts ints to be an element from a list.
    
    This metahandler returns a tuple of strings from the element select from the list.
    The tuple is of the form (rbp_name, rbp_motif)
    """

    def __init__(self, elements):
        assert len(elements) > 0, f"At least on RBP is needed to do insertions."
        self.elements = elements
    
    def generate(
        self,
        r: Source,
        g: Grammar,
        rec,
        new_symbol,
        depth: int,
        base_type,
        context: dict[str, str],
    ):
        # Randomly select an RBP
        rbp_selected = self.elements[r.randint(0, len(self.elements) - 1)]

        # Randomly select a motif from the selected RBP
        rbp_name = rbp_selected[0]
        motif = rbp_selected[1][r.randint(0, len(rbp_selected[1]) - 1)]
        rec((rbp_name, motif))

    def __class_getitem__(self, args):
        return CustomIntListInsertions(*args)

    def __repr__(self):
        return f"[{self.elements}]"