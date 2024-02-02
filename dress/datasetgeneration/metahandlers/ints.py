from __future__ import annotations

from typing import Iterable, Union, TypeVar

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.metahandlers.base import MetaHandlerGenerator
from itertools import chain
min = TypeVar("min", covariant=True)
max = TypeVar("max", covariant=True)

class IntRangeExcludingSomeValues(MetaHandlerGenerator):
    """
    IntRangeExcludingSomeValues(a,b,c) restricts ints to be between a and b, 
    but making sure that c (which can be an iterable) is not selected

    IntRangeExcludingSomeValues(0,20,[range(2,5),range(10,15), 18]) will generate
    an integer between 0 and 20, but excluding the values [2,3,4,10,11,12,13,14,18]
    """

    def __init__(self, min: int, max: int, exclude: Union[int, Iterable[int], Iterable[range]]):
        assert min <= max, "min must be <= max"
        
        msg = "All exclude indexes must be between min and max"
        if isinstance(exclude, int):
            assert min <=exclude <= max, msg
            self.exclude = [exclude]
        else:
            # for e in exclude:
            #     if isinstance(e, int):
            #         assert min <= e <= max, msg
            #     else:   
            #         assert all(min <= _e <= max for _e in e), msg             
            self.exclude = list(chain(*[x if isinstance(x, Iterable) else [x] for x in exclude]))
            
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

        val = r.randint(self.min, self.max)
        while val in self.exclude:
            val = r.randint(self.min, self.max)
       
        rec(val)

    def __class_getitem__(self, args):
        return IntRangeExcludingSomeValues(*args)

    def __repr__(self):
        return f"[{self.min},..,{self.max}] excluding {self.exclude}"