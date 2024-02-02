from __future__ import annotations

from geneticengine.core.grammar import Grammar
from geneticengine.core.random.sources import Source
from geneticengine.metahandlers.base import MetaHandlerGenerator

class RandomNucleotides(MetaHandlerGenerator):
    """
    RandomNucleotides(s) generates a random string of s nucleotides
    sampled from the expected GC content frequency in the human genome
    """

    def __init__(self, max_size: int):

        assert max_size > 0, "max size must be > 0"
        self.max_size = max_size
        self.nucleotides = ["A", "C", "G", "T"]
        self.weights = [0.3, 0.2, 0.2, 0.3]
        
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

        size = r.randint(1, self.max_size)
        sequence = "".join(r.choice_weighted(self.nucleotides, self.weights) for _ in range(size))
        rec(sequence)