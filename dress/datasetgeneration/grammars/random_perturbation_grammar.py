from abc import ABC
import abc
from dataclasses import dataclass
import itertools
from typing import Annotated, Tuple, Union

import pandas as pd
from dress.datasetgeneration.grammars.utils import _get_forbidden_zones, _get_location_map
from dress.datasetgeneration.metahandlers.ints import IntRangeExcludingSomeValues
from dress.datasetgeneration.metahandlers.strings import RandomNucleotides
from geneticengine.core.random.sources import RandomSource
from geneticengine.metahandlers.ints import IntRange
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.metahandlers.vars import VarRange
from geneticengine.core.decorators import weight
from typing import List
from geneticengine.core.grammar import Grammar, extract_grammar
import pyranges as pr

NUCLEOTIDES = ["A", "C", "G", "T"]


class DiffUnit(ABC):
    pass

    @abc.abstractmethod
    def perturb(self, seq: str, position_tracker: int) -> str:
        ...

    @abc.abstractmethod
    def adjust_index(self, ss_idx: List[list]) -> List[list]:
        ...

    @abc.abstractmethod
    def get_size(self) -> int:
        ...

    def get_location(self, loc_map) -> str:
        for loc, _range in loc_map.items():
            if isinstance(_range[1], int) and self.position <= _range[1]:
                return loc

    def get_distance_to_cassette(self, loc_map) -> int:
        cass = loc_map["Exon_cassette"]
        return min(
            abs(self.position - cass[0]),
            abs(self.position - cass[1]),
        )

def create_random_grammar(
    input_seq: dict,
    excluded_regions: list = None,
    **kwargs
) -> Tuple[Grammar, List[range]]:
    """Creates a grammar for the original sequence

    Args:
        input_seq (dict): Info about the input sequence
        excluded_regions (list, optional): List of ranges that cannot be perturbed. Defaults to None.
    Returns:
        Grammar: Returns the grammar that specifies the production rules of the language
        List[range]: List of ranges that cannot be perturbed
    """
    seqsize = len(input_seq["seq"])
    max_diff_units = kwargs.get("max_diff_units", 6)
    max_insertion_size = kwargs.get("max_insertion_size", 5)
    max_deletion_size = kwargs.get("max_deletion_size", 5)
    snv_weight = kwargs.get("snv_weight", 0.33)
    insertion_weight = kwargs.get("insertion_weight", 0.33)
    deletion_weight = kwargs.get("deletion_weight", 0.33)

    LOCATION_MAP = _get_location_map(input_seq)
    if excluded_regions is None:
        EXCLUDED_REGIONS = _get_forbidden_zones(
            input_seq,
            region_ranges=LOCATION_MAP,
            acceptor_untouched_range=kwargs.get("acceptor_untouched_range", [-10, 2]),
            donor_untouched_range=kwargs.get("donor_untouched_range", [-3, 6]),
            untouched_regions=kwargs.get("untouched_regions", None),
            model=kwargs.get("model", "spliceai"),
        )
    else:
        EXCLUDED_REGIONS = excluded_regions

    @dataclass
    class DiffSequence(object):
        diffs: Annotated[list[DiffUnit], ListSizeBetween(1, max_diff_units)]

        def exclude_forbidden_regions(
            self, forbidden_regions: list
        ) -> Union["DiffSequence", None]:
            """
            Excludes diff units overlapping with a set
            of forbidden regions

            Args:
                forbidden_regions (list): List with ranges within the
                sequence that cannot be mutated

            Returns:
                Union[DiffSequence, None]: Returns DiffSequence object itself or None,
                if the individual is empty after excluding forbidden regions
            """
            flat_forbidden = list(itertools.chain(*[x for x in forbidden_regions]))

            self.diffs.sort(key=lambda x: x.position)  # type: ignore
            to_exclude = []
            for i, d in enumerate(self.diffs):
                if isinstance(d, Deletion):
                    _r1 = range(d.position, d.position + d.get_size())

                    for _r2 in forbidden_regions:
                        if _r1.start < _r2.stop and _r1.stop > _r2.start:
                            to_exclude.append(i)
                            break
                else:
                    if d.position in flat_forbidden:  # type: ignore
                        to_exclude.append(i)

            self.diffs[:] = [d for i, d in enumerate(self.diffs) if i not in to_exclude]

            if len(self.diffs) == 0:
                return None

            return self

        def remove_diffunit_overlaps(self, seq: str, rs: RandomSource) -> Union["DiffSequence", None]:
            """
            Clean individual phenotypes by removing redundant
            SNVs (e.g, substitution of A by A) and excluding
            overlapping diff units.

            Prioritizes the longest diff unit, meaning that
            deletions tend to be selected when overlaps
            are found, since they will be bigger than SNVs and
            Insertions

            Args:
                seq (str): Original sequence
                rs (RandomSource): RandomSource object
            """

            self.diffs.sort(key=lambda x: x.position)  # type: ignore
            _diffs = self.diffs.copy()

            ranges = []
            for d in _diffs:
                if isinstance(d, SNV) and d.is_redundant(seq):
                    d.sample_new_nucleotide(rs)

                if isinstance(d, Deletion):
                    current_r = [
                        1,
                        d.position,
                        d.position + d.get_size(),
                        d.get_size(),
                        str(d),
                    ]

                elif isinstance(d, (Insertion, SNV)):
                    current_r = [1, d.position, d.position + 1, d.get_size(), str(d)]

                else:
                    continue

                ranges.append(current_r)

            if len(self.diffs) == 0:
                raise ValueError("Individual should never be empty here")

            elif len(self.diffs) > 1:

                cols = ["Chromosome", "Start", "End", "length", "phen"]
                _df = pd.DataFrame(ranges, columns=cols)
                _df["id"] = _df.index
                gr = pr.PyRanges(_df)

                to_keep = (
                    gr.cluster(slack=-1)
                    .as_df()
                    .groupby("Cluster")
                    .apply(lambda x: x.loc[x.length.idxmax()])
                    .id.to_list()
                )
                self.diffs[:] = [self.diffs[i] for i in to_keep]

            return self


        def apply_diff(
            self, seq: str, ss_indexes: List[list]
        ) -> Tuple[str, List[list]]:
            """
            Applies the DiffUnits to the original sequence
            so that new sequences are generated.

            Because diffUnits may contain insertions and deletions,
            splice site positions are adjusted accordingly

            Args:
                seq (str): Original sequence
                ss_indexes (List[list]): List of lists with the positions
                where splice sites are located

            Returns:
                Tuple[str, List[list]]: New sequence with its adjusted
                splice site positions
            """
            tracker = 0

            for d in self.diffs:
                seq = d.perturb(seq, position_tracker=tracker)

                if isinstance(d, Insertion):
                    tracker += d.get_size()
                elif isinstance(d, Deletion):
                    tracker -= d.get_size()

                ss_indexes = d.adjust_index(ss_indexes)

            assert len(self.diffs) > 0
            return seq, ss_indexes

        def __str__(self):
            return "|".join([str(d) for d in self.diffs])

    @dataclass
    class SNV(DiffUnit):
        position: Annotated[int, IntRangeExcludingSomeValues(0, seqsize - 1, exclude=EXCLUDED_REGIONS)]
        nucleotide: Annotated[str, VarRange(NUCLEOTIDES)]

        def perturb(self, seq: str, position_tracker: int) -> str:
            assert self.position + position_tracker < len(seq)
            seq2 = list(seq)
            seq2[self.position + position_tracker] = self.nucleotide
            return "".join(seq2)

        def adjust_index(self, ss_indexes: List[list]) -> List[list]:
            return ss_indexes

        def is_redundant(self, seq: str) -> bool:
            return True if seq[self.position] == self.nucleotide else False
        
        def sample_new_nucleotide(self, rs: RandomSource) -> str:
            self.nucleotide = rs.choice([nuc for nuc in NUCLEOTIDES if nuc != self.nucleotide])

        def get_size(self) -> int:
            return 1

        def __str__(self):
            return f"SNV[{self.position},{self.nucleotide},{self.get_location(LOCATION_MAP)},{self.get_distance_to_cassette(LOCATION_MAP)}]"

    @dataclass
    class Deletion(DiffUnit):
        position: Annotated[int, IntRangeExcludingSomeValues(0, seqsize - max_deletion_size, exclude=EXCLUDED_REGIONS)]
        size: Annotated[int, IntRange(min=1, max=max_deletion_size)]

        def perturb(self, seq: str, position_tracker: int) -> str:
            assert self.position + self.size + position_tracker <= len(seq)
            return (
                seq[: self.position + position_tracker]
                + seq[self.position + self.size + position_tracker :]
            )

        def adjust_index(self, ss_indexes: List[list]) -> List[list]:
            _ss_idx = list(itertools.chain(*ss_indexes))
            adj = [
                ss - self.size if isinstance(ss, int) and self.position < ss else ss
                for ss in _ss_idx
            ]
            return [adj[0:2], adj[2:4], adj[4:6]]

        def get_size(self) -> int:
            return self.size

        def __str__(self):
            return f"Deletion[{self.position},{self.position + self.size - 1},{self.get_location(LOCATION_MAP)},{self.get_distance_to_cassette(LOCATION_MAP)}]"

    @dataclass
    class Insertion(DiffUnit):
        position: Annotated[int, IntRangeExcludingSomeValues(0, seqsize, exclude=EXCLUDED_REGIONS)]
        nucleotides: Annotated[str, RandomNucleotides(max_size=max_insertion_size)]

        def perturb(self, seq: str, position_tracker: int) -> str:
            assert self.position + position_tracker <= len(seq)
            return (
                seq[: self.position + position_tracker]
                + self.nucleotides
                + seq[self.position + position_tracker :]
            )

        def adjust_index(self, ss_indexes: List[list]) -> List[list]:
            _ss_idx = list(itertools.chain(*ss_indexes))
            adj = [
                ss + self.get_size()
                if isinstance(ss, int) and self.position < ss
                else ss
                for ss in _ss_idx
            ]
            return [adj[0:2], adj[2:4], adj[4:6]]

        def get_size(self) -> int:
            return len(self.nucleotides)

        def __str__(self):
            return f"Insertion[{self.position},{self.nucleotides},{self.get_location(LOCATION_MAP)},{self.get_distance_to_cassette(LOCATION_MAP)}]"

    g = extract_grammar(
        [
            weight(snv_weight)(SNV),
            weight(deletion_weight)(Deletion),
            weight(insertion_weight)(Insertion),
        ],
        DiffSequence,
    )
    g._type = "random"
    return g, EXCLUDED_REGIONS
