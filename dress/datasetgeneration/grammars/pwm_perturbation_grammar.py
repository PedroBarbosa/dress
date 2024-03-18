from abc import ABC
import abc
from dataclasses import dataclass
import itertools
from typing import Annotated, Tuple, Union

import pandas as pd
from dress.datasetevaluation.representation.motifs.search import FimoSearch, PlainSearch
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
MOTIF_SEARCH_OPTIONS = {
    "fimo": FimoSearch,
    "plain": PlainSearch,
}


class DiffUnit(ABC):
    pass

    @abc.abstractmethod
    def perturb(self, seq: str, position_tracker: int) -> str: ...

    @abc.abstractmethod
    def adjust_index(self, ss_idx: List[list]) -> List[list]: ...

    @abc.abstractmethod
    def sample_new_position(self, r: RandomSource) -> int: ...

    @abc.abstractmethod
    def get_size(self) -> int: ...


def create_motif_grammar(
    input_seq: dict,
    **kwargs,
) -> Grammar:
    """Creates a grammar for the original sequence of length seqsize

    Args:
        input_seq (dict): Info about the input sequence

    Returns:
        Grammar: Returns the grammar that specifies the production rules of the language
    """
    print(input_seq)

    original_seq = pd.DataFrame(
        {
            "Seq_id": [input_seq["seq_id"]],
            "Sequence": [input_seq["seq"]],
            "Splice_site_positions": [
                ";".join([str(x) for sublist in input_seq["ss_idx"] for x in sublist])
            ],
            "Score": [input_seq["score"]],
            "Delta_score": [0]
        }
    )
    motif_searcher = MOTIF_SEARCH_OPTIONS.get(kwargs.get("motif_search"))
    motif_search = motif_searcher(dataset=original_seq, **kwargs)
    motif_search.tabulate_occurrences(write_output=False)
    motif_hits = motif_search.motif_results

    seqsize = len(input_seq["seq"])

    @dataclass
    class DiffSequence(object):
        diffs: Annotated[list[DiffUnit], ListSizeBetween(1, max_diff_units)]

        def exclude_forbidden_regions(
            self, forbidden_regions: list, r: RandomSource
        ) -> Union["DiffSequence", None]:
            """
            Excludes diff units overlapping with a set
            of forbidden regions

            Args:
                forbidden_regions (list): List with ranges within the
                sequence that cannot be mutated
                r (RandomSource): Random source generator

            Returns:
                Union[DiffSequence, None]: Returns DiffSequence object itself or None,
                if the individual is empty after excluding forbidden regions
            """
            flat_forbidden = list(itertools.chain(*[x for x in forbidden_regions]))

            self.diffs.sort(key=lambda x: x.position)  # type: ignore
            to_exclude = []
            for i, d in enumerate(self.diffs):
                if isinstance(d, RandomDeletion):
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

        def clean(self, seq: str, r: RandomSource) -> Union["DiffSequence", None]:
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
                r (RandomSource): Random source generator
            """

            self.diffs.sort(key=lambda x: x.position)  # type: ignore
            _diffs = self.diffs.copy()

            ranges = []
            for d in _diffs:
                if isinstance(d, SNV) and d.is_redundant(seq):
                    self.diffs.remove(d)
                    continue

                if isinstance(d, RandomDeletion):
                    current_r = [
                        1,
                        d.position,
                        d.position + d.get_size(),
                        d.get_size(),
                        str(d),
                    ]

                elif isinstance(d, (RandomInsertion, SNV)):
                    current_r = [1, d.position, d.position + 1, d.get_size(), str(d)]

                else:
                    continue

                ranges.append(current_r)

            if len(self.diffs) == 0:
                return None

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

        def draw_valid_diff_unit(
            self, diffs: List[DiffUnit], to_exclude: List[range], r: RandomSource
        ) -> None:
            """
            Draws a new diff unit that does not overlap with a list of ranges

            Args:
                diffs (List[DiffUnit]): List of diff units
                to_exclude (List[range]): List of ranges with the positions to exclude
                r (RandomSource): Random source generator
            """
            _diff_unit = diffs[r.randint(0, len(diffs) - 1)]
            new_pos = _diff_unit.sample_new_position(r)

            def _condition():
                return (
                    any(
                        p in to_exclude
                        for p in [new_pos, new_pos + _diff_unit.get_size()]  # type: ignore
                        if isinstance(_diff_unit, RandomDeletion)
                    )
                    or new_pos in to_exclude
                )

            while _condition():
                new_pos = _diff_unit.sample_new_position(r)

            _diff_unit.position = new_pos  # type: ignore
            self.diffs = [_diff_unit]

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

                if isinstance(d, RandomInsertion):
                    tracker += d.get_size()
                elif isinstance(d, RandomDeletion):
                    tracker -= d.get_size()

                ss_indexes = d.adjust_index(ss_indexes)

            assert len(self.diffs) > 0
            return seq, ss_indexes

        def __str__(self):
            return "|".join([str(d) for d in self.diffs])

    @dataclass
    class SNV(DiffUnit):
        position: Annotated[int, IntRange(0, seqsize - 1)]
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

        def sample_new_nucleotide(self, r: RandomSource) -> str:
            nuc = [n for n in NUCLEOTIDES if n != self.nucleotide]
            return r.choice(nuc)

        def sample_new_position(self, r: RandomSource) -> int:
            return r.randint(0, seqsize - 1)

        def get_size(self) -> int:
            return 1

        def __str__(self):
            return f"SNV[{self.position},{self.nucleotide}]"

    @dataclass
    class RandomDeletion(DiffUnit):
        position: Annotated[int, IntRange(0, seqsize - max_deletion_size)]
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

        def sample_new_position(self, r: RandomSource) -> int:
            return r.randint(0, seqsize - max_deletion_size)

        def get_size(self) -> int:
            return self.size

        def __str__(self):
            return f"RandomDeletion[{self.position},{self.position + self.size - 1}]"

    @dataclass
    class RandomInsertion(DiffUnit):
        position: Annotated[int, IntRange(0, seqsize)]
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
                (
                    ss + self.get_size()
                    if isinstance(ss, int) and self.position < ss
                    else ss
                )
                for ss in _ss_idx
            ]
            return [adj[0:2], adj[2:4], adj[4:6]]

        def sample_new_position(self, r: RandomSource) -> int:
            return r.randint(0, seqsize)

        def get_size(self) -> int:
            return len(self.nucleotides)

        def __str__(self):
            return f"RandomInsertion[{self.position},{self.nucleotides}]"

    return extract_grammar(
        [
            weight(snv_weight)(SNV),
            weight(deletion_weight)(RandomDeletion),
            weight(insertion_weight)(RandomInsertion),
        ],
        DiffSequence,
    )
