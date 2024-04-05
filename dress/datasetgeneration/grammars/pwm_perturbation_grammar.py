from abc import ABC
import abc
from dataclasses import dataclass
import itertools
from typing import Annotated, Tuple, Union

import pandas as pd
import pyranges as pr
from dress.datasetevaluation.representation.motifs.search import FimoSearch, PlainSearch
from dress.datasetgeneration.dataset import Dataset
from dress.datasetgeneration.metahandlers.ints import (
    CustomIntListDeletions,
    CustomIntListInsertions,
    IntListExcludingSomeValues,
    IntRangeExcludingSomeValues,
)
from dress.datasetgeneration.grammars.utils import (
    _get_forbidden_zones,
    _get_location_map,
)
from geneticengine.core.random.sources import RandomSource
from geneticengine.metahandlers.lists import ListSizeBetween
from geneticengine.metahandlers.vars import VarRange
from geneticengine.core.decorators import weight
from typing import List
from geneticengine.core.grammar import Grammar, extract_grammar

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
    def get_size(self) -> int: ...


def create_motif_grammar(
    input_seq: dict,
    excluded_regions: list = None,
    **kwargs,
) -> Tuple[Grammar, List[range]]:
    """Creates a grammar for the original sequence

    Args:
        input_seq (dict): Info about the input sequence
        excluded_regions (list, optional): List of ranges that cannot be perturbed. Defaults to None.
    Returns:
        Grammar: Returns the grammar that specifies the production rules of the language
        List[range]: List of ranges that cannot be perturbed
    """

    max_diff_units = kwargs.get("max_diff_units", 6)
    seqsize = len(input_seq["seq"])
    kwargs["logger"].info(f"Scanning motifs in the original sequence")
    motif_searcher = MOTIF_SEARCH_OPTIONS.get(kwargs.get("motif_search", "fimo"))
    motif_search = motif_searcher(dataset=Dataset(_dict_to_df(input_seq)), **kwargs)

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

    MOTIF_INFO_SNV, MOTIF_INFO_DELS, MOTIF_INFO_INS = _structure_motif_info(
        motif_search, seqsize, EXCLUDED_REGIONS, **kwargs
    )

    @dataclass
    class DiffSequence(object):
        diffs: Annotated[list[DiffUnit], ListSizeBetween(1, max_diff_units)]

        def exclude_forbidden_regions(
            self, forbidden_regions: list
        ) -> Union["DiffSequence", None]:
            """
            Excludes diff units overlapping with a set
            of forbidden regions. The start position
            should be taken into account by metahandlers in the grammar
            itself, but the end position of some grammar nodes
            (Deletion, Ablation, Subststitution) may span forbidden region

            Args:
                forbidden_regions (list): List with ranges within the
                sequence that cannot be mutated

            Returns:
                Union[DiffSequence, None]: Returns DiffSequence object itself or None,
                if the individual is empty after excluding forbidden regions
            """
            flat_forbidden = list(itertools.chain(*[x for x in forbidden_regions]))
            self.diffs.sort(
                key=lambda x: (
                    x.position[0] if isinstance(x.position, tuple) else x.position
                )
            )
            to_exclude = []
            for i, d in enumerate(self.diffs):
                if isinstance(d, (MotifDeletion, MotifAblation, MotifSubstitution)):
                    if isinstance(d, MotifSubstitution):
                        _r1 = range(d.position, d.position + d.get_size())
                    else:
                        _r1 = range(d.position[0], d.position[0] + d.get_size())

                    if any(
                        _r1.start < _r2.stop and _r1.stop > _r2.start
                        for _r2 in forbidden_regions
                    ):
                        to_exclude.append(i)
                else:
                    if d.position in flat_forbidden:  # type: ignore
                        to_exclude.append(i)

            self.diffs[:] = [d for i, d in enumerate(self.diffs) if i not in to_exclude]

            if len(self.diffs) == 0:
                return None

            return self

        def remove_diffunit_overlaps(self, seq: str, rs: RandomSource) -> Union["DiffSequence", None]:
            """
            Clean individual phenotypes by excluding
            overlapping diff units.

            Prioritizes the longest diff unit, meaning that
            deletions or insertions tend to be selected over SNVs

            Args:
                seq (str): Original sequence
                rs (RandomSource): RandomSource object
            """
            self.diffs.sort(
                key=lambda x: (
                    x.position[0] if isinstance(x.position, tuple) else x.position
                )
            )
            _diffs = self.diffs.copy()
            ranges = []
            for d in _diffs:
                if isinstance(d, MotifSNV) and d.is_redundant(seq):
                    d.sample_new_nucleotide(rs)
                    
                if isinstance(d, MotifDeletion):
                    current_r = [
                        1,
                        d.position[0],
                        d.position[1],
                        d.get_size(),
                        str(d),
                    ]

                elif isinstance(
                    d, (MotifInsertion, MotifSubstitution, MotifAblation, MotifSNV)
                ):
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

                if isinstance(d, MotifDeletion):
                    tracker -= d.get_size()
                elif isinstance(d, MotifInsertion):
                    tracker += d.get_size()

                ss_indexes = d.adjust_index(ss_indexes)

            assert len(self.diffs) > 0
            return seq, ss_indexes

        def __str__(self):
            return "|".join([str(d) for d in self.diffs])

    @dataclass
    class MotifSNV(DiffUnit):
        position: Annotated[
            int,
            IntListExcludingSomeValues(
                MOTIF_INFO_SNV.position.unique().tolist(), exclude=EXCLUDED_REGIONS
            ),
        ]
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
            _info = MOTIF_INFO_SNV[MOTIF_INFO_SNV.position == self.position].iloc[0]
            return f"MotifSNV[{self.position},{_info.rbp_name},{_info.ref_nuc}>{self.nucleotide},{_info.location},{_info.distance_to_cassette},{_info.position_in_motif}]"

    @dataclass
    class MotifDeletion(DiffUnit):
        position: Annotated[tuple, CustomIntListDeletions(MOTIF_INFO_DELS)]

        def perturb(self, seq: str, position_tracker: int) -> str:
            assert self.position[0] < self.position[1]
            size = self.position[1] - self.position[0]
            assert self.position[0] + size + position_tracker <= len(seq)

            return (
                seq[: self.position[0] + position_tracker]
                + seq[self.position[0] + size + position_tracker :]
            )

        def adjust_index(self, ss_indexes: List[list]) -> List[list]:
            _ss_idx = list(itertools.chain(*ss_indexes))

            adj = [
                (
                    ss - self.get_size()
                    if isinstance(ss, int) and self.position[0] < ss
                    else ss
                )
                for ss in _ss_idx
            ]
            return [adj[0:2], adj[2:4], adj[4:6]]

        def get_size(self) -> int:
            return self.position[1] - self.position[0]

        def __str__(self):
            _info = MOTIF_INFO_DELS[self.position[2]]
            rbps = ">5RBPs" if _info[5][0].count(";") + 1 > 5 else _info[5][0]
            return f"MotifDeletion[{self.position[0]},{rbps},{self.get_size()},{_info[2]},{min(_info[3], _info[4])}]"

    @dataclass
    class MotifInsertion(DiffUnit):
        position: Annotated[
            int, IntRangeExcludingSomeValues(0, seqsize - 1, exclude=EXCLUDED_REGIONS)
        ]
        nucleotides: Annotated[tuple, CustomIntListInsertions(MOTIF_INFO_INS)]

        def perturb(self, seq: str, position_tracker: int) -> str:
            assert self.position + position_tracker <= len(seq)
            return (
                seq[: self.position + position_tracker]
                + self.nucleotides[1]
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

        def get_size(self) -> int:
            return len(self.nucleotides[1])

        def get_location(self) -> str:
            for loc, _range in LOCATION_MAP.items():
                if isinstance(_range[1], int) and self.position <= _range[1]:
                    return loc

        def get_distance_to_cassette(self) -> int:
            cass = LOCATION_MAP["Exon_cassette"]
            return min(
                abs(self.position - cass[0]),
                abs(self.position - cass[1]),
            )

        def __str__(self):
            return f"MotifInsertion[{self.position},{self.nucleotides[0]},{self.get_size()},{self.get_location()},{self.get_distance_to_cassette()}]"

    @dataclass
    class MotifAblation(DiffUnit):
        position: Annotated[
            tuple, CustomIntListDeletions(MOTIF_INFO_DELS, is_ablation=True)
        ]

        def perturb(self, seq: str, position_tracker: int) -> str:
            assert self.position[0] < self.position[1]
            size = self.position[1] - self.position[0]
            assert self.position[0] + size + position_tracker <= len(seq)

            return (
                seq[: self.position[0] + position_tracker]
                + self.position[3]
                + seq[self.position[0] + size + position_tracker :]
            )

        def adjust_index(self, ss_indexes: List[list]) -> List[list]:
            _ss_idx = list(itertools.chain(*ss_indexes))

            adj = [
                (
                    ss - self.get_size()
                    if isinstance(ss, int) and self.position[0] < ss
                    else ss
                )
                for ss in _ss_idx
            ]
            return [adj[0:2], adj[2:4], adj[4:6]]

        def get_size(self) -> int:
            return self.position[1] - self.position[0]

        def __str__(self):
            _info = MOTIF_INFO_DELS[self.position[2]]
            rbps = ">5RBPs" if _info[5][0].count(";") + 1 > 5 else _info[5][0]
            return f"MotifAblation[{self.position[0]},{rbps},{self.get_size()},{_info[2]},{min(_info[3], _info[4])}]"

    @dataclass
    class MotifSubstitution(DiffUnit):
        position: Annotated[
            int, IntRangeExcludingSomeValues(0, seqsize - 1, exclude=EXCLUDED_REGIONS)
        ]
        nucleotides: Annotated[tuple, CustomIntListInsertions(MOTIF_INFO_INS)]

        def perturb(self, seq: str, position_tracker: int) -> str:
            assert self.position + position_tracker <= len(seq)

            return (
                seq[: self.position + position_tracker]
                + self.nucleotides[1]
                + seq[self.position + position_tracker + len(self.nucleotides[1]) :]
            )

        def adjust_index(self, ss_indexes: List[list]) -> List[list]:
            return ss_indexes

        def get_size(self) -> int:
            return len(self.nucleotides[1])

        def get_location(self) -> str:
            for loc, _range in LOCATION_MAP.items():
                if isinstance(_range[1], int) and self.position <= _range[1]:
                    return loc

        def get_distance_to_cassette(self) -> int:
            cass = LOCATION_MAP["Exon_cassette"]
            return min(
                abs(self.position - cass[0]),
                abs(self.position - cass[1]),
            )

        def __str__(self):
            return f"MotifSubstitution[{self.position},{self.nucleotides[0]},{self.get_size()},{self.get_location()},{self.get_distance_to_cassette()}]"

    g = extract_grammar(
        [
            weight(kwargs["snv_weight"])(MotifSNV),
            weight(kwargs["deletion_weight"])(MotifDeletion),
            weight(kwargs["insertion_weight"])(MotifInsertion),
            weight(kwargs["motif_ablation_weight"])(MotifAblation),
            weight(kwargs["motif_substitution_weight"])(MotifSubstitution),
        ],
        DiffSequence,
    )
    g._type = "motif_based"
    return g, EXCLUDED_REGIONS


def _dict_to_df(d: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Run_id": [
                d["seq_id"]
                .replace(":", "_")
                .replace("(+)", "")
                .replace("(-)", "")
                .replace("-", "_")
            ],
            "Seed": [0],
            "Seq_id": [d["seq_id"]],
            "Phenotype": ["wt"],
            "Sequence": [d["seq"]],
            "Splice_site_positions": [
                ";".join([str(x) for sublist in d["ss_idx"] for x in sublist])
            ],
            "Score": [d["score"]],
            "Delta_score": [0],
        }
    )


def _structure_motif_info(
    motif_search_obj: Union[FimoSearch, PlainSearch],
    seqsize: int,
    excluded_regions: list,
    **kwargs,
) -> pd.DataFrame:
    _excluded_r = set(itertools.chain(*[x for x in excluded_regions]))
    motif2ins = [[rbp, motifs] for rbp, motifs in motif_search_obj.motifs.items()]
    motif_hits = motif_search_obj.motif_results
    motif_intervals = (
        motif_hits.groupby(
            [
                "Start",
                "End",
                "location",
                "distance_to_cassette_acceptor",
                "distance_to_cassette_donor",
            ]
        )
        .apply(
            lambda x: [
                ";".join(x.RBP_name.tolist()),
                list(set(x.RBP_motif.tolist()))[0],
            ]
        )
        .reset_index()
        .rename(columns={0: "rbp_info"})
    )

    per_position_info = pd.concat(motif_intervals.apply(_expand_row, axis=1).tolist())
    per_position_info_agg = per_position_info.groupby("position").apply(
        _aggregate_multi_position_info
    )
    kwargs["logger"].info(
        f"{per_position_info_agg.shape[0]}/{seqsize} ({per_position_info_agg.shape[0] / seqsize * 100:.2f}%) positions with motifs"
    )
    motif2snv = per_position_info_agg[~per_position_info_agg.position.isin(_excluded_r)]

    kwargs["logger"].info(
        f"{per_position_info_agg.shape[0] - motif2snv.shape[0]} positions with motifs will not be used due to overlapping with excluded regions."
    )
    motif2dels = motif_intervals.values.tolist()
    motif2dels = [
        el
        for el in motif2dels
        if not any(x in _excluded_r for x in range(el[0], el[1]))
    ]
    return motif2snv, motif2dels, motif2ins


def _expand_row(row):

    positions = range(row.Start, row.End)
    return pd.DataFrame(
        {
            "position": positions,
            "location": [row.location] * len(positions),
            "distance_to_cassette": min(
                row.distance_to_cassette_acceptor, row.distance_to_cassette_donor
            ),
            "ref_nuc": list(row.rbp_info[1]),
            "position_in_motif": range(1, len(positions) + 1),
            "rbp_name": [row.rbp_info[0]] * len(positions),
            "rbp_motif": [row.rbp_info[1]] * len(positions),
        }
    )


def _aggregate_multi_position_info(group: pd.DataFrame) -> pd.Series:
    """Aggregate multiple RBP information at a single position

    Args:
        group (pd.DataFrame): Motif hits at a single position

    Returns:
        pd.Series: Aggregated information for the position
    """

    # If just one motif hit at this position
    if group.shape[0] == 1:
        group = group.iloc[0]
        n_rbps = len(group.rbp_name.split(";"))
        # Rename the RBP name if there are more than 5 RBPs
        if n_rbps > 5:
            group.rbp_name = ">5RBPs"
        return group
    else:
        _pos = group.position.iloc[0]
        _loc = group.location.iloc[0]
        _distance = group.distance_to_cassette.iloc[0]
        _ref_nuc = group.ref_nuc.iloc[0]
        _motif = group.rbp_motif.iloc[0]

        # If there are multiple motif hits at this position, dont report position and aggregate RBP names
        rbp_list = group.rbp_name.str.split(";")
        unique_rbp_names = list(set(name for sublist in rbp_list for name in sublist))
        if len(unique_rbp_names) > 5:
            rbp_name = ">5RBPs"
        else:
            rbp_name = ";".join(unique_rbp_names)
        return pd.Series(
            {
                "position": _pos,
                "location": _loc,
                "distance_to_cassette": _distance,
                "ref_nuc": _ref_nuc,
                "position_in_motif": "-",
                "rbp_name": rbp_name,
                "rbp_motif": _motif,
            }
        )
