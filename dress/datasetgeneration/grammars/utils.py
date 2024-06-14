import os
import shutil
from typing import List, Union
from geneticengine.core.random.sources import RandomSource
import numpy as np
import pandas as pd

from dress.datasetevaluation.representation.motifs.search import FimoSearch, PlainSearch
from dress.datasetgeneration.dataset import Dataset

MOTIF_SEARCH_OPTIONS = {
    "fimo": FimoSearch,
    "plain": PlainSearch,
}


def _get_forbidden_zones(
    input_seq: dict,
    region_ranges: dict,
    acceptor_untouched_range: List[int] = [-10, 2],
    donor_untouched_range: List[int] = [-3, 6],
    untouched_regions: Union[List[str], None] = None,
    model: str = "spliceai",
) -> List[range]:
    """
    Get intervals in the original sequence not allowed to be mutated.

    It restricts mutations to be outside the vicinity of splice sites.

    It restricts mutations to be outside of specific regions provided by
    the user (e.g, a whole exon).

    It also avoids creating mutations in regions outside of the black
    box model resolution. In the case of `spliceai`, the resolution
    will be 5000 bp left|right from the splice sites of the cassette
    exon.

    Args:
        input_seq (dict): Original sequence
        region_ranges (dict): Mapping of the indices where each region locates
        acceptor_untouched_range (List[int]): Range of positions surrounding
        the acceptor splice site that will not be mutated
        donor_untouched_range (List[int]): Range of positions surrounding
        the donor splice site that will not be mutated
        untouched_regions (Union[List[int], None]): Avoid mutating entire
        regions of the sequence. Defaults to None.
        model (str): Black box model
    Returns:
        List: Non-overlapping sorted intervals that will not be perturbed
    """
    seq = input_seq["seq"]
    ss_idx = input_seq["ss_idx"]
    acceptor_range = list(map(int, acceptor_untouched_range))
    donor_range = list(map(int, donor_untouched_range))

    model_resolutions = {"spliceai": 5000, "pangolin": 5000}
    try:
        resolution = model_resolutions[model]
    except KeyError:
        resolution = model.context

    out = []

    # Forbid to explore Intron_downstrem_2 or Intron_upstream_2
    upst_accept = ss_idx[0][0]
    down_donor = ss_idx[2][1]
    if isinstance(upst_accept, int) and upst_accept > 0:
        out.append(range(0, upst_accept))

    if isinstance(down_donor, int) and down_donor < len(seq) - 1:
        out.append(range(down_donor + 1, len(seq)))

    # Forbid to explore at splice site regions
    for ss in ss_idx:
        # If splice site of upstream and|or downstream exon(s)
        # is out of bounds (<NA>), or if [0, 0] is given, skip it
        if isinstance(ss[0], int) and any(x != 0 for x in acceptor_range):
            _range1 = range(ss[0] + acceptor_range[0], ss[0] + acceptor_range[1] + 1)
            out.append(_range1)

        if isinstance(ss[1], int) and any(x != 0 for x in donor_range):
            _range2 = range(ss[1] + donor_range[0], ss[1] + donor_range[1] + 1)
            out.append(_range2)

    # Forbid to explore outside the model resolution
    cassette_exon = region_ranges["Exon_cassette"]

    if cassette_exon[0] > resolution:
        out.append(range(0, cassette_exon[0] - resolution))

    if len(seq) - cassette_exon[1] - 1 > resolution:
        out.append(range(cassette_exon[1] + resolution, len(seq) - 1))

    if untouched_regions:
        for region in untouched_regions:

            try:
                region = region.capitalize()
                _range = region_ranges[region]

            except KeyError:
                raise ValueError(f"Region {region} not recognized.")

            if all(x == "<NA>" for x in _range):
                continue
            elif _range[0] == "<NA>":
                _range[0] = 0
            elif _range[1] == "<NA>":
                _range[1] = len(seq) - 1

            out.append(range(_range[0], _range[1] + 1))

    if out:
        out = sorted(out, key=lambda x: x.start)
        merged = [out[0]]
        for interval in out[1:]:
            _start, _end = interval.start, interval.stop
            previous = merged[-1]
            if _start <= previous.stop:
                merged[-1] = range(previous.start, max(previous.stop, _end))
            else:
                merged.append(interval)

        return merged

    return out


def _get_location_map(input_seq: dict) -> dict:
    """
    Returns a dictionary mapping the indices of
    each region of the sequence
    """

    ss_idx = input_seq["ss_idx"]
    return {
        "Intron_upstream_2": (
            (0, ss_idx[0][0] - 1) if isinstance(ss_idx[0][0], int) else ("<NA>", "<NA>")
        ),
        "Exon_upstream": (ss_idx[0][0], ss_idx[0][1]),
        "Intron_upstream": (
            ss_idx[0][1] + 1 if isinstance(ss_idx[0][1], int) else 0,
            ss_idx[1][0] - 1,
        ),
        "Exon_cassette": (ss_idx[1][0], ss_idx[1][1]),
        "Intron_downstream": (
            ss_idx[1][1] + 1,
            (
                ss_idx[2][0] - 1
                if isinstance(ss_idx[2][0], int)
                else len(input_seq["seq"]) - 1
            ),
        ),
        "Exon_downstream": (ss_idx[2][0], ss_idx[2][1]),
        "Intron_downstream_2": (
            (ss_idx[2][1] + 1, len(input_seq["seq"]) - 1)
            if isinstance(ss_idx[2][1], int)
            else ("<NA>", "<NA>")
        ),
    }


def _run_motif_scan_on_wt_seq(input_seq: dict, **kwargs):
    """
    Run motif scan on the original sequence

    Args:
        input_seq (dict): Original sequence information
        **kwargs: Additional arguments

    Returns:
        MotifSearch: MotifSearch object
        dict: Updated kwargs
    """
    kwargs["logger"].info(f"Scanning motifs in the original sequence")
    motif_searcher = MOTIF_SEARCH_OPTIONS.get(kwargs.get("motif_search", "fimo"))

    dt = Dataset(_dict_to_df(input_seq))
    motifs_outfn = os.path.join(
        kwargs["outdir"], f"{kwargs['outbasename']}_original_seq_motifs.tsv.gz"
    )
    try:
        motif_search = motif_searcher(dataset=dt, **kwargs)
        motif_results = motif_search.motif_results

    except ValueError as e:
        kwargs["logger"].warning(
            f"Could not find any motifs in the original sequence with the current settings. "
            f"Evolution will be performed with insertions only (--insertion_weight {kwargs['insertion_weight']} "
            f"and --motif_substitution_weight {kwargs['motif_substitution_weight']}. Other diff units will be ignored.)"
        )
        kwargs["snv_weight"] = 0
        kwargs["deletion_weight"] = 0
        kwargs["motif_ablation_weight"] = 0
        motif_search = motif_searcher(dataset=dt, do_scan=False, **kwargs)
        motif_cols = [
            "Seq_id",
            "RBP_name",
            "RBP_motif",
            "Start",
            "End",
            "p-value",
            "q-value",
            "RBP_name_motif",
            "distance_to_cassette_acceptor",
            "distance_to_cassette_donor",
            "is_in_exon",
            "location",
            "distance_to_acceptor",
            "distance_to_donor",
        ]
        motif_results = pd.DataFrame(columns=motif_cols)

    motif_results.to_csv(
        motifs_outfn,
        compression="gzip",
        sep="\t",
        index=False,
    )
    shutil.rmtree(motif_search.outdir)
    return motif_search, kwargs


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


def random_seq(
    seq: str, rs: Union[RandomSource, None] = None, num_shufs: int = 1
) -> Union[str, List[str]]:
    """
    Create a random sequence of the same length as the input sequence.

    Args:
        seq (str): Sequence to extract the target size
        rs (Union[RandomSource, None]): RandomSource object. Defaults to None.
        num_shufs (int): Number of shuffles to create. Defaults to 1.

    Returns:
        Union[str, List[str]]: a single random sequence if num_shufs is 1, or a list of random sequences if num_shufs > 1
    """
    nucs = ["A", "C", "G", "T"]
    if not rs:
        rs = RandomSource(0)

    all_results = []
    for _ in range(num_shufs):
        all_results.append("".join(rs.choice(nucs) for _ in range(len(seq))))
    return all_results if num_shufs > 1 else all_results[0]


def shuffle(
    seq: str, rs: Union[RandomSource, None] = None, num_shufs: int = 1
) -> Union[str, List[str]]:
    """
    Shuffle the given sequence.

    Args:
        seq (str): Sequence to shuffle
        rs (Union[RandomSource, None]): RandomSource object. Defaults to None.
        num_shufs (int): Number of shuffles to create. Defaults to 1.

    Returns:
        Union[str, List[str]]: a single shuffled sequence if num_shufs is 1, or a list of shuffled sequences if num_shufs > 1
    """
    if not rs:
        rs = RandomSource(0)
    all_results = []
    for _ in range(num_shufs):
        all_results.append("".join(rs.shuffle(list(seq))))
    return all_results if num_shufs > 1 else all_results[0]


def dinuc_shuffle(
    seq: str, rs: Union[RandomSource, None] = None, num_shufs: int = 1
) -> Union[str, List[str]]:
    """
    Adapted from https://github.com/kundajelab/deeplift/blob/master/deeplift/dinuc_shuffle.py

    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.

    Args:
        seq (str): a string of length L
        rs (Union[RandomSource, None]): a geneticEngine RandomSource object, to use for performing shuffles
        num_shufs (int): the number of shuffles to create. Defaults to 1.

    Returns:
        Union[str, List[str]]: a single shuffled sequence if num_shufs is 1, or a list of shuffled sequences if num_shufs > 1
    """
    assert num_shufs > 0
    arr = string_to_char_array(seq)

    if not rs:
        rs = RandomSource(0)

    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    all_results = []

    for _ in range(num_shufs):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rs.shuffle(list(range(len(inds) - 1)))  # Keep last index same

            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        all_results.append(char_array_to_string(chars[result]))

    return all_results if num_shufs > 1 else all_results[0]


def string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    """
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)


def char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    return arr.tostring().decode("ascii")
