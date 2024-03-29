from typing import List, Union

REGION_MAP = {'exon_upstream': 'Exon_upstream',
              'intron_upstream': 'Intron_upstream',
              'target_exon': 'Exon_cassette',
              'intron_downstream': 'Intron_downstream',
              'exon_downstream': 'Exon_downstream'}

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
        location_map (dict): Mapping of the indices where each region locates
        acceptor_untouched_range (List[int]): Range of positions surrounding
        the acceptor splice site that will not be mutated
        donor_untouched_range (List[int]): Range of positions surrounding
        the donor splice site that will not be mutated
        untouched_regions (Union[List[int], None]): Avoid mutating entire
        regions of the sequence. Defaults to None.
        model (str): Black box model
    Returns:
        List: Restricted intervals that will not be mutated
    """
    seq = input_seq["seq"]
    ss_idx = input_seq["ss_idx"]
    acceptor_range = list(map(int, acceptor_untouched_range))
    donor_range = list(map(int, donor_untouched_range))

    model_resolutions = {"spliceai": 5000, "pangolin": 5000}

    resolution = model_resolutions[model]
    out = []

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
    cassette_exon = region_ranges['Exon_cassette']

    if cassette_exon[0] > resolution:
        out.append(range(0, cassette_exon[0] - resolution))

    if len(seq) - cassette_exon[1] - 1 > resolution:
        out.append(range(cassette_exon[1] + resolution, len(seq) - 1))
        
    if untouched_regions:
        for region in untouched_regions:
            try:
                region = REGION_MAP[region]
            except KeyError:
                raise ValueError(
                    f"Region {region} not recognized. "
                    f"Choose from {list(region_ranges.keys())}"
                )
            _range = region_ranges[region]
            if all(x == "<NA>" for x in _range):
                continue
            elif _range[0] == '<NA>':
                _range[0] = 0
            elif _range[1] == '<NA>':
                _range[1] = len(seq) - 1

            out.append(range(_range[0], _range[1] + 1))

    return out
    #return sorted(list(set(num for _range in out for num in _range)))

def _get_location_map(input_seq: dict) -> dict:
    """
    Returns a dictionary mapping the indices of 
    each region of the sequence
    """

    ss_idx = input_seq["ss_idx"]
    return {
        "Intron_upstream_2": (0, ss_idx[0][0] - 1) if isinstance(ss_idx[0][0], int) else ('<NA>', '<NA>'),
        "Exon_upstream": (ss_idx[0][0], ss_idx[0][1]),
        "Intron_upstream": (ss_idx[0][1] + 1 if isinstance(ss_idx[0][1], int) else 0, ss_idx[1][0] - 1),
        "Exon_cassette": (ss_idx[1][0], ss_idx[1][1]),
        "Intron_downstream": (ss_idx[1][1] + 1, ss_idx[2][0] - 1 if isinstance(ss_idx[2][0], int) else len(input_seq['seq']) -1),
        "Exon_downstream": (ss_idx[2][0], ss_idx[2][1]),
        "Intron_downstream_2": (ss_idx[2][1] + 1, len(input_seq["seq"]) - 1) if isinstance(ss_idx[2][1], int) else ('<NA>', '<NA>')
    }