import itertools
from dress.datasetgeneration.grammars.pwm_perturbation_grammar import (
    create_motif_grammar,
)
from dress.datasetgeneration.grammars.utils import (
    _get_forbidden_zones,
    _get_location_map,
    dinuc_shuffle,
    random_seq,
    shuffle,
)
from .grammars.random_perturbation_grammar import create_random_grammar
from .config_evolution import configureEvolution
from .black_box.model import DeepLearningModel, SpliceAI, Pangolin
from geneticengine.core.random.sources import RandomSource


SHUFFLE_FUN = {"dinuc_shuffle": dinuc_shuffle, "shuffle": shuffle, "random": random_seq}


def get_score_of_input_sequence(input_seq: dict, **kwargs) -> dict:
    """
    Returns the model score for the input sequence

    Args:
        input_seq (dict): Dictionary with several attributes about the
    the original sequence. Those are sequence ID (seq_id), the
    sequence (seq) and the splice site indexes (ss_idx)

    Returns:
        dict: Updated dictionary with the model score
        DeepLearningModel: Model object
    """
    seq_id = input_seq["seq_id"]
    seq = input_seq["seq"]
    ss_idx = input_seq["ss_idx"]
    dry_run = input_seq["dry_run"]
    model = kwargs["model"]

    if dry_run:
        input_seq["score"] = 0.5
    else:
        metric = kwargs["model_scoring_metric"]
        batch_size = kwargs["batch_size"]
        if model == "spliceai":
            model = SpliceAI(batch_size=batch_size,
                            scoring_metric=metric)

        elif model == "pangolin":
            model = Pangolin(
                scoring_metric=metric,
                batch_size=batch_size,
                mode=kwargs["pangolin_mode"],
                tissue=kwargs["pangolin_tissue"],
            )
        elif isinstance(model, DeepLearningModel):
            pass

        else:
            raise ValueError(f"Model {model} not supported")

        raw_pred = model.run([seq], original_seq=True)
        score = model.get_exon_score({seq_id: raw_pred}, ss_idx={seq_id: ss_idx})

        input_seq["score"] = score[seq_id]

    kwargs["logger"].info(f"Score: {input_seq['score']:.4f}")
    return input_seq, model


def _shuffle(input_seq, excluded_ranges, shuffle_func, **kwargs) -> str:
    """
    Shuffle the input sequence by keeping the excluded ranges untouched
    """
    shuffled_seq = ""
    last_processed_end = 0

    for interval in excluded_ranges:
        if interval.start > last_processed_end:
            shuffled_seq += shuffle_func(
                input_seq["seq"][last_processed_end : interval.start], **kwargs
            )

        shuffled_seq += input_seq["seq"][interval.start : interval.stop]
        last_processed_end = interval.stop

    if last_processed_end < len(input_seq["seq"]):
        shuffled_seq += shuffle_func(input_seq["seq"][last_processed_end:], **kwargs)

    return shuffled_seq


def shuffle_input_sequence(input_seq: dict, **kwargs) -> dict:

    rs = RandomSource(kwargs.get("seed", 0))
    location_map = _get_location_map(input_seq)
    excluded_r = _get_forbidden_zones(
        input_seq,
        region_ranges=location_map,
        acceptor_untouched_range=kwargs["acceptor_untouched_range"],
        donor_untouched_range=kwargs["donor_untouched_range"],
        untouched_regions=kwargs["untouched_regions"],
        model=kwargs["model"],
    )

    if kwargs["shuffle_input"]:
        shuff_func = SHUFFLE_FUN[kwargs["shuffle_input"]]
        if excluded_r:
            shuffled_seq = _shuffle(input_seq, excluded_r, shuff_func, rs=rs)

        else:
            shuffled_seq = shuff_func(input_seq["seq"], rs=rs)

        assert len(shuffled_seq) == len(
            input_seq["seq"]
        ), f"Shuffled sequence has different length compared to the original. {len(shuffled_seq)} != {len(input_seq['seq'])}"
        kwargs["logger"].info(
            f"Calculating score of shuffled ({kwargs['shuffle_input']}) sequence"
        )

        input_seq["seq"] = shuffled_seq
        input_seq, _ = get_score_of_input_sequence(input_seq, **kwargs)

    return input_seq, excluded_r, rs


def do_evolution(
    input_seq: dict,
    excluded_regions: list,
    **kwargs,
):
    """
    Evolution from a single genomic sequence with GeneticEngine

    Args:
        input_seq (dict): Dictionary with several attributes about the
        the original sequence. Those are sequence ID (seq_id), the
        sequence (seq), its SpliceAI score (score) and the splice site indexes (ss_idx)
        excluded_regions (list): List of forbidden zones in the sequence
    """

    if kwargs["which_grammar"] == "random":
        grammar, excluded_r = create_random_grammar(
            input_seq=input_seq, excluded_regions=excluded_regions, **kwargs
        )

    elif kwargs["which_grammar"] == "motif_based":
        grammar, excluded_r = create_motif_grammar(
            input_seq=input_seq,
            excluded_regions=excluded_regions,
            **kwargs,
        )

    alg, archive = configureEvolution(
        input_seq=input_seq,
        grammar=grammar,
        excluded_regions=excluded_regions,
        **kwargs,
    )
    assert excluded_r == excluded_regions
    kwargs["logger"].info("Evolution started")
    alg.evolve()
    return archive
