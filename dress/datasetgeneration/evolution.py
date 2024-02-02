from loguru import logger
from .grammars.with_indels_grammar import create_grammar
from .config_evolution import configureEvolution
from .black_box.model import (
    SpliceAI, Pangolin
)


def get_score_of_input_sequence(input_seq: dict,
                                model: str,
                                metric: str) -> dict:
    """
    Returns the model score for the input sequence

    Args:
        input_seq (dict): Dictionary with several attributes about the
    the original sequence. Those are sequence ID (seq_id), the
    sequence (seq) and the splice site indexes (ss_idx)
        model (str): Model to use for scoring the sequence    
        metric (str): Metric to score the sequence taking 
        into account the predictions for each splice site of an exon
        
    Returns:
        dict: Updated dictionary with the model score
    """
    seq_id = input_seq["seq_id"]
    seq = input_seq["seq"]
    ss_idx = input_seq["ss_idx"]
    dry_run = input_seq["dry_run"]

    if dry_run:
        input_seq["score"] = 0.5

    elif model == "spliceai":
        model = SpliceAI(scoring_metric=metric)
        
    elif model == "pangolin":
        model = Pangolin(scoring_metric=metric)
    
    else:
        raise ValueError(f"Model {model} not supported")

    raw_pred = model.run([seq], original_seq=True)
    score = model.get_exon_score({seq_id: raw_pred}, 
                                 ss_idx={seq_id: ss_idx})
    
    input_seq["score"] = score[seq_id]
    del model
    return input_seq


def do_evolution(
    input_seq: dict,
    **kwargs,
):
    """
    Evolution from a single genomic sequence with GeneticEngine

    Args:
        input_seq (dict): Dictionary with several attributes about the
        the original sequence. Those are sequence ID (seq_id), the
        sequence (seq), its SpliceAI score (score) and the splice site indexes (ss_idx)
        
    """

    grammar = create_grammar(
        max_diff_units=kwargs["max_diff_units"],
        snv_weight=kwargs["snv_weight"],
        insertion_weight=kwargs["insertion_weight"],
        deletion_weight=kwargs["deletion_weight"],
        max_insertion_size=kwargs["max_insertion_size"],
        max_deletion_size=kwargs["max_deletion_size"],
        input_seq=input_seq
    )

    alg, archive = configureEvolution(
        input_seq=input_seq,
        grammar=grammar,
        **kwargs,
    )

    logger.info("Evolution started")
    alg.evolve()
    return archive