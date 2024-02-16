from typing import List, Union

from dress.datasetevaluation.dataset import Dataset, PairedDataset


def structure_dataset(
    generated_datasets: List[str], original_seqs: List[str] = None, **kwargs
) -> Union[Dataset, PairedDataset]:
    """Configs the datasets to be evaluated

    Args:
        generated_datasets (List[str]): Generated dataset(s) to be evaluated
        original_seqs (List[str]): Original sequence(s) used to generate the dataset(s)

    Returns:
        Dataset: Configured dataset(s)
    """
    _g1 = kwargs["groups"][0] if kwargs["groups"] else "1"
    
    def _create_single_dataset(data, original_seq, group):
        if original_seq:
            return Dataset(data, original_seq, group)
        else:
            return Dataset(data, group=group)

    dataset1 = _create_single_dataset(
        generated_datasets[0], original_seqs[0] if original_seqs else None, _g1
    )

    if generated_datasets[1]:
        _g2 = kwargs["groups"][1] if kwargs["groups"] else "2"
        dataset2 = _create_single_dataset(
            generated_datasets[1],
            original_seqs[1] if original_seqs else None,
            _g2,
        )
        return PairedDataset(dataset1, dataset2)

    return dataset1
