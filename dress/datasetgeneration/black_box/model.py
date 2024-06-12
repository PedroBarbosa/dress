import os
from tqdm import tqdm
from loguru import logger
from typing import Dict, Tuple, Union, List, Literal
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf  # noqa: E402
import torch  # noqa: E402

from keras.utils import pad_sequences  # noqa: E402
from spliceai.utils import one_hot_encode  # noqa: E402
from .singleton_model import (  # noqa: E402
    batch_function_spliceai,  # noqa: E402
    batch_function_pangolin,  # noqa: E402
)  # noqa: E402

predict_batch_spliceai = None
predict_batch_pangolin = None


class DeepLearningModel(object):
    def __init__(self, context: int, batch_size: int, scoring_metric: str):
        """Initializes DeepLearningModel class

        Args:
            context (int): Number of basepairs to pad at both ends of the sequences.
            batch_size (int): Number of sequences to predict at once at each batch.
            scoring_metric (str): Metric to use for exon scoring (mean, max). If mean, the mean of the acceptor and
        donor scores is used. If max, the maximum of the acceptor and donor scores is used.
        """
        self.context = context
        self.batch_size = batch_size
        if scoring_metric == "mean":
            self._apply_metric = lambda x: round(np.mean(x, axis=0), 4)
        elif scoring_metric == "max":
            self._apply_metric = lambda x: round(np.max(x, axis=0), 4)
        elif scoring_metric == "min":
            self._apply_metric = lambda x: round(np.min(x, axis=0), 4)

    def data_preparation(
        self, seqs: List[str], model: str
    ) -> Tuple[np.ndarray, List[int]]:
        """Prepares the data to run the model

        Args:
            seqs (List[str]): List with sequences
            model (str): Model name

        Returns:
            np.ndarray: Batches to make model inferences
            seq_lengths (list): List with seq lengths
        """
        logger.debug(
            "Processing the input seqs (one hot encoding, padding, batch split)"
        )
        seq_lengths = [len(s) for s in seqs]
        max_len = max(seq_lengths)
        x = pad_sequences([one_hot_encode(s) for s in seqs], maxlen=max_len)

        logger.debug(f"Max seq size in population: {max_len}")
        npad = ((0, 0), (self.context, self.context), (0, 0))
        x = np.pad(x, pad_width=npad)
        batches = np.split(x, np.arange(self.batch_size, len(x), self.batch_size))
        logger.debug(
            f"Making {model} inferences. Number of batches: {len(batches)}; Max number of seqs per batch: {self.batch_size}"
        )
        return batches, seq_lengths


class SpliceAI(DeepLearningModel):
    def __init__(
        self,
        context: int = 5000,
        batch_size: int = 64,
        scoring_metric: Literal["mean", "max", "min"] = "mean",
    ):
        """SpliceAI model class"""
        super().__init__(context, batch_size, scoring_metric)
        self._init_model()

    def run(
        self, seqs: List[str], original_seq: bool = False
    ) -> Union[Dict[int, List[np.ndarray]], List[np.ndarray]]:
        """Runs SpliceAI on the given sequences and returns raw predictions.

        Args:
            seqs (List[str]): List of sequencees to get predictions
            original_seq (bool): If True, refers to the setting of running SpliceAI in the original sequence

        Returns:
            Union[Dict[int, List[np.ndarray]], List[np.ndarray]]: Returns SpliceAI predictions
        """

        preds = []
        batches, seq_lengths = self.data_preparation(seqs, "SpliceAI")
        max_len = max(seq_lengths)

        for _i, batch in enumerate(tqdm(batches)):
            n_seqs = self.batch_size * _i
            batch_tf = tf.convert_to_tensor(batch, dtype=tf.int32)
            raw_preds = self.predict_batch_spliceai(batch_tf)
            batch_preds = [
                x.numpy()[max_len - seq_lengths[i + n_seqs] :]
                for i, x in enumerate(raw_preds)
            ]

            preds.extend(batch_preds)

        return preds[0] if original_seq else {id: p for id, p in enumerate(preds)}

    def get_exon_score(
        self,
        raw_preds: Dict[int, np.ndarray],
        ss_idx: Union[Dict[int, List[list]], List[list]],
    ) -> Dict[int, float]:
        """Get SpliceAI score for an exon

        Args:
            raw_preds (Dict[int, np.ndarray]): Dict with seq IDs as keys and predictions as values
            ss_idx (Union[Dict[int, List[list]], List[list]]): Splice site indexes of the exon triplet

        Returns:
            Dict[int, float]: Dict with seq IDs as keys and exon scores as values
        """

        # If ss_idx is a list (with updated ss_idx for each individual in a population)
        if isinstance(ss_idx, list):
            ss_idx = {i: v for i, v in enumerate(ss_idx)}

        out = {}

        for seq_id, pred in raw_preds.items():
            if seq_id not in ss_idx.keys():
                logger.error("{} seq ID not in splice site idx file.".format(seq_id))
            cassette = ss_idx[seq_id][1]
            acceptor = pred[:, 1][cassette[0]]
            donor = pred[:, 2][cassette[1]]
            out[seq_id] = self._apply_metric([acceptor, donor])

        return out

    def _init_model(self):
        self.predict_batch_spliceai = batch_function_spliceai()


class Pangolin(DeepLearningModel):
    def __init__(
        self,
        context: int = 5000,
        batch_size: int = 64,
        scoring_metric: Literal["mean", "max", "min"] = "mean",
        mode: Literal["ss_usage", "ss_probability"] = "ss_usage",
        tissue: Union[
            Literal["heart", "liver", "brain", "testis"],
            List[Literal["heart", "liver", "brain", "testis"]],
        ] = None,
    ):
        """Pangolin model class"""
        super().__init__(context, batch_size, scoring_metric)
        if mode == "ss_usage":
            self.model_nums = [1, 3, 5, 7]
        elif mode == "ss_probability":
            self.model_nums = [0, 2, 4, 6]
        else:
            logger.error(
                "Invalid Pangolin mode. Please choose from ss_usage, ss_probability"
            )
            exit(1)

        if tissue:
            t_map_idx = {"heart": 0, "liver": 1, "brain": 2, "testis": 3}

            try:
                if isinstance(tissue, (list, tuple)):
                    self.model_nums = [self.model_nums[t_map_idx[t]] for t in tissue]
                else:
                    self.model_nums = [self.model_nums[t_map_idx[tissue]]]
            except KeyError:
                logger.error(
                    "Invalid tissue type. Please choose from heart, liver, brain, testis"
                )
                exit(1)
        self._init_model()

    def run(
        self, seqs: List[str], original_seq: bool = False
    ) -> Union[Dict[int, List[np.ndarray]], List[np.ndarray]]:
        """Runs Pangolin on the given sequences and returns raw predictions.

        Args:
            seqs (List[str]): List of sequences to get predictions
            original_seq (bool): If True, refers to the setting of running Pangolin in the original sequence

        Returns:
            Union[Dict[int, List[np.ndarray]], List[np.ndarray]]: Pangolin predictions
        """
        preds = []
        batches, seq_lengths = self.data_preparation(seqs, "Pangolin")
        max_len = max(seq_lengths)

        for _i, seqs in enumerate(tqdm(batches)):
            n_seqs = self.batch_size * _i
            seqs = seqs.transpose(0, 2, 1)
            batch = torch.from_numpy(seqs).float()
            if torch.cuda.is_available():
                batch = batch.to(torch.device("cuda"))

            raw_preds = self.predict_batch_pangolin(batch)
            batch_preds = [
                x[max_len - seq_lengths[i + n_seqs] :] for i, x in enumerate(raw_preds)
            ]
            preds.extend(batch_preds)

        return preds[0] if original_seq else {id: p for id, p in enumerate(preds)}

    def get_exon_score(
        self,
        raw_preds: Dict[int, np.ndarray],
        ss_idx: Union[Dict[int, List[list]], List[list]],
    ) -> Dict[int, float]:
        """Get Pangolin score for an exon

        Args:
            raw_preds (Dict[int, np.ndarray]): Dict with seq IDs as keys and predictions as values
            ss_idx (Union[Dict[int, List[list]], List[list]]): Splice site indexes of the exon triplet

        Returns:
            Dict[int, float]: Dict with seq IDs as keys and exon scores as values
        """

        # If ss_idx is a list (with updated ss_idx for each individual in a population)
        if isinstance(ss_idx, list):
            ss_idx = {i: v for i, v in enumerate(ss_idx)}

        out = {}

        for seq_id, pred in raw_preds.items():
            if seq_id not in ss_idx.keys():
                logger.error("{} seq ID not in splice site idx file.".format(seq_id))

            cassette = ss_idx[seq_id][1]
            acceptor = pred[cassette[0]]
            donor = pred[cassette[1]]

            out[seq_id] = self._apply_metric([acceptor, donor])
        return out

    def _init_model(self):
        self.predict_batch_pangolin = batch_function_pangolin(self.model_nums)
