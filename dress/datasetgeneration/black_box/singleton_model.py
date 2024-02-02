from typing import Any, Callable
from loguru import logger
import numpy as np
import tensorflow as tf
from spliceai.utils import load_model, resource_filename
import torch
from pangolin.model import Pangolin, L, W, AR

def batch_function_pangolin(model_nums: list) -> Callable[[Any], Any]:
    """
    Returns a predict_batch that makes inferences for the given model numbers.
    
    Adapted from https://github.com/tkzeng/Pangolin/blob/main/scripts/custom_usage.py
    """
 
    INDEX_MAP = {0:1, 1:2, 2:4, 3:5, 4:7, 5:8, 6:10, 7:11}
    
    logger.debug("Loading the models")
    # Splice site usage models, if Psplice (like SpliceAI), should use 0, 2, 4, 6
    model_nums = [1, 3, 5, 7]
    models = []
    for i in model_nums:
        for j in range(1, 6):
            model = Pangolin(L, W, AR)
            if torch.cuda.is_available():
                model.cuda()
                weights = torch.load(resource_filename("pangolin","models/final.%s.%s.3" % (j, i)))
            else:
                weights = torch.load(resource_filename("pangolin","models/final.%s.%s.3" % (j, i)),
                                 map_location=torch.device('cpu'))
            model.load_state_dict(weights)
            model.eval()
            models.append(model)
    logger.debug("Done")
    
    def predict_batch(batch: torch.Tensor) -> np.ndarray:
        per_tissue_preds = []
        
        for j, model_num in enumerate(model_nums):
            score = []
            
            # Average across 5 models
            for model in models[5*j:5*j+5]:
                with torch.no_grad():
                    score.append(model(batch)[:, INDEX_MAP[model_num], :].cpu().numpy())
                    
            per_tissue_preds.append(np.mean(score, axis=0))
            
        return np.mean(per_tissue_preds, axis=0)

    return predict_batch


def batch_function_spliceAI() -> Callable[[Any], Any]:
    """
    Returns a predict_batch function that executes in tensorflow.
    """
    logger.debug("Loading the models")
    paths = ("models/spliceai{}.h5".format(x) for x in range(1, 6))
    models = [
        load_model(resource_filename("spliceai", x), compile=False) for x in paths
    ]
    logger.debug("Done")

    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None, None, 4], dtype=tf.int32),)
    )
    def predict_batch(batch: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean([model(batch) for model in models], axis=0)

    return predict_batch