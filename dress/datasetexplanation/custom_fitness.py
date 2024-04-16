import sqlite3
from typing import List

import pandas as pd
from geneticengine.algorithms.gp.individual import Individual

from sklearn.metrics import r2_score, root_mean_squared_error


class FitnessEvaluator(object):
    def __init__(
        self,
        dataset: pd.DataFrame,
        db: sqlite3.Connection,
        fitness_function: str = "r2",
    ):
        """Define a fitness function to use when evaluating an individual

        Args:

        """
        self.y = dataset.Score
        self.data = dataset.drop("Score", axis=1)
        self.db = db
        _ff = {
            "r2": self.r2,
            "rmse": self.rmse,
        }
        self.fitness_function = _ff[fitness_function]

    def r2(self, ind: Individual) -> float:
        """Calculates the r2 of an individual
        Args:
            ind (Individual): Population of explanations to evaluate

        Returns:
            float: R-square value of the model
        """
        y_preds = ind.predict(X=self.data, y=self.y, cursor=self.db)
        return r2_score(self.y, y_preds)
        
    def rmse(self, ind: List[Individual]) -> float:
        """Calculates the RMSE of the population
        Args:
            ind (Individual): Population of explanations to evaluate
            
        Returns:
            float: RMSE value of the model
        """
        y_preds = ind.predict(X=self.data, y=self.y, cursor=self.db)
        return root_mean_squared_error(self.y, y_preds)