import abc
from typing import Union
import pandas as pd
from sklearn.tree import DecisionTreeClassifier as SklearnDT
import numpy as np


class Classification(abc.ABC):
    def __init__(self) -> None:
        pass

    def __call__(
        self, data: Union[pd.DataFrame, np.ndarray]
    ) -> Union["Classification", None]:
        """Fit a Classification model

        Args:
            data (Union[pd.DataFrame, np.ndarray]): Dataset to fit
        """
        self.X = (
            data.iloc[:, :-1].values if isinstance(data, pd.DataFrame) else data[:, :-1]
        )
        self.y = (
            data.iloc[:, -1].values if isinstance(data, pd.DataFrame) else data[:, -1]
        )

        self.estimator = self.estimator.fit(self.X, self.y)

        return self

class DecisionTreeClassifier(SklearnDT):
    def __init__(self, **kwargs) -> None:
        """Create a Decision Tree model

        Args:
            **kwargs: Arguments for the Decision Tree model
        """
        pass