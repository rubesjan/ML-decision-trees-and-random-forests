"""
Random forest class using decision tree classifier from sklearn
"""

from random import randint
import pandas as pd
from pandas.core.api import DataFrame
from pandas.core.api import Series
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class MyRandomForest:
    """
    Random forest class
    params:
    n_estimators: int
        number of trees
    max_samples: int
        number of samples to be chosen from given dataset for each tree
    max_depth: int
        maximal depth of each tree
    random_state: int
        fix random state
    min_samples_leaf: int
        minimum number of samples required to be a leaf node
    """

    def __init__(self,
                 n_estimators: int,
                 max_samples: int,
                 max_depth: int,
                 min_samples_leaf: int = 5,
                 random_state=randint(0, 100)):
        """
        Constructs random forest with given hyperparameters
        """
        if not isinstance(n_estimators, int):
            raise TypeError(
                f"Invalid n_estimators. Epected int, got: {type(n_estimators)}")

        if n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer!")

        if not isinstance(max_samples, int):
            raise TypeError(
                f"Invalid max_samples. Expected int, got: {type(max_samples)}")

        if max_samples < min_samples_leaf:
            raise ValueError(
                "max_samples must be lower than min_samples_leaf!")

        if not isinstance(max_depth, int):
            raise TypeError(
                f"Invalid max_depth. Expected int, got: {type(max_depth)}")

        if max_depth < 1:
            raise ValueError("max_depth must be a positive integer!")

        if not isinstance(min_samples_leaf, int):
            raise TypeError(
                f"Invalid min_samples_leaf. Expected int got: {type(min_samples_leaf)}")

        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer!")

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.rstate = random_state
        self.trees = None

    def is_fitted(self) -> bool:
        """
        returns True if tree was fitted, False otherwise
        """
        if self.trees is None:
            return False
        return True

    def fit(self,
            x: DataFrame,
            y: Series):
        """
        Trains all n_estimators trees in the forest on the picked datasets using bootstrap
        """

        if self.is_fitted():
            raise SyntaxError("Forest is already fitted!")

        if not isinstance(x, DataFrame) or not isinstance(y, Series):
            raise TypeError("Invalid data!")

        if len(x) != len(y):
            raise ValueError(
                f"num of x rows: {len(x)} != num of y rows: {len(y)}")

        self.trees = []
        y.name = "correct_y"
        all_data = pd.concat([x, y], axis=1)
        for i in range(self.n_estimators):
            i += 0
            xdata = all_data.sample(n=self.max_samples, replace=True,
                                    random_state=self.rstate, axis=0)  # bootstrap
            ydata = xdata["correct_y"]  # create y
            xdata.drop("correct_y", axis=1, inplace=True)
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.rstate,
                                          min_samples_leaf=self.min_samples_leaf).fit(xdata, ydata)
            self.trees.append(tree)  # add tree to the list
        return self

    def predict_prob(self,
                     x: DataFrame) -> np.array:
        """
        Returns predicted probability of the given rows:
            P(y=0) in the first column
            P(y=1) in the second column
        """
        if not self.is_fitted():
            raise SyntaxError("Forest was not fitted!")

        if not isinstance(x, DataFrame):
            raise TypeError(
                f"Invalid input! Expected DataFrame, got: {type(x)}")

        # matrix with predicted values
        ypredicted = np.zeros((x.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            # in column is predicted values using one tree
            ypredicted[:, i] = tree.predict(x)

        # count mean predicted value for each row
        for i in range(ypredicted.shape[0]):
            ypredicted[i, 1] = np.mean(ypredicted[i, :])
            ypredicted[i, 0] = 1 - ypredicted[i, 1]

        return ypredicted[:, 0:2]

    def predict(self,
                x: DataFrame) -> np.array:
        """
        Predicts y of all rows in the x DataFrame
        """
        if not self.is_fitted():
            raise SyntaxError("Forest was not fitted!")

        if not isinstance(x, DataFrame):
            raise TypeError(
                f"Invalid input! Expected DataFrame, got: {type(x)}")

        ypredicted = self.predict_prob(x)
        for i in range(len(ypredicted)):
            if ypredicted[i, 1] >= 0.5:
                ypredicted[i, 0] = 1
            else:
                ypredicted[i, 0] = 0

        return ypredicted[:, 0]
