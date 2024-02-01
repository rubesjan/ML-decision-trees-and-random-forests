"""
decision tree class with its properties
"""

import pandas as pd
from pandas.core.api import DataFrame
from pandas.core.api import Series
import numpy as np
from id3.metrics import gini, entropy
from id3.tree_construction import get_node


class MyDecisionTree():
    """decision tree classifier

    Parameters:
    max_depth: int
        maximal depth of the tree
    min_samples_leaf: int
        minimum number of samples required to be a leaf node
    criterion: {"gini", "entropy"}
        criterion to measure degree of disorder in a given set (pandas series)
    """

    def __init__(self,
                 max_depth: int = None,
                 min_samples_leaf: int = 1,
                 criterion: str = "gini"):
        """
        construct tree with given hyperparameters
        """
        if not isinstance(max_depth, int):
            raise TypeError(
                f"Invalid max_depth! Expected int, got: {type(max_depth)}")

        if max_depth < 1:
            raise ValueError("max_depth must be a positive integer!")

        if not isinstance(min_samples_leaf, int):
            raise TypeError(
                f"Invalid min_samples_leaf! Expected int got: {type(min_samples_leaf)}")

        if min_samples_leaf < 0:
            raise ValueError("min_samples_leaf must be a positive integer!")

        if not isinstance(criterion, str):
            raise TypeError(
                f"Invalid criterion! Expected string, got: {type(criterion)}")
        if criterion.lower() == "gini":
            self.metric = gini
        elif criterion.lower() == "entropy":
            self.metric = entropy
        else:
            raise ValueError("Invalid criterion! Expected 'gini' or 'entropy'")

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def is_fitted(self) -> bool:
        """
        returns True if tree was fitted, False otherwise
        """
        if self.tree is None:
            return False
        return True

    def fit(self,
            x: pd.DataFrame,
            y: pd.Series):
        """
        Trains the tree on the given data.
        parameters:
        x: DataFrame
            data
        y: Series
            real value
        """
        if self.is_fitted():
            raise SyntaxError("Tree is already fitted!")

        if not isinstance(x, DataFrame) or not isinstance(y, Series):
            raise TypeError("Invalid data!")

        if len(x) != len(y):
            raise ValueError(
                f"num of x rows: {len(x)} != num of y rows: {len(y)}")

        self.tree = get_node(x, y, self.max_depth,
                             self.min_samples_leaf, self.metric)
        return self

    def predict_prob(self, x: DataFrame):
        """
        returns predicted probability of the given rows:
        P(y=0) in the first column
        P(y=1) in the second column
        """
        if not self.is_fitted():
            raise SyntaxError("Tree was not fitted!")

        if not isinstance(x, DataFrame):
            raise TypeError(
                f"Invalid input! Expected DataFrame got: {type(x)}")

        res = np.zeros((x.shape[0], 2))
        # for every row in the given DataFrame
        for i, index in enumerate(x.index):
            res[i, 1] = self.tree.predict_prob(x.loc[index])
            res[i, 0] = 1 - res[i, 1]
        return res

    def predict(self, x: pd.DataFrame):
        """
        return predicted y value for all rows in the given DataFrame
        """

        if not self.is_fitted():
            raise SyntaxError("Tree was not fitted!")

        if not isinstance(x, DataFrame):
            raise TypeError(
                f"Invalid input! Expected DataFrame got: {type(x)}")

        res = np.zeros(x.shape[0])
        # for every row in the given DataFrame
        for i, index in enumerate(x.index):
            res[i] = self.tree.predict(x.loc[index])
        return res
