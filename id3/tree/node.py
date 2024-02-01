"""
polymorphic nodes of decision tree
"""

import pandas as pd
from pandas.core.api import Series

class Node:
    """
    Abstract class of all nodes in decision tree
    """
    def predict_prob(self, xtrain: Series) -> float:
        """
        Abstract method
        predict probability P(y=1)
        """

    def predict(self, xtrain: Series) -> int:
        """
        Abstract method
        prediction of y of the given record
        """

class SplitNode(Node):
    """
    Internal node in a decision tree
    """
    def __init__(self,
                 threshold: float,
                 column: str,
                 left: Node = None,
                 right: Node = None):
        """
        Construct internal node in a decision tree
        parameters:
        threshold: float
            threshold value of in the given column
        column: string
            column in which the split is decided
        left: node
            left child - if value in the column is lower than threshold
        right: node
            right child - if value in the column is greater than threshold

        """
        self.threshold = threshold
        self.column = column
        self.left = left
        self.right = right

    def predict_prob(self, xtrain: Series):
        """
        Abstract method
        predict probability P(y=1)
        """
        if xtrain[self.column] <= self.threshold:
            return self.left.predict_prob(xtrain)
        return self.right.predict_prob(xtrain)

    def predict(self, xtrain: Series):
        """
        Abstract method
        prediction of y of the given record
        """
        if xtrain[self.column] <= self.threshold:
            return self.left.predict(xtrain)
        return self.right.predict(xtrain)

class LeafNode(Node):
    """
    Leaf node in a decision tree
    """
    def __init__(self,
                 y: pd.Series):
        """
        Construct leaf node from the given set of y (only the mean value is necessary)
        """
        self.true_proba = y.mean()

    def predict_prob(self, xtrain: Series):
        """
        Abstract method
        predict probability P(y=1)
        """
        return self.true_proba

    def predict(self, xtrain: Series):
        """
        Abstract method
        prediction of y of the given record
        """
        if self.true_proba > 0.5:
            return 1
        return 0
