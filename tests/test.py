"""
pytest tests
"""

import pytest
import pep_import  # eddited import path
from forest.forest_class import MyRandomForest
from id3.tree.node import LeafNode
from id3.tree_construction import information_gain, get_best_ig
from id3.model_class import MyDecisionTree
from id3.metrics import gini, entropy
import numpy as np
from pandas.core.api import DataFrame, Series


def test_gini():
    """
    Test gini index calculation
    """

    assert gini(Series([0, 0, 0, 0, 0, 0], dtype=int)) == 0
    assert gini(Series([1, 1, 1, 1, 1, 1, 1], dtype=int)) == 0
    assert gini(Series([])) == 0
    assert gini(Series([1, 1, 1, 1, 0, 0])) - 4/9 <= 0.001
    assert gini(Series([1, 0, 0, 0, 0, 0])) - 1/3 <= 0.001
    assert gini(Series([0, 0, 0, 1, 1, 1])) - 0.5 <= 0.001


def test_entropy():
    """
    Test entropy calculation
    """

    assert entropy(Series([0, 0, 0, 0, 0, 0], dtype=int)) == 0
    assert entropy(Series([1, 1, 1, 1, 1, 1, 1], dtype=int)) == 0
    assert entropy(Series([1, 1, 1, 0, 0, 0], dtype=int)) == 1
    assert entropy(Series([1, 1, 1, 1, 0, 0])) - 0.92 <= 0.001
    assert entropy(Series([1, 0, 0, 0, 0, 0])) - 0.81 <= 0.001
    assert entropy(Series([0, 0, 0, 1, 1, 1])) == 1


def test_decision_tree_exceptions():
    """
    Test if invalid arguments in decision tree raises an exception
    """

    with pytest.raises(TypeError):
        tree = MyDecisionTree(
            max_depth="a", min_samples_leaf=10, criterion="gini")

    with pytest.raises(TypeError):
        tree = MyDecisionTree(
            max_depth=10, min_samples_leaf="ten", criterion="gini")

    with pytest.raises(ValueError):
        tree = MyDecisionTree(
            max_depth=5, min_samples_leaf=10, criterion="gini index")

    with pytest.raises(ValueError):
        tree = MyDecisionTree(
            max_depth=5, min_samples_leaf=-1, criterion="gini")


def test_decision_tree():
    """
    Test simple decision tree predictions
    """

    tree = MyDecisionTree(max_depth=2, min_samples_leaf=1, criterion="entropy")
    tree.fit(x=DataFrame([[10], [10], [0], [0]]), y=Series([1, 1, 0, 0]))
    res = tree.predict(DataFrame([[1], [2], [3], [4]]))
    assert len(res) == 4
    assert isinstance(res, np.ndarray)
    for prediction in res:
        assert prediction == 0
    res = tree.predict(DataFrame([[6], [7], [8], [9]]))
    assert isinstance(res, np.ndarray)
    for prediction in res:
        assert prediction == 1


def test_random_forest():
    """
    Test simple random forest predictions
    """

    tree = MyRandomForest(n_estimators=2, max_samples=2,
                          max_depth=2, min_samples_leaf=1)
    tree.fit(x=DataFrame([[10], [10], [10], [10]]), y=Series([1, 1, 1, 1]))
    res = tree.predict(DataFrame([[1], [2], [3], [4]]))
    assert len(res) == 4
    assert isinstance(res, np.ndarray)
    for prediction in res:
        assert prediction == 1


def test_information_gain():
    """
    Test information gain calculation
    """

    first = Series([0, 0, 0, 0])
    second = Series([1, 1, 1, 1])
    assert information_gain(first, second, 0.5, gini) - 0.5 <= 0.001
    assert information_gain(first, second, 1, entropy) == 1


def test_get_best_ig():
    """
    Test get_best_ig function to get the best possible criterion for a split in decision tree
    """

    xtrain = DataFrame([[10], [10], [0], [0]], columns=["first"])
    ytrain = Series([0, 0, 1, 1])
    res = get_best_ig(xtrain, ytrain, 2, entropy)
    assert len(res) == 1
    assert res[0] == {"col": "first", "ig": 1, "threshold": 5}


def test_leaf_node():
    """
    Test creation of a leaf node and prediction
    """

    node = LeafNode(Series([0, 0, 0, 0]))
    assert node.predict(Series([0])) == 0
    assert node.predict_prob(Series([0])) == 0
