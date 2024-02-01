"""
creates decision tree with given dataset, metric and hyperparameters
"""

from random import randint
from pandas.core.api import DataFrame
from pandas.core.api import Series
from id3.tree.node import Node, LeafNode, SplitNode


def information_gain(y1: Series, y2: Series, init_inf: float, metric) -> float:
    """
    compute information gain, when y is splitted to the two given sets (y1 and y2)

    using uniform sample weight
    """
    t1 = len(y1)
    t2 = len(y2)
    t1 = t1 / (t1 + t2)
    t2 = 1 - t1
    y1_val = metric(y1)
    y2_val = metric(y2)
    i_gain = init_inf - t1*y1_val - t2*y2_val
    return i_gain


def get_best_ig(xtrain: DataFrame,
                ytrain: Series,
                min_samples_leaf: int,
                metric) -> list:
    """
    Returns list of column and threshold with the highes information gain possible

    params:
    xtrain: DataFrame
        train data
    ytrain: Series
        real value
    min_samples_leaf: int
        minimum number of samples required to be a leaf node
    metric: function
        function to compute degree of disorder in a set
    """

    init_ig = metric(ytrain)  # equals 0 only if all instances are same class

    best = {  # best criterion to split the given set
        "ig": 0,
        "col": None,
        "threshold": None}
    all_best = [best]
    for col in xtrain.columns:  # for all columns
        values = sorted(xtrain[col].unique())
        # all thresholds to try
        thresholds = [(values[i-1] + values[i]) /
                      2 for i in range(1, len(values))]
        for threshold in thresholds:  # for all thresholds
            mask = xtrain[col] < threshold
            y1 = ytrain[mask]
            y2 = ytrain[~mask]

            # using this threshold would create split with |y| < min_samples_leaf
            if len(y1) < min_samples_leaf or len(y2) < min_samples_leaf:
                continue

            # information gain using this split
            i_gain = information_gain(y1, y2, init_ig, metric)
            if i_gain > best["ig"]:
                best["ig"] = i_gain
                best["col"] = col
                best["threshold"] = threshold
                all_best.clear()
                all_best.append(best)
            elif i_gain == best["ig"]:
                all_best.append(
                    {"ig": i_gain, "col": col, "threshold": threshold})
    return all_best


def get_node(
        xtrain: DataFrame,
        ytrain: Series,
        max_depth: int,
        min_samples_leaf: int,
        metric) -> Node:  # : callable[[Series], float]
    """
    Recursively creates a decision tree with given hyperparameters and returns root node
    using polymorphic nodes from tree/nodes.py

    parameters:
    xtrain: DataFrame
        data
    ytrain: Series
        true value
    max_depth: int
        max_depth of the subtree from this node
    min_samples_leaf: int
        minimum number of samples required to be a leaf node
    metric: function
        function to compute degree of disorder in a set

    returns:
    SplitNode
        if |y| >= min_samples_leaf in both sons and exists column
        with threshold with significant information gain
    LeafNode
        if data does not have any column with threshold with
        significant information gain or one of the sons would have |y| < min_samples_leaf
    """

    if len(ytrain) != len(xtrain):
        raise TypeError(
            f"invalid data shape - x shape = {xtrain.shape}, y shape = {ytrain.shape}")

    if len(ytrain) < min_samples_leaf:
        raise TypeError("Not enough data!")

    init_ig = metric(ytrain)  # equals 0 only if all instances are same class

    if max_depth < 1 or init_ig == 0:
        return LeafNode(ytrain)

    all_best = get_best_ig(xtrain, ytrain, min_samples_leaf, metric)

    best = all_best[0]
    if len(all_best) > 1:
        best = all_best[randint(0, len(all_best) - 1)]

    if best["ig"] > 0:  # information gain is significant
        mask = xtrain[best["col"]] < best["threshold"]
        left = get_node(xtrain[mask], ytrain[mask],
                        max_depth-1, min_samples_leaf, metric)
        right = get_node(xtrain[~mask], ytrain[~mask],
                         max_depth-1, min_samples_leaf, metric)
        return SplitNode(best["threshold"], best["col"], left, right)

    # no criterion to gain any more information - create a leaf
    return LeafNode(ytrain)
