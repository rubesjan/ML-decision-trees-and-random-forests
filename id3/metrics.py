"""
basics metrics for computing diversification
"""

from pandas.core.api import Series
import numpy as np

def gini(y: Series) -> float:
    """
    compute gini index of the given set
    """
    if len(y) == 0:
        return 0
    p = y.mean()
    return 2*p*(1-p)

def entropy(y: Series) -> float:
    """
    compute entropy of the given set
    """
    if len(y) == 0:
        return 0
    p1 = y.mean()
    p0 = 1 - p1
    if p1 == 0 or p0 == 0:
        return 0
    return (-1) * p1 * np.log2(p1) - (p0) * np.log2(p0)
