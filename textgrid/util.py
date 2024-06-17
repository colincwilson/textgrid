import numpy as np
from scipy import stats


def summary_stats(v):
    """ Summary statistics for numpy array. """
    res = stats.describe(v)
    return {
        'mean': res.mean,
        'var': res.variance,
        'min': res.minmax[0],
        'max': res.minmax[1]
    }
