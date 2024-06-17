import numpy as np
from scipy import stats


def summary_stats(v):
    """ Summary statistics for numpy array. """
    v_ = [x for x in v if not np.isnan(x)]
    res = stats.describe(v_)
    prop_nan = float(len(v) - len(v_)) / float(len(v))
    return {
        'mean': res.mean,
        'var': res.variance,
        'min': res.minmax[0],
        'max': res.minmax[1],
        'prop_nan': prop_nan
    }


def speaking_rate_calc(vowels, min_dur=500.0):
    """
    Local speaking rate computed 
    from list of vowel intervals.
        min_dur: minimum duration [ms]
    """
    n = len(vowels)
    if n == 0:
        return np.nan
    min_time = np.min([v[1]['start'] for v in vowels])
    max_time = np.max([v[1]['end'] for v in vowels])
    dur = (max_time - min_time)
    if dur < (min_dur / 1000.0):
        return np.nan
    rate = np.round(float(n) / dur, 2)
    return rate
