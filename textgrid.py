#!/usr/bin/python

from collections import namedtuple
import pandas as pd

Tier = namedtuple("Tier", ["name", "entries"])

Entry = namedtuple("Entry", ["begin", "end", "label", "tier"])


def read_textgrid(filename, fileEncoding="utf-8"):
    """
    Reads a TextGrid file into pandas data frame.
    Each entry is a dictionary with keys "begin", "end", "label", "tier"

    Points and intervals use the same format, 
    but the value for "begin" and "end" are the same

    Optionally, supply fileEncoding as argument. This defaults to "utf-8", tested with 'utf-16-be'.
    """
    try:
        if hasattr(filename, "readlines"):
            content = _read(filename)
        else:
            with open(str(filename), "r", encoding=fileEncoding) as f:
                content = _read(f)
    except:
        raise TypeError("filename must be a string or a readable buffer")

    interval_lines = [
        i for i, line in enumerate(content)
        if line.startswith("intervals [") or line.startswith("points [")
    ]

    tier_lines = []
    tier_names = []
    for i, line in enumerate(content):
        if line.startswith("name ="):
            tier_lines.append(i)
            tier_names.append(line.split('"')[-2])

    interval_tiers = _find_tiers(interval_lines, tier_lines, tier_names)
    assert len(interval_lines) == len(interval_tiers)
    intervals = [
        _build_entry(i, content, t)
        for i, t in zip(interval_lines, interval_tiers)
    ]

    grid = pd.DataFrame(intervals)
    grid['dur'] = grid['end'] - grid['begin']
    grid = grid[['begin', 'end', 'dur', 'label', 'tier']]

    return grid


def _find_tiers(interval_lines, tier_lines, tiers):
    tier_pairs = zip(tier_lines, tiers)
    cur_tline, cur_tier = next(tier_pairs)
    next_tline, next_tier = next(tier_pairs, (None, None))
    tiers = []
    for il in interval_lines:
        if next_tline is not None and il > next_tline:
            cur_tline, cur_tier = next_tline, next_tier
            next_tline, next_tier = next(tier_pairs, (None, None))
        tiers.append(cur_tier)
    return tiers


def _read(f):
    return [x.strip() for x in f.readlines()]


def _build_entry(i, content, tier):
    """
    takes the ith line that begin an interval and returns
    a dictionary of values
    """
    begin = _get_float_val(content[i + 1])  # addition is cheap typechecking
    if content[i].startswith("intervals ["):
        offset = 1
    else:
        offset = 0  # for "point" objects
    end = _get_float_val(content[i + 1 + offset])
    label = _get_str_val(content[i + 2 + offset])
    return Entry(begin=begin, end=end, label=label, tier=tier)


def _get_float_val(string):
    """
    returns the last word in a string as a float
    """
    return float(string.split()[-1])


def _get_str_val(string):
    """
    returns the last item in quotes from a string
    """
    return string.split('"')[-2]


def interval_at(grid, timepoint, tier=None):
    """
    Data frame of intervals overlapping timepoint (inclusive) 
    on all tiers or specified tier
    """
    if tier is not None:
        grid = grid[(grid['tier'] == tier)]
    intervals = grid[((grid['begin'] <= timepoint) &
                      (grid['end'] >= timepoint))]
    intervals = intervals.reset_index(drop=True)
    return intervals


def intervals_between(grid, begin, end, tier=None):
    """
    Data frame of intervals between timepoints (inclusive) 
    on all tiers or specified tier
    """
    if tier is not None:
        grid = grid[(grid['tier'] == tier)]
    intervals = grid[((grid['begin'] >= begin) & (grid['end'] <= end))]
    intervals = intervals.reset_index(drop=True)
    return intervals
