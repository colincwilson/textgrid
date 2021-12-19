#!/usr/bin/python

import re, sys
from pathlib import Path
from collections import namedtuple
import pandas as pd

Tier = namedtuple("Tier", ["name", "entries"])

Entry = namedtuple("Entry", ["begin", "end", "label", "tier"])

vowel_regex = '^[AEIOUaeiou].[012]$'


def read(filename, fileEncoding="utf-8"):
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
    grid['dur'] = 1000.0 * (grid['end'] - grid['begin'])  # ms
    grid['file'] = Path(filename).stem
    grid = grid[['file', 'tier', 'label', 'begin', 'end', 'dur']]

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


def interval_at(grid, timepoint, speaker=None, tier=None):
    """
    Data frame of intervals overlapping timepoint (inclusive) 
    for all speakers or specified speaker, on all tiers or 
    specified tier
    """
    if speaker is not None:
        grid = grid[(grid['speaker'] == speaker)]
    if tier is not None:
        grid = grid[(grid['tier'] == tier)]
    intervals = grid[((grid['begin'] <= timepoint) &
                      (grid['end'] >= timepoint))]
    intervals = intervals.reset_index(drop=True)
    return intervals


def intervals_between(grid, begin, end, speaker=None, tier=None):
    """
    Data frame of intervals between timepoints (inclusive) 
    for all speakers or specified speaker, on all tiers or 
    specified tier
    """
    if speaker is not None:
        grid = grid[(grid['speaker'] == speaker)]
    if tier is not None:
        grid = grid[(grid['tier'] == tier)]
    intervals = grid[((grid['begin'] >= begin) & (grid['end'] <= end))]
    intervals = intervals.reset_index(drop=True)
    return intervals


def previous_interval(grid,
                      interval,
                      label='label',
                      speaker='speaker',
                      skip=['sp'],
                      max_skip_dur=500.0):
    """
    Return interval before specified one on the same tier of grid
    """
    return _adjacent_interval(
        grid, interval, label, speaker, skip, max_skip_dur, direction='before')


def following_interval(grid,
                       interval,
                       label='label',
                       speaker='speaker',
                       skip=['sp'],
                       max_skip_dur=500.0):
    """
    Return interval after specified one on the same tier of grid
    """
    return _adjacent_interval(
        grid, interval, label, speaker, skip, max_skip_dur, direction='after')


def _adjacent_interval(grid,
                       interval,
                       label='label',
                       speaker='speaker',
                       skip=['sp'],
                       max_skip_dur=500.0,
                       direction=None):
    """
    Return interval before/after specified one on the same tier of grid
    grid: textgrid as pd data frame
    interval: row of grid
    label (str): key of labels in grid (default = 'label')
    speaker (str): key of speaker in grid (default = 'speaker')
    skip (list): skip one short instance of sp/pause/etc. (default = ['sp'])
    max_skip_dur (ms): do not skip long instances of sp/pause/etc. (default = 500ms)
    """
    if direction is None:
        raise Error('Must specify direction for _adjacent_interval()')
    idx = interval.name

    # Handle direction and grid edges
    if direction == 'before':
        if idx == 0:
            return None
        deltas = [-1, -2]
    if direction == 'after':
        if idx == grid.index[-1]:
            return None
        deltas = [1, 2]

    for delta in deltas:
        interval2 = grid.iloc[idx + delta]
        # Require matching tier
        if interval2['tier'] != interval['tier']:
            return None
        # Optionally require matching speaker fields
        if (speaker is not None) and (interval2[speaker] != interval[speaker]):
            return None
        # Skip short pauses, fail on long ones
        if interval2[label] in skip:
            if interval2['dur'] > max_skip_dur:
                return None
            continue
        # Return closest non-pause interval
        else:
            return interval2
    return None


def speaking_rate_before(grid, interval, label='word', window=1000.0):
    """
    Speaking rate (syllable / second) in specified window prior to interval
    label (str): key of word labels in grid (default = 'word')
    window (ms): duration of window (default = 1000.0)
    """
    # Words (from all speakers) in window prior to interval
    begin = interval['begin'] - window / 1000.0
    if begin < 0.0:
        begin = 0.0
    end = interval['begin']
    words = intervals_between(grid, begin, end, tier='word')
    nwords = len(words)
    if nwords == 0:
        return None

    # Contiguous words with same speaker as interval
    speaker = interval['speaker']
    words_contig = []
    for i in range(nwords - 1, -1, -1):
        word = words.iloc[i]
        if word['speaker'] != speaker:
            break
        words_contig.append(word)
    if len(words_contig) == 0:
        return None
    words_contig.reverse()

    # Window duration and speaking rate
    window_begin = words_contig[0]['begin']
    window_end = words_contig[-1]['end']
    window_dur = (window_end - window_begin)  # seconds
    phons_contig = intervals_between(
        grid, window_begin, window_end, tier='phone')
    if len(phons_contig) == 0:
        return None
    vowels_contig = phons_contig[(
        phons_contig['label'].str.contains(vowel_regex))]
    if len(vowels_contig) == 0:
        return None
    speaking_rate = float(len(vowels_contig)) / float(window_dur)

    return speaking_rate


def main():
    fgrid = Path.home() / 'Sounds/SCOTUS/fave_align/2013/12-1168_11025.TextGrid'
    grid = read(fgrid)
    grid['speaker'] = [re.sub(' -.*', '', x) for x in grid['tier']]
    grid['tier'] = [re.sub('.*- ', '', x) for x in grid['tier']]
    grid['word'] = grid['label']
    interval = grid.iloc[-100]
    speaking_rate = speaking_rate_before(grid, interval)
    print(f'{speaking_rate} syllables / second')
    print(1.0 / speaking_rate * 1000.0)


if __name__ == '__main__':
    main()