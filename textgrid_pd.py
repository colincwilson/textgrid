import re, sys
from pathlib import Path
from collections import namedtuple
import pandas as pd
import tgt

Tier = namedtuple("Tier", ["name", "entries"])

Entry = namedtuple("Entry", ["start", "end", "label", "tier"])

vowel_regex = '^[AEIOUaeiou].[012]$'


def read(filename, fileEncoding="utf-8"):
    """
    Reads a TextGrid file into pandas data frame.
    Each entry is a dictionary with keys "start", "end", "label", "tier"

    Points and intervals use the same format, 
    but the value for "start" and "end" are the same

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
    grid['dur'] = 1000.0 * (grid['end'] - grid['start'])  # ms
    grid['file'] = Path(filename).stem
    grid = grid[['file', 'tier', 'label', 'start', 'end', 'dur']]

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
    start = _get_float_val(content[i + 1])  # addition is cheap typechecking
    if content[i].startswith("intervals ["):
        offset = 1
    else:
        offset = 0  # for "point" objects
    end = _get_float_val(content[i + 1 + offset])
    label = _get_str_val(content[i + 2 + offset])
    return Entry(start=start, end=end, label=label, tier=tier)


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


def write(dat, fout, speaker=None, tiers=None):
    """
    Write out pandas dataframe as a TextGrid file (long format).
        fout (str/path): output file
        speaker (str): name of speaker field (default is None)
        tiers (list): tier names (default is None)
    """
    tier_list = []
    if speaker is not None:
        for spkr in dat[speaker].unique():
            dat1 = dat[(dat[speaker] == spkr)]
            tier_list.extend(_make_tiers(dat1, speaker=spkr, tiers=tiers))
    else:
        tier_list = _make_tiers(dat, speaker=None, tiers=tiers)

    start_time = 0.0
    end_time = dat['end'].max()
    grid = tgt.core.TextGrid(filename=fout)
    for tier in tier_list:
        tier.start_time = start_time
        tier.end_time = end_time
        grid.add_tier(tier)
    tgt.io.write_to_file(grid, filename=fout, format='long')
    return grid


def _make_tiers(dat, speaker, tiers):
    """
    Make one or more tiers for a speaker.
    """
    tier_list = []
    if tiers is not None:
        for tier in tiers:
            dat1 = dat[(dat['tier'] == tier)]
            tier_list.append(_make_tier(dat1, speaker=speaker, tier=tier))
    else:
        tier_list.append(_make_tier(dat, speaker, 'utterance'))
    return tier_list


def _make_tier(dat, speaker, tier):
    """
    Make one tier for a speaker.
    """
    tier_name = f'{speaker} - {tier}' \
        if speaker is not None else f'{tier}'
    tier = tgt.core.IntervalTier(name=tier_name)
    for i, row in dat.iterrows():
        try:
            interval = tgt.core.Interval(row['start'], row['end'], row['label'])
            tier.add_interval(interval)
        except:
            print(f'Error creating interval from {row}')
    return tier


def interval_at(grid, timepoint, speaker=None, tier=None):
    """
    Data frame of intervals overlapping timepoint (inclusive) 
    for all speakers or specified speaker, on all tiers or 
    specified tier.
        grid (dataframe)
        timepoint (float): time point (s)
        speaker (str): speaker name
        tier (str): tier name
    """
    if speaker is not None:
        grid = grid[(grid['speaker'] == speaker)]
    if tier is not None:
        grid = grid[(grid['tier'] == tier)]
    intervals = grid[((grid['start'] <= timepoint) &
                      (grid['end'] >= timepoint))]
    intervals = intervals.reset_index(drop=True)
    return intervals


def intervals_between(grid, start, end, speaker=None, tier=None):
    """
    Data frame of intervals between timepoints (inclusive) 
    for all speakers or specified speaker, on all tiers or 
    specified tier.
        grid (dataframe)
        start (float): start time (s)
        end (float): end time (s)
        speaker (str): speaker name
        tier (str): tier name
    """
    if speaker is not None:
        grid = grid[(grid['speaker'] == speaker)]
    if tier is not None:
        grid = grid[(grid['tier'] == tier)]
    intervals = grid[((grid['start'] >= start) & (grid['end'] <= end))]
    intervals = intervals.reset_index(drop=True)
    return intervals


def previous_interval(grid,
                      interval,
                      label='label',
                      speaker='speaker',
                      skip=['sp', ''],
                      max_skip_dur=500.0):
    """
    Return interval before specified one on the same tier of grid.
    """
    return _adjacent_interval(
        grid, interval, label, speaker, skip, max_skip_dur, direction='before')


def following_interval(grid,
                       interval,
                       label='label',
                       speaker='speaker',
                       skip=['sp', ''],
                       max_skip_dur=500.0):
    """
    Return interval after specified one on the same tier of grid.
    """
    return _adjacent_interval(
        grid, interval, label, speaker, skip, max_skip_dur, direction='after')


def _adjacent_interval(grid,
                       interval,
                       label='label',
                       speaker='speaker',
                       skip=['sp', ''],
                       max_skip_dur=500.0,
                       direction=None):
    """
    Return interval before/after specified one on the same tier of grid.
        grid (dataframe)
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


def speaking_rate_before(grid,
                         interval,
                         word_tier='word',
                         phone_tier='phone',
                         window=1000.0):
    """
    Speaking rate (syllable / second) in specified window prior to interval.
        label (str): key of word labels in grid (default = 'word')
        window (ms): duration of window (default = 1000.0)
    """
    # Words (from all speakers) in window prior to interval.
    start = interval['start'] - window / 1000.0
    if start < 0.0:
        start = 0.0
    end = interval['start']
    words = intervals_between(grid, start, end, tier=word_tier)
    nwords = len(words)
    if nwords == 0:
        return None

    # Contiguous words with same speaker as interval.
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

    # Window duration and speaking rate.
    window_start = words_contig[0]['start']
    window_end = words_contig[-1]['end']
    window_dur = (window_end - window_start)  # seconds
    phons_contig = intervals_between(
        grid, window_start, window_end, tier=phone_tier)
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
    write(
        grid,
        Path.home() / 'Desktop/tmp.TextGrid',
        speaker='speaker',
        tiers=['word', 'phone'])


if __name__ == '__main__':
    main()