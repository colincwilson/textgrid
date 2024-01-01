import re, sys
from pathlib import Path
from collections import namedtuple
import polars as pl
import tgt

Tier = namedtuple("Tier", ["name", "entries"])

Entry = namedtuple("Entry", ["start", "end", "label", "tier"])

vowel_regex = '^[AEIOUaeiou].[012]$'
mfa_tier_regex = '^(.+) - (phones|words|utterances)$'

# # # # # # # # # #


def read(filename, fileEncoding="utf-8"):
    """
    Read TextGrid into polars data frame.
    """
    filename = str(filename)
    grid = tgt.io.read_textgrid(filename)
    dat = []
    for tier in grid.tiers:
        tier_name = tier.name
        if re.search(mfa_tier_regex, tier_name):
            speaker = re.sub(mfa_tier_regex, '\\1', tier_name)
            tier_name = re.sub(mfa_tier_regex, '\\2', tier_name)
        else:
            speaker = '<null>'

        for interval in tier.annotations:
            dat.append((speaker,tier_name,  \
                interval.text, interval.start_time, interval.end_time))

    dat = pl.DataFrame(dat,
                       schema=['speaker', 'tier', 'label', 'start', 'end'])
    dat = dat.with_columns(file=pl.lit(filename))
    dat = dat[['file', 'speaker', 'tier', 'label', 'start', 'end']]

    return dat


def write(dat, filename, speakers=None, tiers=None):
    """
    Write polars data frame as TextGrid.
        filename (str/path): output file
        speakers (list): speakers to include (None => all)
        tiers (list): tiers to include (None => all)
    """

    # Make textgrid.
    start_time = 0.0
    end_time = dat['end'].max()
    grid = tgt.core.TextGrid(filename)

    # Make tiers.
    tiers = _make_tiers(dat, speakers, tiers)
    for tier in tiers:
        tier.start_time = start_time
        tier.end_time = end_time
        grid.add_tier(tier)

    tgt.io.write_to_file(grid, filename, format='long')  # checkme: format
    return grid


def _make_tiers(dat, speakers, tiers):
    """ Make tier for each speaker x tier combo. """
    if speakers is None:
        speakers = list(set(dat['speaker']))
    if tiers is None:
        tiers = list(set(dat['tier']))
    ret = []
    for speaker in speakers:
        for tier in tiers:
            dat1 = dat.filter((pl.col('speaker') == speaker)
                              & (pl.col('tier') == tier))
            if len(dat1) == 0:
                continue
            ret.append(_make_tier(dat1, speaker, tier))
    return ret


def _make_tier(dat, speaker, tier):
    """ Make tier for one speaker x tier combo. """
    if speaker == '<null>':
        tier_name = tier
    else:
        tier_name = f'{speaker} - {tier}'
    tier = tgt.core.IntervalTier(name=tier_name)
    for row in dat.rows(named=True):
        interval = tgt.core.Interval(row['start'], row['end'], row['label'])
        try:
            tier.add_interval(interval)
        except:
            print(f'Error adding interval {interval}')
    return tier


def interval_at(dat, timepoint, speakers=None, tiers=None):
    """
    Data frame of intervals overlapping timepoint (inclusive) 
    for all speakers or specified speakers, on all tiers or 
    specified tiers.
        dat: polars dataframe
        timepoint (float): time point (s)
        speaker (list): speakers to include (None => all)
        tier (str): tiers to include (None => all)
    """
    dat1 = dat
    if speakers is not None:
        dat1 = dat1.filter(pl.col('speaker').is_in(speakers))
    if tiers is not none:
        dat1 = dat1.filter(pl.col('tier').is_in(tiers))
    dat1 = dat1.filter((pl.col('start') <= timepoint)
                       & (pl.col('end') >= timepoint))
    return dat1


def intervals_between(dat, start, end, speaker=None, tier=None):
    """
    Data frame of intervals between timepoints (inclusive) 
    for all speakers or specified speakers, on all tiers or 
    specified tiers.
        dat: polars dataframe
        start (float): start time (s)
        end (float): end time (s)
        speaker (list): speakers to include (None => all)
        tier (str): tiers to include (None => all)
    """
    dat1 = dat
    if speaker is not None:
        dat1 = dat1.filter(pl.col('speaker').is_in(speakers))
    if tier is not None:
        dat1 = dat1.filter(pl.col('tier').is_in(tiers))
    dat1 = dat1.filter((pl.col('start') >= start) & (pl.col('end') <= end))
    return dat1


# # # # # # # # # #

# FIXME: convert to polars


def previous_interval(grid,
                      interval,
                      label='label',
                      speaker='speaker',
                      skip=['sp', ''],
                      max_skip_dur=500.0):
    """
    Return interval before specified one on the same tier of grid.
    """
    return _adjacent_interval(grid,
                              interval,
                              label,
                              speaker,
                              skip,
                              max_skip_dur,
                              direction='before')


def following_interval(grid,
                       interval,
                       label='label',
                       speaker='speaker',
                       skip=['sp', ''],
                       max_skip_dur=500.0):
    """
    Return interval after specified one on the same tier of grid.
    """
    return _adjacent_interval(grid,
                              interval,
                              label,
                              speaker,
                              skip,
                              max_skip_dur,
                              direction='after')


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
    phons_contig = intervals_between(grid,
                                     window_start,
                                     window_end,
                                     tier=phone_tier)
    if len(phons_contig) == 0:
        return None
    vowels_contig = phons_contig[(
        phons_contig['label'].str.contains(vowel_regex))]
    if len(vowels_contig) == 0:
        return None
    speaking_rate = float(len(vowels_contig)) / float(window_dur)

    return speaking_rate


def main():
    fgrid = Path.home(
    ) / 'Sounds/SCOTUS/fave_align/2013/12-1168_11025.TextGrid'
    grid = read(fgrid)
    grid['speaker'] = [re.sub(' -.*', '', x) for x in grid['tier']]
    grid['tier'] = [re.sub('.*- ', '', x) for x in grid['tier']]
    grid['word'] = grid['label']
    interval = grid.iloc[-100]
    speaking_rate = speaking_rate_before(grid, interval)
    print(f'{speaking_rate} syllables / second')
    print(1.0 / speaking_rate * 1000.0)
    write(grid,
          Path.home() / 'Desktop/tmp.TextGrid',
          speaker='speaker',
          tiers=['word', 'phone'])


if __name__ == '__main__':
    main()
