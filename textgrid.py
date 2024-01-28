import re, sys
import numpy as np
from pathlib import Path
from collections import namedtuple
import polars as pl
import tgt

# # # # # # # # # #

# Regexes.
ARPABET_VOWELS = '^[AEIOUaeiou].?$'
ARPABET_VOWELS_STRESS = '^[AEIOUaeiou].[012]$'
MFA_TIERS = '^(.+) - (phones|words|utterances)$'

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
        if re.search(MFA_TIERS, tier_name):
            speaker = re.sub(MFA_TIERS, '\\1', tier_name)
            tier_name = re.sub(MFA_TIERS, '\\2', tier_name)
        else:
            speaker = '<null>'

        for interval in tier.annotations:
            dat.append((speaker, tier_name,  \
                interval.text, interval.start_time, interval.end_time))

    dat = pl.DataFrame(dat,
                       schema=['speaker', 'tier', 'label', 'start', 'end'])
    dat = dat.with_columns( \
        filename=pl.lit(filename),
        dur_ms=1000.0*(pl.col('end') - pl.col('start')))

    dat = dat[[
        'filename', 'speaker', 'tier', 'label', 'start', 'end', 'dur_ms'
    ]]
    dat = dat.sort(['speaker', 'tier', 'start', 'end'])

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


def combine_tiers(dat):
    """
    Combine word and phone tiers within speaker.
    """
    # Word tier.
    dat_word = dat \
        .filter(pl.col('tier') == "words") \
        .rename({"label": "word", "start": "word_start",
                 "end": "word_end"}) \
        .sort(['filename', 'speaker', 'word_start'])

    dat_word = dat_word[[
        'filename', 'speaker', 'word', 'word_start', 'word_end'
    ]]

    # Assign consecutive ids to words.
    dat_word = dat_word.with_columns( \
        pl.Series(name='word_id', \
            values=list(range(len(dat_word))))
        )

    # Phone tier.
    dat_phon = dat \
        .filter(pl.col('tier') == "phones") \
        .sort(['filename', 'speaker', 'start'])

    # Assign non-decreasing word ids to phones.
    i = 0
    n = len(dat_phon)
    word_ids = [-1] * n
    for word in dat_word.iter_rows(named=True):
        # checkme: use filtering instead?
        while i < n:
            phon = dat_phon.row(i, named=True)
            if phon['filename'] != word['filename']:
                break
            if phon['speaker'] != word['speaker']:
                break
            if phon['start'] < word['word_start']:
                i += 1
                continue
            if phon['end'] <= word['word_end']:
                word_ids[i] = word['word_id']
                i += 1
                continue
            break

    dat_phon = dat_phon.with_columns( \
        pl.Series(name='word_id', values=word_ids))

    # Report phones with missing word ids.
    dat_miss = dat_phon.filter(pl.col('word_id') == -1)
    if len(dat_miss) > 0:
        print(f'Phones missing word ids ({len(dat_miss)}):')
        print(dat_miss)

    # Merge words and phones on word ids.
    #ret = dat_phon.join(dat_word, \
    #    on=['filename', 'speaker', 'word_id'])
    ret = dat_word.join(dat_phon, \
        on=['filename', 'speaker', 'word_id'])

    # Reorder tiers.
    #ret = ret[['filename', 'speaker', 'word_id', 'word', 'word_start', 'word_end', '']]

    return ret


def interval_at(dat, timepoint, speakers=None, tiers=None):
    """
    Data frame of intervals overlapping timepoint (inclusive) 
    for all speakers or specified speakers, on all tiers or 
    specified tiers.
        dat: polars dataframe
        timepoint (float): time point (s)
        speakers (list): speakers to include (None => all)
        tier (str): tiers to include (None => all)
    """
    dat1 = dat
    if speakers is not None:
        dat1 = dat1.filter(pl.col('speaker').is_in(speakers))
    if tiers is not none:
        dat1 = dat1.filter(pl.col('tier').is_in(tiers))
    dat1 = dat1.filter((pl.col('start') <= timepoint),
                       (pl.col('end') >= timepoint))
    return dat1


def intervals_between(dat, start, end, speakers=None, tier=None):
    """
    Data frame of intervals between timepoints (inclusive) 
    for all speakers or specified speakers, on all tiers or 
    specified tiers.
        dat: polars dataframe
        start (float): start time (s)
        end (float): end time (s)
        speakers (list): speakers to include (None => all)
        tier (str): tiers to include (None => all)
    """
    dat1 = dat
    if speakers is not None:
        dat1 = dat1.filter( \
            pl.col('speaker').is_in(speakers))
    if tier is not None:
        dat1 = dat1.filter( \
            pl.col('tier').is_in(tiers))
    dat1 = dat1.filter( \
        (pl.col('start') >= start),
        (pl.col('end') <= end))
    return dat1


def preceding(dat1, dat, skip=['sp', ''], max_sep=500.0):
    """
    Get preceding interval in dat, with same filename / speaker / 
    tier, for each interval in dat1. Skip designated labels and
    limit search by maximum separation (in ms).
    """
    dat = dat.filter(~pl.col('label').is_in(skip))
    missing = dat.clear(n=1)
    max_sep_s = max_sep / 1000.0

    dats = []
    for row in dat1.iter_rows(named=True):
        _dat = dat.filter( \
                pl.col('filename') == row['filename'],
                pl.col('speaker') == row['speaker'],
                pl.col('tier') == row['tier'],
                pl.col('end') <= row['start'],
                (row['start'] - pl.col('end')) <= max_sep_s
            )
        if len(_dat) == 0:
            dats.append(missing)
        else:
            dats.append(_dat.tail(1))

    dat0 = pl.concat(dats)
    return dat0


def following(dat1, dat, skip=['sp', ''], max_sep=500.0):
    """
    Get following interval in dat, with same filename / speaker /
    tier, for each interval in dat1. Skip designated labels and
    limit search by maximum separation (in ms).
    """
    dat = dat.filter(~pl.col('label').is_in(skip))
    missing = dat.clear(n=1)
    max_sep_s = max_sep / 1000.0

    dats = []
    for row in dat1.iter_rows(named=True):
        _dat = dat.filter( \
                pl.col('filename') == row['filename'],
                pl.col('speaker') == row['speaker'],
                pl.col('tier') == row['tier'],
                pl.col('start') >= row['end'],
                (row['end'] - pl.col('start')) <= max_sep_s
            )
        if len(_dat) == 0:
            dats.append(missing)
        else:
            dats.append(_dat.head(1))

    dat2 = pl.concat(dats)
    return dat2


# FIXME: convert to polars


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
        phons_contig['label'].str.contains(ARPABET_VOWELS))]
    if len(vowels_contig) == 0:
        return None
    speaking_rate = float(len(vowels_contig)) / float(window_dur)

    return speaking_rate


# Test.
def main():
    import matplotlib.pyplot as plt
    import seaborn as sns

    grid_file = Path.home() / \
        'Sounds/WhiteHousePressBriefings/data/mfa_out/11⧸20⧸23： Press Briefing by Press Secretary Karine Jean-Pierre [FYZztiGyz4g].TextGrid'
    grid = read(grid_file)

    # Tokens of 'the'.
    grid1 = grid.filter( \
        pl.col('tier') == 'words',
        pl.col('label') == 'the')
    print(grid1)

    # Following words.
    grid2 = following(grid1, grid)
    print(grid2)

    grid1 = grid1[[
        'filename', 'speaker', 'tier', 'label', 'start', 'end', 'dur_ms'
    ]]
    grid2 = grid2[['speaker', 'tier', 'label', 'start', 'end', 'dur_ms']]
    grid2.columns = ['_' + x for x in grid2.columns]

    grid12 = pl.concat([grid1, grid2], how='horizontal')
    print(grid12)

    sns.scatterplot(x='dur_ms', y='_dur_ms', data=grid12)
    plt.show()


if __name__ == '__main__':
    main()
