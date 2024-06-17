# Praat textgrids represented as polars dataframes and
# networkx graphs.
# Notes:
# - Time units are in seconds (s) unless otherwise
#   indicated as milliseconds (ms).
#

import re, sys
import numpy as np
from pathlib import Path
import polars as pl
import tgt

# # # # # # # # # #

# Regexes.
ARPABET_VOWELS = '^[AEIOUaeiou].[012]?$'
#ARPABET_VOWELS_STRESS = '^[AEIOUaeiou].[012]$'
MFA_TIERS = '^(.+?)( - utterance)? - ((phone|word|utterance)s?)$'

# # # # # # # # # #


def read(filename, fileEncoding="utf-8", verbose=True):
    """
    Read TextGrid into polars data frame.
    # todo: round timestamps to two decimal places
    """
    filename = str(filename)
    grid = tgt.io.read_textgrid(filename)
    dat = []
    for tier in grid.tiers:
        tier_name = tier.name
        if re.search(MFA_TIERS, tier_name):
            speaker = re.sub(MFA_TIERS, '\\1', tier_name)
            tier_name = re.sub(MFA_TIERS, '\\3', tier_name)
        else:
            speaker = '<null>'

        for interval in tier.annotations:
            dat.append((speaker, tier_name,  \
                interval.text, interval.start_time, interval.end_time))

    dat = pl.DataFrame(dat, \
        schema=['speaker', 'tier', 'label', 'start', 'end'])
    dat = dat \
        .with_columns( \
            filename=pl.lit(filename),
            dur_ms=1000.0*(pl.col('end') - pl.col('start'))) \
        .with_columns( \
            pl.col('start').round(3),
            pl.col('end').round(3),
            pl.col('dur_ms').round(0))

    dat = dat[[ \
        'filename', 'speaker', 'tier', 'label',
        'start', 'end', 'dur_ms']]
    dat = dat.sort(['start', 'speaker'])

    if verbose:
        speakers = dat['speaker'].unique().to_list()
        tiers = dat['tier'].unique().to_list()
        n = len(dat)

        print(f'\nTextgrid {filename}\n'
              f'* Speakers: {speakers}\n'
              f'* Tiers: {tiers}\n')
        print(f'dat {dat}')
        print()

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

    # checkme: format
    tgt.io.write_to_file(grid, filename, format='long')
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
        interval = tgt.core.Interval( \
            row['start'],
            row['end'],
            row['label'])
        try:
            tier.add_interval(interval)
        except:
            print(f'Error adding interval {interval}')
    return tier


def combine_tiers(dat):
    """
    Combine word and phone tiers.
    """
    # Word tier.
    dat_word = dat \
        .filter(pl.col('tier') == 'words') \
        .rename({'label': 'word', 'start': 'word_start',
        'end': 'word_end', 'dur_ms': 'word_dur_ms'}) \
        .drop(['tier', 'label']) \
        .sort(['filename', 'word_start'])

    # Assign consecutive ids to words.
    dat_word = dat_word.with_columns( \
        pl.Series(name='word_id', \
            values=list(range(len(dat_word)))))

    # Phone tier.
    dat_phon = dat \
        .filter(pl.col('tier') == 'phones') \
        .rename({'label': 'phone'}) \
        .drop(['tier']) \
        .sort(['filename', 'start'])

    # Assign non-decreasing word ids to phones.
    # todo: index phones within words
    i = 0
    n = len(dat_phon)
    word_ids = [-1] * n
    for word in dat_word.iter_rows(named=True):
        word_id = word['word_id']
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
                word_ids[i] = word_id
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
    ret = dat_word \
        .join(dat_phon, \
            on=['filename', 'speaker', 'word_id']) \
        .sort(['filename', 'word_start', 'start'])

    # Reorder columns.
    ret = ret[['filename', 'speaker', 'word', 'word_id', \
               'word_start', 'word_end', 'word_dur_ms', \
               'phone', 'start', 'end', 'dur_ms']]

    return ret


def intervals_at(dat, timepoint, speakers=None, tiers=None):
    """
    Data frame of intervals overlapping timepoint (inclusive) 
    for all speakers or specified speakers, on all tiers or 
    specified tiers.
        dat: polars dataframe
        timepoint (float): time point
        speakers (list): speakers to include (None => all)
        tier (str): tiers to include (None => all)
    """
    dat1 = dat
    if speakers is not None:
        dat1 = dat1.filter(pl.col('speaker').is_in(speakers))
    if tiers is not None:
        dat1 = dat1.filter(pl.col('tier').is_in(tiers))
    dat1 = dat1.filter((pl.col('start') <= timepoint),
                       (pl.col('end') >= timepoint))
    return dat1


def intervals_between(dat, start, end, speakers=None, tiers=None):
    """
    Data frame of intervals between timepoints (inclusive) 
    for all speakers or specified speakers, on all tiers or 
    specified tiers.
        dat: polars dataframe
        start (float): start time
        end (float): end time
        speakers (list): speakers to include (None => all)
        tiers (str): tiers to include (None => all)
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


def preceding(dat1, dat, \
    tier=None, skip=['sp', ''], pattern='', max_sep=500.0):
    """
    Get closest preceding interval (row) in dat with the same 
    filename, same speaker, and (by default) same tier for each 
    interval (row) in dat1. Optionally specify tier of matches. 
    Skip designated labels, include only labels that match 
    pattern, and limit search by maximum separation [ms].
    """
    if tier is not None:
        dat = dat.filter(pl.col('tier') == tier)

    dat = dat.filter( \
        ~pl.col('label').is_in(skip),
        pl.col('label').str.contains(pattern))

    dats = []
    missing = dat.clear(n=1)
    max_sep_s = max_sep / 1000.0
    for row in dat1.iter_rows(named=True):
        if (row['filename'] is None) or (row['speaker'] is None) \
            or (row['tier'] is None) or (row['start'] is None):
            dats.append(missing)
            continue

        dat_ = dat.filter( \
            pl.col('filename') == row['filename'],
            pl.col('speaker') == row['speaker'],
            pl.col('start') <= row['start'],
            pl.col('start') >= (row['start'] - max_sep_s))

        if tier is None:
            dat_ = dat_.filter( \
                pl.col('tier') == row['tier'])

        if len(dat_) == 0:
            dats.append(missing)
        else:
            dats.append(dat_.tail(1))

    ret = pl.concat(dats)
    return ret


def following(dat1, dat, \
    tier=None, skip=['sp', ''], pattern='', max_sep=500.0):
    """
    Get closest following interval (row) in dat with the same 
    filename, same speaker, and (by default) same tier for each 
    interval (row) in dat1. Optionally specify tier of matches. 
    Skip designated labels, include only labels that match 
    pattern, and limit search by maximum separation [ms].
    """
    if tier is not None:
        dat = dat.filter(pl.col('tier') == tier)

    dat = dat.filter( \
        ~pl.col('label').is_in(skip),
        pl.col('label').str.contains(pattern))

    dats = []
    missing = dat.clear(n=1)
    max_sep_s = max_sep / 1000.0
    for row in dat1.iter_rows(named=True):
        if (row['filename'] is None) or (row['speaker'] is None) \
            or (row['tier'] is None) or (row['end'] is None):
            dats.append(missing)
            continue

        dat_ = dat.filter( \
            pl.col('filename') == row['filename'],
            pl.col('speaker') == row['speaker'],
            pl.col('end') >= row['end'],
            pl.col('end') <= (row['end'] + max_sep_s))

        if tier is None:
            dat_ = dat_.filter(pl.col('tier') == row['tier'])

        if len(dat_) == 0:
            dats.append(missing)
        else:
            dats.append(dat_.head(1))

    ret = pl.concat(dats)
    return ret


def speaking_rate(dat1, dat, window=1000.0, side='before'):
    """
    Local speaking rate (vowels/second) in dat, before or 
    after each interval (row) in dat1, within window of
    specified duration [ms].
    """
    if side == 'before':
        return speaking_rate_before(dat1, dat, window)
    if side == 'after':
        return speaking_rate_after(dat1, dat, window)
    return None


def speaking_rate_before(dat1, dat, window=1000.0):
    """
    Speaking rate (vowels/second) in dat before each 
    interval (row) in dat1 within specified window.
        window: duration of window (default = 1000.0) [ms]
    """
    dat = dat.filter( \
        pl.col('tier') == 'phones',
        pl.col('label').str.contains(ARPABET_VOWELS))

    val = []
    window_s = window / 1000.0
    for row in dat1.iter_rows(named=True):
        dat_ = dat.filter( \
                pl.col('filename') == row['filename'],
                pl.col('speaker') == row['speaker'],
                pl.col('end') <= row['start'],
                (row['start'] - pl.col('start')) <= window_s)
        n = len(dat_)
        if n == 0:
            val.append(np.nan)
        else:
            val.append(1000.0 * float(n) / window)

    dat1 = dat1.with_columns( \
        pl.Series(name='rate_before', values=val))

    return dat1


def speaking_rate_after(dat1, dat, window=1000.0):
    """
    Speaking rate (vowels/second) in dat after each
    interval (row) in dat1 within specified window.
        window: duration of window (default = 1000.0) [ms]
    """
    dat = dat.filter( \
        pl.col('tier') == 'phones',
        pl.col('label').str.contains(ARPABET_VOWELS))

    val = []
    window_s = window / 1000.0
    for row in dat1.iter_rows(named=True):
        dat_ = dat.filter( \
                pl.col('filename') == row['filename'],
                pl.col('speaker') == row['speaker'],
                pl.col('start') >= row['end'],
                (pl.col('end') - row['end']) <= window_s)
        n = len(dat_)
        if n == 0:
            val.append(np.nan)
        else:
            val.append(float(n) / window_s)

    dat1 = dat1.with_columns( \
        pl.Series(name='rate_after', values=val))

    return dat1


# Test.
def main():
    import matplotlib.pyplot as plt
    import seaborn as sns

    grid_file = Path.home() / \
        'Sounds/WhiteHousePressBriefings/data2/mfa_out/11⧸20⧸23： Press Briefing by Press Secretary Karine Jean-Pierre [FYZztiGyz4g].TextGrid'
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
        'filename', 'speaker', 'tier', 'label', 'start_ms', 'end_ms', 'dur_ms'
    ]]
    grid2 = grid2[['speaker', 'tier', 'label', 'start_ms', 'end_ms', 'dur_ms']]
    grid2.columns = ['_' + x for x in grid2.columns]

    grid12 = pl.concat([grid1, grid2], how='horizontal')
    print(grid12)

    sns.scatterplot(x='dur_ms', y='_dur_ms', data=grid12)
    plt.show()


if __name__ == '__main__':
    main()
