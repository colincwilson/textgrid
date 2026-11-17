# Praat textgrid represented as a Polars dataframe.
# Notes:
# - Time units are in seconds (s) unless otherwise
#   specified as in milliseconds (ms).
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
WORD_TIER = '^word(s)?$'
PHON_TIER = '^phon(e)?(s)?$'

# # # # # # # # # #


def read(filename, encoding="utf-8", verbose=True):
    """
    Read TextGrid into polars data frame.
    # todo: round timestamps to two decimal places
    """
    filename = str(filename)
    grid = tgt.io.read_textgrid(filename, encoding=encoding)
    grid_df = []
    for tier in grid.tiers:
        tier_name = tier.name
        if re.search(MFA_TIERS, tier_name):
            speaker = re.sub(MFA_TIERS, '\\1', tier_name)
            tier_name = re.sub(MFA_TIERS, '\\3', tier_name)
        else:
            speaker = '<null>'

        for interval in tier.annotations:
            grid_df.append((speaker, tier_name,  \
                interval.text, interval.start_time, interval.end_time))

    grid_df = pl.DataFrame(grid_df, \
        schema=['speaker', 'tier', 'label', 'start', 'end'],
        orient='row')

    grid_df = grid_df \
        .with_columns( \
            filename=pl.lit(filename),
            dur_ms=1000.0*(pl.col('end') - pl.col('start'))) \
        .with_columns( \
            pl.col('start').round(3),
            pl.col('end').round(3),
            pl.col('dur_ms').round(0))

    grid_df = grid_df[[ \
        'filename', 'speaker', 'tier', 'label',
        'start', 'end', 'dur_ms']]
    grid_df = grid_df.sort(['start', 'speaker'])

    if verbose:
        speakers = grid_df['speaker'].unique().to_list()
        tiers = grid_df['tier'].unique().to_list()
        n = len(grid_df)

        print()
        print('-' * 20)
        print(f'* Textgrid: {filename}\n'
              f'* Speakers: {speakers}\n'
              f'* Tiers: {tiers}\n')
        print(f'* Grid DF:\n{grid_df}')
        print()

    return grid_df


# Alias.
read_textgrid = import_textgrid = read


def write(grid_df, filename, speakers=None, tiers=None):
    """
    Write polars data frame as TextGrid.
        filename (str/path): output file
        speakers (list): speakers to include (None => all)
        tiers (list): tiers to include (None => all)
    """

    # Make textgrid.
    start_time = 0.0
    end_time = grid_df['end'].max()
    grid = tgt.core.TextGrid(filename)

    # Make tiers.
    tiers = _make_tiers(grid_df, speakers, tiers)
    for tier in tiers:
        tier.start_time = start_time
        tier.end_time = end_time
        grid.add_tier(tier)

    # checkme: format
    tgt.io.write_to_file(grid, filename, format='long')
    return grid


# Alias.
write_textgrid = export_textgrid = write


def _make_tiers(grid_df, speakers=None, tiers=None):
    """ Make tier for each speaker x tier combo. """
    if speakers is None:
        if 'speaker' in grid_df.columns:
            speakers = grid_df['speaker'].unique(maintain_order=True)
        else:
            speakers = ['<null>']
    if tiers is None:
        if 'tier' in grid_df.columns:
            tiers = grid_df['tier'].unique(maintain_order=True)
        else:
            tiers = ['label']
    ret = []
    for speaker in speakers:
        spkr = grid_df
        if 'speaker' in grid_df.columns:
            spkr_df = grid_df.filter(pl.col('speaker') == speaker)
        for tier in tiers:
            spkr_tier_df = spkr_df
            if 'tier' in grid_df.columns:
                spkr_tier_df = spkr_df.filter(pl.col('tier') == tier)
            if len(spkr_tier_df) == 0:
                continue
            ret.append(_make_tier(spkr_tier_df, speaker, tier))
    return ret


def _make_tier(grid_df, speaker, tier):
    """ Make tier for one speaker x tier combo. """
    if speaker == '<null>':
        tier_name = tier
    else:
        tier_name = f'{speaker} - {tier}'
    tier = tgt.core.IntervalTier(name=tier_name)
    for row in grid_df.rows(named=True):
        interval = tgt.core.Interval( \
            row['start'],
            row['end'],
            row['label'])
        try:
            tier.add_interval(interval)
        except:
            print(f'Error adding interval {interval}')
    return tier


def combine_tiers(grid_df, verbose=False):
    """
    Combine word and phone tiers.
    """
    # Word tier.
    word_df = grid_df \
        .filter(pl.col('tier').str.contains(WORD_TIER)) \
        .rename({'label': 'word', 'start': 'word_start',
        'end': 'word_end', 'dur_ms': 'word_dur_ms'}) \
        .drop(['tier']) \
        .sort(['filename', 'word_start'])

    # Assign consecutive ids to words.
    word_df = word_df.with_columns( \
        pl.Series(name='word_id', \
            values=list(range(len(word_df)))))
    if verbose:
        print(word_df.head())

    # Phone tier.
    phon_df = grid_df \
        .filter(pl.col('tier').str.contains(PHON_TIER)) \
        .rename({'label': 'phone'}) \
        .drop(['tier']) \
        .sort(['filename', 'start'])

    # Merge words and phones on filename, speaker, and time overlap.
    ret = phon_df.join_where( \
        word_df,
        pl.col('filename') == pl.col('filename'),
        pl.col('speaker') == pl.col('speaker'),
        pl.col('start') >= pl.col('word_start'),
        pl.col('end') <= pl.col('word_end')
    )

    # # Report phones with missing word ids.
    # dat_miss = dat_phon.filter(pl.col('word_id') == -1)
    # if len(dat_miss) > 0:
    #     print(f'Phones missing word ids ({len(dat_miss)}):')
    #     print(dat_miss)

    # # Merge words and phones on word ids.
    # ret = dat_word \
    #     .join(dat_phon, \
    #         on=['filename', 'speaker', 'word_id']) \
    #     .sort(['filename', 'word_start', 'start'])

    # Reorder columns.
    ret = ret[['filename', 'speaker', 'word', 'word_id', \
               'word_start', 'word_end', 'word_dur_ms', \
               'phone', 'start', 'end', 'dur_ms']]

    # Sort rows by filename, speaker, word start, and phone start.
    ret = ret.sort(['filename', 'speaker', 'word_start', 'start'])

    return ret


def intervals_at(grid_df, timepoint, speakers=None, tiers=None):
    """
    Data frame of intervals overlapping timepoint (inclusive) 
    for all speakers or specified speakers, on all tiers or 
    specified tiers.
        grid_df: polars dataframe
        timepoint (float): time point
        speakers (list): speakers to include (None => all)
        tier (str): tiers to include (None => all)
    """
    grid_df1 = grid_df
    if speakers is not None:
        grid_df1 = grid_df1.filter(pl.col('speaker').is_in(speakers))
    if tiers is not None:
        grid_df1 = grid_df1.filter(pl.col('tier').is_in(tiers))
    grid_df1 = grid_df1.filter((pl.col('start') <= timepoint),
                               (pl.col('end') >= timepoint))
    return grid_df1


def intervals_between(grid_df, start, end, speakers=None, tiers=None):
    """
    Data frame of intervals between timepoints (inclusive) 
    for all speakers or specified speakers, on all tiers or 
    specified tiers.
        grid_df: polars dataframe
        start (float): start time
        end (float): end time
        speakers (list): speakers to include (None => all)
        tiers (str): tiers to include (None => all)
    """
    grid_df1 = grid_df
    if speakers is not None:
        grid_df1 = grid_df1.filter( \
            pl.col('speaker').is_in(speakers))
    if tier is not None:
        grid_df1 = grid_df1.filter( \
            pl.col('tier').is_in(tiers))
    grid_df1 = grid_df1.filter( \
        (pl.col('start') >= start),
        (pl.col('end') <= end))
    return grid_df1


def preceding(grid_df1, grid_df2, \
    tier=None, skip=['sp', ''], pattern='', max_sep=500.0):
    """
    For each interval (row) in grid_df1, get closest preceding interval
    (row) in grid_df2 with matching filename, speaker, and (by default)
    tier. Optionally specify tier of matches.
    Skip designated labels, include only labels that match
    pattern, and limit search by maximum separation [ms].
    todo: use polars join_asof() with strategy="nearest"
    """
    if tier is not None:
        grid_df2 = grid_df2.filter(pl.col('tier') == tier)

    grid_df2 = grid_df2.filter( \
        ~pl.col('label').is_in(skip),
        pl.col('label').str.contains(pattern))

    grid_df = []
    missing = grid_df2.clear(n=1)
    max_sep_s = max_sep / 1000.0
    for row in grid_df1.iter_rows(named=True):
        if (row['filename'] is None) or (row['speaker'] is None) \
            or (row['tier'] is None) or (row['start'] is None):
            dats.append(missing)
            continue

        grid_df2_ = grid_df2.filter( \
            pl.col('filename') == row['filename'],
            pl.col('speaker') == row['speaker'],
            pl.col('start') <= row['start'],
            pl.col('start') >= (row['start'] - max_sep_s))

        if tier is None:
            grid_df2_ = grid_df2_.filter( \
                pl.col('tier') == row['tier'])

        if len(grid_df2_) == 0:
            grid_df.append(missing)
        else:
            grid_df.append(grid_df2_.tail(1))

    ret = pl.concat(grid_df)
    return ret


def following(grid_df1, grid_df2, \
    tier=None, skip=['sp', ''], pattern='', max_sep=500.0):
    """
    For each interval (row) in grid_df1, get closest following interval
    (row) in grid_df2 with matching filename, speaker, and (by default) tier. Optionally specify tier of matches.
    Skip designated labels, include only labels that match 
    pattern, and limit search by maximum separation [ms].
    todo: use polars join_asof() with strategy="nearest"
    """
    if tier is not None:
        grid_df2 = grid_df2.filter(pl.col('tier') == tier)

    grid_df2 = grid_df2.filter( \
        ~pl.col('label').is_in(skip),
        pl.col('label').str.contains(pattern))

    grid_df = []
    missing = grid_df2.clear(n=1)
    max_sep_s = max_sep / 1000.0
    for row in grid_df1.iter_rows(named=True):
        if (row['filename'] is None) or (row['speaker'] is None) \
            or (row['tier'] is None) or (row['end'] is None):
            grid_df.append(missing)
            continue

        grid_df2_ = grid_df2.filter( \
            pl.col('filename') == row['filename'],
            pl.col('speaker') == row['speaker'],
            pl.col('end') >= row['end'],
            pl.col('end') <= (row['end'] + max_sep_s))

        if tier is None:
            grid_df2_ = grid_df2_.filter(pl.col('tier') == row['tier'])N)) 

        if len(grid_df2_) == 0:
            grid_df.append(missing)
        else:
            grid_df.append(grid_df2_.head(1))

    ret = pl.concat(grid_df)
    return ret


def preceding_word(grid_df, word_tier='word'):
    """
    Code preceding words with matching filename and speaker.
    Apply to result of combine_tiers().
    """
    word_prec = f'{word_tier}_prec'

    grid_df = grid_df.with_columns(
        pl.col(word_tier).shift(1).alias(word_prec))

    grid_df = grid_df.with_columns(
        pl.when((pl.col('filename') == pl.col('filename').shift(1)) & (pl.col('speaker') == pl.col('speaker').shift(1))) \
            .then(pl.col(word_prec)) \
            .otherwise(pl.lit(None)) \
            .alias(word_prec))
    return grid_df


def following_word(grid_df, word_tier='word'):
    """
    Code following words with matching filename and speaker.
    Apply to result of combine_tiers().
    """
    word_succ = f'{word_tier}_succ'

    grid_df = grid_df.with_columns(
        pl.col(word_tier).shift(-1).alias(word_succ))

    grid_df = grid_df.with_columns(
        pl.when((pl.col('filename') == pl.col('filename').shift(-1)) & (pl.col('speaker') == pl.col('speaker').shift(-1))) \
            .then(pl.col(word_succ)) \
            .otherwise(pl.lit(None)) \
            .alias(word_succ))
    return grid_df


def preceeding_phon(grid_df, phon_tier='phone'):
    """
    Code preceding phones with matching filename and speaker, marking
    that they belong to different words with '#'.
    Apply to result of combine_tiers().
    """
    phon_prec = f'{phon_tier}_prec'

    grid_df = grid_df.with_columns( \
        pl.col(phon_tier).shift(1).alias(phon_prec))

    grid_df = grid_df.with_columns(
        pl.when(pl.col('word_id') == pl.col('word_id').shift(1)) \
            .then(pl.col(phon_prec)) \
            .otherwise(pl.col(phon_prec) + ' #') \
            .alias(phon_prec))

    grid_df = grid_df.with_columns(
        pl.when(pl.when((pl.col('filename') == pl.col('filename').shift(1)) & (pl.col('speaker') == pl.col('speaker').shift(1)))) \
            .then(pl.col(phon_prec)) \
            .otherwise(pl.lit(None)) \
            .alias(phon_prec))

    return grid_df


def following_phon(grid_df, phon_tier='phone'):
    """
    Code following phones, indicating whether they belong
    to the same word with '#'. Apply to result of combine_tiers().
    """
    phon_succ = f'{phon_tier}_succ'

    grid_df = grid_df.with_columns(pl.col('phone').shift(-1).alias(phon_suc))

    grid_df = grid_df.with_columns(
        pl.when(pl.col('word_id') == pl.col('word_id').shift(-1)) \
            .then(phon_succ) \
            .otherwise('# '+ pl.col(phon_succ)) \
            .alias(phon_succ))

    grid_df = grid_df.with_columns(
        pl.when(pl.when((pl.col('filename') == pl.col('filename').shift(-1)) & (pl.col('speaker') == pl.col('speaker').shift(-1)))) \
            .then(pl.col(phon_succ)) \
            .otherwise(pl.lit(None)) \
            .alias(phon_succ))

    return grid_df


def speaking_rate(grid_df1, grid_df2, window=1000.0, side='before'):
    """
    Local speaking rate (vowels/second) before or after
    each interval (row) in grid_df1 based on vowel count
    in grid_df2 within window of specified duration [ms].
    todo: reuse grid_df1 if grid_df2 is unspecified
    """
    if side == 'before':
        return speaking_rate_before(grid_df1, grid_df2, window)
    if side == 'after':
        return speaking_rate_after(grid_df1, grid_df2, window)
    return None


def speaking_rate_before(grid_df1, grid_df2, window=1000.0):
    """
    Local speaking rate (vowels/second) before each
    interval (row) in grid_df1 calculated from vowel count
    in grid_df2 within window of specified duration [ms].
        window: duration of window (default = 1000.0) [ms]
    """
    grid_df2 = grid_df2.filter( \
        pl.col('tier').str.contains(PHON_TIER),
        pl.col('label').str.contains(ARPABET_VOWELS))

    val = []
    window_s = window / 1000.0
    for row in grid_df1.iter_rows(named=True):
        grid_df2_ = grid_df2.filter( \
                pl.col('filename') == row['filename'],
                pl.col('speaker') == row['speaker'],
                pl.col('end') <= row['start'],
                (row['start'] - pl.col('start')) <= window_s)
        n = len(grid_df2_)
        if n == 0:
            val.append(np.nan)
        else:
            val.append(1000.0 * float(n) / window)

    grid_df1 = grid_df1.with_columns( \
        pl.Series(name='rate_before', values=val))

    return grid_df1


def speaking_rate_after(grid_df1, grid_df2, window=1000.0):
    """
    Local speaking rate (vowels/second) after each
    interval (row) in grid_df1 calculated from vowel count
    in grid_df2 within window of specified duration [ms].
        window: duration of window (default = 1000.0) [ms]
    """
    grid_df2 = grid_df2.filter( \
        pl.col('tier').str.contains(PHON_TIER),
        pl.col('label').str.contains(ARPABET_VOWELS))

    val = []
    window_s = window / 1000.0
    for row in grid_df1.iter_rows(named=True):
        grid_df2_ = grid_df2.filter( \
                pl.col('filename') == row['filename'],
                pl.col('speaker') == row['speaker'],
                pl.col('start') >= row['end'],
                (pl.col('end') - row['end']) <= window_s)
        n = len(grid_df2_)
        if n == 0:
            val.append(np.nan)
        else:
            val.append(float(n) / window_s)

    grid_df1 = grid_df1.with_columns( \
        pl.Series(name='rate_after', values=val))

    return grid_df1


# Test.
def main():
    import matplotlib.pyplot as plt
    import seaborn as sns

    grid_file = Path.home() / \
        'Sounds/WhiteHousePressBriefings/example2/11⧸20⧸23： Press Briefing by Press Secretary Karine Jean-Pierre [FYZztiGyz4g].TextGrid'
    grid_df = read(grid_file)

    # Tokens of 'the'.
    grid_df1 = grid_df.filter( \
        pl.col('tier') == 'words',
        pl.col('label') == 'the')

    # Following words.
    grid_df2 = following(grid_df1, grid_df)
    print(grid_df2)

    grid_df1 = grid_df1[[ \
        'filename', 'speaker', 'tier', 'label',
        'start', 'end', 'dur_ms']]

    grid_df2 = grid_df2[[ \
        'speaker', 'tier', 'label',
        'start', 'end', 'dur_ms']]
    grid_df2.columns = ['_' + x for x in grid_df2.columns]  # fixme

    grid_df12 = pl.concat([grid_df1, grid_df2], how='horizontal_extend')
    print(grid_df12)

    sns.scatterplot(x='dur_ms', y='_dur_ms', data=grid_df12)
    plt.show()


if __name__ == '__main__':
    main()
