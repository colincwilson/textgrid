# Determiner duration by frequency of following word.

import sys
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from textgrid import textgrid

# # # # # # # # # #
# Subtlex frequencies.
subtlex_file = '~/Languages/00English/SUBTLEX_US/SUBTLEX-US_frequency_list_with_PoS_and_Zipf_information.csv'
subtlex = pl.read_csv(subtlex_file, ignore_errors=True)
subtlex = subtlex[['Word', 'SUBTLWF', 'Dom_PoS_SUBTLEX']]
subtlex_noun = subtlex \
    .filter(pl.col('Dom_PoS_SUBTLEX') == 'Noun') \
    .drop('Dom_PoS_SUBTLEX')
print(subtlex_noun)


# # # # # # # # # #
# Determiner + Noun data.
def process(grid_file, det='the'):
    grid_df = textgrid.read(grid_file)
    #print(dat.columns)

    # Combine word and phone tiers.
    word_df = textgrid.combine_tiers(grid_df)

    # Tokens of 'the'.
    the_df = grid_df.filter( \
        pl.col('tier') == 'words',
        pl.col('label') == det)

    # Preceding words.
    prec1_df = textgrid.preceding(the_df, grid_df)
    prec2_df = textgrid.preceding(prec1_df, grid_df)

    # Following words.
    succ1_df = textgrid.following(the_df, grid_df)
    succ2_df = textgrid.following(succ1_df, grid_df)

    the_df = the_df.with_columns( \
        word_prev2=prec2_df['label'],
        word_dur_ms_prev2=prec2_df['dur_ms'],
        word_prev1=prec1_df['label'],
        word_dur_ms_prev1=prec1_df['dur_ms'],
        word_next1=succ1_df['label'],
        word_dur_ms_next1=succ1_df['dur_ms'],
        word_next2=succ2_df['label'],
        word_dur_ms_next2=succ2_df['dur_ms'])

    # Speaking rate before each 'the'.
    the_df = textgrid.speaking_rate( \
        the_df, grid_df, side='before')

    # Merge with words+phones combo.
    the_df = the_df \
        .drop('tier') \
        .rename(
            {'label': 'word',
            'start': 'word_start',
            'end': 'word_end',
            'dur_ms': 'word_dur_ms'}) \
        .join( \
            word_df,
            on = ['filename', 'speaker', 'word', 'word_start', 'word_end', 'word_dur_ms'])

    # Merge with subtlex nouns.
    the_df = the_df.join( \
        subtlex_noun,
        left_on = ['word_next1'],
        right_on = ['Word'])

    print(the_df.head(10))
    print(the_df.columns)

    # Select vowels.
    the_df = the_df.filter( \
        pl.col('phone').str.contains('[012]'))

    return the_df


# # # # # # # # # #
# Grid.
grid_files = (
    Path.home() /
    'Sounds/WhiteHousePressBriefings/media/data2/mfa_align').glob('*.TextGrid')
grid_files = list(grid_files)[1:]
print(list(grid_files))

# grid_file = Path.home() / \
#     'Sounds/WhiteHousePressBriefings/data2/mfa_align/11⧸20⧸23： Press Briefing by Press Secretary Karine Jean-Pierre [FYZztiGyz4g].TextGrid'

the_df = None
for grid_file in grid_files:
    the_df1 = process(grid_file)
    if the_df is None:
        the_df = the_df1
        continue
    try:
        the_df = pl.concat([the_df, the_df1])
    except:
        print(f'Error on {grid_file}')

#dat_the = pl.concat(dat_the1)
the_df.write_csv('~/Downloads/tmp.csv')

# sys.exit(0)

# print(dat_the.select(pl.corr('SUBTLWF', 'dur_ms')))
# sns.relplot(dat_the, x='SUBTLWF', y='dur_ms')
# plt.show()

# print(dat_the)
# print(dat_the.columns)

# print(dat_the['phone'].value_counts())

# print(dat_the[[
#     'word', 'word_prev2', 'word_prev1', 'word_next1', 'word_next2', 'phone'
# ]].head())

# #print(dat_the.filter(pl.col('word_prev1') == 'of'))
