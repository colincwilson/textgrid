# Determiner duration by frequency of following word.

import sys
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import textgrid

# # # # # # # # # #
# Subtlex frequencies.
subtlex_file = '~/Languages/00English/SUBTLEX_US/SUBTLEX-US_frequency_list_with_PoS_and_Zipf_information.csv'
subtlex = pl.read_csv(subtlex_file, ignore_errors=True)
subtlex = subtlex[['Word', 'SUBTLWF', 'Dom_PoS_SUBTLEX']]
subtlex_noun = subtlex.filter( \
    pl.col('Dom_PoS_SUBTLEX') == 'Noun')
print(subtlex_noun)

# # # # # # # # # #
# Grid.
grid_file = Path.home() / \
    'Sounds/WhiteHousePressBriefings/data2/mfa_align/11⧸20⧸23： Press Briefing by Press Secretary Karine Jean-Pierre [FYZztiGyz4g].TextGrid'

dat = textgrid.read(grid_file)
#print(dat.columns)

# Combine word and phone tiers.
dat_combo = textgrid.combine_tiers(dat)
#print(dat_combo)
#print(dat_combo.columns)

# Tokens of 'the'.
dat_the = dat.filter( \
    pl.col('tier') == 'words',
    pl.col('label') == 'the')
#print(dat_the)

# Preceding words.
dat_prev1 = textgrid.preceding(dat_the, dat)
dat_prev2 = textgrid.preceding(dat_prev1, dat)

# Following words.
dat_next1 = textgrid.following(dat_the, dat)
dat_next2 = textgrid.following(dat_next1, dat)

dat_the = dat_the.with_columns( \
    word_prev2=dat_prev2['label'],
    word_dur_ms_prev2=dat_prev2['dur_ms'],
    word_prev1=dat_prev1['label'],
    word_dur_ms_prev1=dat_prev1['dur_ms'],
    word_next1=dat_next1['label'],
    word_dur_ms_next1=dat_next1['dur_ms'],
    word_next2=dat_next2['label'],
    word_dur_ms_next2=dat_next2['dur_ms'])
#print(dat_the)
#print(dat_the.columns)

# Speaking rate before each 'the'.
dat_the = textgrid.speaking_rate( \
    dat_the, dat, side='before')

# print(dat_the)
# print(dat_the.columns)
# print(len(dat_the))

# Merge with combo.
dat_the = dat_the \
    .rename({'label': 'word', 'start': 'word_start', 'end': 'word_end', 'dur_ms': 'word_dur_ms'}) \
    .drop('tier')

# print(dat_the)
# print(dat_the.columns)

dat_the = dat_the.join( \
    dat_combo,
    on = ['filename', 'speaker', 'word', 'word_start', 'word_end', 'word_dur_ms'])

# Merge with subtlex nouns.
dat_the = dat_the.join( \
    subtlex_noun,
    left_on = ['word_next1'],
    right_on = ['Word'])

# Select vowels.
dat_the = dat_the.filter( \
    pl.col('phone').str.contains('[012]'))

print(len(dat_the))

dat_the.write_csv('~/Downloads/tmp.csv')

sys.exit(0)

print(dat_the.select(pl.corr('SUBTLWF', 'dur_ms')))
sns.relplot(dat_the, x='SUBTLWF', y='dur_ms')
plt.show()

print(dat_the)
print(dat_the.columns)

print(dat_the['phone'].value_counts())

print(dat_the[[
    'word', 'word_prev2', 'word_prev1', 'word_next1', 'word_next2', 'phone'
]].head())

#print(dat_the.filter(pl.col('word_prev1') == 'of'))
