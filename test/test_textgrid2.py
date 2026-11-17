import sys
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from textgrid import textgrid

grid_file = Path.home() / \
    'Sounds/WhiteHousePressBriefings/example2/11⧸20⧸23： Press Briefing by Press Secretary Karine Jean-Pierre [FYZztiGyz4g].TextGrid'

grid_df = textgrid.read(grid_file)

# Combine word and phone tiers.
grid_combo = textgrid.combine_tiers(grid_df)
#print(grid_combo.columns)
print(grid_combo)

# Tokens of 'the'.
grid_df1 = grid_df.filter( \
    pl.col('tier') == 'words',
    pl.col('label') == 'the')
print(grid_df1)

# Preceding words.
grid_df0 = textgrid.preceding(grid_df1, grid_df)
print(grid_df0)

# Following words.
grid_df2 = textgrid.following(grid_df1, grid_df)
print(grid_df2)

# Speaking rate before.
grid_df3 = textgrid.speaking_rate(grid_df, grid_df, side='before')
print(grid_df3)

# Speaking rate after.
grid_df4 = textgrid.speaking_rate(grid_df3, grid_df, side='after')
print(len(grid_df4))
print(grid_df4)

grid_df4.write_csv('~/Downloads/tmp.csv')

sns.relplot(grid_df4, x='rate_before', y='rate_after')
plt.show()
