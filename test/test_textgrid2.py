import sys
import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

import textgrid

grid_file = Path.home() / \
    'Sounds/WhiteHousePressBriefings/data2/mfa_align/11⧸20⧸23： Press Briefing by Press Secretary Karine Jean-Pierre [FYZztiGyz4g].TextGrid'

grid = textgrid.read(grid_file)

# Combine word and phone tiers.
grid_combo = textgrid.combine_tiers(grid)
#print(grid_combo.columns)
print(grid_combo)

# Tokens of 'the'.
grid1 = grid.filter( \
    pl.col('tier') == 'words',
    pl.col('label') == 'the')
print(grid1)

# Preceding words.
grid0 = textgrid.preceding(grid1, grid)
print(grid0)

# Following words.
grid2 = textgrid.following(grid1, grid)
print(grid2)

# Speaking rate before.
grid3 = textgrid.speaking_rate(grid, grid, side='before')
print(grid3)

# Speaking rate after.
grid4 = textgrid.speaking_rate(grid3, grid, side='after')
print(len(grid4))
print(grid4)

grid4.write_csv('~/Downloads/tmp.csv')

sns.relplot(grid4, x='rate_before', y='rate_after')
plt.show()
