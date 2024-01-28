import sys
import polars as pl
from pathlib import Path

import textgrid

grid_file = Path.home() / \
    'Sounds/WhiteHousePressBriefings/data/mfa_align/11⧸20⧸23： Press Briefing by Press Secretary Karine Jean-Pierre [FYZztiGyz4g].TextGrid'

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
grid3 = textgrid.speaking_rate(grid1, grid, side='before')
print(grid3)

# Speaking rate after.
grid4 = textgrid.speaking_rate(grid1, grid, side='after')
print(grid4)

#print(list(zip(grid0['label'], grid1['label'], grid2['label'])))
