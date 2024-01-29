from pathlib import Path
import textgrid

year = 2014
scotus_dir = Path.home() / f'Sounds/SCOTUS/mfa_align/{year}'
scotus_grids = list(scotus_dir.glob('*.TextGrid'))

grid = textgrid.read(str(scotus_grids[0]))
print(grid.head())

grid = grid.sort(['start'])
print(grid.head())

intervals = textgrid.intervals_at(grid,
                                  3.0,
                                  speakers=['John G. Roberts, Jr.'],
                                  tiers=['words'])
print(intervals)
