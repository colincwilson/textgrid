from pathlib import Path
import textgrid

year = 2014
scotus_dir = Path.home() / f'Sounds/SCOTUS/fave_align/{year}'
scotus_grids = list(scotus_dir.glob('*.TextGrid'))

grid = textgrid.read_textgrid(str(scotus_grids[0]))
print(grid.head())

intervals = textgrid.get_intervals(grid, 'John G. Roberts, Jr. - phone', 2.164)
print(intervals)