# docme
import re, sys
import igraph
import networkx
import polars as pl
from pathlib import Path

from textgrid import textgrid
from textgrid import textgraph

# # # # # # # # # #
# Read textgrid.
grid_file = Path.home() / \
    'Projects/SCOTUSProductionPlanning/textgrids_tagged/2013/12-574.TextGrid'
dat = textgrid.read(grid_file)

graph = textgraph.to_graph(dat)
print(graph.summary())

# Local speaking rate.
textgraph.speaking_rate(graph, side='before', window=1000.0)
textgraph.speaking_rate(graph, side='after', window=1000.0)

textgraph.to_dat(graph)

sys.exit(0)

# Subset words.
dat_word = dat.filter(pl.col('tier') == 'words')
print(f'dat_word {dat_word}', end='\n\n')

# Subset phones.
dat_phon = dat.filter(pl.col('tier') == 'phones')
print(f'dat_phon {dat_phon}', end='\n\n')

# Subset tokens of 'the'.
dat_the = dat_word.filter( \
    pl.col('label').str.contains('^[Tt]he$'))
print(f'dat_the {dat_the}', end='\n\n')

# Words following 'the' tokens.
dat_the_x = textgrid.succeeding(dat_the, dat)
print(f'dat_the_x {dat_the_x}', end='\n\n')
