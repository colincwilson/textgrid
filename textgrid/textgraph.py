# Functions for operating on igraph representations of textgrids.
# see: https://igraph.org/
import re, sys
import igraph
import numpy as np
import polars as pl
from collections import deque

from .textgrid import combine_tiers
from .textgrid import ARPABET_VOWELS
from .util import *


def to_graph(dat):
    """
    Convert dataframe to graph with word and phone 
    intervals as nodes, bidirectional edges between 
    adjacent words/phones in the same speaker 'turn', 
    and bidirectional edges from words to the phones 
    that they contain.
    # todo: check/fix 'turn' segmentation
    """

    graph = igraph.Graph()

    # Combine word and phone tiers.
    dat = combine_tiers(dat)
    print(dat)
    print(dat.columns)

    # Get words tiers.
    dat_word = dat[['filename', 'speaker', 'word', 'word_id', \
                    'word_start', 'word_end', 'word_dur_ms']] \
                .unique() \
                .sort(['filename', 'word_id'])
    print(dat_word)

    # Nodes and edges with their attributes.
    node_index = 0
    edges = []
    node_attrs = {key: [] for key in ['speaker', 'tier', 'word_id', \
        'label', 'start', 'end', 'dur_ms']}
    edge_attrs = {'label': []}

    # Word nodes and edges.
    speaker_ = None
    word_index_ = None
    word_id2index = {}  # Map dataframe word_id -> node index.
    for row in dat_word.iter_rows(named=True):
        speaker = row['speaker']
        word_id = row['word_id']
        node_attrs['speaker'].append(speaker)
        node_attrs['tier'].append('word')
        node_attrs['word_id'].append(word_id)
        node_attrs['label'].append(row['word'])
        node_attrs['start'].append(row['word_start'])
        node_attrs['end'].append(row['word_end'])
        node_attrs['dur_ms'].append(row['word_dur_ms'])
        word_id2index[word_id] = node_index
        # note: insertion ordered preserved
        if (word_index_ is not None) and (speaker == speaker_):
            # todo: check 'sp', 'sil' intervals
            edges.append((word_index_, node_index))
            edge_attrs['label'].append('succ')
            edges.append((node_index, word_index_))
            edge_attrs['label'].append('prec')
        speaker_ = speaker
        word_index_ = node_index
        node_index += 1

    # Phone nodes and edges.
    speaker_ = None
    word_index_ = None
    phone_index_ = None
    phone_idx = 0  # Phone index within word.
    for row in dat.iter_rows(named=True):
        speaker = row['speaker']
        word_id = row['word_id']
        word_index = word_id2index[word_id]
        # Reset phone_idx at word boundaries.
        if word_index != word_index_:
            phone_idx = 0
        node_attrs['speaker'].append(speaker)
        node_attrs['tier'].append('phone')
        node_attrs['word_id'].append(word_id)
        node_attrs['label'].append(row['phone'])
        node_attrs['start'].append(row['start'])
        node_attrs['end'].append(row['end'])
        node_attrs['dur_ms'].append(row['dur_ms'])

        edges.append((node_index, word_index))
        edge_attrs['label'].append('word')
        edges.append((word_index, node_index))
        edge_attrs['label'].append(f'phone{phone_idx}')
        # note: insertion ordered preserved
        if (phone_index_ is not None) and (speaker == speaker_):
            edges.append((phone_index_, node_index))
            edge_attrs['label'].append('succ')
            edges.append((node_index, phone_index_))
            edge_attrs['label'].append('prec')
        speaker_ = speaker
        word_index_ = word_index
        phone_index_ = node_index
        node_index += 1

    # Create graph.
    graph = igraph.Graph( \
        node_index,
        edges,
        directed=True,
        vertex_attrs=node_attrs,
        edge_attrs=edge_attrs)

    return graph


def to_dat(graph):
    """ Convert graph to dataframe. """
    dat = [v.attributes() for v in graph.vs]
    dat = pl.DataFrame(dat)
    print(dat)
    return dat


# # # # # # # # # #


def filter_nodes(graph, nodes=None, tier=None, label=None, regex=None, \
                 func=None, noneify=False):
    """
    Filter an iterable of nodes with tier / exact label /
    regex / boolean function. If noneify is True, replace 
    nodes that would be filtered with placeholder None.
    """
    if graph is None:
        return []
    if nodes is None:
        nodes = list(graph.vs)

    ret = []
    for node in nodes:
        # Skip nones.
        if node is None:
            if noneify: ret.append(None)
            continue
        # Get node data (aka attributes).
        if isinstance(node, int):
            node = graph.vs[node]
        node_dat = node.attributes()

        # Filter.
        match = True
        # Match tier.
        if (tier is not None) and (node_dat['tier'] != tier):
            match = False
        # Match exact label.
        if match and (label is not None) and (node_dat['label'] != label):
            match = False
        # Match label regex.
        if match and (regex is not None) and \
            not re.search(regex, node_dat['label']):
            match = False
        # Match data predicate.
        if match and (func is not None) and not func(node_dat):
            match = False

        # Accumulate.
        if match:
            ret.append(node)
        elif noneify:
            ret.append(None)

    return ret


def get_nodes(graph, **kwargs):
    """
    Get nodes in graph with optional filtering
    (see filter_nodes for options).
    """
    return filter_nodes(graph, **kwargs)


def node_access_decorator(func):
    """
    Decorator for node access functions: handle NAs,
    lift to lists, conver nodes to ids.
    """

    def _wrapper(graph, node, **kwargs):
        if graph is None or node is None:
            return None
        if isinstance(node, list):
            return [_wrapper(graph, n, **kwargs) for n in node]
        if isinstance(node, int):
            node = graph.vs[node]
        return func(graph, node, **kwargs)

    return _wrapper


@node_access_decorator
def get_adj(graph, node, reln='succ', skip=['sp', ''], max_sep=None):
    """
    Get preceding or following node (assumed unique).
        max_sep: maximum separation [ms]
    """
    start_time = get_start_time(graph, node)
    end_time = get_end_time(graph, node)
    if max_sep is not None:
        max_sep_s = max_sep / 1000.0
    while 1:
        edges = [e for e in node.out_edges() if e['label'] == reln]
        if len(edges) == 0:
            return None
        node_adj = graph.vs[edges[0].target]
        if max_sep is not None:
            if reln == 'prec':
                start_time_ = get_start_time(graph, node_adj)
                if (start_time - start_time_) > max_sep_s:
                    return None
            elif reln == 'succ':
                end_time_ = get_end_time(graph, node_adj)
                if (end_time_ - end_time) > max_sep_s:
                    return None
        if node_adj['label'] in skip:
            node = node_adj
        else:
            break
    return node_adj


def get_prec(graph, node, **kwargs):
    """ Get preceding node (assumed unique). """
    return get_adj(graph, node, reln='prec', **kwargs)


def get_succ(graph, node, **kwargs):
    """ Get following node (assumed unique). """
    return get_adj(graph, node, reln='succ', **kwargs)


@node_access_decorator
def get_window(graph, node, nprec=1, nsucc=1, **kwargs):
    """ Get window of preceding and following nodes. """
    node_prec = []
    if nprec > 0:
        node_ = node
        for _ in range(nprec):
            node_ = get_prec(graph, node_, **kwargs)
            node_prec.append(node_)
        node_prec = node_prec[::-1]

    node_succ = []
    if nsucc > 0:
        node_ = node
        for _ in range(nsucc):
            node_ = get_succ(graph, node_, **kwargs)
            node_succ.append(node_)

    return (node_prec, node_succ)


@node_access_decorator
def get_phones(graph, word):
    """ Get phones within word. """
    edges = [e for e in word.out_edges() \
        if e['label'].startswith('phone')]
    if len(edges) == 0:
        return None
    phones = [graph.vs[e.target] for e in edges]
    return phones


@node_access_decorator
def get_initial_phone(graph, word):
    """ Get first phone of word. """
    phones = get_phones(graph, word)
    if phones is None:
        return None
    return phones[0]


@node_access_decorator
def get_final_phone(graph, word):
    """ Get last phone of word. """
    phones = get_phones(graph, word)
    if phones is None:
        return None
    return phones[-1]  # negative indices, bless


@node_access_decorator
def get_word(graph, phone):
    """ Get word containing phone. """
    edges = [e for e in phone.out_edges() \
        if e['label'] == 'word']
    if len(edges) == 0:
        return None
    word = graph.vs[edges.target]
    return word


# todo: decorator
def get_attr(graph, thing, attr=None, default=None):
    """ Get attribute of node or edge. """
    # Null node/edge.
    if thing is None:
        return None
    # List of nodes/edges.
    if isinstance(thing, list):
        return [get_attr(x) for x in thing]
    # Node index.
    if isinstance(thing, int):
        thing = graph.vs[thing]
    # Base case.
    # todo: avoid dictionary creation
    return thing.attributes().get(attr, default)


def get_speaker(graph, node):
    """ Get speaker of node. """
    return get_attr(graph, node, 'speaker')


def get_tier(graph, node):
    """ Get tier of node. """
    return get_attr(graph, node, 'tier')


def get_start_time(graph, node):
    """ Get start time of node. """
    return get_attr(graph, node, 'start')


def get_end_time(graph, node):
    """ Get end time of node. """
    return get_attr(graph, node, 'end')


def get_label(graph, thing):
    """ Get label of node or edge. """
    return get_attr(graph, thing, 'label')


# # # # # # # # # #


def speaking_rate(graph,
                  vowel_regex=None,
                  window=1000.0,
                  side='before',
                  verbose=True):
    """
    Add local speaking rate (vowels/second) to phone and word 
    nodes, in preceding (before) or following (after) context, 
    within specified window [ms].
    """
    # Vowel phones.
    if vowel_regex is None:
        vowel_regex = ARPABET_VOWELS
    phones = get_nodes(graph, tier='phone')

    # Process side.
    before = (side == 'before')
    if before:
        phone = phones[0]
        phones = phones[1:]
        get_next = lambda x: get_succ(graph, x, skip=[])
        rate_side = 'rate_before'
    else:
        phone = phones[-1]
        phones = reversed(phones[:-1])
        get_next = lambda x: get_prec(graph, x, skip=[])
        rate_side = 'rate_after'

    # Initialize.
    rates = []  # for summary statistics.
    window_s = window / 1000.0

    rate = np.nan
    rates.append(rate)
    phone[rate_side] = rate
    vowels_ = deque()
    if re.search(vowel_regex, phone['label']):
        vowels_.append(phone)
    phone_ = phone

    for phone in phones:
        # Clear at turn changes.
        if phone != get_next(phone_):
            rate = np.nan
            rates.append(rate)
            phone[rate_side] = rate
            vowels_.clear()
            if re.search(vowel_regex, phone['label']):
                vowels_.append(phone)
            phone_ = phone
            continue

        # Pop stale vowels.
        n = len(vowels_)
        for _ in range(n):
            if before:
                delta = (phone['start'] - vowels_[0]['end'])
            else:
                delta = (vowels_[0]['start'] - phone['end'])
            if delta > window_s:
                vowels_.popleft()

        # Local speaking rate.
        n = len(vowels_)
        if n == 0:
            rate = np.nan
            rates.append(rate)
            phone[rate_side] = rate
        else:
            rate = speaking_rate_calc(vowels_)
            rates.append(rate)
            phone[rate_side] = rate
        if re.search(vowel_regex, phone['label']):
            vowels_.append(phone)
        phone_ = phone

    # Set word rate equal to phone rate.
    for word in get_nodes(graph, tier='word'):
        phones = get_phones(graph, word)
        if before:
            # Rate before word is rate before first phone.
            word[rate_side] = phones[0][rate_side]
        else:
            # Rate after word is rate after last phone.
            word[rate_side] = phones[-1][rate_side]

    if verbose:
        print(f'* Speaking rate statistics: '
              f'{summary_stats(np.array(rates))}')

    return graph


# Aliases.
prec = get_prev = get_prec
succ = get_next = get_succ
window = get_window
phones = get_phones
first_phone = initial_phone = get_initial_phone
last_phone = final_phone = get_final_phone
attr = get_attr
get_start = get_start_time
get_end = get_end_time
