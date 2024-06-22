# Functions for operating on networkx representations of textgrids.
# todo: TextGraph class backed by networkx DiGraph
import re, sys
import networkx as nx
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

    graph = nx.DiGraph()

    # Combine word and phone tiers.
    dat = combine_tiers(dat)
    print(dat)
    print(dat.columns)

    # Word nodes and edges.
    dat_word = dat[['filename', 'speaker', 'word', 'word_id', \
                    'word_start', 'word_end', 'word_dur_ms']] \
                .unique() \
                .sort(['filename', 'word_id'])
    print(dat_word)

    speaker_ = None
    word_id_ = None
    for row in dat_word.iter_rows(named=True):
        speaker = row['speaker']
        word_id = row['word_id']
        graph.add_node( \
            word_id, speaker=speaker, tier='word',
            label=row['word'], start=row['word_start'],
            end=row['word_end'], dur_ms=row['word_dur_ms'])
        # note: insertion ordered preserved in python 3.7+
        if (word_id_ is not None) and (speaker == speaker_):
            # todo: check 'sp', 'sil' intervals
            graph.add_edge(word_id_, word_id, label='succ')
            graph.add_edge(word_id, word_id_, label='prec')
        speaker_ = speaker
        word_id_ = word_id

    # Phone nodes and edges.
    speaker_ = None
    word_id_ = None
    phone_id_ = None
    phone_id = dat['word_id'].max() + 1  # Cumulative phone index.
    phone_idx = 0  # Phone index within word.
    for row in dat.iter_rows(named=True):
        speaker = row['speaker']
        word_id = row['word_id']
        # Reset phone_idx at word boundaries.
        if word_id != word_id_:
            phone_idx = 0
        graph.add_node( \
            phone_id, speaker=speaker, tier='phone',
            label=row['phone'], start=row['start'],
            end=row['end'], dur_ms=row['dur_ms'])
        graph.add_edge(phone_id, word_id, label='word')
        graph.add_edge(word_id, phone_id, label=f'phone{phone_idx}')
        # note: insertion ordered preserved in python 3.7+
        if (phone_id_ is not None) and (speaker == speaker_):
            graph.add_edge(phone_id_, phone_id, label='succ')
            graph.add_edge(phone_id, phone_id_, label='prec')
        speaker_ = speaker
        word_id_ = word_id
        phone_id_ = phone_id
        phone_id += 1
        phone_idx += 1

    return graph


def to_dat(graph):
    """ Convert graph to dataframe. """
    rows = [d for (idx, d) in graph.nodes(data=True)]
    dat = pl.DataFrame(rows)
    print(dat)
    return dat


# # # # # # # # # #


def get_nodes(graph, tier=None, label=None, regex=None):
    """ Get nodes on tier by exact label or regex. """
    if tier is None and label is None:
        return graph.nodes(data=True)
    if label is None and regex is None:
        nodes = [(n,d) for (n,d) in graph.nodes(data=True) \
                    if d['tier'] == tier]
        return nodes
    if label is not None:
        nodes = [(n, d) for (n, d) in graph.nodes(data=True) \
                    if d['tier'] == tier and d['label'] == label]
        return nodes
    nodes = [(n, d) for (n, d) in graph.nodes(data=True) \
                if d['tier'] == tier and re.search(regex, d['label'])]
    return nodes


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
            node_id = node
        else:
            node_id = node[0]
        return func(graph, node_id, **kwargs)

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
        edges = [(u, v, d) for (u, v, d) in \
                    graph.edges(node, data=True) \
                    if d['label'] == reln]
        if len(edges) == 0:
            return None
        node_adj = edges[0][1]  # v of first edge (u, v, d)
        if max_sep is not None:
            if reln == 'prec':
                start_time_ = get_start_time(graph, node_adj)
                if (start_time - start_time_) > max_sep_s:
                    return None
            elif reln == 'succ':
                end_time_ = get_end_time(graph, node_adj)
                if (end_time_ - end_time) > max_sep_s:
                    return None
        if get_attr(graph, node_adj, 'label') in skip:
            node = node_adj
        else:
            break
    return (node_adj, graph.nodes[node_adj])


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
def get_phones(graph, word_node):
    """ Get phones within word. """
    edges = [(u, v, d) for (u, v, d) in \
                graph.edges(word_node, data=True) \
                if d['label'].startswith('phone')]
    if len(edges) == 0:
        return None
    phones = [v for (u, v, d) in edges]
    phones = [(v, graph.nodes[v]) for v in phones]
    return phones


@node_access_decorator
def get_initial_phone(graph, word_node):
    """ Get first phone of word. """
    phones = get_phones(graph, word_node)
    if phones is None:
        return None
    return phones[0]


@node_access_decorator
def get_final_phone(graph, word_node):
    """ Get last phone of word. """
    phones = get_phones(graph, word_node)
    if phones is None:
        return None
    return phones[-1]  # negative indices, bless


@node_access_decorator
def get_word(graph, phone_node):
    """ Get word containing phone. """
    edges = [(u, v, d) for (u, v, d) in \
                graph.edges(phone_node, data=True) \
                if d['label'] == 'word']
    if len(edges) == 0:
        return None
    word = [v for (u, v, d) in edges][0]
    word = (word, graph.nodes[word])
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
        return graph.nodes[thing].get(attr, default)
    # Base case.
    return thing[-1].get(attr, default)


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
    phone[1][rate_side] = rate
    vowels_ = deque()
    if re.search(vowel_regex, phone[1]['label']):
        vowels_.append(phone)
    phone_ = phone

    for phone in phones:
        # Clear at turn changes.
        if phone != get_next(phone_):
            rate = np.nan
            rates.append(rate)
            phone[1][rate_side] = rate
            vowels_.clear()
            if re.search(vowel_regex, phone[1]['label']):
                vowels_.append(phone)
            phone_ = phone
            continue

        # Pop stale vowels.
        n = len(vowels_)
        for _ in range(n):
            if before:
                delta = (phone[1]['start'] - vowels_[0][1]['end'])
            else:
                delta = (vowels_[0][1]['start'] - phone[1]['end'])
            if delta > window_s:
                vowels_.popleft()

        # Local speaking rate.
        n = len(vowels_)
        if n == 0:
            rate = np.nan
            rates.append(rate)
            phone[1][rate_side] = rate
        else:
            rate = speaking_rate_calc(vowels_)
            rates.append(rate)
            phone[1][rate_side] = rate
        if re.search(vowel_regex, phone[1]['label']):
            vowels_.append(phone)
        phone_ = phone

    # Set word rate equal to phone rate.
    for word in get_nodes(graph, 'word'):
        phones = get_phones(graph, word)
        if before:
            # Rate before word is rate before first phone.
            word[1][rate_side] = phones[0][1][rate_side]
        else:
            # Rate after word is rate after last phone.
            word[1][rate_side] = phones[-1][1][rate_side]

    if verbose:
        print(f'* Speaking rate statistics: '
              f'{summary_stats(np.array(rates))}')

    return


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
