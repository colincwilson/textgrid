# Functions for operating on networkx representations of textgrids.
import re
import networkx as nx

from .textgrid import combine_tiers


def to_graph(dat):
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

    speaker_prec = None
    word_prec = None
    for row in dat_word.iter_rows(named=True):
        speaker = row['speaker']
        word_id = row['word_id']
        graph.add_node( \
            word_id, speaker=speaker, tier='word',
            label=row['word'], start_ms=row['word_start'],
            end_ms=row['word_end'], dur_ms=row['word_dur_ms'])
        if speaker == speaker_prec:
            graph.add_edge(word_prec, word_id, label='succ')
            graph.add_edge(word_id, word_prec, label='prec')
        speaker_prec = speaker
        word_prec = word_id

    # Phone nodes and edges.
    speaker_prec = None
    word_prec = None
    phone_prec = None
    phone = dat['word_id'].max() + 1  # Cumulative phone index.
    phone_idx = 0  # Phone index within word.
    for row in dat.iter_rows(named=True):
        speaker = row['speaker']
        word_id = row['word_id']
        # Reset phone_idx at word boundaries.
        if word_id != word_prec:
            phone_idx = 0
        graph.add_node( \
            phone, speaker=speaker, tier='phone',
            label=row['phone'], start_ms=row['start'],
            end_ms=row['end'], dur_ms=row['dur_ms'])
        graph.add_edge(word_id, phone, label=f'phone{phone_idx}')
        graph.add_edge(phone, word_id, label='word')
        phone_idx += 1
        if speaker == speaker_prec:
            graph.add_edge(phone_prec, phone, label='succ')
            graph.add_edge(phone, phone_prec, label='prec')
        phone += 1
        speaker_prec = speaker
        word_prec = word_id
        phone_prec = phone

    return graph


def get_nodes(graph, tier=None, label=None, regex=None):
    """
    Get nodes on tier by exact label or regex.
    """
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


def get_prec(graph, node):
    """
    Get preceding node (assumed unique).
    """
    # Null node.
    if node is None:
        return None
    # List of nodes.
    if isinstance(node, list):
        return [get_prec(graph, n) for n in node]
    # Base case.
    node_id = node[0]
    edges = [(u, v, d) for (u, v, d) in \
                graph.edges(node_id, data=True) \
                if d['label'] == 'prec']
    if len(edges) == 0:
        return None
    node_prec = [v for (u, v, d) in edges][0]
    return (node_prec, graph.nodes[node_prec])


# Alias.
prec = get_prev = get_prec


def get_succ(graph, node):
    """
    Get following node (assumed unique).
    """
    # Null node.
    if node is None:
        return None
    # List of nodes.
    if isinstance(node, list):
        return [get_succ(graph, n) for n in node]
    # Base case.
    node_id = node[0]
    edges = [(u, v, d) for (u, v, d) in \
                graph.edges(node_id, data=True) \
                if d['label'] == 'succ']
    if len(edges) == 0:
        return None
    node_succ = [v for (u, v, d) in edges][0]
    return (node_succ, graph.nodes[node_succ])


# Alias.
succ = get_next = get_succ


def get_window(graph, node, nprec=1, nsucc=1):
    # Null node.
    if node is None:
        return None
    # List of nodes.
    if isinstance(node, list):
        return [get_window(graph, n, nprec, nsucc) \
                    for n in node]
    # Base case.
    node_prec = []
    if nprec > 0:
        node_ = node
        for _ in range(nprec):
            node_ = get_prec(graph, node_)
            node_prec.append(node_)
        node_prec = node_prec[::-1]

    node_succ = []
    if nsucc > 0:
        node_ = node
        for _ in range(nsucc):
            node_ = get_succ(graph, node_)
            node_succ.append(node_)

    return (node_prec, node_succ)


# Alias.
window = get_window


def get_phones(graph, word_node):
    """
    Get phone within word.
    """
    # Null node.
    if word_node is None:
        return None
    # List of nodes.
    if isinstance(word_node, list):
        return [get_phones(graph, n) for n in word_node]
    # Base case.
    node_id = word_node[0]
    edges = [(u, v, d) for (u, v, d) in \
                graph.edges(node_id, data=True) \
                if d['label'].startswith('phone')]
    if len(edges) == 0:
        return None
    phones = [v for (u, v, d) in edges]
    phones = [(v, graph.nodes[v]) for v in phones]
    return phones


def get_word(graph, phone_node):
    """
    Get word containing phone.
    """
    # Null node.
    if phone_node is None:
        return None
    # List of nodes.
    if isinstance(phone_node, list):
        return [get_word(graph, n) for n in phone_node]
    # Base case.
    node_id = phone_node[0]
    edges = [(u, v, d) for (u, v, d) in \
                graph.edges(node_id, data=True) \
                if d['label'].startswith('phone')]
    if len(edges) == 0:
        return None
    phones = [v for (u, v, d) in edges]
    phones = [(v, graph.nodes[v]) for v in phones]
    return phones


# Alias.
phones = get_phones


def get_attr(graph, thing, attr, default=None):
    """
    Get attribute of node or edge.
    """
    # Null node/edge.
    if thing is None:
        return None
    # List of nodes/edges.
    if isinstance(thing, list):
        return [get_attr(x) for x in thing]
    # Base case.
    return thing[-1].get(attr, default)


attr = get_attr
