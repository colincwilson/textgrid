# Functions for operating on networkx representations of textgrids.


def get_nodes(graph, tier=None, label=None):
    """
    Get nodes on tier with label.
    """
    if tier is None and label is None:
        return graph.nodes(data=True)
    if label is None:
        nodes = [(n,d) for (n,d) in graph.nodes(data=True) \
                    if d['tier'] == tier]
        return nodes
    nodes = [(n, d) for (n, d) in graph.nodes(data=True) \
                if d['tier'] == tier and d['label'] == label]
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


get_prev = get_prec


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


get_next = get_succ


def get_phones(graph, word_node):
    """
    Get phones within word.
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
