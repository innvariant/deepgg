import torch
import itertools
import pickle
import random
import networkx as nx

from copy import deepcopy
from networkx.utils import py_random_state
from torch.utils.data import Dataset


class ConstructionAction:
    add_node = 0
    add_edge = 1
    remove_node = 2
    remove_edge = 3


class ConstructionSequenceDataset(Dataset):
    @staticmethod
    def from_pickle_file(fname):
        with open(fname, 'rb') as f:
            pickled_sequences = pickle.load(f)
        return ConstructionSequenceDataset(pickled_sequences)

    def __init__(self, generator, device='cpu'):
        super(ConstructionSequenceDataset, self).__init__()

        # Load from generator into memory
        self.dataset = [torch.tensor(seq, device=device) for seq in generator]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def construction_sequence_to_graph(sequence, start_node_from: int = 0):
    g = nx.Graph()

    action_idx = 0
    while action_idx < len(sequence):
        action = sequence[action_idx]

        if ConstructionAction.add_node == action:
            g.add_node(start_node_from + len(g.nodes))
            action_idx += 1
        elif ConstructionAction.add_edge == action:
            action_idx += 1
            source = start_node_from + sequence[action_idx]
            action_idx += 1
            target = start_node_from + sequence[action_idx]
            g.add_edge(source, target)
            action_idx += 1
        elif ConstructionAction.remove_edge == action:
            action_idx += 1
            u = start_node_from + sequence[action_idx]
            action_idx += 1
            v = start_node_from + sequence[action_idx]
            g.remove_edge(u, v)
            action_idx += 1
        elif ConstructionAction.remove_node == action:
            action_idx += 1
            the_node = start_node_from + sequence[action_idx]
            g.remove_node(start_node_from + the_node)
            action_idx += 1
        else:
            raise ValueError('Sequence error (unknown action) on action idx #{idx}: a={action}'.format(idx=action_idx, action=action))
    return g


def graph_to_construction_sequence(graph: nx.Graph, root_node=None, node_offset: int = 0, traversal='bfs'):
    num_components = nx.algorithms.components.number_connected_components(graph)
    if num_components > 1:
        subgraphs = [graph.subgraph(c) for c in sorted(nx.connected_components(graph))]
        current_offset = node_offset
        sequences = []
        for subgraph in subgraphs:
            sequences.append(graph_to_construction_sequence(subgraph, node_offset=current_offset, traversal=traversal))
            current_offset += len(subgraph.nodes)
        return list(itertools.chain(*sequences))

    if root_node is None:
        root_node = random.choice(list(graph.nodes))

    construction_sequence = []
    last_node_offset = node_offset
    node_map = {root_node: last_node_offset}
    visits = set()

    #"""
    traverser = nx.dfs_edges if traversal != 'bfs' else nx.bfs_edges
    for source, target in traverser(graph, source=root_node):
        if target not in node_map:
            construction_sequence.append(ConstructionAction.add_node)
            last_node_offset += 1
            node_map[target] = last_node_offset
        visits.add(target)
        for visit in [n for n in graph.neighbors(target) if n not in visits]:
            if visit not in node_map:
                construction_sequence.append(ConstructionAction.add_node)
                last_node_offset += 1
                node_map[visit] = last_node_offset
                construction_sequence.extend([ConstructionAction.add_edge, node_map[target], node_map[visit]])
            if not graph.is_directed():
                construction_sequence.extend([ConstructionAction.add_edge,  node_map[visit], node_map[target]])
    #"""
    """
    for source_node, to_visit in nx.bfs_successors(graph, source=root_node):
        for visit in to_visit:
            visits.add(visit)
            if visit not in node_map:
                construction_sequence.append(ConstructionAction.add_node)
                last_node_offset += 1
                node_map[visit] = last_node_offset
            for target in [n for n in graph.neighbors(visit) if n not in visits]:
                if target not in node_map:
                    construction_sequence.append(ConstructionAction.add_node)
                    last_node_offset += 1
                    node_map[target] = last_node_offset
                construction_sequence.extend([ConstructionAction.add_edge, node_map[visit], node_map[target]])
                if not graph.is_directed():
                    construction_sequence.extend([ConstructionAction.add_edge,  node_map[target], node_map[visit]])
    """

    return construction_sequence


def generate_ba_model_construction_sequence_dataset(
        graph_size_min:int,
        graph_size_max:int,
        ba_model_m_min:int,
        ba_model_m_max:int,
        n_samples,
        fname):

    samples = []
    for _ in range(n_samples):
        size = random.randint(graph_size_min, graph_size_max)
        ba_model_m = random.randint(ba_model_m_min, ba_model_m_max)
        samples.append(generate_ba_model_construction_sequence(size, ba_model_m))

    with open(fname, 'wb') as f:
        pickle.dump(samples, f)


@py_random_state(2)
def generate_ba_model_construction_sequence(n, m, seed=None):
    if m < 1 or m >= n:
        raise nx.NetworkXError(f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}")

    # Construction sequence
    sequence = []

    # Add m initial nodes (m0 in barabasi-speak)
    G = nx.empty_graph(m)
    sequence.extend([ConstructionAction.add_node] * m)  # create nodes m times and don't create edges

    # Target nodes for new edges
    targets = list(range(m))
    # List of existing nodes, with nodes repeated once for each adjacent edge
    repeated_nodes = []
    # Start adding the other n-m nodes. The first node is m.
    source = m
    while source < n:
        sequence.append(ConstructionAction.add_node)
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        for t in targets:
            sequence.append(ConstructionAction.add_edge)
            sequence.append(source)
            sequence.append(t)

        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = nx.random_graphs._random_subset(repeated_nodes, m, seed)
        source += 1

    return sequence


@py_random_state(3)
def generate_ws_model_construction_sequence(n, k, p, seed=None):
    # Based on nx.watts_strogatz_graph
    if k > n:
        raise nx.NetworkXError("k>n, choose smaller k or larger n")

    # TODO: If k == n, the graph is complete not Watts-Strogatz
    if k == n:
        return nx.complete_graph(n)

    G = nx.Graph()
    # Construction sequence
    sequence = []
    nodes = list(range(n))  # nodes are labeled 0 to n-1
    sequence.extend([ConstructionAction.add_node] * n)
    # connect each node to k/2 neighbors
    for j in range(1, k // 2 + 1):
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        G.add_edges_from(zip(nodes, targets))
        for source, target in zip(nodes, targets):
            sequence.extend([ConstructionAction.add_edge, source, target])

    # rewire edges from each node
    # loop over all nodes in order (label) and neighbors in order (distance)
    # no self loops or multiple edges allowed
    for j in range(1, k // 2 + 1):  # outer loop is neighbors
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last in list
        # inner loop in node order
        for u, v in zip(nodes, targets):
            if seed.random() < p:
                w = seed.choice(nodes)
                # Enforce no self-loops or multiple edges
                while w == u or G.has_edge(u, w):
                    w = seed.choice(nodes)
                    if G.degree(u) >= n - 1:
                        break  # skip this rewiring
                else:
                    G.remove_edge(u, v)
                    sequence.extend([ConstructionAction.remove_edge, u, v])
                    G.add_edge(u, w)
                    sequence.extend([ConstructionAction.add_edge, u, w])
    return sequence


@py_random_state(2)
def generate_er_model_construction_sequence(n, p, seed=None, directed=False):
    # Based on nx.gnp_random_graph
    # Construction sequence
    sequence = []

    if directed:
        edges = itertools.permutations(range(n), 2)
        G = nx.DiGraph()
    else:
        edges = itertools.combinations(range(n), 2)
        G = nx.Graph()

    # Add n vertices to graph in both representations
    G.add_nodes_from(range(n))
    sequence.extend([ConstructionAction.add_node] * n)
    if p <= 0:
        return sequence

    for e in edges:
        if seed.random() < p:
            G.add_edge(*e)
            sequence.extend([ConstructionAction.add_edge, e[0], e[1]])
    return sequence


def construction_sequence_evolution(sequence):
    g = nx.Graph()
    evolution = [deepcopy(g)]

    action_idx = 0
    while action_idx < len(sequence):
        action = sequence[action_idx]

        if action is ConstructionAction.add_node:
            g.add_node(len(g.nodes))
            action_idx += 1
        elif action is ConstructionAction.add_edge:
            action_idx += 1
            source = sequence[action_idx]
            action_idx += 1
            target = sequence[action_idx]
            g.add_edge(source, target)
            action_idx += 1
        elif action is ConstructionAction.remove_node:
            action_idx += 1
            vertex = sequence[action_idx]
            g.remove_node(vertex)
            action_idx += 1
        else:
            raise ValueError('Sequence error on action idx #%s' % action_idx)

        evolution.append(deepcopy(g))
    return evolution

