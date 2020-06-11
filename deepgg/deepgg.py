import torch
import torch.nn as nn
import torch.distributions as dists
import dgl
import numpy as np

from enum import Enum
from dgl.nn.pytorch import GraphConv
from functools import partial
from torch.functional import F
from torch.distributions import Categorical


class MessageKey(object):
    repr_vertex = 'dgg_hv'
    repr_edge = 'dgg_he'
    repr_message = 'dgg_m'
    repr_activation = 'dgg_a'


def get_node_features(graph: dgl.DGLGraph):
    return graph.ndata[MessageKey.repr_vertex]


def get_node_feature(graph: dgl.DGLGraph, node, default=None):
    return graph.nodes[node].data[MessageKey.repr_vertex] if MessageKey.repr_vertex in graph.nodes[node].data else default


def set_nodes_features(graph: dgl.DGLGraph, features):
    graph.ndata[MessageKey.repr_vertex] = features


def get_nodes_features(nodes: dgl.NodeBatch):
    return nodes.data[MessageKey.repr_vertex]


def get_source_node_feature(edge: dgl.EdgeBatch):
    return edge.src[MessageKey.repr_vertex]


def get_edge_feature(edge: dgl.EdgeBatch):
    return edge.data[MessageKey.repr_edge]


def get_activations(graph: dgl.DGLGraph):
    return graph.ndata[MessageKey.repr_activation]


class DeepGGModule(nn.Module):
    _log_losses: list

    def prepare_training(self):
        self._log_losses = []

    def add_log_loss(self, loss: torch.Tensor):
        self._log_losses.append(loss)

    def get_log_loss(self):
        if len(self._log_losses) < 1:
            return torch.tensor(0)
        return torch.stack(self._log_losses).sum()

    def get_log_losses(self):
        return self._log_losses


class DeepStateModule(DeepGGModule):
    def __init__(self, state_embedding_size, actions):
        super(DeepStateModule, self).__init__()

        self.register_buffer('_actions', torch.tensor(actions).view(-1, 1))
        self._next_state_action = nn.Linear(state_embedding_size, len(actions))

    def map_action_to_index(self, action):
        try:
            return (self._actions.view(-1) == action).nonzero()[0]
        except IndexError:
            return None

    def map_index_to_action(self, action_idx):
        return self._actions[action_idx]

    def forward(self, embedding, action=None):
        next_state_logits = self._next_state_action(embedding)

        if self.training:
            next_state_log_loss = F.cross_entropy(next_state_logits, self.map_action_to_index(action))
            self.add_log_loss(next_state_log_loss)
            return action
        else:
            next_state_probs = F.softmax(next_state_logits, dim=1)
            next_action_ix = Categorical(next_state_probs).sample().item()
            return self.map_index_to_action(next_action_ix)


class GraphEmbed(nn.Module):
    def __init__(self, node_hidden_size):
        super(GraphEmbed, self).__init__()

        self.graph_hidden_size = 2 * node_hidden_size

        self.register_buffer('_embedding_unknown_graph', torch.zeros(1, self.graph_hidden_size))

        # Embed graphs
        self.node_gating = nn.Sequential(
            nn.Linear(node_hidden_size, 1),
            nn.Sigmoid()
        )
        self.node_to_graph = nn.Linear(node_hidden_size, self.graph_hidden_size)
        self._new_node_gating = nn.Sequential(
            nn.Linear(node_hidden_size, 7),
            nn.Sigmoid()
        )
        self._conv = GraphConv(7, self.graph_hidden_size)

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        if g.number_of_nodes() == 0:
            return self._embedding_unknown_graph
        else:
            # Node features are stored as hv in ndata.
            hvs = get_node_features(g)
            res = self._conv(g, self._new_node_gating(hvs)).mean(0, keepdim=True)

            return res


class GraphProp(nn.Module):
    def __init__(self, num_prop_rounds, node_hidden_size):
        super(GraphProp, self).__init__()

        self.num_prop_rounds = num_prop_rounds

        # Setting from the paper
        self.node_activation_hidden_size = 2 * node_hidden_size

        message_funcs = []
        node_update_funcs = []
        self.reduce_funcs = []

        for t in range(num_prop_rounds):
            # input being [hv, hu, xuv]
            message_funcs.append(nn.Linear(2 * node_hidden_size + node_hidden_size, self.node_activation_hidden_size))

            self.reduce_funcs.append(partial(self.dgmg_reduce, round=t))
            node_update_funcs.append(nn.GRUCell(self.node_activation_hidden_size, node_hidden_size))

        self.message_funcs = nn.ModuleList(message_funcs)
        self.node_update_funcs = nn.ModuleList(node_update_funcs)

    def dgmg_msg(self, edges):
        """For an edge u->v, return concat([h_u, x_uv])"""
        return {MessageKey.repr_message: torch.cat([edges.src[MessageKey.repr_vertex], edges.data[MessageKey.repr_edge]], dim=1)}

    def dgmg_reduce(self, nodes, round):
        hv_old = get_nodes_features(nodes)
        m = nodes.mailbox[MessageKey.repr_message]
        message = torch.cat([hv_old.unsqueeze(1).expand(-1, m.size(1), -1), m], dim=2)
        node_activation = (self.message_funcs[round](message)).mean(1)

        return {MessageKey.repr_activation: node_activation}

    def forward(self, g):
        if g.number_of_edges() > 0:
            for t in range(self.num_prop_rounds):
                g.update_all(message_func=self.dgmg_msg, reduce_func=self.reduce_funcs[t])
                new_node_features = self.node_update_funcs[t](get_activations(g), get_node_features(g))

                set_nodes_features(g, new_node_features)


class ChooseNodeAction(DeepGGModule):
    def __init__(self, graph_hidden_size: int, node_hidden_size: int):
        super(ChooseNodeAction, self).__init__()

        self._choose_node = nn.Linear(graph_hidden_size + 2 * node_hidden_size, 1)

    def forward(self, graph: dgl.DGLGraph, graph_state_embedding: torch.Tensor, extra_node_state: torch.Tensor = None, choose_node=None):
        num_possible_nodes = graph.number_of_nodes()
        possible_nodes = range(graph.number_of_nodes())
        possible_nodes_embed = graph.nodes[possible_nodes].data[MessageKey.repr_vertex]
        per_node_extra_node_state = extra_node_state.expand(num_possible_nodes, -1)
        per_node_graph_state_embedding = graph_state_embedding.expand(num_possible_nodes, -1)
        per_node_choice = self._choose_node(torch.cat([per_node_graph_state_embedding, per_node_extra_node_state, possible_nodes_embed], dim=1))
        choices_logit = per_node_choice.view(-1, num_possible_nodes)
        choices_probs = F.softmax(choices_logit, dim=1)

        if not self.training:
            return Categorical(choices_probs).sample()
        else:
            assert choose_node < num_possible_nodes
            log_loss = F.cross_entropy(choices_logit, choose_node.view(1))/np.log(num_possible_nodes)
            self.add_log_loss(log_loss)
            return choose_node


class DeepGGActions:
    stop = 2
    add_node = 0
    add_edge = 1


class DeepGG(nn.Module):
    v_min: int = 0

    def __init__(self, v_max, node_hidden_size, num_prop_rounds):
        super(DeepGG, self).__init__()

        # Graph configuration
        self.v_max = v_max

        # Graph embedding module
        self._graph_embed = GraphEmbed(node_hidden_size)

        # Graph propagation module
        self._graph_prop = GraphProp(num_prop_rounds, node_hidden_size)
        #self._graph_prop2 = GraphProp2(num_prop_rounds, node_hidden_size)

        self._choose_node = ChooseNodeAction(self._graph_embed.graph_hidden_size, node_hidden_size)

        self._node_type_embed = nn.Embedding(2, node_hidden_size)
        self._initialize_hv = nn.Linear(self._graph_embed.graph_hidden_size, node_hidden_size)
        #self._init_node_activation = torch.zeros(1, 2 * node_hidden_size)
        self.register_buffer('_init_node_activation', torch.zeros(1, 2 * node_hidden_size))
        #self._unknown_node = torch.zeros(node_hidden_size)
        self.register_buffer('_unknown_node', torch.zeros(node_hidden_size))
        self._initialize_edge = nn.Linear(2 * node_hidden_size, node_hidden_size)

        # States
        action_values = [
            DeepGGActions.add_node,
            DeepGGActions.add_edge,
            DeepGGActions.stop
        ]
        self.register_buffer('_actions', torch.tensor(action_values))
        self._state_adding_node = DeepStateModule(self._graph_embed.graph_hidden_size, action_values)
        self._state_adding_edge = DeepStateModule(self._graph_embed.graph_hidden_size + node_hidden_size, action_values)

    def _initialize_node_repr(self, node_number, cur_graph_embedding):
        g = self._g
        hv_init = self._initialize_hv(cur_graph_embedding)
        g.nodes[node_number].data[MessageKey.repr_vertex] = hv_init
        g.nodes[node_number].data[MessageKey.repr_activation] = self._init_node_activation
        return hv_init

    def _initialize_edge_repr(self, src, dest):
        g = self._g
        repr_src = self._g.nodes[src].data[MessageKey.repr_vertex]
        repr_dest = self._g.nodes[dest].data[MessageKey.repr_vertex]
        he_init = self._initialize_edge(torch.cat([repr_src, repr_dest], dim=1))
        g.edges[src, dest].data[MessageKey.repr_edge] = he_init

    def forward_train(self, actions):
        self.prepare_for_train()

        cur_graph_embedding = self._graph_embed(self._g)
        current_state = self._state_adding_node(cur_graph_embedding, self.current_action(actions))
        last_added_node_idx = -1

        while not DeepGGActions.stop == int(current_state):
            cur_graph_embedding = self._graph_embed(self._g)

            if last_added_node_idx < 0 or DeepGGActions.add_node == int(current_state):
                self._g.add_nodes(1)
                last_added_node_idx = self._g.number_of_nodes() - 1
                self._initialize_node_repr(last_added_node_idx, cur_graph_embedding)

                current_state = self._state_adding_node(cur_graph_embedding, self.get_action_and_increment_step(actions))

            elif DeepGGActions.add_edge == int(current_state):
                last_added_node_embedding = self._g.nodes[last_added_node_idx].data[MessageKey.repr_vertex]

                # Choose source node
                expected_node_src = self.get_action_and_increment_step(actions)
                node_choice_src = self._choose_node(self._g, cur_graph_embedding, last_added_node_embedding, expected_node_src)

                embedding_src_node = self._g.nodes[expected_node_src].data[MessageKey.repr_vertex]
                expected_node_dest = self.get_action_and_increment_step(actions)
                node_choice_dest = self._choose_node(self._g, cur_graph_embedding, embedding_src_node, expected_node_dest)

                # Add edge to graph
                self._g.add_edge(node_choice_src, node_choice_dest)
                self._initialize_edge_repr(node_choice_src, node_choice_dest)

                # Update graph propagation
                self._graph_prop(self._g)

                # Update current graph state and compute next state
                cur_graph_embedding = self._graph_embed(self._g)
                state_add_edge = torch.cat([cur_graph_embedding, last_added_node_embedding], dim=1)
                current_state = self._state_adding_edge(state_add_edge, self.get_action_and_increment_step(actions))

        return self._get_log_loss()

    def forward_inference(self):
        cur_graph_embedding = self._graph_embed(self._g)
        current_state = self._state_adding_node(cur_graph_embedding)
        last_added_node_idx = -1
        construction_sequence = []

        while not DeepGGActions.stop == current_state and self._g.number_of_nodes() < self.v_max + 1:
            cur_graph_embedding = self._graph_embed(self._g)

            if last_added_node_idx < 0 or DeepGGActions.add_node == current_state:
                self._g.add_nodes(1)
                last_added_node_idx = self._g.number_of_nodes() - 1
                self._initialize_node_repr(last_added_node_idx, cur_graph_embedding)
                construction_sequence.append(DeepGGActions.add_node)

                current_state = self._state_adding_node(cur_graph_embedding)

            elif DeepGGActions.add_edge == current_state:
                last_added_node_embedding = get_node_feature(self._g, last_added_node_idx, self._unknown_node)

                # Choose source and destination nodes
                node_choice_src = self._choose_node(self._g, cur_graph_embedding, last_added_node_embedding)
                embedding_src_node = self._g.nodes[node_choice_src].data[MessageKey.repr_vertex]
                node_choice_dest = self._choose_node(self._g, cur_graph_embedding, embedding_src_node)

                # Add edge to graph if it not already exists
                if not self._g.has_edge_between(node_choice_src, node_choice_dest):
                    self._g.add_edge(node_choice_src, node_choice_dest)
                    self._initialize_edge_repr(node_choice_src, node_choice_dest)
                    construction_sequence.append(DeepGGActions.add_edge)
                    construction_sequence.append(int(node_choice_src.detach().cpu().numpy()))
                    construction_sequence.append(int(node_choice_dest.detach().cpu().numpy()))

                # Update graph propagation
                self._graph_prop(self._g)

                # Update current graph state and compute next state
                cur_graph_embedding = self._graph_embed(self._g)
                state_add_edge = torch.cat([cur_graph_embedding, last_added_node_embedding], dim=1)
                current_state = self._state_adding_edge(state_add_edge)

            if DeepGGActions.stop == current_state and self._g.number_of_nodes() < self.v_min:
                current_state = DeepGGActions.add_node

        return self._g, construction_sequence

    def _get_log_loss(self):
        loss = 0
        for module in self.modules():
            if isinstance(module, DeepGGModule):
                loss += module.get_log_loss()
        return loss

    def forward(self, graph: dgl.DGLGraph(), actions=None):
        self._g = graph

        self._g.set_n_initializer(dgl.init.zero_initializer)
        self._g.set_e_initializer(dgl.init.zero_initializer)

        if self.training:
            return self.forward_train(actions=actions)
        else:
            return self.forward_inference()

    def prepare_for_train(self):
        self._current_step = 0
        self.modules_prepare_training()

    @property
    def current_step(self):
        return self._current_step

    def current_action(self, actions):
        return actions[self._current_step] if self._current_step < len(actions) else DeepGGActions.stop

    def increment_step(self):
        old_step_count = self._current_step
        self._current_step += 1

        return old_step_count

    def get_action_and_increment_step(self, actions):
        step = self.increment_step()
        return actions[step] if step < len(actions) else DeepGGActions.stop

    def modules_prepare_training(self):
        for module in self.modules():
            if isinstance(module, DeepGGModule):
                module.prepare_training()
