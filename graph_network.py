import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

# Typing alias.
ReducerFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]  # Updated name

def build_mlp(hidden_size: int, num_hidden_layers: int, output_size: int, layer_norm: bool = True) -> nn.Module:
    """Builds an MLP with optional layer normalization."""
    layers = []
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(hidden_size, hidden_size))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers)

class EncodeProcessDecode(nn.Module):
    """Encode-Process-Decode function approximator for learnable simulator."""

    def __init__(
            self,
            latent_size: int,
            mlp_hidden_size: int,
            mlp_num_hidden_layers: int,
            num_message_passing_steps: int,
            output_size: int,
            reducer: ReducerFn = torch.sum,  # Updated to a more general type
            name: str = "EncodeProcessDecode"):
        """Inits the model.

        Args:
            latent_size: Size of the node and edge latent representations.
            mlp_hidden_size: Hidden layer size for all MLPs.
            mlp_num_hidden_layers: Number of hidden layers in all MLPs.
            num_message_passing_steps: Number of message passing steps.
            output_size: Output size of the decode node representations.
            reducer: Reduction function to be used when aggregating
                the edges in the nodes in the interaction network.
            name: Name of the model.
        """

        super().__init__()

        self._latent_size = latent_size
        self._mlp_hidden_size = mlp_hidden_size
        self._mlp_num_hidden_layers = mlp_num_hidden_layers
        self._num_message_passing_steps = num_message_passing_steps
        self._output_size = output_size
        self._reducer = reducer
        self._name = name

        self._networks_builder()

    def forward(self, input_graph: dict) -> torch.Tensor:
        """Forward pass of the learnable dynamics model."""

        # Encode the input_graph.
        latent_graph_0 = self._encode(input_graph)

        # Do `m` message passing steps in the latent graphs.
        latent_graph_m = self._process(latent_graph_0)

        # Decode from the last latent graph.
        return self._decode(latent_graph_m)

    def _networks_builder(self):
        """Builds the networks."""

        def build_mlp_with_layer_norm():
            return build_mlp(
                hidden_size=self._mlp_hidden_size,
                num_hidden_layers=self._mlp_num_hidden_layers,
                output_size=self._latent_size,
                layer_norm=True
            )

        # The encoder graph network independently encodes edge and node features.
        self._encoder_network = GraphIndependent(
            edge_model_fn=build_mlp_with_layer_norm,
            node_model_fn=build_mlp_with_layer_norm
        )

        # Create `num_message_passing_steps` graph networks with unshared parameters
        # that update the node and edge latent features.
        self._processor_networks = nn.ModuleList()
        for _ in range(self._num_message_passing_steps):
            self._processor_networks.append(
                InteractionNetwork(
                    edge_model_fn=build_mlp_with_layer_norm,
                    node_model_fn=build_mlp_with_layer_norm,
                    reducer=self._reducer
                )
            )

        # The decoder MLP decodes node latent features into the output size.
        self._decoder_network = build_mlp(
            hidden_size=self._mlp_hidden_size,
            num_hidden_layers=self._mlp_num_hidden_layers,
            output_size=self._output_size,
            layer_norm=False
        )

    def _encode(self, input_graph: dict) -> dict:
        """Encodes the input graph features into a latent graph."""

        # Copy the globals to all of the nodes, if applicable.
        if "globals" in input_graph:
            broadcasted_globals = input_graph["globals"].repeat(input_graph["nodes"].shape[0], 1)
            nodes = torch.cat([input_graph["nodes"], broadcasted_globals], dim=-1)
        else:
            nodes = input_graph["nodes"]

        # Encode the node and edge features.
        latent_graph_0 = self._encoder_network(
            {"nodes": nodes, "edges": input_graph["edges"],
             "senders": input_graph["senders"], "receivers": input_graph["receivers"]}
        )
        return latent_graph_0

    def _process(self, latent_graph_0: dict) -> dict:
        """Processes the latent graph with several steps of message passing."""

        # Do `m` message passing steps in the latent graphs.
        latent_graph_prev_k = latent_graph_0
        latent_graph_k = latent_graph_0
        for processor_network_k in self._processor_networks:
            latent_graph_k = self._process_step(
                processor_network_k, latent_graph_prev_k
            )
            latent_graph_prev_k = latent_graph_k

        latent_graph_m = latent_graph_k
        return latent_graph_m

    def _process_step(self, processor_network_k: nn.Module, latent_graph_prev_k: dict) -> dict:
        """Single step of message passing with node/edge residual connections."""

        # One step of message passing.
        latent_graph_k = processor_network_k(latent_graph_prev_k)

        # Add residuals.
        latent_graph_k["nodes"] = latent_graph_k["nodes"] + latent_graph_prev_k["nodes"]
        latent_graph_k["edges"] = latent_graph_k["edges"] + latent_graph_prev_k["edges"]
        return latent_graph_k

    def _decode(self, latent_graph: dict) -> torch.Tensor:
        """Decodes from the latent graph."""
        return self._decoder_network(latent_graph["nodes"])

class GraphIndependent(nn.Module):
    """GraphIndependent layer from the Graph Nets library."""
    def __init__(self, edge_model_fn: Callable, node_model_fn: Callable):
        super().__init__()
        self.edge_model_fn = edge_model_fn()
        self.node_model_fn = node_model_fn()

    def forward(self, graph: dict) -> dict:
        return {
            "nodes": self.node_model_fn(graph["nodes"]),
            "edges": self.edge_model_fn(graph["edges"]),
            "senders": graph["senders"],
            "receivers": graph["receivers"]
        }

class InteractionNetwork(nn.Module):
    """Interaction Network layer from the Graph Nets library."""
    def __init__(self, edge_model_fn: Callable, node_model_fn: Callable, reducer: ReducerFn = torch.sum):
        super().__init__()
        self.edge_model_fn = edge_model_fn()
        self.node_model_fn = node_model_fn()
        self.reducer = reducer

    def forward(self, graph: dict) -> dict:
        senders = graph["senders"]
        receivers = graph["receivers"]
        edges = graph["edges"]
        nodes = graph["nodes"]

        sender_nodes = nodes[senders]
        receiver_nodes = nodes[receivers]

        # Edge update
        edge_inputs = torch.cat([sender_nodes, receiver_nodes, edges], dim=-1)
        updated_edges = self.edge_model_fn(edge_inputs)

        # Node update
        # Aggregate edge outputs from connected nodes
        num_nodes = nodes.shape[0]
        agg_received_edges = torch.zeros_like(nodes).index_add_(0, receivers, updated_edges)

        node_inputs = torch.cat([nodes, agg_received_edges], dim=-1)
        updated_nodes = self.node_model_fn(node_inputs)

        return {
            "nodes": updated_nodes,
            "edges": updated_edges,
            "senders": senders,
            "receivers": receivers
        }