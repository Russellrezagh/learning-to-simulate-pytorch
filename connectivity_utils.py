import torch
from typing import Tuple

def compute_connectivity_for_batch_pyfunc(
    positions: torch.Tensor,
    n_node: torch.Tensor,
    radius: float,
    add_self_edges: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes connectivity for a batch of graphs using a radius-based approach.

    Args:
        positions: Positions of nodes in the batch of graphs.
                   Shape: [num_nodes_in_batch, num_dims].
        n_node: Number of nodes for each graph in the batch.
                Shape: [num_graphs_in_batch].
        radius: Radius of connectivity.
        add_self_edges: Whether to include self edges or not.

    Returns:
        senders: Sender indices [num_edges_in_batch].
        receivers: Receiver indices [num_edges_in_batch].
        n_edge: Number of edges per graph [num_graphs_in_batch].
    """

    # Calculate pairwise distances using cdist.
    distances = torch.cdist(positions, positions, p=2.0)

    # Create a mask for distances within the radius.
    mask = distances <= radius

    if not add_self_edges:
        # Remove self-edges
        self_edge_mask = torch.eye(distances.shape[0], dtype=torch.bool, device=positions.device)
        mask = mask & ~self_edge_mask

    # Get sender and receiver indices from the mask.
    receivers, senders = torch.where(mask)  # Note: senders and receivers are swapped

    # Calculate the number of edges per graph.
    n_edge = torch.bincount(receivers, minlength=positions.shape[0])

    # Filter out zero-count nodes from n_edge.
    n_edge = n_edge.nonzero().squeeze()

    return senders, receivers, n_edge