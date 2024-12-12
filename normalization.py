import numpy as np
import torch
from typing import Dict

class Normalizer:
    """Performs continuous, online normalization of a given variable."""

    def __init__(self, name: str, device: torch.device, noise_std: float = 0.0, normalizer_fn = None):
        """Initializes the normalizer.

        Args:
            name: Name of the normalizer.
            noise_std: Standard deviation of the noise to add to the normalized
            variable.
        """
        self.name = name
        self.noise_std = noise_std
        self.device = device
        self.normalizer_fn = normalizer_fn

    def __call__(
        self,
        tensor: torch.Tensor,
        node_per_graph: torch.Tensor,
        is_training: bool = True,
        current_sum: torch.Tensor = None,
        num_accumulated: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Normalizes a given tensor.

        The normalizer tracks the mean and standard deviation of the input across
        all previous calls.

        Args:
            tensor: Tensor to normalize.
            node_per_graph: Tensor of number of nodes per graph.
            is_training: Whether the model is being trained.
            current_sum: Accumulated sum across previous calls. Can be None.
            num_accumulated: Accumulated number of elements across previous calls.
            Can be None.

        Returns:
            A dictionary with the normalized tensor, updated sum, and updated
            number of accumulated elements.
        """
        # Computes or updates the tracked statistics for the tensor.
        if is_training or current_sum is None:
            current_sum, num_accumulated = self.update_sum_and_count(
                tensor, node_per_graph, current_sum, num_accumulated
            )
            current_mean = current_sum / torch.max(
                num_accumulated, torch.tensor(1.0, device=tensor.device)
            )
            current_sq_sum = self._compute_sq_sum(
                tensor, node_per_graph, current_mean
            )
            current_std = torch.sqrt(
                current_sq_sum
                / torch.max(
                    num_accumulated - 1, torch.tensor(1.0, device=tensor.device)
                )
            )
        else:
            current_mean = current_sum / torch.max(
                num_accumulated, torch.tensor(1.0, device=tensor.device)
            )
            current_std = torch.sqrt(
                self._compute_sq_sum(tensor, node_per_graph, current_mean)
                / torch.max(
                    num_accumulated - 1, torch.tensor(1.0, device=tensor.device)
                )
            )

        # Normalizes the tensor.
        if self.normalizer_fn is not None:
            tensor = self.normalizer_fn(tensor, current_mean, current_std)
        else:
            tensor = (tensor - current_mean) / current_std

        # Add noise to the normalized tensor.
        if is_training and self.noise_std > 0.0:
            tensor += torch.randn_like(tensor) * self.noise_std

        return {
            f"{self.name}_normalized": tensor,
            f"{self.name}_sum": current_sum,
            f"{self.name}_num_accumulated": num_accumulated,
        }

    def _compute_sq_sum(
        self,
        tensor: torch.Tensor,
        node_per_graph: torch.Tensor,
        current_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the sum of squares of the difference from the mean."""
        num_graphs = node_per_graph.shape[0]
        num_nodes = tensor.shape[0]

        # Compute mean using the number of nodes in each graph
        mean = torch.zeros(num_nodes, device=tensor.device).scatter_add(0, self._batch_indices(node_per_graph), tensor)
        mean = mean / node_per_graph.repeat_interleave(node_per_graph)
        mean = torch.repeat_interleave(mean, node_per_graph, dim=0)

        return torch.sum((tensor - mean) ** 2, dim=list(range(1, tensor.dim())))

    def update_sum_and_count(
        self,
        tensor: torch.Tensor,
        node_per_graph: torch.Tensor,
        current_sum: torch.Tensor,
        num_accumulated: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates the accumulated sum and number of elements."""
        # If there is no previous sum, compute initial statistics
        if current_sum is None:
            current_sum = torch.zeros(tensor.shape[1:], device=self.device)
            num_accumulated = torch.tensor(0.0, device=self.device)

        # Update sum and count
        current_sum += torch.sum(tensor, dim=0)
        num_accumulated += float(tensor.shape[0])

        return current_sum, num_accumulated

    def _batch_indices(self, num_nodes: torch.Tensor) -> torch.Tensor:
        """Computes an ক্রম of indices for scattering operations.

        For example, if we have a tensor of shape [10, 3] and num_nodes is [2, 3, 5],
        this function will return [0, 0, 1, 1, 1, 2, 2, 2, 2, 2].

        Args:
            num_nodes: Tensor of number of nodes in each graph.

        Returns:
            Tensor of indices.
        """
        cumsum = torch.cumsum(torch.cat([torch.tensor([0], device=num_nodes.device), num_nodes]), dim=0)
        batch_indices = torch.zeros(cumsum[-1], dtype=torch.long, device=num_nodes.device)
        for i in range(len(num_nodes)):
            batch_indices[cumsum[i]:cumsum[i + 1]] = i
        return batch_indices

def batch_normalize_tensor(tensor: torch.Tensor, batch: torch.Tensor, epsilon=1e-8):
    """Apply per-batch normalization."""
    mean = scatter_mean(tensor, batch, dim=0)
    mean = mean.index_select(0, batch)
    var = scatter_mean((tensor - mean) ** 2, batch, dim=0)
    var = var.index_select(0, batch)
    return (tensor - mean) / torch.sqrt(var + epsilon)