import torch

class SpatialTissueEnvironment:
    """
    High-performance spatial neighborhood solver to calculate cell-to-cell 
    interaction matrices using PyTorch.
    """
    def __init__(self, coords: torch.Tensor, sigma: float = 1.0):
        """
        coords: Tensor of shape (N_cells, 2) or (N_cells, 3).
        sigma: Paracrine signaling radius for the RBF kernel.
        """
        self.coords = coords
        self.sigma = sigma
        self.N = coords.shape[0]
        
        # Pre-compute the RBF Kernel Matrix K_ij
        # K_ij = exp(-||x_i - x_j||^2 / 2sigma^2)
        # Using torch.cdist for highly optimized pairwise distance computation
        dist_matrix = torch.cdist(coords, coords, p=2.0)
        
        # Zero out diagonal so a cell doesn't spatially couple with itself
        mask = torch.eye(self.N, device=coords.device).bool()
        
        self.K = torch.exp(-(dist_matrix ** 2) / (2.0 * (self.sigma ** 2)))
        self.K.masked_fill_(mask, 0.0)

    def get_neighborhood_influence(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Returns a weighted composition vector of the hidden states of all surrounding cells.
        hidden_states: Tensor of shape (N_cells, num_hidden)
        Returns: Tensor of shape (N_cells, num_hidden) representing sum_j K_ij h_j
        """
        # Matrix multiplication: K is (N, N), hidden_states is (N, H)
        # Result is (N, H)
        return torch.matmul(self.K, hidden_states)
