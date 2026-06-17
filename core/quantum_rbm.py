import torch
import torch.nn as nn
import torch.nn.functional as F
import opt_einsum as oe

class QuantumInspiredRBM(nn.Module):
    """
    Quantum-Inspired Tensor Network RBM.
    Compresses the standard dense weight matrix into low-rank MPS-style tensors.
    """
    def __init__(self, num_visible: int, num_hidden: int, rank: int = 4):
        super(QuantumInspiredRBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.rank = rank
        
        # Decomposed weight factors: W \approx A @ B
        # A: (num_visible, rank)
        # B: (rank, num_hidden)
        self.A = nn.Parameter(torch.randn(num_visible, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, num_hidden) * 0.01)
        
        self.v_bias = nn.Parameter(torch.zeros(num_visible))
        self.h_bias = nn.Parameter(torch.zeros(num_hidden))
        
        self.W_spatial = nn.Parameter(torch.randn(num_hidden, num_hidden) * 0.01)
        self.gamma = 1.0

    def get_effective_W(self) -> torch.Tensor:
        """Contract tensors to get the effective dense weight matrix for compatibility."""
        return torch.matmul(self.A, self.B)
        
    def calculate_quantum_free_energy(self, v: torch.Tensor, temperature: float = 100.0) -> torch.Tensor:
        """
        Calculates Free Energy using tensor contraction without instantiating full W.
        """
        v_term = torch.matmul(v, self.v_bias)
        
        # Tensor network contraction: v @ (A @ B)
        # Contract v with A first, then with B using opt_einsum
        vA = oe.contract('bv,vr->br', v, self.A)
        hidden_activation = oe.contract('br,rh->bh', vA, self.B) + self.h_bias
        
        h_term = torch.sum(torch.nn.functional.softplus(hidden_activation / temperature), dim=1)
        return -(v_term + h_term)
        
    def calculate_plasticity_entropy(self, v: torch.Tensor, epsilon: float = 1e-9, temperature: float = 100.0) -> torch.Tensor:
        """
        Quantum version of plasticity entropy calculation.
        """
        vA = oe.contract('bv,vr->br', v, self.A)
        hidden_activation = oe.contract('br,rh->bh', vA, self.B) + self.h_bias
        p_h_given_v = torch.sigmoid(hidden_activation / temperature)
        
        entropy = -torch.sum(
            p_h_given_v * torch.log(p_h_given_v + epsilon) + 
            (1.0 - p_h_given_v) * torch.log(1.0 - p_h_given_v + epsilon), 
            dim=1
        )
        return entropy

    def copy_from_dense(self, dense_rbm):
        """Initializes A and B from a dense W matrix using Singular Value Decomposition."""
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(dense_rbm.W, full_matrices=False)
            self.A.copy_(U[:, :self.rank] * torch.sqrt(S[:self.rank]))
            self.B.copy_(Vh[:self.rank, :] * torch.sqrt(S[:self.rank]).unsqueeze(1))
            self.v_bias.copy_(dense_rbm.v_bias)
            self.h_bias.copy_(dense_rbm.h_bias)
            self.W_spatial.copy_(dense_rbm.W_spatial)
