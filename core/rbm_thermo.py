import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiOmicRBM(nn.Module):
    """
    Multi-Omic Restricted Boltzmann Machine.
    Visible layer represents joint multi-omic state.
    Hidden layer represents latent regulatory states.
    """
    def __init__(self, num_visible: int, num_hidden: int):
        super(MultiOmicRBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        
        # Weights (W)
        self.W = nn.Parameter(torch.randn(num_visible, num_hidden) * 0.01)
        # Visible bias (b)
        self.v_bias = nn.Parameter(torch.zeros(num_visible))
        # Hidden bias (c)
        self.h_bias = nn.Parameter(torch.zeros(num_hidden))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Calculates the probability of hidden nodes being active given visible nodes.
        p(h=1|v) = sigmoid(c + vW)
        """
        return torch.sigmoid(F.linear(v, self.W.t(), self.h_bias))

    def sample_hidden(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden states given visible states.
        """
        p_h_given_v = self.forward(v)
        h_sample = torch.bernoulli(p_h_given_v)
        return p_h_given_v, h_sample

    def sample_visible(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible states given hidden states.
        p(v=1|h) = sigmoid(b + hW^T)
        """
        p_v_given_h = torch.sigmoid(F.linear(h, self.W, self.v_bias))
        v_sample = torch.bernoulli(p_v_given_h)
        return p_v_given_h, v_sample

    def compute_energy(self, v: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Physical energy function: E(v, h) = -b^T v - c^T h - v^T W h
        Expects v of shape (Batch, V), h of shape (Batch, H).
        Returns scalar energy per sample, shape (Batch,)
        """
        v_term = torch.matmul(v, self.v_bias)
        h_term = torch.matmul(h, self.h_bias)
        # v^T W h is (B, 1) if computed per sample. Efficiently: sum(v * (h @ W^T), dim=1)
        w_term = torch.sum(torch.matmul(v, self.W) * h, dim=1)
        
        return -(v_term + h_term + w_term)

    def calculate_free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Returns scalar free energy (basin of attraction) for any given cell state vector.
        F(v) = -b^T v - sum_j log(1 + exp(c_j + v^T W_{:, j}))
        Returns scalar free energy per sample, shape (Batch,)
        """
        v_term = torch.matmul(v, self.v_bias)
        # linear projection of v onto h space + h_bias -> c + vW
        hidden_activation = F.linear(v, self.W.t(), self.h_bias)
        
        # Softplus is exactly log(1 + exp(x))
        h_term = torch.sum(F.softplus(hidden_activation), dim=1)
        
        return -(v_term + h_term)

    def calculate_plasticity_entropy(self, v: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
        """
        Quantifies cellular plasticity (stemness vs. differentiation) using thermodynamic entropy.
        S = - sum p(h|v) * log(p(h|v))
        Returns scalar S (High S = multipotent/plastic state, Low S = locked/differentiated state).
        """
        # Activation probabilities of the hidden layer p(h|v) = sigmoid(c + vW)
        hidden_activation = torch.nn.functional.linear(v, self.W.t(), self.h_bias)
        p_h_given_v = torch.sigmoid(hidden_activation)
        
        # Shannon entropy calculation
        entropy = -torch.sum(p_h_given_v * torch.log(p_h_given_v + epsilon), dim=1)
        
        return entropy

    def contrastive_divergence(self, v_data: torch.Tensor, k: int = 1, lr: float = 0.01) -> float:
        """
        Train using Contrastive Divergence (CD-k) to learn weights W.
        Returns the reconstruction error (pseudo-loss).
        """
        batch_size = v_data.shape[0]
        
        # Positive phase
        ph_prob, ph_sample = self.sample_hidden(v_data)
        
        # Negative phase (Gibbs sampling)
        v_k = v_data
        for _ in range(k):
            _, h_k = self.sample_hidden(v_k)
            _, v_k = self.sample_visible(h_k)
            
        nh_prob, _ = self.sample_hidden(v_k)
        
        # Update weights and biases
        # W_update = v_data^T ph_prob - v_k^T nh_prob
        # Since v_data is (B, V) and ph_prob is (B, H), their outer product summed over batch is (V, H)
        w_grad = (torch.matmul(v_data.t(), ph_prob) - torch.matmul(v_k.t(), nh_prob)) / batch_size
        v_bias_grad = torch.mean(v_data - v_k, dim=0)
        h_bias_grad = torch.mean(ph_prob - nh_prob, dim=0)
        
        with torch.no_grad():
            self.W += lr * w_grad
            self.v_bias += lr * v_bias_grad
            self.h_bias += lr * h_bias_grad
            
        # Compute reconstruction error for monitoring
        recon_error = F.mse_loss(v_data, v_k)
        return recon_error.item()

    def fit(self, data_tensor: torch.Tensor, epochs: int = 10, batch_size: int = 64, k: int = 1, lr: float = 0.01) -> list[float]:
        """
        Utility training loop.
        """
        dataset = torch.utils.data.TensorDataset(data_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        history = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                v_batch = batch[0]
                loss = self.contrastive_divergence(v_batch, k=k, lr=lr)
                epoch_loss += loss
            history.append(epoch_loss / len(loader))
            
        self.cache_base_weights()
            
        return history

    def cache_base_weights(self):
        """
        Caches the trained pristine weights for use during epigenetic erosion.
        """
        self.base_W = self.W.detach().clone()
        self.base_v_bias = self.v_bias.detach().clone()
        self.base_h_bias = self.h_bias.detach().clone()

    def apply_epigenetic_erosion(self, erosion_factor: float):
        """
        Models biological aging by flattening the energy landscape.
        erosion_factor: 0.0 (young) to 1.0 (dead).
        """
        erosion_factor = min(max(erosion_factor, 0.0), 1.0)
        
        with torch.no_grad():
            self.W.copy_(self.base_W * (1.0 - erosion_factor))
            self.v_bias.copy_(self.base_v_bias * (1.0 - erosion_factor))
            self.h_bias.copy_(self.base_h_bias * (1.0 - erosion_factor))
