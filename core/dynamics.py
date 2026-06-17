import torch
from .rbm_thermo import MultiOmicRBM

class LangevinSimulator:
    """
    Simulates the physical trajectory of a cell moving through the energy landscape 
    over time after a perturbation, using Langevin dynamics (Euler-Maruyama method).
    """
    def __init__(self, rbm: MultiOmicRBM, temperature: float = 1.0):
        self.rbm = rbm
        self.rbm.eval()
        self.temperature = temperature

    def step_forward(self, v_current: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        """
        Calculates the next state of the cell vector over delta_t.
        v_{t+1} = v_t - nabla F(v) * dt + sqrt(2 * T * dt) * noise
        """
        # Ensure we don't mutate the original in-place unintentionally
        v = v_current.clone()
        
        # Calculate analytical gradient nabla F(v)
        # dF/dv = -b - W * sigmoid(c + v W)^T
        hidden_activation = torch.nn.functional.linear(v, self.rbm.W.t(), self.rbm.h_bias)
        h_prob = torch.sigmoid(hidden_activation)
        grad_f = -self.rbm.v_bias - torch.matmul(h_prob, self.rbm.W.t())
        
        # Generate Gaussian thermal noise
        # Shape matches v
        noise = torch.randn_like(v)
        
        # Langevin step
        deterministic_step = -grad_f * dt
        stochastic_step = torch.sqrt(torch.tensor(2.0 * self.temperature * dt)) * noise
        
        v_next = v + deterministic_step + stochastic_step
        
        # Constrain to biological bounds (e.g., [0, 1] if normalized probabilities/counts)
        v_next = torch.clamp(v_next, 0.0, 1.0)
        
        return v_next

    def simulate_trajectory(self, v_start: torch.Tensor, steps: int = 100, dt: float = 0.01, target_gene_idx: int = None, target_gene_value: float = None) -> torch.Tensor:
        """
        Simulates the trajectory of a cell from v_start.
        Returns a time-series matrix of shape (steps + 1, num_visible).
        If target_gene_idx is provided, forces that node to target_gene_value 
        at each step to simulate an active clamp (knockout/overexpression).
        """
        trajectory = [v_start.clone().squeeze()]
        v_current = v_start.clone().unsqueeze(0) if v_start.dim() == 1 else v_start.clone()
        
        # Apply initial clamp if active perturbation
        if target_gene_idx is not None and target_gene_value is not None:
            v_current[:, target_gene_idx] = target_gene_value
            trajectory[0] = v_current.clone().squeeze()
            
        for _ in range(steps):
            v_current = self.step_forward(v_current, dt=dt)
            
            # Enforce perturbation clamp during relaxation
            if target_gene_idx is not None and target_gene_value is not None:
                v_current[:, target_gene_idx] = target_gene_value
                
            trajectory.append(v_current.clone().squeeze())
            
        # Stack into (steps+1, V)
        return torch.stack(trajectory)
