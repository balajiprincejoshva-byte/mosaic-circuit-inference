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

    def step_forward(self, v_current: torch.Tensor, dt: float = 0.01, spatial_env=None) -> torch.Tensor:
        """
        Calculates the next state of the cell vector over delta_t.
        v_{t+1} = v_t - nabla F(v) * dt + sqrt(2 * T * dt) * noise
        If spatial_env is provided, calculates the coupled spatial gradient.
        """
        # Ensure v_current requires grad to compute the exact analytical gradient via PyTorch
        v = v_current.clone().detach().requires_grad_(True)
        
        # Calculate Free Energy (incorporates Mean-Field spatial coupling if provided)
        energy = self.rbm.calculate_free_energy(v, spatial_env)
        
        # Autograd dynamically computes the exact gradient nabla F(v)
        # We sum the energies to compute independent gradients for all cells simultaneously in parallel
        grad_f = torch.autograd.grad(energy.sum(), v)[0]
        
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

    def simulate_trajectory(self, v_start: torch.Tensor, steps: int = 100, dt: float = 0.01, target_gene_idx: int = None, target_gene_value: float = None, spatial_env=None) -> torch.Tensor:
        """
        Simulates the trajectory of a cell or tissue grid from v_start.
        Returns a time-series matrix of shape (steps + 1, ..., num_visible).
        """
        trajectory = [v_start.clone().squeeze()]
        v_current = v_start.clone().unsqueeze(0) if v_start.dim() == 1 else v_start.clone()
        
        # Apply initial clamp if active perturbation
        if target_gene_idx is not None and target_gene_value is not None:
            v_current[:, target_gene_idx] = target_gene_value
            trajectory[0] = v_current.clone().squeeze()
            
        for _ in range(steps):
            v_current = self.step_forward(v_current, dt=dt, spatial_env=spatial_env)
            
            # Enforce perturbation clamp during relaxation
            if target_gene_idx is not None and target_gene_value is not None:
                v_current[:, target_gene_idx] = target_gene_value
                
            trajectory.append(v_current.clone().squeeze())
            
        # Stack into (steps+1, ...)
        return torch.stack(trajectory)
