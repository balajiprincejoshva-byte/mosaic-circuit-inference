import torch
from .rbm_thermo import MultiOmicRBM

class TargetOptimizer:
    """
    Reverse-engineers the optimal multi-gene perturbation required to push a cell 
    into a target state safely using a gradient-based optimization on the Free Energy landscape.
    """
    def __init__(self, rbm: MultiOmicRBM, target_state: torch.Tensor, avoidance_states: list[torch.Tensor]):
        self.rbm = rbm
        self.rbm.eval()
        self.target_state = target_state
        self.avoidance_states = avoidance_states
        
    def optimize(self, steps: int = 100, lr: float = 0.1, lambda_avoid: float = 2.0, alpha_l1: float = 0.5) -> tuple[torch.Tensor, list[tuple[int, float]]]:
        """
        Optimizes a continuous perturbation vector (delta_v) to minimize target free energy 
        while maximizing avoidance free energy and enforcing sparsity.
        
        Returns:
            optimal_delta_v: The optimized continuous perturbation vector.
            top_targets: A list of tuples (gene_idx, dosage) for the top 3 absolute magnitudes.
        """
        num_visible = self.target_state.shape[-1]
        
        # Initialize continuous perturbation vector
        delta_v = torch.zeros(num_visible, requires_grad=True)
        
        optimizer = torch.optim.Adam([delta_v], lr=lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            
            # Simulate the state after applying the perturbation delta_v
            # The state must remain within [0, 1] biological bounds
            perturbed_target = torch.clamp(self.target_state + delta_v, 0.0, 1.0)
            
            # 1. Minimize Free Energy of the target basin
            target_energy = self.rbm.calculate_free_energy(perturbed_target.unsqueeze(0)).squeeze()
            
            # 2. Maximize Free Energy of the avoidance basins (Penalty if they drop)
            avoid_penalty = 0.0
            if self.avoidance_states:
                for avoid_s in self.avoidance_states:
                    perturbed_avoid = torch.clamp(avoid_s + delta_v, 0.0, 1.0)
                    avoid_e = self.rbm.calculate_free_energy(perturbed_avoid.unsqueeze(0)).squeeze()
                    # We penalize LOW avoidance energy, so we subtract it from the loss
                    avoid_penalty += avoid_e
            
            # 3. Enforce Sparsity (L1 Regularization) to ensure we find a few specific drug targets
            l1_penalty = torch.norm(delta_v, p=1)
            
            # Total Loss Objective
            loss = target_energy - lambda_avoid * avoid_penalty + alpha_l1 * l1_penalty
            
            loss.backward()
            optimizer.step()
            
        # Optimization complete. Detach and extract optimal vector
        optimal_delta_v = delta_v.detach().clone()
        
        # Identify top 3 highest magnitude perturbations
        # (Assuming indices correspond to global features: RNA, ATAC, ADT)
        magnitudes = torch.abs(optimal_delta_v)
        top_indices = torch.topk(magnitudes, k=3).indices.tolist()
        
        top_targets = []
        for idx in top_indices:
            dosage = optimal_delta_v[idx].item()
            if abs(dosage) > 1e-4: # Only include if it's a meaningful perturbation
                top_targets.append((idx, dosage))
                
        return optimal_delta_v, top_targets
