import torch
from .rbm_thermo import MultiOmicRBM

class PerturbationSimulator:
    """
    Predicts cell fate transitions by simulating gene knockouts or overexpression.
    Uses an analytical gradient solver on the free energy landscape.
    """
    def __init__(self, rbm: MultiOmicRBM):
        """
        Initialize with a trained Multi-Omic RBM.
        """
        self.rbm = rbm
        self.rbm.eval() # Ensure in evaluation mode

    def compute_analytical_gradient(self, v: torch.Tensor, target_gene_idx: int) -> torch.Tensor:
        """
        Computes the analytical gradient of the free energy landscape F(v) 
        with respect to the target gene's visible node: nabla_{v_target} F(v).
        
        F(v) = -b^T v - sum_j softplus(c_j + W_{:,j}^T v)
        dF/dv_i = -b_i - sum_j sigmoid(c_j + W_{:,j}^T v) * W_{i, j}
        """
        b_i = self.rbm.v_bias[target_gene_idx]
        
        # c + v W  (shape: Batch x Hidden)
        hidden_activation = torch.nn.functional.linear(v, self.rbm.W.t(), self.rbm.h_bias)
        
        # sigmoid(c + v W) (shape: Batch x Hidden)
        h_prob = torch.sigmoid(hidden_activation)
        
        # W_{target, :} (shape: Hidden)
        w_i = self.rbm.W[target_gene_idx, :]
        
        # sum_j sigmoid(c_j + v^T W_{:,j}) * W_{i, j}
        # h_prob is (B, H), w_i is (H,) -> sum over H
        interaction_grad = torch.matmul(h_prob, w_i) # (Batch,)
        
        grad = -b_i - interaction_grad
        return grad

    def _calculate_safety(self, dest_state: torch.Tensor, avoidance_states: list[torch.Tensor]) -> tuple[dict[int, float], float]:
        """
        Calculates Delta E_off-target and Safety Score.
        """
        if not avoidance_states:
            return {}, 100.0
            
        dest_energy = self.rbm.calculate_free_energy(dest_state.unsqueeze(0)).item()
        off_target_energies = {}
        min_barrier = float('inf')
        
        for i, avoid_v in enumerate(avoidance_states):
            avoid_energy = self.rbm.calculate_free_energy(avoid_v.unsqueeze(0)).item()
            barrier = avoid_energy - dest_energy
            off_target_energies[i] = barrier
            if barrier < min_barrier:
                min_barrier = barrier
                
        if min_barrier < 0:
            safety_score = 0.0
        else:
            safety_score = min(100.0, (min_barrier / 10.0) * 100.0)
            
        return off_target_energies, safety_score

    def simulate_knockout(self, cell_vector: torch.Tensor, target_gene_idx: int, steps: int = 10, step_size: float = 0.1, avoidance_states: list[torch.Tensor] = None) -> dict:
        """
        Simulates a knockout by forcing the target node to 0 and relaxing the state.
        Now evaluates safety against avoidance_states.
        """
        if cell_vector.dim() == 1:
            v_current = cell_vector.unsqueeze(0).clone()
        else:
            v_current = cell_vector.clone()
            
        initial_energy = self.rbm.calculate_free_energy(v_current).item()
        
        # Apply Knockout perturbation
        v_current[:, target_gene_idx] = 0.0
        
        for _ in range(steps):
            hidden_activation = torch.nn.functional.linear(v_current, self.rbm.W.t(), self.rbm.h_bias)
            h_prob = torch.sigmoid(hidden_activation)
            full_grad = -self.rbm.v_bias - torch.matmul(h_prob, self.rbm.W.t())
            
            v_current = v_current - step_size * full_grad
            v_current[:, target_gene_idx] = 0.0
            v_current = torch.clamp(v_current, 0.0, 1.0)
            
        final_energy = self.rbm.calculate_free_energy(v_current).item()
        delta_E = final_energy - initial_energy
        dest_state = v_current.squeeze(0)
        
        off_target_energies, safety_score = self._calculate_safety(dest_state, avoidance_states)
        
        return {
            'delta_E': delta_E,
            'destination_state': dest_state,
            'off_target_shifts': off_target_energies,
            'safety_score': safety_score
        }

    def simulate_overexpression(self, cell_vector: torch.Tensor, target_gene_idx: int, steps: int = 10, step_size: float = 0.1, avoidance_states: list[torch.Tensor] = None) -> dict:
        """
        Simulates overexpression by forcing the target node to 1.0.
        Evaluates safety against avoidance_states.
        """
        if cell_vector.dim() == 1:
            v_current = cell_vector.unsqueeze(0).clone()
        else:
            v_current = cell_vector.clone()
            
        initial_energy = self.rbm.calculate_free_energy(v_current).item()
        
        # Apply Overexpression perturbation
        v_current[:, target_gene_idx] = 1.0
        
        for _ in range(steps):
            hidden_activation = torch.nn.functional.linear(v_current, self.rbm.W.t(), self.rbm.h_bias)
            h_prob = torch.sigmoid(hidden_activation)
            full_grad = -self.rbm.v_bias - torch.matmul(h_prob, self.rbm.W.t())
            
            v_current = v_current - step_size * full_grad
            v_current[:, target_gene_idx] = 1.0
            v_current = torch.clamp(v_current, 0.0, 1.0)
            
        final_energy = self.rbm.calculate_free_energy(v_current).item()
        delta_E = final_energy - initial_energy
        dest_state = v_current.squeeze(0)
        
        off_target_energies, safety_score = self._calculate_safety(dest_state, avoidance_states)
        
        return {
            'delta_E': delta_E,
            'destination_state': dest_state,
            'off_target_shifts': off_target_energies,
            'safety_score': safety_score
        }
