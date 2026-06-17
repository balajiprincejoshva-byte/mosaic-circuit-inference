import unittest
import torch
import sys
import os

# Ensure core is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.rbm_thermo import MultiOmicRBM
from core.dynamics import LangevinSimulator
from core.inverse_design import TargetOptimizer

class TestMosaicPhysics(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.num_visible = 60
        self.num_hidden = 15
        self.rbm = MultiOmicRBM(self.num_visible, self.num_hidden)
        # Mock training setup
        dummy_data = torch.rand(10, self.num_visible)
        self.rbm.fit(dummy_data, epochs=1, batch_size=5) # This will cache base_weights
        
        self.base_vector = torch.rand(self.num_visible)

    def test_epigenetic_erosion(self):
        """
        Test 1: Initialize MultiOmicRBM, calculate base energy, apply erosion_factor=0.9,
        and assert that the new energy landscape has structurally flattened.
        """
        # Get base energy
        base_energy = self.rbm.calculate_free_energy(self.base_vector.unsqueeze(0)).item()
        
        # Apply heavy erosion (aging)
        self.rbm.apply_epigenetic_erosion(0.9)
        
        # Get aged energy
        aged_energy = self.rbm.calculate_free_energy(self.base_vector.unsqueeze(0)).item()
        
        # As weights and biases -> 0, the energy should approach 0 (flattened landscape)
        # Specifically, F(v) = -b^T v - sum log(1 + exp(c + W^T v)).
        # If b, c, W -> 0, F(v) -> - sum log(2).
        expected_flat_energy = -self.num_hidden * torch.log(torch.tensor(2.0)).item()
        
        # Assert the aged energy is closer to the flat energy than the original energy
        diff_base = abs(base_energy - expected_flat_energy)
        diff_aged = abs(aged_energy - expected_flat_energy)
        
        self.assertTrue(diff_aged < diff_base or diff_aged < 1e-1, 
                        "Epigenetic erosion failed to structurally flatten the energy landscape.")

    def test_langevin_convergence(self):
        """
        Test 2: Run LangevinSimulator for 50 steps.
        Assert output tensor has no NaN values and strictly matches (steps+1, num_features).
        """
        simulator = LangevinSimulator(self.rbm, temperature=0.1)
        steps = 50
        
        trajectory = simulator.simulate_trajectory(self.base_vector, steps=steps, dt=0.01)
        
        # Assert shape: (steps + 1, num_visible)
        self.assertEqual(trajectory.shape, (steps + 1, self.num_visible), 
                         "Langevin trajectory shape mismatch.")
        
        # Assert no NaN values
        self.assertFalse(torch.isnan(trajectory).any(), 
                         "Langevin dynamics produced NaN values (Hallucination detected).")

    def test_inverse_design_sparsity(self):
        """
        Test 3: Run TargetOptimizer Adam loop for 10 iterations.
        Assert delta_v is sparse (L1 penalty) and gradients calculated/cleared properly.
        """
        # Create an avoidance state
        avoidance_states = [torch.rand(self.num_visible)]
        
        optimizer = TargetOptimizer(self.rbm, self.base_vector, avoidance_states)
        
        # Run optimization for 10 steps with a high L1 penalty to enforce sparsity
        optimal_delta_v, top_targets = optimizer.optimize(steps=10, lr=0.1, lambda_avoid=1.0, alpha_l1=5.0)
        
        # Assert shape
        self.assertEqual(optimal_delta_v.shape, (self.num_visible,), "Delta_V shape mismatch.")
        
        # Assert that delta_v is disconnected from the computational graph (gradients cleared/detached)
        self.assertFalse(optimal_delta_v.requires_grad, "Delta_V should be detached from autograd.")
        
        # Assert sparsity: With high L1, many values should be close to 0
        # Check if the number of near-zero values is significant
        near_zeros = torch.sum(torch.abs(optimal_delta_v) < 1e-3).item()
        
        # Assert at least some values were zeroed out (sparse)
        self.assertTrue(near_zeros > 0, "Inverse design failed to produce a sparse perturbation vector.")

if __name__ == '__main__':
    unittest.main()
