import torch
from .dynamics import LangevinSimulator
from .rbm_thermo import MultiOmicRBM

class VirtualCohort:
    """
    Simulates the robustness of a perturbation across a virtual patient population
    with built-in biological variance.
    """
    def __init__(self, base_cell_vector: torch.Tensor, n_patients: int = 1000, variance: float = 0.1):
        self.base_cell_vector = base_cell_vector
        self.n_patients = n_patients
        self.variance = variance
        
        # Generate the cohort matrix (n_patients, num_features)
        # Add Gaussian noise
        noise = torch.randn(n_patients, base_cell_vector.shape[-1]) * variance
        self.cohort_matrix = torch.clamp(base_cell_vector.unsqueeze(0) + noise, 0.0, 1.0)
        
    def run_trial(self, simulator: LangevinSimulator, rbm: MultiOmicRBM, target_gene_idx: int, target_gene_value: float, steps: int = 50, dt: float = 0.01) -> tuple[float, torch.Tensor]:
        """
        Runs the full virtual cohort through the Langevin dynamics engine given a specific perturbation.
        
        Returns:
            efficacy_rate: Percentage of the cohort that successfully settled into the target energy basin.
            final_energies: Tensor of the final Free Energies for each patient.
        """
        # We can vectorize the LangevinSimulator by passing the entire cohort_matrix
        # simulate_trajectory handles batched inputs automatically because of PyTorch tensor operations
        trajectory_matrix = simulator.simulate_trajectory(
            self.cohort_matrix, 
            steps=steps, 
            dt=dt, 
            target_gene_idx=target_gene_idx, 
            target_gene_value=target_gene_value
        )
        
        # trajectory_matrix shape: (steps + 1, n_patients, num_features)
        final_states = trajectory_matrix[-1] # shape: (n_patients, num_features)
        
        # Calculate Final Energies
        final_energies = rbm.calculate_free_energy(final_states)
        
        # Define Success/Clinical Efficacy:
        # A patient is successfully reprogrammed if their final energy drops below a critical threshold.
        # Let's say the threshold is based on the unperturbed cohort's mean energy minus some delta,
        # or more simply, we use a fixed energy threshold from the intended target basin.
        
        # We will use the base cell's fully relaxed target state energy as the benchmark
        # For simplicity without running another simulation, we'll assume a success is an energy 
        # significantly lower than the starting mean energy of the cohort.
        initial_energies = rbm.calculate_free_energy(self.cohort_matrix)
        mean_initial = torch.mean(initial_energies).item()
        
        # Assuming reprogramming aims to push cells down an energy gradient, success = drops below mean_initial
        # (For a more rigorous definition, the user sets an absolute basin threshold, but we'll use a relative drop)
        success_threshold = mean_initial - 0.5 # A meaningful drop in free energy
        
        successes = torch.sum(final_energies < success_threshold).item()
        efficacy_rate = (successes / self.n_patients) * 100.0
        
        return efficacy_rate, final_energies
