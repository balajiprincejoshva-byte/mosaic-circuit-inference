import torch

class SystemicOrganNetwork:
    """
    Simulates the systemic collateral toxicity of localized therapeutic perturbations.
    Models multiple organ thermodynamic states.
    """
    def __init__(self, num_features: int):
        self.num_features = num_features
        # Mock tissue profiles representing distinct homeostatic attractors
        # We use random normal distributions centered at different means
        self.organ_profiles = {
            "Target_Tissue": torch.randn(num_features) * 1.5 + 0.5,
            "Cardiac_Tissue": torch.randn(num_features) * 2.0 + 1.0,
            "Neural_Tissue": torch.randn(num_features) * 1.5 - 0.5,
            "Hepatic_Tissue": torch.randn(num_features) * 3.0 + 2.0
        }
        
    def calculate_systemic_toxicity(self, rbm, perturbation_vector: torch.Tensor, is_quantum=False):
        """
        Calculates toxicity based on free energy shifts (\Delta E) across organ profiles.
        Returns a dictionary of risk scores (0-100).
        """
        risks = {}
        for organ, profile in self.organ_profiles.items():
            profile_batch = profile.unsqueeze(0)
            perturbed_profile = (profile + perturbation_vector).unsqueeze(0)
            
            if is_quantum and hasattr(rbm, 'calculate_quantum_free_energy'):
                base_energy = rbm.calculate_quantum_free_energy(profile_batch).item()
                pert_energy = rbm.calculate_quantum_free_energy(perturbed_profile).item()
            else:
                # If using standard RBM, calculate standard free energy
                base_energy = rbm.calculate_free_energy(profile_batch).item()
                pert_energy = rbm.calculate_free_energy(perturbed_profile).item()
                
            # Delta E = E(pert) - E(base)
            # Positive Delta E means the perturbation destabilized the homeostatic basin.
            delta_E = pert_energy - base_energy
            
            # Map Delta E to a 0-100 risk score
            # A large positive delta indicates high toxicity / instability
            risk_score = min(max(delta_E * 2.5 + 10.0, 0.0), 100.0)
            
            # Add a bit of natural variance for the simulation dashboard
            risk_score += torch.rand(1).item() * 5.0
            
            risks[organ] = min(risk_score, 100.0)
            
        return risks
