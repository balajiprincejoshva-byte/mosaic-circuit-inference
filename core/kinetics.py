import torch

class MicroenvironmentEnv:
    """
    OpenAI Gym-style environment simulator for time-series cellular trajectories 
    and drug resistance development.
    """
    def __init__(self, initial_state: torch.Tensor, max_steps: int=20, mutation_std: float=0.02):
        self.initial_state = initial_state.clone()
        self.state = initial_state.clone()
        self.max_steps = max_steps
        self.current_step = 0
        self.mutation_std = mutation_std

    def reset(self) -> torch.Tensor:
        """Resets the environment to the initial state."""
        self.state = self.initial_state.clone()
        self.current_step = 0
        return self.state

    def step(self, action: torch.Tensor) -> tuple:
        """
        Applies therapeutic dosage (action) to the current cellular state, 
        incorporating stochastic mutation to model emerging drug resistance.
        Returns: next_state, done
        """
        self.current_step += 1
        
        # 1. Apply Action (Therapeutic Dosage)
        # Action is a continuous dosage bounded [-1, 1]. We scale it
        # so it models gradual kinetic integration rather than instant flipping.
        dosage_impact = action * 0.15 
        new_state = self.state + dosage_impact
        
        # 2. Apply Drug Resistance Mutation
        # Resistance scales over time as the tumor evolves against the therapy.
        adaptive_mutation_std = self.mutation_std * (1.0 + (self.current_step / self.max_steps))
        mutation_noise = torch.randn_like(new_state) * adaptive_mutation_std
        new_state = new_state + mutation_noise
        
        # 3. Restrict biological state bounds
        # Cellular expression vectors must remain valid probabilities/normalized scales [0, 1]
        self.state = torch.clamp(new_state, 0.0, 1.0)
        
        done = (self.current_step >= self.max_steps)
        
        return self.state, done
