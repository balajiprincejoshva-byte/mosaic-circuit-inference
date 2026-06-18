import torch
import torch.nn as nn
import torch.optim as optim

class TherapeuticActor(nn.Module):
    """
    Policy network mapping tissue state vectors to continuous dosage actions.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super(TherapeuticActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Bounded dosages between -1.0 and 1.0
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

class TherapeuticCritic(nn.Module):
    """
    Value network estimating the expected cumulative reward of a state-action pair.
    """
    def __init__(self, state_dim: int, action_dim: int):
        super(TherapeuticCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=1))

class RLEngine:
    """
    Orchestrates the Actor-Critic reinforcement learning control loop.
    """
    def __init__(self, num_features: int, lr_actor: float=1e-3, lr_critic: float=1e-3):
        self.actor = TherapeuticActor(num_features, num_features)
        self.critic = TherapeuticCritic(num_features, num_features)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = 0.99
        self.num_features = num_features

    def select_action(self, state: torch.Tensor, noise_std: float=0.05) -> torch.Tensor:
        """
        Selects an action based on the current policy with optional exploration noise.
        """
        with torch.no_grad():
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
            action = self.actor(state)
            if noise_std > 0:
                noise = torch.randn_like(action) * noise_std
                action = torch.clamp(action + noise, -1.0, 1.0)
            return action.squeeze(0)

    def calculate_reward(self, rbm, sys_net, next_state: torch.Tensor, action: torch.Tensor, is_quantum: bool=False) -> tuple:
        """
        Calculates the immediate reward for the current state.
        R = - (alpha * FreeEnergy + beta * sum(Toxicity))
        Returns: total_reward, free_energy, total_toxicity
        """
        v_batch = next_state.unsqueeze(0)
        
        # 1. Evaluate Attractor Stability (Free Energy)
        if is_quantum and hasattr(rbm, 'calculate_quantum_free_energy'):
            free_energy = rbm.calculate_quantum_free_energy(v_batch).item()
        else:
            free_energy = rbm.calculate_free_energy(v_batch).item()
            
        # 2. Evaluate Systemic Toxicity based on the dosage action
        sys_risks = sys_net.calculate_systemic_toxicity(rbm, action, is_quantum)
        total_toxicity = sum(sys_risks.values())
        
        # Alpha and Beta coefficients to scale the reward
        alpha = 0.1
        beta = 0.05
        
        # We want to MINIMIZE free energy (deepen the attractor basin) and MINIMIZE toxicity.
        # So we penalize highly positive free energy and high toxicity.
        reward = -(alpha * free_energy + beta * total_toxicity)
        
        return reward, free_energy, total_toxicity, sys_risks

    def train_step(self, state: torch.Tensor, action: torch.Tensor, reward: float, next_state: torch.Tensor, done: bool) -> tuple:
        """
        Single step DDPG/Actor-Critic update.
        """
        state = state.unsqueeze(0) if len(state.shape)==1 else state
        next_state = next_state.unsqueeze(0) if len(next_state.shape)==1 else next_state
        action = action.unsqueeze(0) if len(action.shape)==1 else action
        reward_t = torch.tensor([[reward]], dtype=torch.float32)
        done_t = torch.tensor([[float(done)]], dtype=torch.float32)
        
        # Critic update
        with torch.no_grad():
            next_action = self.actor(next_state)
            target_q = reward_t + (1.0 - done_t) * self.gamma * self.critic(next_state, next_action)
            
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return critic_loss.item(), actor_loss.item()
