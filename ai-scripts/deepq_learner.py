import torch
import torch.nn as nn
import random
from collections import deque
import torch.nn.functional as F
import torch.optim as optim
import copy
from config import Logger


Logger.init()
logging = Logger.get_logger(__name__) 

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), # layer 1 (hidden)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),  # layer 2 (hidden)
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # (layer 3 -> output)
        )

    def forward(self, x):
        return self.net(x)


class DQNLearner:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hiddem_dim: int,
        enemy_id: str,
        learning_rate,
        discount_factor,
        epsilon,
        mutation_sd,
        device="cpu"
    ):
        # hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hiddem_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.mutation_sd = mutation_sd
        self.device = device
        self.enemy_id = enemy_id

        # Neural nets
        self.q_net = QNetwork(state_dim, action_dim, hiddem_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim, hiddem_dim).to(device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.step_count = 0
        self.target_update_interval = 60  # every 1 sec


    def choose_action(self, state: dict, valid_action_indices: list):
        # state_dict, state_keys = state[0], state[1]
        state_vector = list(state.values())
        state_t = torch.tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)

        # explore
        if random.random() < self.epsilon:
            action_idx = random.choice(valid_action_indices)
        else:
            with torch.no_grad():
                q_values = self.q_net(state_t)[0]
                masked_q = q_values.clone()
                # mask invalid actions
                masked_q = torch.full_like(q_values, -1e9)
                masked_q[valid_action_indices] = q_values[valid_action_indices]
                action_idx = masked_q.argmax().item()

        return int(action_idx)

    def apply_reward(self, reward, new_state, action_idx, state):
        # update target net periodically 
        if self.step_count % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        

        state_t = torch.tensor(list(state.values()), dtype=torch.float32, device=self.device).unsqueeze(0)
        new_state_t = torch.tensor(list(new_state.values()), dtype=torch.float32, device=self.device).unsqueeze(0)

        # Q(s,a)
        q_values = self.q_net(state_t) # shape [1 x 5]
        q_sa = q_values[0, action_idx] # scalar -- shape []

        # max_a' Q(s',a')
        # double DQN
        with torch.no_grad():
            # action selection by online net
            next_action = self.q_net(new_state_t).argmax(dim=1)

            # action evaluation by target net
            max_next_q = self.target_net(new_state_t)[0, next_action]
            max_next_q = max_next_q.squeeze(0)


        target = reward + self.discount_factor * max_next_q
        loss = F.mse_loss(q_sa, target)

        self.optimizer.zero_grad()   # reset gradient
        loss.backward()              # new gradients (backprop)
        self.optimizer.step()        # update weights

    def __repr__(self):
        return f"DQNLearner(enemy_id={self.enemy_id})"




class SharedDeepQLearner(DQNLearner):
    def __init__(self, state_dim, hidden_dim, action_dim, discount_factor, learning_rate, epsilon, mutation_sd, device="cpu"):
        super().__init__(
            state_dim=state_dim,
            hiddem_dim=hidden_dim,
            action_dim=action_dim,
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            epsilon=epsilon,
            mutation_sd=mutation_sd,
            enemy_id="shared",
            device=device
        )

    def merge_from(self, other_net, weight=1.0):
        """Add weighted parameters from another network"""
        with torch.no_grad():
            for p_shared, p_other in zip(
                self.q_net.parameters(),
                other_net.parameters()
            ):
                p_shared.add_(p_other, alpha=weight)

    def average_all(self, learners_with_fitness):
        """
        learners_with_fitness: list[(DQNLearner, fitness)]
        """
        total_fitness = sum(f for _, f in learners_with_fitness)
        if total_fitness == 0:
            return

        self.reset()

        # weighted sum
        for learner, fitness in learners_with_fitness:
            w = fitness / total_fitness
            self.merge_from(learner.q_net, weight=w)

    def spawn(self, enemy_id: str) -> DQNLearner:
        """
        Create a new independent DQNLearner initialized
        with the shared network weights.
        """

        learner = DQNLearner(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hiddem_dim=self.hidden_dim,
            enemy_id=enemy_id,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            epsilon=self.epsilon,
            mutation_sd=self.mutation_sd,
            device=self.device,
        )

        # copy weights ONLY (not optimizer state)
        learner.q_net.load_state_dict(self.q_net.state_dict())
        learner.target_net.load_state_dict(self.target_net.state_dict())

        return learner
    
    def spawn_mutated(self, enemy_id: str) -> DQNLearner:
        learner = self.spawn(enemy_id)

        with torch.no_grad():
            for p in learner.q_net.parameters():
                p.add_(torch.randn_like(p) * self.mutation_sd)

        return learner
    
    # makes every parameter 0, only use if adding stuff afterwards (not training)
    def reset(self):
        with torch.no_grad():
            for p in self.q_net.parameters():
                p.zero_()
            


    def crossover(self, parents):
        self.average_all(parents)