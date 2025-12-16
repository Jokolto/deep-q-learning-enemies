import random
from collections import deque
from config import Logger

MAX_PENDING = 300

Logger.init()
logging = Logger.get_logger(__name__) 


class QLearner:
    def __init__(self, enemy_id:str="", learning_rate=0.1, discount_factor=0.95):
        self.q_table = {}
        self.enemy_id = enemy_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.pending_actions = deque()  # queue of (state, action)


    def get_q_value(self, state: str, action: str) -> float:
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        return self.q_table[state][action]

    def apply_reward(self, reward, new_state, action_to_reward, state_to_reward):
        if action_to_reward is None and state_to_reward is None:
            if not self.pending_actions:
                logging.debug("No pending actions to apply reward to.", self.pending_actions)
                return  # Nothing to apply reward to
            state, action = self.pending_actions.popleft()
            old_value = self.get_q_value(state, action)

        state, action = (state_to_reward, action_to_reward)
        old_value = self.get_q_value(state, action)
        # Calculate max future Q-value for the new state
        if new_state in self.q_table and self.q_table[new_state]:
            max_future_q = max(self.q_table[new_state].values())
        else:
            max_future_q = 0.0

        new_value = old_value + self.learning_rate * (reward + self.discount_factor * max_future_q - old_value) # Update Q-value according to Q-learning formula
        
        self.q_table[state][action] = new_value

    def choose_action(self, state: str, valid_actions: list[str], epsilon: float = 0.1, random_q: bool = False) -> str:  # last parameter is true in base and ga_only configs
        self.last_state = state

        if random_q:
            epsilon = 0.0 
            if state not in self.q_table:
                self.q_table[state] = {action: random.uniform(-1, 1) for action in valid_actions} # random q values for actions that persists for agents

         # Explore if epsilon hits or state is unknown
        if random.random() < epsilon or state not in self.q_table or not self.q_table[state]:
            action = random.choice(valid_actions)
        else:
            action = max(self.q_table[state], key=self.q_table[state].get)
        
        # Store for later reward
        self.pending_actions.append((state, action))
        
        if len(self.pending_actions) > MAX_PENDING:
            self.pending_actions.popleft()

        return action
    
    def __repr__(self) -> str:
        return f"QLearner(enemy_id={self.enemy_id})"
       


class SharedQLearner(QLearner):
    def __init__(self):
        super().__init__(enemy_id="shared")

    # averaging crossover used earlier
    def merge_from(self, other, weight=1.0):
        for state, actions in other.q_table.items():
            if state not in self.q_table:
                self.q_table[state] = {}

            for action, q in actions.items():
                if action not in self.q_table[state]:
                    self.q_table[state][action] = 0.0

                self.q_table[state][action] += q * weight

    def average_all(self, learners_with_fitness):
        total_fitness = sum(f for _, f in learners_with_fitness)
        if total_fitness == 0:
            logging.warning("No fitness data to average, skipping merge.")
            return  # Avoid div-by-zero

        for learner, fitness in learners_with_fitness:
            self.merge_from(learner, weight=(fitness / total_fitness))
    #


    def per_state_crossover(self, parents, mutation_prob=0.05, mutation_range=0.1):
        ''' new crossover, where top candidates individual with highest fitness produce new q table. For each state random parents q values are taken '''
        # Collect all states across all candidates
        all_states = set()
        for candidate in parents:
            all_states.update(candidate.q_table.keys())
        
        for state in all_states:
            # Candidates that have this state
            candidates_with_state = [c for c in parents if state in c.q_table]
            # Pick one randomly
            parent = random.choice(candidates_with_state)
            self.q_table[state] = parent.q_table[state].copy()

            # Apply mutation
            for action, value in self.q_table[state].items():
                if random.random() < mutation_prob:
                    # Add or subtract a small random number
                    self.q_table[state][action] += random.uniform(-mutation_range, mutation_range)


    