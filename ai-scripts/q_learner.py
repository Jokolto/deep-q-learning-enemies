import random
import copy
import math
from collections import deque
from config import Logger

MAX_PENDING = 300

Logger.init()
logging = Logger.get_logger(__name__) 


class QLearner:
    def __init__(self, enemy_id:str, learning_rate, discount_factor, epsilon, mutation_prob, mutation_sd, random_policy: bool = False):
        self.q_table = {}
        self.enemy_id = enemy_id
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.mutation_prob = mutation_prob
        self.mutation_sd = mutation_sd
        self.random_policy = random_policy # true in base and ga_only configs
        if random_policy:
            self.epsilon = 0.0

        self.pending_actions = deque()  # queue of (state, action)


    def get_q_value(self, state: str, action: str) -> float:
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        return self.q_table[state][action]

    def apply_reward(self, reward, new_state_not_d, action_to_reward, state_to_reward_not_d):
        new_state = encode_state_tabular(new_state_not_d)
        state_to_reward = encode_state_tabular(state_to_reward_not_d)
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

    def choose_action(self, state_not_d: dict, valid_actions: list[str]) -> str:  
        state = encode_state_tabular(state_not_d)
        self.last_state = state

        if self.random_policy:
            if state not in self.q_table:
                self.q_table[state] = {action: random.uniform(-1, 1) for action in valid_actions} # random q values for actions that persists for agents

         # Explore if epsilon hits or state is unknown
        if random.random() < self.epsilon or state not in self.q_table or not self.q_table[state]:
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
    def __init__(self, learning_rate, discount_factor, epsilon, mutation_prob, mutation_sd, random_policy):
        super().__init__(enemy_id="shared", learning_rate=learning_rate, discount_factor=discount_factor, epsilon=epsilon,
                          mutation_prob=mutation_prob, mutation_sd=mutation_sd, random_policy=random_policy)

    def crossover(self, parents):
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

    
    def mutate(self, mutation_prob, mutation_sd):
        for state, actions in self.q_table.items():
            for action, q in actions.items():
                if random.random() < mutation_prob:
                    actions[action] += random.gauss(0.0, mutation_sd)

    def spawn(self, enemy_id):
        new_agent = copy.deepcopy(self)
        new_agent.pending_actions.clear()
        new_agent.enemy_id = enemy_id
        return new_agent
    
    def spawn_mutated(self, enemy_id):
        new_agent = self.spawn(enemy_id)
        new_agent.mutate(self.mutation_prob, self.mutation_sd)
        return new_agent

    def reset(self):
        self.q_table = {}



def clamp(v, vmin, vmax):
    return max(vmin, min(v, vmax))



def discretize_linear(value, vmin, vmax, bins):
    """
    Maps a continuous value into an integer bin.
    """
    value = clamp(value, vmin, vmax)
    norm = (value - vmin) / (vmax - vmin)
    return int(norm * (bins - 1))


def encode_state_tabular(state: dict) -> str:
    # state looks like state = { "weapon_type": 0.0, "player_weapon_type": 0.0, "pos_x": 0.0, "pos_y": 0.0, "dist_to_player": 0.0, "angle_to_player": 0.0, "bullet_dist": 0.0, "bullet_angle": 0.0, "dist_ally": 0.0, "angle_ally": 0.0 }
    # --- configuration  ---
    DIST_MAX = 1000.0
    BULLET_DIST_MAX = 800.0

    DIST_BINS = 5
    ANGLE_BINS = 8

    # --- categorical ---
    wt = int(state["weapon_type"])
    pw = int(state["player_weapon_type"])

    # --- distances ---
    d = discretize_linear(state["dist_to_player"], 0.0, DIST_MAX, DIST_BINS)
    bd = discretize_linear(state["bullet_dist"], 0.0, BULLET_DIST_MAX, DIST_BINS)
    ad = discretize_linear(state["dist_ally"], 0.0, DIST_MAX, DIST_BINS)

    # --- angles (wrapped to [-pi, pi]) ---
    a = discretize_linear(
        ((state["angle_to_player"] + math.pi) % (2 * math.pi)) - math.pi,
        -math.pi, math.pi, ANGLE_BINS
    )

    ba = discretize_linear(
        ((state["bullet_angle"] + math.pi) % (2 * math.pi)) - math.pi,
        -math.pi, math.pi, ANGLE_BINS
    )

    # --- final compact state string ---
    return (
        f"wt{wt}"
        f"pw{pw}"
        f"d{d}"
        f"a{a}"
        f"bd{bd}"
        f"ba{ba}"
        f"ad{ad}"
    )