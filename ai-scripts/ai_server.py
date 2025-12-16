import socket
import json
import threading
import copy
import signal
import sys
import pandas as pd
import argparse
import random
import numpy as np

from q_learner import QLearner, SharedQLearner
from config import ServerConfig, RewardConfig, Logger


Logger.init()
logging = Logger.get_logger(__name__) 


class AIServer:
    def __init__(self, server_cfg, reward_cfg, csv_file='', run_id=0, exp_config='gen_q_learning'):
        self.agents = {}  # enemy_id (str): QLearner
        self.fitnesses = {}  # enemy_id (str) : float
        self.shared_brain = SharedQLearner()
        self.running = True

        # configs
        self.server_cfg = server_cfg
        self.reward_cfg = reward_cfg

        # experiment info, not really used, as it is provided indirectly by client
        self.run_id = run_id
        self.exp_config = exp_config

        # for data handling
        self.csv_file = csv_file
        self.df = pd.DataFrame()

    def handle_client(self, conn, addr):
        with conn:
            logging.info(f"[Connection] Accepted connection from {addr}")
            buffer = ""
            while True:
                data = conn.recv(self.server_cfg.BUFFER_SIZE)
                if not data:
                    logging.info(f"[Connection] Client {addr} disconnected.")
                    break
                try:
                    buffer += data.decode("utf-8")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.strip() == "":
                            continue
                        message = json.loads(line)
                        self.handle_message(message, conn)
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decode error: {e} while processing data {repr(data.decode('utf-8'))}")
                    continue
    
    def handle_message(self, msg, conn):
        msg_type = msg.get("type")
        data = msg.get("data")

        match msg_type:
            case "STATE":
                self.handle_state_msg(data, conn)
            case "REWARD":
                self.handle_reward_msg(data)
            case "FITNESS":
                self.handle_fitness_msg(data)
            case "WAVE_END":
                self.handle_wave_end()
            case "LOG":
                self.handle_log_msg(data)
            case "SHUTDOWN":
                self.shutdown(signal.SIGINT, None)
            case _:
                logging.debug(f"Unknown message type: {msg_type}")
                

    def handle_state_msg(self, data, conn):
        msg = {
                "type": "ACTION",
                "data": {}}
                   
        for enemy_id, enemy_info in data.items():
            state = enemy_info["state"]
            valid_actions = enemy_info["valid_actions"]


            agent = self.get_or_create_agent(str(enemy_id))  # q table

            # Choose action based on current state and valid actions
            random_q_condition = self.exp_config in ['base', 'ga_only'] 
            action = agent.choose_action(state, valid_actions, random_q=random_q_condition)

            msg["data"][str(enemy_id)] = action

        logging.debug(f"Chosen actions for enemies: {msg['data']}")
        # Send the action back to the client  
        message = json.dumps(msg) + "\n"
        conn.sendall(message.encode("utf-8"))


    def handle_reward_msg(self, data):
        if self.exp_config == 'base':
            return
        for enemy_id_str, events in data.items():
            enemy_id = int(enemy_id_str)
            agent = self.get_or_create_agent(enemy_id_str)

            for event in events:
                event_type = event["event_type"]
                new_state = event["new_state"]
                action_to_reward = event["action_to_reward"]
                state_to_reward = event["state_to_reward"]


                reward = self.reward_cfg.get(event_type)
                
                if reward is None:
                    logging.warning(f"Unknown event type '{event_type}' for enemy {enemy_id}, no reward applied.")
                else:
                    logging.debug(f"Applying reward {reward} of event {event_type} for action {action_to_reward} to agent {enemy_id}.")
                    agent.apply_reward(reward, new_state, action_to_reward, state_to_reward)
    
    def handle_wave_end(self):
        
        # to see table 
        # Q = self.agents['0'].q_table
        # q_values = [v for actions in Q.values() for v in actions.values()]
        # print(f"Q stats → min: {np.min(q_values):.3f}, max: {np.max(q_values):.3f}, mean: {np.mean(q_values):.3f}")

        self.shared_brain.q_table = {}  # Reset shared brain

        if self.exp_config in ["base", "q_only"]:
            self.agents.clear()
            self.fitnesses.clear()
            return

        # selection
        top_two_ids = [k for k, v in sorted(self.fitnesses.items(), key=lambda item: item[1], reverse=True)[:2]]
        top_two = [self.agents[enemy_id] for enemy_id in top_two_ids]

        # crossover + mutation
        self.shared_brain.per_state_crossover(top_two, self.server_cfg.MUTATION_PROB, self.server_cfg.MUTATION_RANGE)
        # print(f"shared q table: {self.shared_brain.q_table}")
        
        # older approach> Merge all agents' Q-tables into the shared brain
        # learners = [(agent, self.fitnesses.get(agent.enemy_id, 0.0)) for agent in self.agents.values()]
        # logging.debug(f"Merging {learners} into shared brain.")
        # self.shared_brain.average_all(learners)
        # logging.debug(f"Shared brain updated from wave. Resulting Shared Q-table: {self.shared_brain.q_table}")
        
        # Clear agents and fitnesses for the next wave
        self.agents.clear()
        self.fitnesses.clear()

    def handle_fitness_msg(self, data):
        # Update fitnesses for each agent
        for enemy_id_str, fitness in data.items():
            self.fitnesses[enemy_id_str] = fitness
        logging.debug(f"Resulting fitnesses: {self.fitnesses}")
        self.handle_wave_end()  # Process the end of the wave after receiving fitness data
        
    def handle_log_msg(self, data):
        if not self.csv_file:
            return
        # Expecting `data` to already be a flat dict with all columns
        self.df = pd.concat([self.df, pd.DataFrame([data])], ignore_index=True)
        # Save every time so you don’t lose data if crash
        self.df.to_csv(self.csv_file, index=False)

    def get_or_create_agent(self, enemy_id: str):
        if enemy_id not in self.agents:
            agent = copy.deepcopy(self.shared_brain)
            agent.enemy_id = enemy_id
            self.agents[enemy_id] = agent
        return self.agents[enemy_id]
    

    def run(self):
        self.running = True
        logging.info(f"[Python AI Server] Starting server on on {self.server_cfg.HOST}:{self.server_cfg.PORT}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.server_cfg.HOST, self.server_cfg.PORT))
            s.listen()
            logging.info(f"[Python AI Server] Server listening on {self.server_cfg.HOST}:{self.server_cfg.PORT}")
            while self.running:
                conn, addr = s.accept()
                self.handle_client(conn, addr)
                # threading.Thread(target=self.handle_client, args=(conn, addr)).start()

    def shutdown(self, signal_num, frame):
        logging.info("[SERVER] Shutting down server...")
        self.running = False
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10000)
    parser.add_argument("--run_id", type=int, default=0)  # not really used in server currently, client sends this indirectly in data
    parser.add_argument("--config", type=str, default='gen_q_learning') # not really used in server currently, client sends this indirectly in data
    parser.add_argument("--learning_rate", type=float, default=0.2)
    parser.add_argument("--discount_factor", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--mutation_prob", type=float, default=0.05)
    parser.add_argument("--mutation_range", type=float, default=0.1)
    parser.add_argument("--output_csv", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--reward_dict', type=str, default='{}')

    args = parser.parse_args()

    random.seed(args.seed)

    srv_cgf = ServerConfig(
        port=args.port,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon,
        mutation_prob=args.mutation_prob,
        mutation_range=args.mutation_range
    )

    # default now, but making possible to expand experiments with varying rewards
    reward_dict = json.loads(args.reward_dict)
    reward_cfg = RewardConfig()
    if len(args.reward_dict) >= 1:
        reward_cfg.update_rewards(reward_dict)
   

    server = AIServer(server_cfg=srv_cgf, reward_cfg=reward_cfg, csv_file=args.output_csv)
    signal.signal(signal.SIGINT, server.shutdown)
    server.run()


if __name__ == "__main__":
    main()
    
