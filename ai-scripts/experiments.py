import subprocess
import time
import random
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import Logger

# config
PROJECT_PATH = r'C:\projects\matura\source' 
SERVER_PATH = r'C:\projects\matura\python_ai_scripts\ai_server.py'
GODOT_PATH = "godot" # works cause i have registered godot in path

Logger.init()
logging = Logger.get_logger(__name__) 

def start_logger(
    port: int,
    run_id: int = 0,
    output_csv: str = '',
    seed: int = 0,
    config_name: str = "q_only",
    # q learning hyperparameters
    learning_rate: float = 0.2,
    discount_factor: float = 0.9,
    epsilon: float = 0.2,
    reward_dict: dict = {},
    # ga hyperparamters
    mutation_prob = 0.05,
    mutation_range = 0.1,
):
    """Start the Python server that logs experiment data."""
    cmd = [
        sys.executable,
        SERVER_PATH,
        "--port", str(port),
        "--run_id", str(run_id),
        "--config", config_name,
        "--learning_rate", str(learning_rate),
        "--discount_factor", str(discount_factor),
        "--epsilon", str(epsilon),
        "--mutation_prob", str(mutation_prob),
        "--mutation_range", str(mutation_range),
        '--output_csv', str(output_csv),
        "--seed", str(seed),
        "--reward_dict", str(reward_dict)
    ]
    
    proc = subprocess.Popen(
        cmd,
        # stdout=subprocess.PIPE,  # runs without output of the server,
        # stderr=subprocess.PIPE
    )
    return proc

def run_godot(run_id, seed, config, port, waves_amount):
    """Launch Godot instance headlessly."""
    cmd = [
        GODOT_PATH,   
        "--headless",
        "--path", PROJECT_PATH,
        "--audio-driver", "Dummy",
        "--",
        f"--run_id={run_id}",
        f"--seed={seed}",
        f"--config={config}",
        f"--port={port}",
        f"--waves={waves_amount}",
    ]
    return subprocess.Popen(cmd)

# experiment helpers
def run_single_experiment(repetition, cfg, port, run_id, seed, until_wave, output_csv, mutation_prob, mutation_range, learning_rate, discount_factor, epsilon, reward_dict):
    logging.info(f"=== Starting experiment {run_id}: repetition {repetition} ({cfg}) ===")

    logger = start_logger(port, run_id=run_id, config_name=cfg, output_csv=output_csv, seed=seed, 
                          mutation_prob=mutation_prob, mutation_range=mutation_range, 
                          learning_rate=learning_rate, discount_factor=discount_factor, epsilon=epsilon, reward_dict=reward_dict)
    godot = run_godot(run_id, seed, cfg, port, waves_amount=until_wave)

    godot.wait()
    logger.terminate()

    logging.info(f"Experiment {run_id} ({cfg}, repetition {repetition}) finished")
    time.sleep(0.2)
    return (run_id, True, "ok")

def prepare_experiments(output_csv_base, repeats, until_wave, configs, port_base, mutation_prob, mutation_range, learning_rate, discount_factor, epsilon, reward_dict):
    """Prepare a list of experiment dicts with parameters."""
    experiments = []
    n_exp = 0

    for repetition in range(1, repeats+1):
        seed = random.randint(0, 999999)
        for cfg in configs:
            n_exp += 1
            port = port_base + n_exp
            output_csv = os.path.join(output_csv_base, cfg, f'cfg_{cfg}_run_{repetition}.csv')
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)

            experiments.append({
                "repetition": repetition,
                "cfg": cfg,
                "port": port,
                "run_id": n_exp,
                "seed": seed,
                "until_wave": until_wave,
                "output_csv": output_csv,
                "mutation_prob": mutation_prob,
                "mutation_range": mutation_range,
                "learning_rate": learning_rate, 
                "discount_factor": discount_factor, 
                "epsilon": epsilon, 
                "reward_dict": reward_dict
            })

    logging.info(f"Prepared {len(experiments)} experiments.")
    return experiments

# main experiment method. some leak errors after each godot is exited are expected, i solved some of them but due to time not all of them, they are harmless anyway.
def run_experiments(parallel=False):
    output_csv_base = r'C:\projects\matura\python_ai_scripts\data'
    repeats = 5  # repeats every config experiment n times to have more consitant data. 22 min for 1 rep with until wave = 15, kill_time = 30 sec. If not parallel.
    until_wave = 10 # after this closes godot and server to end an experiment.
    
    learning_rate: float = 0.2
    discount_factor: float = 0.9
    epsilon: float = 0.2
    reward_dict: dict = {} # empty to use dict from config without change.
      
    mutation_prob = 0.05
    mutation_range = 0.1


    configs = [ "base", "q_only",  "ga_only", "gen_q_learning"]
    # what each config does. Config effects work only if EXPERIMENTING in godot is true:
    # base - enemies have random q values without rewards. Implemented in godot with no_q_learning parameter and in python with condition of config
    # q_only - enemies learn intra wave, but not with each wave. Implemented in python server with not filling shared q table
    # ga_only - enemies have random q values without rewards, but best are selected at wave end to be reproduced in next with some mutation. uses no_q_learning parameter in godot and in python with condition of config
    # gen_q_learning - uses both algorithms, default in release.

    port_base = 10000
    
    experiments = prepare_experiments(output_csv_base, repeats, until_wave, configs, port_base, mutation_prob, mutation_range, learning_rate, discount_factor, epsilon, reward_dict)

    if parallel:
        logging.info("Running experiments in parallel...")
        max_workers = min(20, len(experiments))  # prevent overload
        with ProcessPoolExecutor(max_workers=min(max_workers, len(experiments))) as ex:
            futures = { ex.submit(run_single_experiment, **exp): exp for exp in experiments }
            for fut in as_completed(futures):
                run_id, success, msg = fut.result() 
                if not success:
                    print(f"[{run_id}] FAILED: {msg}")
    else:
        logging.info("Running experiments sequentially...") 
        for e in experiments:
            run_single_experiment(**e)

    logging.info(f"All experiments finished!")

if __name__ == "__main__":
    run_experiments(parallel=True)
    
