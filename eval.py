import os
import time
import csv
import random
import numpy as np
import torch

from datetime import datetime
from skyfield.api import load, utc, EarthSatellite

from env import OCO2Env
from agent import DQNAgent
from random_policy import run_random_policy
from greedy_policy import run_greedy_policy
from satellite_utils import get_oco2_positions_16days, compute_visibility_all

# =========================
# Config
# =========================
SEED = 42
# Set to at least 100 for statistically significant comparison
EVAL_EPISODES = 100 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", DEVICE)

CKPT_DIR = "checkpoints"
DQN_BEST_PATH = os.path.join(CKPT_DIR, "dqn_agent_best.pt")
DQN_LAST_PATH = os.path.join(CKPT_DIR, "dqn_agent_last.pt")

# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def action_idx_to_vec(action_idx: int) -> np.ndarray:
    """Converts discrete action index (0..10) to MultiBinary(10) vector."""
    if action_idx == 0:
        return np.zeros(10, dtype=int)
    gs_idx = action_idx - 1
    vec = np.zeros(10, dtype=int)
    vec[gs_idx] = 1
    return vec

def build_env(seed: int) -> OCO2Env:
    set_seed(seed)
    # TLE data for OCO-2
    line1 = "1 40059U 14035A   25009.89311847  .00001165  00000+0  26847-3 0  9990"
    line2 = "2 40059  98.2255 313.7501 0001333  70.4411 289.6932 14.57138477559845"
    satellite = EarthSatellite(line1, line2, "OCO 2")
    ts = load.timescale()
    satellite.ts = ts

    stations = {
        "KSAT Singapore": (1.3661, 103.9915),
        "KSAT Svalbard": (78.22875, 15.39964),
        "SANSA Hartebeesthoek": (-25.88974, 27.68533),
        "KSAT Punta Arenas, Chile": (-52.93279, -70.85021),
        "Mingenew Ground Station, Australia": (-29.1994, 115.44225),
        "SSC Space US South Point, Hawaii": (21.54675, -158.23982),
        "NASA's Alaska Satellite Facility, Fairbanks": (64.97659, -147.51769),
        "NASA's Wallops Ground Station, Virginia": (37.946775, -75.460063),
        "NASA's White Sands Ground Station, New Mexico": (32.5007, -106.6086),
        "NASA's McMurdo Ground Station, Antartica": (77.839130, -166.667083),
    }

    start_date = datetime(2025, 1, 10, 0, 0, 0, tzinfo=utc)
    oco2_positions = get_oco2_positions_16days(start_date, satellite, ts)
    station_names, vis_map = compute_visibility_all(
        satellite, stations, start_date, ts, num_steps_per_cycle=60 * 24 * 16
    )

    env = OCO2Env(
        satellite=satellite,
        stations=stations,
        station_names=station_names,
        start_date=start_date,
        vis_map=vis_map,
        oco2_positions=oco2_positions,
        leo_queue_capacity=60 * 24 * 1,
        gs_queue_capacity=5000,
        cycles=1,
        use_future_los=True
    )
    return env

def run_dqn_eval(
    env: OCO2Env,
    agent: DQNAgent,
    num_episodes: int,
    csv_path: str,
    step_csv_path: str,
    queue_csv_path: str,
    force_epsilon_zero: bool = True,
):
    """Evaluates DQN performance and logs results."""
    rewards = []
    
    # Open CSV files for logging
    f_ep = open(csv_path, "w", newline="")
    w_ep = csv.writer(f_ep)
    w_ep.writerow(["episode", "total_reward"])

    f_step = open(step_csv_path, "w", newline="")
    w_step = csv.writer(f_step)
    w_step.writerow(["episode", "step", "env_minute", "energy"] + [f"aoi_gs_{i}" for i in range(env.num_stations)])

    f_q = open(queue_csv_path, "w", newline="")
    w_q = csv.writer(f_q)
    w_q.writerow(["episode", "step", "env_minute"] + [f"queue_gs_{i}" for i in range(env.num_stations)])

    # Set model to evaluation mode
    agent.qnet.eval()
    old_eps = agent.epsilon
    if force_epsilon_zero:
        agent.epsilon = 0.0

    max_steps = env.episode_length
    t0 = time.time()

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while (not done) and (step_count < max_steps):
            # Select action based on current observation (Epsilon=0 for purely greedy)
            action_idx = agent.select_action(obs)
            action_vec = action_idx_to_vec(action_idx)

            next_obs, reward, done, info = env.step(action_vec)
            obs = next_obs
            total_reward += reward
            step_count += 1

            # Log metrics
            aoi_list = env.aoi_gs.tolist()
            energy_now = env.energy
            w_step.writerow([ep + 1, step_count, env.minute, energy_now, *aoi_list])
            
            q_list = env.queue_gs.tolist()
            w_q.writerow([ep + 1, step_count, env.minute, *q_list])

        rewards.append(total_reward)
        w_ep.writerow([ep + 1, total_reward])
        print(f"[DQN-EVAL] Episode={ep+1}, Reward={total_reward:.3f}")

    # Restore agent state and close files
    agent.epsilon = old_eps
    f_ep.close(); f_step.close(); f_q.close()
    
    sec = time.time() - t0
    print(f"[DONE] DQN-EVAL Avg Reward: {np.mean(rewards):.2f} (Total Time: {sec:.2f}s)")
    return rewards

# =========================
# Main Execution
# =========================
def main():
    set_seed(SEED)

    # 1. Evaluate Random Policy (Pure Random Baseline)
    print("\n[Eval 1/4] Running RANDOM Policy")
    env_random = build_env(SEED)
    run_random_policy(
        env_random,
        num_episodes=EVAL_EPISODES,
        csv_path="random_log.csv",
        step_csv_path="random_steps_log.csv",
        queue_csv_path="random_queue_log.csv",
    )

    # 2. Evaluate Distance-based Greedy Policy (Heuristic Baseline)
    print("\n[Eval 2/4] Running GREEDY (Distance) Policy")
    env_greedy = build_env(SEED)
    run_greedy_policy(
        env_greedy,
        num_episodes=EVAL_EPISODES,
        csv_path="greedy_log.csv",
        step_csv_path="greedy_steps_log.csv",
        queue_csv_path="greedy_queue_log.csv",
    )

    # 3. Evaluate DQN Best Model
    print("\n[Eval 3/4] Running DQN BEST Model")
    env_dqn_best = build_env(SEED)
    obs_dim = env_dqn_best.observation_space.shape[0]
    act_dim = 11

    if os.path.exists(DQN_BEST_PATH):
        agent_best = DQNAgent(obs_dim=obs_dim, act_dim=act_dim, lr=1e-4, device=DEVICE)
        agent_best.load(DQN_BEST_PATH)
        run_dqn_eval(
            env_dqn_best, agent_best,
            num_episodes=EVAL_EPISODES,
            csv_path="dqn_best_log.csv",
            step_csv_path="dqn_best_steps_log.csv",
            queue_csv_path="dqn_best_queue_log.csv",
        )
    else:
        print(f"[WARN] File not found: {DQN_BEST_PATH}")

    # 4. Evaluate DQN Last Model
    print("\n[Eval 4/4] Running DQN LAST Model")
    if os.path.exists(DQN_LAST_PATH):
        env_dqn_last = build_env(SEED)
        agent_last = DQNAgent(obs_dim=obs_dim, act_dim=act_dim, lr=1e-4, device=DEVICE)
        agent_last.load(DQN_LAST_PATH)
        run_dqn_eval(
            env_dqn_last, agent_last,
            num_episodes=EVAL_EPISODES,
            csv_path="dqn_last_log.csv",
            step_csv_path="dqn_last_steps_log.csv",
            queue_csv_path="dqn_last_queue_log.csv",
        )

    print("\n[ALL DONE]")

if __name__ == "__main__":
    main()