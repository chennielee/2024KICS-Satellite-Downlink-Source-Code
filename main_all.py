import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import csv

# Skyfield and SGP4 related imports
from datetime import datetime, timedelta
from skyfield.api import load, utc, wgs84, EarthSatellite
from skyfield.framelib import itrs
from sgp4.api import Satrec, jday

# Module imports
from env import OCO2Env 
from agent import DQNAgent
from random_policy import run_random_policy
from greedy_policy import run_greedy_policy
from satellite_utils import get_oco2_positions_16days, compute_visibility_all

# Check device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

def main_all():
    """
    main_all() executes: 
    (1) Random Policy, (2) Greedy Policy, and (3) DQN Training.
    Then, it compares the results in visual plots.
    """

    # Set random seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # 1> OCO-2 Satellite TLE Data
    line1 = "1 40059U 14035A   25009.89311847  .00001165  00000+0  26847-3 0  9990"
    line2 = "2 40059  98.2255 313.7501 0001333  70.4411 289.6932 14.57138477559845"
    satellite = EarthSatellite(line1, line2, "OCO 2")

    ts = load.timescale()
    satellite.ts = ts

    # 2> Ground Station Coordinates (lat, lon)
    stations = {
        "KSAT Singapore" : (1.3661, 103.9915),
        "KSAT Svalbard": (78.22875, 15.39964),
        "SANSA Hartebeesthoek": (-25.88974, 27.68533),
        "KSAT Punta Arenas, Chile" : (-52.93279, -70.85021),
        "Mingenew Ground Station, Australia" : (-29.1994, 115.44225), 
        
        "SSC Space US South Point, Hawaii": (21.54675, -158.23982),
        "NASA's Alaska Satellite Facility, Fairbanks" : (64.97659, -147.51769),
        "NASA's Wallops Ground Station, Virginia" : (37.946775, -75.460063),
        "NASA's White Sands Ground Station, New Mexico" : (32.5007,-106.6086),
        "NASA's McMurdo Ground Station, Antartica" : (77.839130, -166.667083)
    }

    station_names_list = list(stations.keys())

    # Calculate 16-day orbit (minute-by-minute)
    start_date = datetime(2025, 1, 10, 0, 0, 0, tzinfo=utc)
    oco2_positions = get_oco2_positions_16days(start_date, satellite, ts)

    # Compute global ground station visibility matrix
    station_names, vis_map = compute_visibility_all(
        satellite, stations, start_date, ts, num_steps_per_cycle=60*24*16
    )

    ##### Environment Creation - Separate instances for each policy #####
    # 1) random_env -> run_random_policy
    # 2) greedy_env -> run_greedy_policy
    # 3) dqn_env    -> DQN Training
    ####################################################################

    print("[DEBUG] Creating env for RANDOM policy...")
    np.random.seed(SEED)
    env_random = OCO2Env(
        satellite=satellite,
        stations=stations,
        station_names=station_names,
        start_date=start_date,
        vis_map=vis_map,
        oco2_positions=oco2_positions,
        leo_queue_capacity=60*24*1,
        gs_queue_capacity=5000,
        cycles=1,
        use_future_los=True
    )

    print("[DEBUG] Creating env for GREEDY policy...")
    np.random.seed(SEED)
    env_greedy = OCO2Env(
        satellite=satellite,
        stations=stations,
        station_names=station_names,
        start_date=start_date,
        vis_map=vis_map,
        oco2_positions=oco2_positions,
        leo_queue_capacity=60*24*1,
        gs_queue_capacity=5000,
        cycles=1,
        use_future_los=True
    )

    print("[DEBUG] Creating env for DQN agent...")
    np.random.seed(SEED)
    env = OCO2Env(
        satellite=satellite,
        stations=stations,
        station_names=station_names,
        start_date=start_date,
        vis_map=vis_map,
        oco2_positions=oco2_positions,
        leo_queue_capacity = 60*24*1,
        gs_queue_capacity = 5000, 
        cycles = 1,
        use_future_los=True 
    )
    print("[DEBUG] All environments initialized successfully.")

    # Open CSV files for logging
    with open("episode_rewards.csv", "w", newline="") as epi_f, \
         open("actions.csv", "w", newline="") as act_f, \
         open("aoi_gs.csv", "w", newline="") as aoi_f, \
         open("queue_usage.csv", "w", newline="") as queue_f:

        # (1) Total reward per episode
        epi_writer = csv.writer(epi_f)
        epi_writer.writerow(["episode", "total_reward"])

        # (2) Step-level action logs
        act_writer = csv.writer(act_f)
        act_header = ["episode", "step", "reward", "chosen_station"] + [f"gs_idx{i}" for i in range(11)]
        act_writer.writerow(act_header)

        # (3) Step-level AoI (Age of Information) and Energy logs
        aoi_writer = csv.writer(aoi_f)
        aoi_header = ["episode", "step", "env_minute", "energy"] + [f"aoi_gs_{j}" for j in range(11)]
        aoi_writer.writerow(aoi_header)

        # (4) Step-level GS Queue usage logs
        queue_writer = csv.writer(queue_f)
        queue_header = ["episode", "step", "env_minute"] + [f"queue_gs_{j}" for j in range(env.num_stations)]
        queue_writer.writerow(queue_header)

        # Execute Random Policy Baseline
        print("[DEBUG] Running RANDOM policy...")
        random_rewards = run_random_policy(
            env_random, 
            num_episodes=100, 
            csv_path="random_log.csv", 
            step_csv_path="random_steps_log.csv",
            queue_csv_path="random_queue_log.csv"
        )

        # Execute Distance-based Greedy Policy Baseline
        print("[DEBUG] Running GREEDY policy...")
        greedy_rewards = run_greedy_policy(
            env_greedy, 
            num_episodes=100, 
            csv_path="greedy_log.csv", 
            step_csv_path="greedy_steps_log.csv", 
            queue_csv_path="greedy_queue_log.csv" 
        )

        # Initialize DQN Agent
        obs_dim = env.observation_space.shape[0]
        act_dim = 11  # 0..10 indices (NoAction + 10 Ground Stations)
        agent = DQNAgent(obs_dim=obs_dim, act_dim=act_dim, lr=1e-4, device=device)

        num_episodes = 100 
        max_steps_per_ep = env.episode_length 
        batch_size = 64 
        all_rewards_dqn = []
        update_freq = 4

        print("[DEBUG] Starting DQN training...")
        for episode in range(num_episodes):
            obs = env.reset()
            total_reward = 0.0
            done = False
            step_count = 0

            while (not done) and (step_count < max_steps_per_ep):
                action_idx = agent.select_action(obs)
                
                # Convert discrete action_idx to MultiBinary(10) vector for environment
                if action_idx == 0:
                    # NoAction: All GS bits set to 0
                    action_vec = np.zeros(10, dtype=int)
                else:
                    # Action 1..10 maps to GS index 0..9
                    gs_idx = action_idx - 1
                    action_vec = np.zeros(10, dtype=int)
                    action_vec[gs_idx] = 1

                next_obs, reward, done, info = env.step(action_vec)

                agent.store_transition(obs, action_idx, reward, next_obs, done)
                obs = next_obs
                total_reward += reward

                # Periodic network update
                if step_count % update_freq == 0:
                    agent.train_on_batch(batch_size)

                # (A) Log Action data to CSV
                act_writer.writerow([
                    episode, step_count, reward, action_idx,
                    *action_vec
                ])

                # (B) Log AoI and Energy data to CSV
                aoi_list = env.aoi_gs.tolist()
                energy_val = env.energy
                aoi_writer.writerow([
                    episode, step_count, env.minute, energy_val,
                    *aoi_list
                ])

                # (C) Log GS Queue status to CSV
                queue_list = env.queue_gs.tolist()
                queue_writer.writerow([
                    episode, step_count, env.minute, 
                    *queue_list
                ])

                step_count += 1

            # Handle remaining n-step transitions at the end of the episode
            agent.finish_n_step()

            all_rewards_dqn.append(total_reward)
            epi_writer.writerow([episode, total_reward])
            print(f"Episode={episode+1}, total_reward={total_reward:.3f}, epsilon={agent.epsilon:.3f}")

        # Save the trained model
        agent.save("dqn_agent.pt")

        # Visualization 1: Raw Training Rewards
        plt.figure(figsize=(10,6))
        plt.plot(all_rewards_dqn, label="DQN")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DQN Training Progress (Raw)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("dqn_raw.png")
        plt.close()

        # Visualization 2: Smoothed Progress Comparison
        plt.figure(figsize=(10,6))

        def smooth_curve(rewards, window=20):
            # Applying simple moving average for visualization
            return np.convolve(rewards, np.ones(window)/window, mode='valid')

        dqn_smooth = smooth_curve(all_rewards_dqn, window=10)
        dqn_smooth2 = smooth_curve(all_rewards_dqn, window=20)

        plt.plot(dqn_smooth, label="DQN (W=10)")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DQN Training Progress (Smoothed)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("dqn_smoothed.png")
        plt.close()

        plt.plot(dqn_smooth2, label="DQN (W=20)")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DQN Training Progress (Higher Smoothing)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("dqn_smoothed2.png")
        plt.close()

        print("[DEBUG] Done! All logs and comparison figures saved.")

if __name__ == "__main__":
    main_all()