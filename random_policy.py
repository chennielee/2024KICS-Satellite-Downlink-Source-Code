import random
import numpy as np
import csv

def run_random_policy(env, num_episodes=100, csv_path=None, step_csv_path=None, queue_csv_path=None):
    """
    Runs the environment using a random policy (no training).
    Selects an action between 0 and 10 at each step and returns total rewards.
    
    - env: Initialized OCO2Env instance.
    - num_episodes: Total number of episodes to run.
    """
    all_rewards = []
    
    # (1) Episode-level logging: Open CSV for total rewards per episode
    if csv_path is not None:
        f_ep = open(csv_path, "w", newline="")
        writer_ep = csv.writer(f_ep)
        writer_ep.writerow(["episode", "total_reward"])
    else:
        f_ep = None
        writer_ep = None

    # (2) Step-level logging: Open CSV for metrics (AoI, Energy, etc.)
    if step_csv_path is not None:
        f_step = open(step_csv_path, "w", newline="")
        writer_step = csv.writer(f_step)
        # Header: [episode, step, env_minute, energy, aoi_gs_0 ... aoi_gs_9]
        header_step = ["episode", "step", "env_minute", "energy"] + [f"aoi_gs_{i}" for i in range(env.num_stations)]
        writer_step.writerow(header_step)
    else:
        f_step = None
        writer_step = None

    # (3) Queue-level logging: Open CSV for GS queue usage
    if queue_csv_path is not None:
        f_queue = open(queue_csv_path, "w", newline="")
        writer_queue = csv.writer(f_queue)
        header_queue = ["episode", "step", "env_minute"] + [f"queue_gs_{i}" for i in range(env.num_stations)]
        writer_queue.writerow(header_queue)
    else:
        f_queue = None
        writer_queue = None

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0.0
        done = False
        step_count = 0
        
        while (not done) and (step_count < env.episode_length):
            # [Decision Logic] Aligned with DQN's discrete action space (0 to 10)
            # Pick an action randomly without pre-checking visibility for a fair baseline.
            action_idx = random.randint(0, env.num_stations) 
            
            # Convert discrete action index to MultiBinary(10) vector for the environment
            action_vec = np.zeros(env.num_stations, dtype=int)
            if action_idx > 0:
                # 0 is 'No Action', 1-10 corresponds to GS indices 0-9
                action_vec[action_idx - 1] = 1
            
            # Apply action to the environment
            next_obs, reward, done, info = env.step(action_vec)
            
            # Accumulate reward and update step count
            total_reward += reward
            step_count += 1

            # Log step-wise AoI and Energy
            if writer_step is not None:
                aoi_list = env.aoi_gs.tolist()
                energy_now = env.energy
                writer_step.writerow([
                    episode + 1, 
                    step_count, 
                    env.minute, 
                    energy_now,
                    *aoi_list
                ])

            # Log step-wise Queue status
            if writer_queue is not None:
                queue_list = env.queue_gs.tolist()
                writer_queue.writerow([
                    episode + 1,
                    step_count,
                    env.minute,
                    *queue_list
                ])

            # Terminate if max episode length is reached
            if step_count >= env.episode_length:
                done = True    
 
        # Store and display episode results
        all_rewards.append(total_reward)
        print(f"[RANDOM] Episode={episode+1}, total_reward={total_reward:.3f}")

        if writer_ep is not None:
            writer_ep.writerow([episode + 1, total_reward])

    if f_ep is not None: f_ep.close()
    if f_step is not None: f_step.close()
    if f_queue is not None: f_queue.close()

    return all_rewards