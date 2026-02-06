import numpy as np
import csv

def run_greedy_policy(env, num_episodes=100, csv_path=None, step_csv_path=None, queue_csv_path=None):
    """
    Distance-based Greedy Policy: 
    Selects the nearest Ground Station (GS) among those currently visible.
    
    - env: OCO2Env instance
    - num_episodes: Number of episodes to simulate
    """
    all_rewards = []

    # (1) Episode-level logging: Total rewards per episode
    if csv_path is not None:
        f_ep = open(csv_path, "w", newline="")
        writer_ep = csv.writer(f_ep)
        writer_ep.writerow(["episode", "total_reward"])
    else:
        f_ep = None
        writer_ep = None

    # (2) Step-level logging: Energy and AoI status
    if step_csv_path is not None:
        f_step = open(step_csv_path, "w", newline="")
        writer_step = csv.writer(f_step)
        header_step = ["episode", "step", "env_minute", "energy"] + [f"aoi_gs_{i}" for i in range(env.num_stations)]
        writer_step.writerow(header_step)
    else:
        f_step = None
        writer_step = None

    # (3) Queue-level logging: Data backlog for each GS
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
        max_steps = env.episode_length 

        while (not done) and (step_count < max_steps):
            # [Fairness Logic] Extracting current status from 'obs'
            # Index mapping (Adjust these indices based on your env._get_obs() implementation)
            # aoi_gs = obs[2:12]
            # dist_gs = obs[12:22]  <-- Assuming distance is provided in observation
            # vis_map = obs[23:33]
            
            # For this script, we use the specific current distances from the environment 
            # to simulate a sensor-based greedy approach.
            minute_idx = env.minute % env.num_steps_per_cycle
            vis_map = obs[23:33] # Get visibility from observation
            dist_row = env.distance_all[minute_idx] # Get current distance context

            # Find indices of GS that are currently visible (Visibility flag == 1)
            visible_indices = np.where(vis_map == 1)[0]
            
            action_vec = np.zeros(env.num_stations, dtype=int)
            
            if len(visible_indices) > 0:
                # Greedy Choice: Select the station with the minimum distance
                visible_distances = dist_row[visible_indices]
                min_dist_idx = np.argmin(visible_distances)
                chosen_station = visible_indices[min_dist_idx]
                
                action_vec[chosen_station] = 1

            # Execute action in the environment
            next_obs, reward, done, info = env.step(action_vec)
            obs = next_obs # Update current observation
            total_reward += reward
            step_count += 1

            # Log step-wise data
            if writer_step is not None:
                aoi_list = env.aoi_gs.tolist()
                energy_now = env.energy
                writer_step.writerow([episode+1, step_count, env.minute, energy_now, *aoi_list])

            if writer_queue is not None:
                queue_list = env.queue_gs.tolist()
                writer_queue.writerow([episode+1, step_count, env.minute, *queue_list])

            if step_count >= env.episode_length:
                done = True

        all_rewards.append(total_reward)
        print(f"[GREEDY-DIST] Episode={episode+1}, Total Reward={total_reward:.3f}")

        if writer_ep is not None:
            writer_ep.writerow([episode+1, total_reward])

    if f_ep: f_ep.close()
    if f_step: f_step.close()
    if f_queue: f_queue.close()

    return all_rewards
