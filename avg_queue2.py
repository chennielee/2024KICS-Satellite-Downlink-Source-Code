import csv
import numpy as np


def compute_queue_std(csv_path, num_stations=10):
    """
    Assumes the CSV format is:
      [episode, step, env_minute, queue_gs_0, ..., queue_gs_(num_stations-1)]

    Procedure:
    1) Filter only the data corresponding to the last 1000 episodes
    2) Compute the standard deviation (np.std) for each row (time step)
       across all ground-station queues
    3) Compute and return the mean of those standard deviations
    """
    with open(csv_path, mode="r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        # Load the entire CSV into memory at once
        rows = list(reader)

    # Convert the episode index (first column) to int for all rows
    ep_list = [int(r[0]) for r in rows]

    # Maximum episode index
    max_ep = max(ep_list) if ep_list else 0

    # Starting episode index (at least 1)
    start_ep = max(1, max_ep - 999)

    # Filter rows belonging to the last 1000 episodes
    # (or all available rows if fewer than 1000 episodes exist)
    last_rows = [r for r in rows if int(r[0]) >= start_ep]

    step_std_list = []
    for row in last_rows:
        # Example row: [ep, step, env_minute, q0, q1, ..., q9]
        usage_arr = []
        for j in range(num_stations):
            q_val = float(row[3 + j])
            usage_arr.append(q_val)

        # Standard deviation for a single time step
        std_t = np.std(usage_arr)
        step_std_list.append(std_t)

    # Mean standard deviation over the filtered (last 1000 episodes) data
    if len(step_std_list) == 0:
        return 0.0
    return float(np.mean(step_std_list))


# Example execution
if __name__ == "__main__":
    csv_file = "random_queue_log.csv"  # e.g., queue_usage.csv
    avg_std = compute_queue_std(csv_file)
    print(
        f"Average standard deviation of ground-station queue usage "
        f"over the last 1000 episodes: {avg_std:.4f}"
    )
