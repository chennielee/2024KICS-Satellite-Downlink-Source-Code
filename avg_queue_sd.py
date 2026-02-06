import csv
import numpy as np


def compute_queue_std(csv_path, num_stations=10):
    """
    Assumes the CSV format is:
      [episode, step, env_minute, queue_gs_0, ..., queue_gs_(num_stations-1)]

    Procedure:
    1) Filter only the rows belonging to the last 1000 episodes
    2) Compute the standard deviation (np.std) across ground-station queues for each row (time step)
    3) Return the mean of those per-step standard deviations
    """
    with open(csv_path, mode="r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        # Load the entire CSV into memory
        rows = list(reader)

    # Convert episode column to int for all rows
    ep_list = [int(r[0]) for r in rows]

    # Maximum episode index
    max_ep = max(ep_list) if ep_list else 0

    # Start episode index for the last 1000 episodes (at least 1)
    start_ep = max(1, max_ep - 999)

    # Filter rows belonging to the last 1000 episodes (or all available if fewer)
    last_rows = [r for r in rows if int(r[0]) >= start_ep]

    step_std_list = []
    for row in last_rows:
        # Example row: [ep, step, min, q0, q1, ..., q9]
        usage_arr = []
        for j in range(num_stations):
            q_val = float(row[3 + j])
            usage_arr.append(q_val)

        # Standard deviation at this time step
        std_t = np.std(usage_arr)
        step_std_list.append(std_t)

    # Mean standard deviation over the filtered (last 1000 episodes) data
    if len(step_std_list) == 0:
        return 0.0
    return float(np.mean(step_std_list))


# Example usage
if __name__ == "__main__":
    csv_file = "random_queue_log.csv"  # e.g., queue_usage.csv
    avg_std = compute_queue_std(csv_file)
    print(f"Average standard deviation of GS queue usage over the last 1000 episodes: {avg_std:.4f}")
