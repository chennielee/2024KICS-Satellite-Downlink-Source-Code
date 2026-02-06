import csv

def compute_aoi_means(csv_file_path):
    """
    Reads an aoi_gs.csv file and computes the mean value for each
    aoi_gs_0 ~ aoi_gs_9 column.

    (It is assumed that aoi_gs_10, aoi_gs_11, and aoi_gs_12 are empty
    and therefore ignored.)
    """
    # Lists to store the sum and count for each aoi_gs_i
    sums = [0.0] * 10    # indices 0..9 -> aoi_gs_0..9
    counts = [0] * 10

    with open(csv_file_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header (column names)

        for row in reader:
            # Defensive check for empty rows
            if not row:
                continue

            # row[4] corresponds to aoi_gs_0, row[5] to aoi_gs_1, ..., row[13] to aoi_gs_9
            # row[14], row[15], row[16] = aoi_gs_10..12 (assumed to be empty)

            # Here, we process only aoi_gs_0 ~ aoi_gs_9
            for i in range(10):
                # Actual column index in the CSV: 4 + i
                val_str = row[4 + i].strip()
                if val_str == "":  # skip empty entries
                    continue
                val = float(val_str)  # convert string to float
                sums[i] += val
                counts[i] += 1

    # Compute and print the mean using the accumulated sums and counts
    for i in range(10):
        if counts[i] > 0:
            mean_val = sums[i] / counts[i]
            print(f"aoi_gs_{i} mean: {mean_val:.2f}")
        else:
            print(f"aoi_gs_{i} mean: no data")


# Example usage
if __name__ == "__main__":
    csv_file = "random_steps_log.csv"  # aoi_gs.csv
    compute_aoi_means(csv_file)
