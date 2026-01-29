import csv
import numpy as np

def compute_queue_std(csv_path, num_stations=10):
    """
    csv_path: 
      [episode, step, env_minute, queue_gs_0, ..., queue_gs_(num_stations-1)] 
    형태라고 가정.
    
    1) 각 행마다 (queue_gs_j / queue_capacity)를 구해,
    2) 그 시점에 대한 표준편차(np.std),
    3) 전체 스텝에 대한 표준편차 평균을 구함.
    """
    step_std_list = []

    with open(csv_path, mode="r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # 헤더 건너뛰기

        for row in reader:
            # row 예: [ep, step, min, q0, q1, ..., q9]
            # q0..q9는 row[3]..row[3+num_stations-1]
            # => row[3]..row[12] 가 지상국 큐 상태
            usage_arr = []
            for j in range(num_stations):
                q_val = float(row[3 + j])
                usage_arr.append(q_val)

            # 한 시점에 대한 표준편차
            std_t = np.std(usage_arr)
            step_std_list.append(std_t)

    # 전체 평균
    if len(step_std_list) == 0:
        return 0.0
    return float(np.mean(step_std_list))


# 실행 예시
if __name__ == "__main__":
    csv_file = "random_queue_log.csv" #queue_usage.csv
    avg_std = compute_queue_std(csv_file)
    print(f"전체 스텝에 대한 지상국 큐 사용률 표준편차 평균: {avg_std:.4f}")
