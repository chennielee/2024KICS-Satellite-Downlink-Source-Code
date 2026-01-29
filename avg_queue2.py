import csv
import numpy as np

def compute_queue_std(csv_path, num_stations=10):
    """
    csv_path:
      [episode, step, env_minute, queue_gs_0, ..., queue_gs_(num_stations-1)]
    형태라고 가정.

    1) 마지막 1000 에피소드에 해당하는 데이터만 추려서
    2) 각 행마다 표준편차(np.std)를 구하고,
    3) 그 표준편차들의 평균을 계산함.
    """
    with open(csv_path, mode="r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # 헤더 건너뛰기
        
        # CSV 전체를 메모리에 한 번에 불러오기
        rows = list(reader)
    
    # 모든 행에서 episode 정보를 int로 변환하여 에피소드 번호 목록 생성
    ep_list = [int(r[0]) for r in rows]
    
    # 가장 큰 에피소드 번호
    max_ep = max(ep_list) if ep_list else 0
    
    # 시작 에피소드(최소 1 이상)
    start_ep = max(1, max_ep - 999)
    
    # 마지막 1000 에피소드(또는 데이터가 더 적으면 가능한 모든)만 필터링
    last_rows = [r for r in rows if int(r[0]) >= start_ep]
    
    step_std_list = []
    for row in last_rows:
        # row 예: [ep, step, min, q0, q1, ..., q9]
        usage_arr = []
        for j in range(num_stations):
            q_val = float(row[3 + j])
            usage_arr.append(q_val)
        
        # 한 시점에 대한 표준편차
        std_t = np.std(usage_arr)
        step_std_list.append(std_t)
    
    # 마지막 1000 에피소드(필터링된 데이터)의 표준편차 평균
    if len(step_std_list) == 0:
        return 0.0
    return float(np.mean(step_std_list))

# 실행 예시
if __name__ == "__main__":
    csv_file = "random_queue_log.csv"  # 예: queue_usage.csv
    avg_std = compute_queue_std(csv_file)
    print(f"마지막 1000 에피소드에 대한 지상국 큐 사용률 표준편차 평균: {avg_std:.4f}")
