##### [NEW FILE] greedy_policy.py #####
import numpy as np
import csv

def run_greedy_policy(env, num_episodes=100, csv_path=None, step_csv_path=None, queue_csv_path=None):
    """
    '가시성(vis_map)가 있는 지상국 중 거리(distance)가 가장 가까운 곳'에만
    전송하는 Greedy 정책.
    
    - env: OCO2Env 인스턴스
    - num_episodes: 수행 에피소드 수
    """
    all_rewards = []

    # (1) episode 에피소드 별 보상 csv
    if csv_path is not None:
        f_ep = open(csv_path, "w", newline="")
        writer_ep = csv.writer(f_ep)
        writer_ep.writerow(["episode", "total_reward"])
    else:
        f_ep = None
        writer_ep = None

    # (2) 스텝별 CSV (AoI, energy 등)
    if step_csv_path is not None:
        f_step = open(step_csv_path, "w", newline="")
        writer_step = csv.writer(f_step)
        header_step = ["episode", "step", "env_minute", "energy"] + [f"aoi_gs_{i}" for i in range(env.num_stations)]
        writer_step.writerow(header_step)
    else:
        f_step = None
        writer_step = None

        # (C) 스텝별 큐 사용량 CSV
    if queue_csv_path is not None:
        f_queue = open(queue_csv_path, "w", newline="")
        writer_queue = csv.writer(f_queue)
        # 예: [episode, step, env_minute, queue_gs_0, ..., queue_gs_9]
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

        max_steps = env.episode_length  # 최대 스텝

        while (not done) and (step_count < max_steps):
            # [1] 현재 시점(minute)에 대한 가시성 정보와 거리를 env 내부에서 추출
            minute_idx = env.minute % env.num_steps_per_cycle
            
            # vis_map[minute_idx] = shape(10,) : 0(가시), -1(비가시)
            vis_row = env.vis_map[minute_idx]  # ex: array of length num_stations(10)
            
            # distance_all[minute_idx] = shape(10,) : 각 지상국까지 거리
            dist_row = env.distance_all[minute_idx]

            # [2] '가시성 있는 지상국'을 찾음
            visible_indices = np.where(vis_row == 0)[0]  # 0인 스테이션 인덱스들
            
            if len(visible_indices) == 0:
                # 가시성 있는 지상국 없음 => NoAction
                action_vec = np.zeros(env.num_stations, dtype=int)
            else:
                # 가시성 있는 지상국 중, distance가 가장 작은 곳 선택
                visible_distances = dist_row[visible_indices]
                min_idx = np.argmin(visible_distances)
                chosen_station = visible_indices[min_idx]

                action_vec = np.zeros(env.num_stations, dtype=int)
                action_vec[chosen_station] = 1

            # [3] env.step()으로 진행
            next_obs, reward, done, info = env.step(action_vec)
            total_reward += reward
            step_count += 1

            # 스텝별 AoI/energy 로깅
            if writer_step is not None:
                aoi_list = env.aoi_gs.tolist()
                energy_now = env.energy
                writer_step.writerow([
                    episode+1,
                    step_count,
                    env.minute,
                    energy_now,
                    *aoi_list
                ])

            # 스텝별 큐 사용량 기록
            if writer_queue is not None:
                queue_list = env.queue_gs.tolist()  # shape=(10,)
                writer_queue.writerow([
                    episode+1,
                    step_count,
                    env.minute,
                    *queue_list
                ])

            if step_count >= env.episode_length:
                done = True



        all_rewards.append(total_reward)
        print(f"[GREEDY]] Episode={episode+1}, total_reward={total_reward:.3f}")

        if writer_ep is not None:
            writer_ep.writerow([episode+1, total_reward])

    # 파일 닫기
    if f_ep is not None:
        f_ep.close()
    if f_step is not None:
        f_step.close()
    if f_queue is not None:
        f_queue.close()

    return all_rewards
