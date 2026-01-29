import random
import numpy as np
import csv

def run_random_policy(env, num_episodes=1000, csv_path=None, step_csv_path=None, queue_csv_path=None):
    """
    주어진 env(OCO2Env)에서 학습 없이, 매 스텝마다 0~10 중 무작위로 액션을 선택하여
    num_episodes만큼 에피소드를 돌린 뒤, 각 에피소드의 총 보상을 리스트로 반환한다.
    
    - env: 이미 초기화된 OCO2Env (또는 동일 인터페이스를 가진 환경)
    - num_episodes: 실행할 에피소드 수
    """
    all_rewards = []
    
    # (1) 에피소드 별 총보상 CSV 열기.
    if csv_path is not None:
        f_ep = open(csv_path, "w", newline="")
        writer_ep = csv.writer(f_ep)
        writer_ep.writerow(["episode", "total_reward"])
    else:
        f_ep = None
        writer_ep = None

     # (2) 스텝별 로깅 CSV 열기 (AoI, 에너지, 등)
    if step_csv_path is not None:
        f_step = open(step_csv_path, "w", newline="")
        writer_step = csv.writer(f_step)
        # 헤더: [episode, step, env_minute, energy, aoi_gs_0, aoi_gs_1, ..., aoi_gs_9]
        header_step = ["episode", "step", "env_minute", "energy"] + [f"aoi_gs_{i}" for i in range(env.num_stations)]
        writer_step.writerow(header_step)
    else:
        f_step = None
        writer_step = None

    # (C) 스텝별 큐 사용량 CSV
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
        
        # 환경에 정의된 최대 스텝 수(에피소드 길이) 가져오기 (예: env.episode_length)
        max_steps_per_ep = env.episode_length
        
        while (not done) and (step_count < max_steps_per_ep):
            minute_idx = env.minute % env.num_steps_per_cycle
            
            # 2) 이 minute_idx에서 vis_map (shape: [num_steps_per_cycle, num_stations])을 확인
            #    -> vis_row[j] == 0 이면 j번 지상국이 가시성 있음,
            #       vis_row[j] == -1 이면 가시성 없음
            vis_row = env.vis_map[minute_idx]
            
            # 3) 가시성이 있는 지상국 인덱스들 뽑기
            visible_indices = np.where(vis_row == 0)[0]
            
            # 4) visible_indices가 비어 있으면 NoAction (액션 0벡터)
            if len(visible_indices) == 0:
                action_vec = np.zeros(env.num_stations, dtype=int)
            else:
                # 그렇지 않다면, 그중 랜덤으로 하나 골라 액션 1
                chosen_station = random.choice(visible_indices)
                action_vec = np.zeros(env.num_stations, dtype=int)
                action_vec[chosen_station] = 1

            """
            # 0..10 사이 무작위 액션 인덱스 선택/ 지상국 가시성 고려하지 않음.
            action_idx = random.randint(0, 10)
            
            # action_idx -> MultiBinary(10) 벡터 변환
            if action_idx == 0:
                # 0번이면 NoAction
                action_vec = np.zeros(10, dtype=int)
            else:
                # 1~10 중 하나면 station_idx = action_idx - 1
                station_idx = action_idx - 1
                action_vec = np.zeros(10, dtype=int)
                action_vec[station_idx] = 1
            """
            
            # 환경에 액션 적용
            next_obs, reward, done, info = env.step(action_vec)
            
            # 보상 누적
            total_reward += reward
            step_count += 1

            # 스텝별 AoI/energy 로깅
            if writer_step is not None:
                aoi_list = env.aoi_gs.tolist()       # shape=(10,)
                energy_now = env.energy             # float
                # episode+1 로 저장하는 것이 일관성이 좋을 수도 있음(원하시는 대로)
                writer_step.writerow([
                    episode+1, 
                    step_count, 
                    env.minute, 
                    energy_now,
                    *aoi_list
                ])

            # 스텝별 큐 사용량 기록
            if writer_queue is not None:
                queue_list = env.queue_gs.tolist()
                writer_queue.writerow([
                    episode+1,
                    step_count,
                    env.minute,
                    *queue_list
            ])

            if step_count >= env.episode_length:
                done = True    
 
        # 이번 에피소드의 총 보상 저장
        all_rewards.append(total_reward)
        print(f"[RANDOM] Episode={episode+1}, total_reward={total_reward:.3f}")

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
