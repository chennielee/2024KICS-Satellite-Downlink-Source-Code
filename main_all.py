import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import csv

# skyfield, sgp4 관련 import
from datetime import datetime, timedelta
from skyfield.api import load, utc, wgs84, EarthSatellite
from skyfield.framelib import itrs
from sgp4.api import Satrec, jday

# 모듈 import
from env import OCO2Env 
from agent import DQNAgent
from random_policy import run_random_policy
from greedy_policy import run_greedy_policy
from satellite_utils import get_oco2_positions_16days, compute_visibility_all

# device 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

def main_all():
    """
    main_all()은 (1) 랜덤 정책 결과(random_rewards), (2) DQN 학습 결과 (dqn_rewards), (3) greedy 정책 결과
    를 얻고, 한 그래프에 표현한다.
    """

    # 난수 시드 설정
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # 1> 위성 OCO-2 TLE
    line1 = "1 40059U 14035A   25009.89311847  .00001165  00000+0  26847-3 0  9990"
    line2 = "2 40059  98.2255 313.7501 0001333  70.4411 289.6932 14.57138477559845"
    satellite = EarthSatellite(line1, line2, "OCO 2")

    ts = load.timescale()
    satellite.ts = ts

    # 2> 지상국 좌표 (lat,lon)
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

    station_names_list = list(stations.keys()) # ['KSAT Svalbard', 'SANSA Hartebeesthoek', ...]

    # 60*24*16일(매 분) 궤도계산
    start_date = datetime(2025, 1, 10, 0, 0, 0, tzinfo=utc)
    oco2_positions = get_oco2_positions_16days(start_date, satellite, ts)

    # 전체 지상국 가시성 행렬
    # compute_visibility_all도 station_names_list 순서에 의존 (내부적으로도 같은 순서)
    station_names, vis_map = compute_visibility_all(
        satellite, stations, start_date, ts, num_steps_per_cycle=60*24*16
    )


    ##### Env 생성 - 3개 서로 다른 env 인스턴스 #####
    #    1) random_env      -> run_random_policy
    #    2) greedy_env   -> run_greedy_policy
    #    3) dqn_env         -> DQN 학습
    ############################################################

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

    print("[DEBUG] Creating env for greedy policy...")
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

    print("[DEBUG] Creating env for DQN agent")
    np.random.seed(SEED)
    env = OCO2Env(
        satellite=satellite,
        stations=stations,
        station_names=station_names, # or station_names_list (둘 다 동일 순서)
        start_date=start_date,
        vis_map=vis_map,
        oco2_positions=oco2_positions,
        leo_queue_capacity = 60*24*1,
        gs_queue_capacity = 5000, # 원래 leo_queue_capacity가 60*24*1인 것 기준으로 15000
        cycles = 1,
        use_future_los=True #<- 옵션 B
    )
    print("[DEBUG] All env creation done")


    # csv 파일 열기
    with open("episode_rewards.csv", "w", newline="") as epi_f, \
         open("actions.csv", "w", newline="") as act_f, \
         open("aoi_gs.csv", "w", newline="") as aoi_f, \
         open("queue_usage.csv", "w", newline="") as queue_f:

        # (1) 에피소드 별 보상
        epi_writer = csv.writer(epi_f)
        epi_writer.writerow(["episode", "total_reward"])

        # (2) 액션 기록
        act_writer = csv.writer(act_f)
        act_header = ["episode", "step", "reward", "chosen_station"] + [f"gs_idx{i}" for i in range(11)]
        act_writer.writerow(act_header)

        # (3) AoI 기록
        aoi_writer = csv.writer(aoi_f)
        aoi_header = ["episode", "step", "env_minute", "energy"] + [f"aoi_gs_{j}" for j in range(11)]
        aoi_writer.writerow(aoi_header)

        # (4) 큐 사용량 기록 (queue_usage.csv)
        queue_writer = csv.writer(queue_f)
        queue_header = ["episode", "step", "env_minute"] + [f"queue_gs_{j}" for j in range(env.num_stations)]
        queue_writer.writerow(queue_header)

        # Run three policies
        print("[DEBUG] Running RANDOM policy...")
        random_rewards = run_random_policy(
            env_random, 
            num_episodes=100, 
            csv_path="random_log.csv", 
            step_csv_path="random_steps_log.csv",
            queue_csv_path="random_queue_log.csv"
        )

        print("[DEBUG] Running greedy policy...")
        greedy_rewards = run_greedy_policy(
            env_greedy, 
            num_episodes=100, 
            csv_path="greedy_log.csv", 
            step_csv_path="greedy_steps_log.csv", 
            queue_csv_path="greedy_queue_log.csv" 
        )

        obs_dim = env.observation_space.shape[0]
        act_dim = 11  # 0..10 (NoAction + 10개 GS)
        agent = DQNAgent(obs_dim=obs_dim, act_dim=act_dim, lr=1e-4, device=device)

        num_episodes = 100 # 50000 -> 10000
        max_steps_per_ep = env.episode_length #16*24*60
        batch_size = 64 #128
        all_rewards_dqn = []

        update_freq = 4

        print("[DEBUG] Starting DQN training...")
        for episode in range(num_episodes):
            obs = env.reset()
            total_reward = 0.0
            done = False
            step_count = 0

            #print(f"[DEBUG] Episode={episode+1} start")

            while (not done) and (step_count < max_steps_per_ep):
                action_idx = agent.select_action(obs)
                # action_idx -> MultiBinary(10) 변환
                if action_idx == 0:
                    # NoAction
                    action_vec = np.zeros(10, dtype=int)
                else:
                    # 1..10 -> station_idx = action_idx - 1
                    gs_idx = action_idx - 1
                    action_vec = np.zeros(10, dtype=int)
                    action_vec[gs_idx] = 1

                next_obs, reward, done, info = env.step(action_vec)

                agent.store_transition(obs, action_idx, reward, next_obs, done)
                obs = next_obs
                total_reward += reward

                ## 수정한 부분!! 원래 if 문 없음
                if step_count % update_freq == 0:
                    agent.train_on_batch(batch_size)

                # (A) CSV에 액션 정보 저장
                act_writer.writerow([
                    episode, step_count, reward, action_idx,
                    *action_vec  # unpack
                ])

                # (B) CSV에 AoI 정보 저장
                # env.aoi_gs가 np.array(길이=env.num_stations)라 가정
                aoi_list = env.aoi_gs.tolist()
                energy_val = env.energy
                aoi_writer.writerow([
                    episode, step_count, env.minute, energy_val,
                    *aoi_list
                ])

                # (C) [NEW CODE] 큐 사용량 CSV
                queue_list = env.queue_gs.tolist()  # shape=(10,)  
                # 10개 지상국에 대한 queue 정보
                queue_writer.writerow([
                    episode, 
                    step_count, 
                    env.minute, 
                    *queue_list
                ])

                step_count += 1
                #if step_count % 1000 == 0:
                #    print(f"  [DEBUG] Episode={episode+1}, step={step_count}, partial_reward={total_reward:.3f}")

            # (1) while문이 끝난 "직후" → 에피소드가 종료됨
            #     남은 n-step 버퍼(예: 2개 남았을 수도)를 마저 처리
            agent.finish_n_step()  # [추가!! 남은 transition 처리]

            # (2) 그 다음, 보상 기록이나 출력
            all_rewards_dqn.append(total_reward)
            epi_writer.writerow([episode, total_reward])  # (C) 에피소드별 보상 기록
            print(f"Episode={episode+1}, total_reward={total_reward:.3f}, epsilon={agent.epsilon:.3f}")


        # 모델 저장 (옵션)
        agent.save("dqn_agent.pt")

        # 결과 시각화1
        plt.figure(figsize=(10,6))
        plt.plot(all_rewards_dqn, label="DQN")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DQN Training Progress")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("dqn_raw.png")
        plt.close()

        # 결과 시각화 2. 
        # 2) 스무딩한 그래프
        plt.figure(figsize=(10,6))

        # window = 20 으로 해도 됨.
        def smooth_curve(rewards, window=20):
            return np.convolve(rewards, np.ones(window)/window, mode='valid')

        random_smooth = smooth_curve(random_rewards, window=10)
        greedy_smooth = smooth_curve(greedy_rewards, window=10)
        dqn_smooth = smooth_curve(all_rewards_dqn, window=10)
        dqn_smooth2 = smooth_curve(all_rewards_dqn, window=20)


        plt.plot(dqn_smooth, label="DQN")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DQN Training Progress")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("dqn_smoothed.png")
        plt.close()

        plt.plot(dqn_smooth2, label="DQN")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DQN Training Progress")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("dqn_smoothed2.png")
        plt.close()


        print("[DEBUG] Done! Saved comparison figures.")


if __name__ == "__main__":
    main_all()