"""
1. 매일 UTC 0시 기준으로 1분마다 oco-2의 위치와 고도각 확보 여부 등을 tracking.
2. 10개 지상국 중 어떤 곳(또는 아무 곳에도) 전송할지 결정.
3. 데이터는 1분마다 쌓임.
4. 한 번에 하나의 지상국만 전송 가능(멀티바이너리 액션).
* 가시성(vis_map)을 obs에 포함하여 agent가 action masking에 활용할 수 있도록 함.

5. 위성 큐(LEO queue)와 지상국 큐(GS queue)에 '최대 용량'을 두어 overflow를 처리.
6. 초기 지상국 AoI는 무작위로 설정.
7. 첨부 식에 따라 AoI를 갱신:
        AoI^LEO(t) = (AoI^LEO(t-1)+1) if a_j(t)=0,
                    0               if a_j(t)=1

        AoI^GS_j(t) = (AoI^GS_j(t-1)+1) if a_j(t)=0, queue_gs_j(t) > 0
                    -1               if a_j(t)=0, queue_gs_j(t) = 0
                    AoI^LEO(t)       if a_j(t)=1, queue_gs_j(t) >= 0


8. 에너지 모델링 추가 :
거리기반 p_transmitter 모델링 + Obsesrvation에 distance 추가

9. num_steps_per_cycle = 60 * 24 * 1 = 1440 (하루)
- cycles를 설정하여, 이틀, 삼일 등을 한 에피소드로 설정할 수 있다.
- 혹은, num_steps_per_cycle을 60*24*16 으로 하면, cycle이 위성의 궤도 주기를 의미하게 된다.

"""

import gym
import numpy as np
from gym import spaces

from satellite_utils import compute_station_itrf_list, compute_distance_all, compute_station_passes, build_future_los_array, get_transmit_power_watt

class OCO2Env(gym.Env):
    metadata = {"render.modes": ["human"]}

    # parameter의 station_names (인덱스 <-> 이름)
    def __init__(self, 
                 satellite, 
                 stations, 
                 station_names, 
                 start_date,
                 vis_map, 
                 oco2_positions, 
                 leo_queue_capacity=60*24, 
                 gs_queue_capacity=5000, 
                 cycles=1,
                 use_future_los = False # false : future_los를 observation이 활용하지 않음. 오로지 경험을 통해 학습. / True : future_los를 observation을 통해 활용가능.
                 ):
      
        # 상속받은 부모 클래스인 gym class 초기화
        super(OCO2Env, self).__init__()

        self.satellite = satellite
        self.stations = stations
        self.station_names = station_names
        self.start_date = start_date
        self.vis_map = vis_map  # shape (60*24*16, 13)
        self.oco2_positions = oco2_positions  # shape (60*24*16, (x,y,z))

        self.num_steps_per_cycle = int(((vis_map.shape[0])/16)*1) # 60*24*16
        self.episode_length = self.num_steps_per_cycle * cycles # 주기 고려
        self.num_stations = vis_map.shape[1] # 10

        # 큐 용량 파라미터
        self.leo_queue_capacity = leo_queue_capacity
        self.gs_queue_capacity = gs_queue_capacity

        # Friis/FSPL 관련 파라미터
        # DR(data rate)에 대해서 Shannon capacity는 고려하지 않음.
        # p_transmitter와 관련하여, Friis/FSPL 개념에 근거하여, 그 송신 전력을 계산한다.
        self.freq_mhz = 2000.0 #주파수(단위 : MHz), 2 GHz 대역 가정
        self.Gt_dBi = 15.0 #송신 안테나 이득(dBi)
        self.Gr_dBi = 20.0 #수신 안테나 이득(dBi)

        # 아래 값은 "최소 요구 수신전력 달성"을 위한 레퍼런스. 
        # # 참고로 최소 요구 SNR 달성을 목표로 할 것이라면 Shannon capacity 까지 고려해서 식을 추가해야 함.
        # 즉, 실제로는 SNR 기반으로 Shannon Capacity를 구해, data_rate을 동적으로 바꾸는 방식으로 확장할 수도 있다. (+ 실제 통신 링크에서는 링크 마진, 시스템 잡음온도, 주파수 대역폭 등을 더 세밀하게 고려할 수 있음.)
        self.required_Pr_dBm = -120.0 # 예 : -100 dBm 이상 필요하다고 가정. 낮출수록 pt_need_w 편차가 줄어들어, 보상이 튀는 현상이 완화될 수 있음.
        self.data_rate = 1.0 # DR (bit/s) 가정 # 실제로는 거리에 다른, Pr의 변화로 인해,shannon capacity에 근거하여 data_rate도 변화하지만 여기서는, 고려하지 않는다.
        self.c_charging = 3.0 # 1분당 7Wh 충전 가정.[PR: 7-> 3]
        self.s_data = 1.0 # 전송할 데이터 크기(비트) - 예시.


        # 매일 지상국까지 거리 계산 (km 단위)
        #  - 기존에 oco2_positions[day]=(x,y,z), station_positions=(gx,gy,gz) 를 이용해
        #  - 24*60*1 x 10개 지상국의 '거리(km)'를 미리 계산해둠.
        self.gs_positions = compute_station_itrf_list(
            self.stations,
            self.station_names
        )


        """
        OCO2Env.__init__에서 “self.gs_positions를 만든 시점”은
        “메인에서 env = OCO2Env(...)를 호출”하는 그 순간이며,
        이미 그때 station_names의 순서가 고정되어 있음.
        """
        self.distance_all = compute_distance_all(
            self.num_steps_per_cycle, 
            self.num_stations, 
            self.oco2_positions, 
            self.gs_positions
        ) 
        # 모든 (min, station_idx)에 대한 거리를 미리 구해, self.distance_all[m,j]에 저장한다.
        # 즉, "가시성"과는 상관없이, 우선 "24*60*16 * 10개 지상국" 거리 전체를 계산해둔다.

        # Observation 구성:
        #  1) day (1)
        #  2) aoi_leo (1)
        #  3) aoi_gs (10)
        #  4) queue_leo (1)
        #  5) queue_gs (10)
        #  6) vis_map_now (10)
        #  7) distance_now (10)
        #  8) leo_queue_usage (1)
        #  9) gs_queue_usage (10)
        # 10) energy (1)
        # 11) (선택) future_los_array (10)

        # 총 길이 = 1 +1 +10 +1 +10 +10 +10 +1 +10 +1 = 55 or 65
        # [옵션 B] future_los 사용 여부
        self.use_future_los = use_future_los
        if self.use_future_los:
            # (A) pass 계산
            self.station_passes = compute_station_passes(
                self.num_steps_per_cycle, 
                self.start_date, 
                self.satellite, 
                self.stations
            )
            # (B) pass -> future_los_array
            self.future_los_array = build_future_los_array(
                self.start_date, 
                self.num_steps_per_cycle, 
                self.num_stations, 
                self.station_names, 
                self.station_passes
            )
            # obs_dim=65
            self.obs_dim = 65
        else:
            # obs_dim=55
            self.obs_dim = 55


        # Action space: 길이 = num_stations (MultiBinary)
        #    예) [0,0,0,0,1,0,0,0,0,0] -> 5번 지상국에만 전송
        self.action_space = spaces.MultiBinary(self.num_stations)

        high_val = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-high_val, high=high_val, shape=(self.obs_dim,), dtype=np.float32
        )

        # 내부 상태
        self.minute = 0 # 이제 minute는 분 단위 인덱스로 동작.
        self.aoi_leo = 0
        self.aoi_gs = np.zeros(self.num_stations, dtype=int)
        self.queue_leo = 0
        self.queue_gs = np.zeros(self.num_stations, dtype=int)
        self.energy = 300.0 #예 : 초기 300 Wh 가정
        self.energy_max = 1000.0 # 편리한 정규화를 위해 에너지 최대 용량 설정함. 총 1kWh 배터리라고 가정.

        self.done = False
        self.reset()


    def reset(self):
        self.minute = 0
        for j in range(self.num_stations):
            if np.random.rand() < 0.3:
                self.aoi_gs[j] = -1 # 30%확률로 데이터 없음,
            else:
                self.aoi_gs[j] = np.random.randint(0,6) # 0~5중 랜덤덤

        # 여기서는 reset() 시, 아직 첫날 이전이므로 "LEO가 데이터가 없다고 가정" -> aoi_leo=-1
        self.aoi_leo = -1  # 데이터가 아예 없다고 가정
        self.queue_leo = 0

        # 지상국의 큐도, aoi_gs 정보에 따라 설정해주어야 함.
        # 데이터가 없는 지상국(AoI=-1)이면 해당 큐를 0으로(사실상 의미 같음)
        for j in range(self.num_stations):
            if self.aoi_gs[j] == -1:
                self.queue_gs[j] = 0
            else:
                self.queue_gs[j] = np.random.randint(10, 30) # 아니면, 모두 aoi_gs 가 -1 이고, 모두, queue_gs = 0 인 상황에서 실험해봐도 됨.

        # 에너지 리셋
        self.energy = 300.0
        self.done = False
        return self._get_obs()

    def _generate_new_data_if_needed(self):
        """
        매일(스텝이 시작될 때) LEO에 새 데이터 1개가 생긴다고 가정.
        만약 LEO가 데이터가 없었다면(aoi_leo=-1), 그걸 0으로 만들어줌.
        """

        if self.queue_leo == 0:
            # 데이터가 없던 상태: 새 데이터 1건 생성
            self.queue_leo = min(self.queue_leo + 1, self.leo_queue_capacity)
            self.aoi_leo = 0  # 가장 최신 데이터
        else:
            # 이미 데이터가 있어도, 큐에 데이터 하나 추가, AoI도 +1
            if self.aoi_leo >= 0:
                self.aoi_leo += 1
                self.queue_leo = min(self.queue_leo + 1, self.leo_queue_capacity) #추가.
            else:
                # 혹시 -1인 상태인데 queue_leo>0이면 모순 -> 보정
                self.aoi_leo = 0


    def step(self, action):
        """
        action: 길이=13의 0/1 벡터

        # (1) _generate_new_data_if_needed()로 새 데이터 발생
        # (2) action 해석 -> 전송/가시성 체크(혹시 몰라서 env에서 한 번 더 하는 것일뿐. 사실상 action masking으로 걸러짐), 큐 업데이트, 보상
        # (3) AoI 갱신 (전송받은 지상국은 aoi= old_aoi_leo 등)
        # (4) 지상국 큐에서 매 스텝마다 1씩 차감
        # (5) AoI 보상 + 큐 보상 계산 + 에너지 보상 계산

        # (6) minute+=1, if self.minute>=self.episode_length: done=True
        # (7) return obs, reward, done, {}

        """

        # 매 스텝 새 데이터 생성 로직
        self._generate_new_data_if_needed()
        old_aoi_leo = self.aoi_leo 

        reward = 0.0
        step_mod = self.minute % self.num_steps_per_cycle
        e_comm_total = 0.0

        # 액션 해석
        chosen_stations = np.where(action == 1)[0]  # 1인 지상국 인덱스들

        # 데이터 전송(큐 업데이트) 로직은 "하나만 선택"일 때에만 수행
        if len(chosen_stations) > 1:
            # 여러 개 동시 선택 -> 큰 페널티 (코드 구조상 여러개 동시 선택하는 상황은 발생하지 않지만 그냥 넣어둠)
            reward -= 10.0
        elif len(chosen_stations) == 1:
            gs_idx = chosen_stations[0]
            
            # 전송 횟수당 Overhead Penalty (Fixed)
            reward -= 0.5
            
            if self.vis_map[step_mod, gs_idx] == 0:

                # 1) queue_leo를 미리 읽어 "만약 전송된다면"의 e_comm 계산
                # [Energy Add] 식1: E_j^comm(t) = (p_transmitter/DR)*s_j(t)*a_j(t)
                # 거리 기반 송신전력
                # -----------------------------
                self.distance_now = self.distance_all[step_mod, gs_idx]
                p_trans_w = get_transmit_power_watt(
                    self.distance_now, 
                    self.freq_mhz, 
                    self.required_Pr_dBm, 
                    self.Gt_dBi, 
                    self.Gr_dBi
                )

                # 전송 결정시, 큐에 있는 데이터 전부 전송-> X* s_j(t)
                # self.queue_leo가 아니라, transmitted data를 큐에 있는 데이터의 개수로 사용해야 함. self.queue_leo는 일단 전송하면 비어버리기 때문.
                # 전송 에너지 계산

                # 이미 reward function 안에 에너지 잔량을 높게 유지하는 항목 reward += w_e * energy_norm 이 있지만, 그것만으로는,
                # '송신전력'(또는 소모 에너지) 자체에 대한 즉각적인 페널티가 크지 않을 수 있음.
                e_comm = (p_trans_w / self.data_rate) * (self.queue_leo * self.s_data)
                

                if e_comm > self.energy:
                    reward -= 3.0
                else:
                    transmitted_data = self.queue_leo
                    self.queue_leo = 0 # 위성 큐에서 전송 완료
                    self.aoi_leo = -1 # 전송 후 데이터 없음.

                    # 지상국 큐 남은 용량 계산
                    possible_space = self.gs_queue_capacity - self.queue_gs[gs_idx]
                    data_to_store = min(transmitted_data, possible_space)
                    self.queue_gs[gs_idx] += data_to_store
                    overflow = transmitted_data - data_to_store

                    if overflow>0:
                        reward -= 5.0 # 유실 발생 시 페널티
                    reward += 2.0

                    e_comm_total += e_comm
                    penalty_factor = 0.001
                    reward -= penalty_factor * e_comm
            else:
                # 가시성 없는데 선택 -> 실패
                reward -= 1.0
        else:
            # 아무것도 선택하지 않음
            reward -= 0.5 # 없어도 됨.

        # 모델링한 식에 따른 AoI 갱신 (지상국)
        old_aoi_gs = self.aoi_gs.copy()

        if len(chosen_stations) == 1:

            # GS AoI
            chosen_j = chosen_stations[0]
            for j in range(self.num_stations):
                # AoI^GS_j(t) = AoI^LEO(t) : 지금 갱신된 LEO AoI = 0 이 아니라, 그 직전의 값이 들어와야, 데이터의 나이를 확실히 고려한 것이 됨.
                # 즉, 전송 시점의 '데이터 나이'(= old AoI LEO)를 지상국에 반영하고 싶다.는 것.
                if j == chosen_j:
                    # 전송받는 지상국이 -1이었다면, 이제 old_aoi_leo를 물려받는다.
                    # (데이터가 없던 곳 -> 데이터 들어옴)
                    if old_aoi_gs[j] == -1:
                        self.aoi_gs[j] = old_aoi_leo
                    else:
                        self.aoi_gs[j] = old_aoi_leo
                else:
                    # 만약 old_aoi_gs[j]가 -1이면, 그대로 -1 유지
                    if old_aoi_gs[j] == -1:
                        self.aoi_gs[j] = -1
                    else:
                        self.aoi_gs[j] = old_aoi_gs[j] + 1

        elif len(chosen_stations) > 1:
            # 여러개 동시 선택 -> 이미 위에서 페널티 받음
            # 이미 agent에서 action masking을 해서, len(chosen_stations) > 1 인 경우는 일어나지 않지 않음.
            # "식에서 정의가 없으므로, "전송 안 한 것"과 동일하게 처리"
            for j in range(self.num_stations):
                if old_aoi_gs[j] != -1:
                    self.aoi_gs[j] = old_aoi_gs[j] + 1
        else:
            # 아무것도 선택 안 한 경우
            self.aoi_leo = old_aoi_leo + 1
            for j in range(self.num_stations):
                if old_aoi_gs[j] != -1:
                    self.aoi_gs[j] = old_aoi_gs[j] + 1

        # 추가) 지상국 큐에서 매 스텝마다 1씩 차감
        for j in range(self.num_stations):
            if self.queue_gs[j] > 0:
                self.queue_gs[j] = max(0, self.queue_gs[j]-1)

        #   E(t) = E(t-1) - sum(E_comm_j) + c
        #   (여러 지상국을 골랐으면 합산, 여기서는 하나면 e_comm_total)
        self.energy = self.energy - e_comm_total + self.c_charging

        # [추가] 에너지가 최대치를 초과하지 않도록 clamp
        if self.energy > self.energy_max:
            self.energy = self.energy_max

        # 위성 에너지가 고갈되었을 경우 음수 리워드 추가
        if self.energy < 0:
            reward -= 5.0  # [MODIFIED] 에너지 고갈에 따른 큰 패널티
            self.energy = 0.0
            # self.done = True  # 조기 종료를 원할시에 활성화


        # 보상 계산:
        # R(t) = w^AoI * R^AoI + w^Q * R^Q + w^E * R^E

        # AoI 기반 부가 보상
        w_a = 0.3 # [pr]
        aoi_reward = 0.0
        for j in range(self.num_stations):
            if self.aoi_gs[j] == -1:
                # 데이터 없음 => 보상 0 (원한다면 음수 줄 수도 있음)
                pass
            elif self.aoi_gs[j] == 0:
                # AoI=0 => 최신 데이터
                aoi_reward += 1.0
            else:
                # AoI>0
                aoi_reward += (1.0 / self.aoi_gs[j])

        # AoI 정규화
        aoi_reward /= self.num_stations # 최대값을 1 근처로 스케일
        # 기존 reward에 AoI 보상 더하기
        reward += w_a * aoi_reward


        # 큐 사용량 표준편차, 잔여공간 평균을 이용한 큐 보상
        # 참고 ) gs_queue_capacity는 모든 station에 대해 동일하다. 이를 station마다 다르게 하고 싶다면, 즉 Q^GS_{max}를 station별로 상이하다고 가정한다면,
        # 그만큼 코드를 station별 array로 처리해야 할 것.

        # 예시 파라미터
        alpha = 0.3   # 표준편차 불이익 계수 PR : 0.3 -> 1.0
        w_q   = 0.6   # 큐 보상 가중치 [PR 0.6 -> 2.0 -> 3.0]
        N = self.num_stations

        # 1) 잔여공간 평균: leftover_avg
        #    leftover_j = (gs_queue_capacity - queue_gs[j])
        leftover_list = [(self.gs_queue_capacity - self.queue_gs[j])
                        for j in range(N)]
        leftover_avg = np.mean(leftover_list)  # Q^GS_{l_avg}(t)

        # 2) 큐 사용공간 표준편차: usage_sd
        usage_sd = np.std(self.queue_gs)       # Q^GS_sd(t)

        # 정규화
        leftover_avg_norm = leftover_avg / self.gs_queue_capacity
        usage_sd_norm = usage_sd / self.gs_queue_capacity

        # 3) 최종 큐 보상
        queue_reward = leftover_avg_norm - alpha * usage_sd_norm
        reward += w_q * queue_reward


        # 에너지 보상 (정규화 추가)
        #    기존 : reward += w_e * self.energy
        #    변경 : 에너지를 [0, self.energy_max] 범위로 정규화

        # [Energy Add] w^E * R^E(t) = w^E * E(t)
        w_e = 0.1   # 에너지 가중치 [PR : 0.1 -> 0.5]
        energy_norm = self.energy / self.energy_max
        reward += w_e * energy_norm

        # 날짜 진행
        self.minute += 1
        if self.minute >= self.episode_length:
            self.minute = self.episode_length - 1  # day가 더 이상 증가하지 않도록
            self.done = True

        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        """
        Obs 구성(길이=55 or 65):
         1) day -> minute으로 수정
         ----
         2) aoi_leo
         3) aoi_gs(10)
         ----
         4) queue_leo
         5) queue_gs(10)
         ----
         6) vis_map_now(10) // Line of Sight (If Agent send when LoS = 0 -> Penalty)
         7) distance_now(10)  // Distance Between 1Agent <-> 1 GS (Used for Energy Efficiency Calculation)
         ----
         8) leo_queue_usage // Ratio
         9) gs_queue_usage(10) // Ratio
        10) energy
        11) 
        """
        obs_list = []
        obs_list.append(float(self.minute))          # minute
        obs_list.append(float(self.aoi_leo))         # aoI_leo
        obs_list.extend(self.aoi_gs.tolist())        # aoI_gs(10)
        obs_list.append(float(self.queue_leo))       # queue_leo
        obs_list.extend(self.queue_gs.tolist())      # queue_gs(10)

        step_mod = self.minute % self.num_steps_per_cycle # Time Now
        vis_map_now = self.vis_map[step_mod].astype(float)
        obs_list.extend(vis_map_now.tolist())

        # distance_now
        distance_now = self.distance_all[step_mod]
        obs_list.extend(distance_now.tolist())

        # 새로 추가된 부분
        # leo_queue_usage, gs_queue_usage (큐 사용량 비율(0~1)) Normalization
        leo_queue_usage = float(self.queue_leo) / self.leo_queue_capacity
        gs_queue_usage = self.queue_gs / self.gs_queue_capacity  # shape=(10,)

        obs_list.append(leo_queue_usage)
        obs_list.extend(gs_queue_usage.tolist())

        # [Energy Add] 에너지 상태 E(t)
        obs_list.append(float(self.energy))

        if self.use_future_los:
            # future_los_array
            step_mod = self.minute % self.num_steps_per_cycle
            fut_los_row = self.future_los_array[step_mod]  # shape=(10,)
            obs_list.extend(fut_los_row.tolist())

        return np.array(obs_list, dtype=np.float32)

    def render(self, mode="human"):
        print(f"Day={self.minute/1440}, Minute={self.minute}, AoI_LEO={self.aoi_leo}, AoI_GS={self.aoi_gs}")
        print(f"Queue_LEO={self.queue_leo}, Queue_GS={self.queue_gs}")