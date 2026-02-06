"""
1. Track OCO-2 position, elevation/visibility status, etc. every minute, using UTC 00:00 as the daily reference.
2. Decide which of the 10 ground stations to transmit to (or transmit to none).
3. New data accumulates every minute.
4. Only one ground station can be selected per step (MultiBinary action, but constrained to single-1).
* Include visibility (vis_map) in the observation so the agent can learn to use it for action masking.

5. Add maximum capacities for the satellite queue (LEO queue) and ground-station queues (GS queues) to handle overflow.
6. Initialize ground-station AoI randomly.
7. Update AoI according to the following equations:
        AoI^LEO(t) = (AoI^LEO(t-1)+1) if a_j(t)=0,
                    0               if a_j(t)=1

        AoI^GS_j(t) = (AoI^GS_j(t-1)+1) if a_j(t)=0, queue_gs_j(t) > 0
                    -1               if a_j(t)=0, queue_gs_j(t) = 0
                    AoI^LEO(t)       if a_j(t)=1, queue_gs_j(t) >= 0


8. Add energy modeling:
Distance-based transmitter power model + include distance in the observation.

9. num_steps_per_cycle = 60 * 24 * 1 = 1440 (one day)
- By setting 'cycles', you can define an episode as multiple days (e.g., 2 days, 3 days, etc.).
- Alternatively, if you set num_steps_per_cycle = 60*24*16, then one cycle corresponds to the satellite orbital period.
"""

import gym
import numpy as np
from gym import spaces

from satellite_utils import compute_station_itrf_list, compute_distance_all, compute_station_passes, build_future_los_array, get_transmit_power_watt

class OCO2Env(gym.Env):
    metadata = {"render.modes": ["human"]}

    # station_names parameter defines the mapping (index <-> name)
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
                 use_future_los = False # False: future_los is not included in the observation (agent must learn purely from experience).
                                        # True : future_los is included in the observation (agent can exploit it explicitly).
                 ):
      
        # Initialize the parent gym.Env
        super(OCO2Env, self).__init__()

        self.satellite = satellite
        self.stations = stations
        self.station_names = station_names
        self.start_date = start_date
        self.vis_map = vis_map  # shape (60*24*16, 13)
        self.oco2_positions = oco2_positions  # shape (60*24*16, (x,y,z))

        self.num_steps_per_cycle = int(((vis_map.shape[0])/16)*1) # 60*24*16
        self.episode_length = self.num_steps_per_cycle * cycles # episode length considering cycles
        self.num_stations = vis_map.shape[1] # 10

        # Queue capacity parameters
        self.leo_queue_capacity = leo_queue_capacity
        self.gs_queue_capacity = gs_queue_capacity

        # Friis/FSPL-related parameters
        # Shannon capacity (data rate adaptation) is NOT considered here.
        # We compute required transmit power based on a Friis/FSPL-style link budget.
        self.freq_mhz = 2000.0 # frequency (MHz), assuming 2 GHz band
        self.Gt_dBi = 15.0 # transmit antenna gain (dBi)
        self.Gr_dBi = 20.0 # receive antenna gain (dBi)

        # Reference minimum received power requirement.
        # If the goal is minimum required SNR, Shannon capacity would need to be included as well.
        # In practice, one could extend this by computing data_rate dynamically from SNR via Shannon capacity.
        # (Real link budgets may also include link margin, system noise temperature, bandwidth, etc.)
        self.required_Pr_dBm = -120.0 # e.g., require received power >= -120 dBm
                                      # Lowering this can reduce variance in pt_need_w and stabilize rewards.
        self.data_rate = 1.0 # assumed data rate (bit/s); kept constant here (no Shannon-capacity adaptation)
        self.c_charging = 3.0 # assume charging of 3 Wh per minute [PR: 7 -> 3]
        self.s_data = 1.0 # data size per unit (bits), example value


        # Precompute distances to all ground stations (in km):
        # Using oco2_positions[minute] = (x,y,z) and station_positions = (gx,gy,gz),
        # Precompute (24*60*1) x (num_stations) distances.
        self.gs_positions = compute_station_itrf_list(
            self.stations,
            self.station_names
        )

        """
        The timing of creating self.gs_positions in OCO2Env.__init__ is
        exactly when the main code calls env = OCO2Env(...),
        and by then the order of station_names is already fixed.
        """
        self.distance_all = compute_distance_all(
            self.num_steps_per_cycle, 
            self.num_stations, 
            self.oco2_positions, 
            self.gs_positions
        ) 
        # Store distances for all (minute, station_idx) pairs in self.distance_all[m, j].
        # This is computed regardless of visibility (LoS); we precompute the full distance grid.

        # Observation :
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
        # 11) (optional) future_los_array (10)

        # Total Length = 1 +1 +10 +1 +10 +10 +10 +1 +10 +1 = 55 or 65
        # [옵션 B] whether to include future_los
        self.use_future_los = use_future_los
        if self.use_future_los:
            # (A) compute station passes
            self.station_passes = compute_station_passes(
                self.num_steps_per_cycle, 
                self.start_date, 
                self.satellite, 
                self.stations
            )
            # (B) convert passes -> future_los_array
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


        # Action space: MultiBinary with length = num_stations
        # Example: [0,0,0,0,1,0,0,0,0,0] -> transmit only to station #5
        self.action_space = spaces.MultiBinary(self.num_stations)

        high_val = np.finfo(np.float32).max
        self.observation_space = spaces.Box(
            low=-high_val, high=high_val, shape=(self.obs_dim,), dtype=np.float32
        )

        # Internal state
        self.minute = 0 # minute is the time-step index (in minutes)
        self.aoi_leo = 0
        self.aoi_gs = np.zeros(self.num_stations, dtype=int)
        self.queue_leo = 0
        self.queue_gs = np.zeros(self.num_stations, dtype=int)
        self.energy = 300.0 # e.g., initial energy = 300 Wh
        self.energy_max = 1000.0 # max battery capacity for convenient normalization (assume 1 kWh)

        self.done = False
        self.reset()


    def reset(self):
        self.minute = 0
        for j in range(self.num_stations):
            if np.random.rand() < 0.3:
                self.aoi_gs[j] = -1 # with 30% probability, no data
            else:
                self.aoi_gs[j] = np.random.randint(0,6) # random AoI in [0, 5]

        # At reset, assume the LEO has no data yet -> aoi_leo = -1
        self.aoi_leo = -1  # 데이터가 아예 없다고 가정
        self.queue_leo = 0

        # Initialize GS queues according to aoi_gs
        # If AoI=-1 (no data), set GS queue to 0 (equivalent meaning)
        for j in range(self.num_stations):
            if self.aoi_gs[j] == -1:
                self.queue_gs[j] = 0
            else:
                self.queue_gs[j] = np.random.randint(10, 30) # Alternatively, you can test the scenario where all AoIs are -1 and all GS queues are 0.

        # Reset energy
        self.energy = 300.0
        self.done = False
        return self._get_obs()

    def _generate_new_data_if_needed(self):
        """
        Assume 1 new data item is generated on the LEO at the beginning of each step.
        If the LEO previously had no data (aoi_leo = -1), set it to 0.
        """

        if self.queue_leo == 0:
            # No data previously: generate one new item
            self.queue_leo = min(self.queue_leo + 1, self.leo_queue_capacity)
            self.aoi_leo = 0  # freshest data
        else:
            # If data already exists, append one more item and increase AoI
            if self.aoi_leo >= 0:
                self.aoi_leo += 1
                self.queue_leo = min(self.queue_leo + 1, self.leo_queue_capacity) #추가.
            else:
                # If aoi_leo is -1 while queue_leo > 0, it is inconsistent -> Modify
                self.aoi_leo = 0


    def step(self, action):
        """
        action: a 0/1 vector of length=13

        (1) Generate new data via _generate_new_data_if_needed()
        (2) Interpret action -> transmit / visibility check (redundant safety check; action masking should handle this),
            queue update, and immediate reward shaping
        (3) Update AoI (the selected GS receives AoI derived from old_aoi_leo)
        (4) Dequeue 1 item from each GS queue per step
        (5) Compute reward components: AoI reward + queue reward + energy reward
        (6) minute += 1; if minute >= episode_length: done = True
        (7) return obs, reward, done, info

        """

        # Generate new data each step
        self._generate_new_data_if_needed()
        old_aoi_leo = self.aoi_leo

        reward = 0.0
        step_mod = self.minute % self.num_steps_per_cycle
        e_comm_total = 0.0

        # Interpret action
        chosen_stations = np.where(action == 1)[0]  # indices where action == 1

        # Transmission logic is applied only when exactly one station is selected
        if len(chosen_stations) > 1:
            # Multiple stations selected -> large penalty
            # (Should not happen if masking is correct, but kept for safety.)
            reward -= 10.0
        elif len(chosen_stations) == 1:
            gs_idx = chosen_stations[0]

            # Fixed overhead penalty per transmission attempt
            reward -= 0.5

            if self.vis_map[step_mod, gs_idx] == 0:

                # 1) Read queue_leo to compute communication energy if transmission occurs
                # [Energy] E_j^comm(t) = (p_transmitter / DR) * s_j(t) * a_j(t)
                # Distance-based transmitter power
                self.distance_now = self.distance_all[step_mod, gs_idx]
                p_trans_w = get_transmit_power_watt(
                    self.distance_now,
                    self.freq_mhz,
                    self.required_Pr_dBm,
                    self.Gt_dBi,
                    self.Gr_dBi,
                )

                # If transmitting, send all queued data: X * s_j(t)
                # Use transmitted_data as the number of items currently in queue (queue_leo will be set to 0 after sending).
                # Compute transmission energy.

                # Although the reward includes a term encouraging high remaining energy (reward += w_e * energy_norm),
                # that alone may not penalize instantaneous transmit energy strongly enough.
                e_comm = (p_trans_w / self.data_rate) * (self.queue_leo * self.s_data)

                if e_comm > self.energy:
                    reward -= 3.0
                else:
                    transmitted_data = self.queue_leo
                    self.queue_leo = 0  # transmission empties the satellite queue
                    self.aoi_leo = -1  # no data after transmission

                    # Compute remaining GS queue capacity
                    possible_space = self.gs_queue_capacity - self.queue_gs[gs_idx]
                    data_to_store = min(transmitted_data, possible_space)
                    self.queue_gs[gs_idx] += data_to_store
                    overflow = transmitted_data - data_to_store

                    if overflow > 0:
                        reward -= 5.0  # penalty for packet/data loss due to overflow
                    reward += 2.0

                    e_comm_total += e_comm
                    penalty_factor = 0.001
                    reward -= penalty_factor * e_comm
            else:
                # Selected when not visible -> failure
                reward -= 1.0
        else:
            # No station selected
            reward -= 0.5  # optional penalty

        # Update GS AoI according to the modeled equations
        old_aoi_gs = self.aoi_gs.copy()

        if len(chosen_stations) == 1:
            chosen_j = chosen_stations[0]
            for j in range(self.num_stations):
                # For the chosen station, we propagate the AoI of the data at transmission time (old_aoi_leo).
                # This explicitly reflects the "data age" at the moment of transmission.
                if j == chosen_j:
                    self.aoi_gs[j] = old_aoi_leo
                else:
                    # If old_aoi_gs[j] == -1 (no data), keep it as -1
                    if old_aoi_gs[j] == -1:
                        self.aoi_gs[j] = -1
                    else:
                        self.aoi_gs[j] = old_aoi_gs[j] + 1

        elif len(chosen_stations) > 1:
            # Multiple stations selected -> already penalized above.
            # There is no definition for this case in the AoI equation, so treat it as "no transmission."
            for j in range(self.num_stations):
                if old_aoi_gs[j] != -1:
                    self.aoi_gs[j] = old_aoi_gs[j] + 1
        else:
            # No station selected
            self.aoi_leo = old_aoi_leo + 1
            for j in range(self.num_stations):
                if old_aoi_gs[j] != -1:
                    self.aoi_gs[j] = old_aoi_gs[j] + 1

        # Additional: dequeue 1 item from each GS queue per step
        for j in range(self.num_stations):
            if self.queue_gs[j] > 0:
                self.queue_gs[j] = max(0, self.queue_gs[j] - 1)

        # Energy update:
        #   E(t) = E(t-1) - sum(E_comm_j) + c_charging
        self.energy = self.energy - e_comm_total + self.c_charging

        # Clamp energy to the maximum capacity
        if self.energy > self.energy_max:
            self.energy = self.energy_max

        # If the satellite energy is depleted, apply an additional penalty
        if self.energy < 0:
            reward -= 5.0  # large penalty for energy depletion
            self.energy = 0.0
            # self.done = True  # enable if you want early termination on depletion


        # Reward composition:
        # R(t) = w^AoI * R^AoI + w^Q * R^Q + w^E * R^E

        # AoI-based reward
        w_a = 0.3  # [PR]
        aoi_reward = 0.0
        for j in range(self.num_stations):
            if self.aoi_gs[j] == -1:
                # No data => 0 reward (could be negative if desired)
                pass
            elif self.aoi_gs[j] == 0:
                # AoI=0 => freshest
                aoi_reward += 1.0
            else:
                aoi_reward += (1.0 / self.aoi_gs[j])

        # Normalize AoI reward
        aoi_reward /= self.num_stations
        reward += w_a * aoi_reward


        # Queue reward using std. deviation of usage and mean leftover space
        # Note: gs_queue_capacity is assumed to be identical for all stations.
        # If you want station-specific Q^GS_max, implement capacities as an array per station.

        alpha = 0.3  # std. deviation penalty coefficient [PR: 0.3 -> 1.0]
        w_q = 0.6    # queue reward weight [PR 0.6 -> 2.0 -> 3.0]
        N = self.num_stations

        # 1) Mean leftover capacity
        #    leftover_j = (gs_queue_capacity - queue_gs[j])
        leftover_list = [(self.gs_queue_capacity - self.queue_gs[j])
                        for j in range(N)]
        leftover_avg = np.mean(leftover_list)  # Q^GS_{l_avg}(t)

        # 2) Standard deviation of queue usage
        usage_sd = np.std(self.queue_gs)       # Q^GS_sd(t)

        # Normalized
        leftover_avg_norm = leftover_avg / self.gs_queue_capacity
        usage_sd_norm = usage_sd / self.gs_queue_capacity

        # 3) Final queue reward
        queue_reward = leftover_avg_norm - alpha * usage_sd_norm
        reward += w_q * queue_reward


        # Energy reward (normalized)
        # Previous: reward += w_e * self.energy
        # Updated: normalize energy into [0, 1] via energy_max

        # Energy : w^E * R^E(t) = w^E * E(t)
        w_e = 0.1   # Energy weight [PR : 0.1 -> 0.5]
        energy_norm = self.energy / self.energy_max
        reward += w_e * energy_norm

        # 날짜 진행
        self.minute += 1
        if self.minute >= self.episode_length:
            self.minute = self.episode_length - 1  # prevent further increase
            self.done = True

        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        """
        Observation (length = 55 or 65):
         1) day -> replaced by minute index
         ----
         2) aoi_leo
         3) aoi_gs (10)
         ----
         4) queue_leo
         5) queue_gs (10)
         ----
         6) vis_map_now (10)   // Line of Sight (if agent transmits when LoS=0 -> penalty)
         7) distance_now (10)  // Distance between satellite and each GS (used for energy calculation)
         ----
         8) leo_queue_usage    // ratio
         9) gs_queue_usage (10)// ratio
        10) energy
        11) (optional) future_los_array (10)
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

        # leo_queue_usage, gs_queue_usage Normalization
        leo_queue_usage = float(self.queue_leo) / self.leo_queue_capacity
        gs_queue_usage = self.queue_gs / self.gs_queue_capacity  # shape=(10,)

        obs_list.append(leo_queue_usage)
        obs_list.extend(gs_queue_usage.tolist())

        # Energy E(t)
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