import numpy as np
from datetime import datetime, timedelta
from skyfield.api import wgs84, utc
from skyfield.framelib import itrs
import math

# 부가적인 좌표/가시성 계산 메서드들
def get_oco2_positions_16days(start_date, satellite, ts):
    print("[DEBUG] get_oco2_positions_16days: Start calculation")
    num_days = 16
    end_date = start_date + timedelta(days=num_days)
    positions = []
    current_date = start_date

    total_minutes = int((end_date - start_date).total_seconds() // 60)  # 예: 16일이면 23040 / 디버그를위해 추가
    count = 0 # 디버그를 위해 추가

    while current_date < end_date :
        t = ts.from_datetime(current_date)
        geocentric = satellite.at(t)
        itrf_xyz = geocentric.frame_xyz(itrs)
        x_km, y_km, z_km = itrf_xyz.km
        positions.append((x_km, y_km, z_km))
        current_date += timedelta(minutes=1)
        
        count += 1
        if count % 1440 == 0:
            print(f"[DEBUG] get_oco2_positions_16days: {count}/{total_minutes} steps processed...")

    print(f"[DEBUG] get_oco2_positions_16days: Done, generated {len(positions)} positions.")
    return positions


def get_station_itrf_dict(stations):
    """
    각 지상국의 ITRF(ECEF) xyz(km) 반환
    """
    gs_dict = {}
    for station_name, (lat, lon) in stations.items():
        location = wgs84.latlon(lat, lon, elevation_m=0.0)
        itrs_distance = location.itrs_xyz
        gs_dict[station_name] = itrs_distance.km
    return gs_dict


def compute_visibility_all(satellite, stations, start_date, ts, num_steps_per_cycle):
    """
    [FIX]
    num_steps_per_cycle= 16일*24*60=23040분 동안
    각 분(minute_idx)에 대해, 10개 지상국과의 가시성(alt>0)을 체크
        alt>0 이면 0(가시), 아니면 -1(비가시)
    """
    station_names = list(stations.keys())
    N = len(station_names)
    vis_map = np.full((num_steps_per_cycle, N), -1, dtype=int)

    current_date = start_date
    for minute_idx in range(num_steps_per_cycle):
        t = ts.from_datetime(current_date)
        for j, (st_name, (lat, lon)) in enumerate(stations.items()):
            station = wgs84.latlon(lat, lon, elevation_m=0.0)
            difference = satellite - station
            topocentric = difference.at(t)
            alt, az, dist = topocentric.altaz()
            if alt.degrees > 0.0:
                vis_map[minute_idx, j] = 0  # 가시
            else:
                vis_map[minute_idx, j] = -1 # 비가시
        current_date += timedelta(minutes=1)

    return station_names, vis_map


def compute_station_itrf_list(stations, station_names):
    """
    station_names 순서대로 stations[station_name] = (lat, lon)을 참조해
    (x,y,z) 튜플을 리스트에 append한 뒤 반환.

    station_names(리스트) 순서대로 stations[st_name]에서 (lat, lon)을 꺼내고 (x, y, z)로 변환하여 gs_list를 만든 뒤 반환.
    결국, 인덱스 j와 station_names[j], 그리고 **gs_list[j]**가 일치하는 것임.
    """
    gs_list = []
    for st_name in station_names:
        lat, lon = stations[st_name] # stations 딕셔너리에서 lat/lon
        loc = wgs84.latlon(lat, lon, elevation_m=0.0)
        itrs_xyz = loc.itrs_xyz
        x_km, y_km, z_km = itrs_xyz.km
        gs_list.append((x_km, y_km, z_km))
    return gs_list


# station_names = station_names_list = gs_list = gs_positions 순서
def compute_distance_all(num_steps_per_cycle, num_stations, oco2_positions, gs_positions):
    """
    self.oco2_positions[t] = t번째 타임스텝(1분 단위)에서의 위성 (x,y,z)
    self.gs_positions[j]   = j번째 지상국 (x,y,z)
    => 거리 계산 -> shape=(23040, num_stations)
    """
    dist_now = np.zeros((num_steps_per_cycle, num_stations), dtype=float)
    for m in range(num_steps_per_cycle):
        sx, sy, sz = oco2_positions[m]  # 위성 위치(km)
        for j in range(num_stations):
            gx, gy, gz = gs_positions[j]
            dx = sx - gx
            dy = sy - gy
            dz = sz - gz
            dist_km = np.sqrt(dx*dx + dy*dy + dz*dz)
            dist_now[m, j] = dist_km
    return dist_now


#//
def compute_station_passes(num_steps_per_cycle, start_date, satellite, stations):
        """
        16일간 매 1분 alt>0 여부를 직접 돌며
        station_passes[st_name] = [(start_t, end_t, dur_min), ...] 구하기
        """
        from collections import defaultdict

        print("[DEBUG] Start compute_station_passes")
        # 1) 16일간 모든 minute에 대한 alt>0 체크 -> results_los
        results_los = []
        num_minutes = num_steps_per_cycle  # 23040
        current_date = start_date 
        end_date = start_date + timedelta(days=16)

        while current_date < end_date:
            t = satellite.ts.from_datetime(current_date) # main에서 satellite에 ts 요소를 만들어서 env에 넘겨줌. 그 satellite을 가지고 env는 이 함수를 호출하는 거고.
            for st_name, (lat, lon) in stations.items():
                station = wgs84.latlon(lat, lon, elevation_m=0.0)
                difference = satellite - station
                topocentric = difference.at(t)
                alt, az, dist = topocentric.altaz()
                visible = (alt.degrees>0.0)
                results_los.append((current_date, st_name, alt.degrees, visible))
            current_date += timedelta(minutes=1)

        # 2) station_los_records
        station_los_records = defaultdict(list)
        for (t, st_name, alt_deg, visible) in results_los:
            station_los_records[st_name].append((t, visible))

        for st_name in station_los_records:
            station_los_records[st_name].sort(key=lambda x: x[0])

        # 3) find_los_passes() 내장
        def find_los_passes(records):
            passes = []
            in_pass = False
            pass_start = None
            for i in range(len(records)):
                cur_t, is_vis = records[i]
                if is_vis and not in_pass:
                    in_pass = True
                    pass_start = cur_t
                elif (not is_vis) and in_pass:
                    pass_end = records[i-1][0]
                    dur_min = (pass_end - pass_start).total_seconds()/60
                    passes.append((pass_start, pass_end, dur_min))
                    in_pass = False
            if in_pass:
                pass_end = records[-1][0]
                dur_min = (pass_end - pass_start).total_seconds()/60
                passes.append((pass_start, pass_end, dur_min))
            return passes

        station_passes = {}
        for stn in station_los_records:
            station_passes[stn] = find_los_passes(station_los_records[stn])

        return station_passes


def build_future_los_array(start_date, num_steps_per_cycle, num_stations, station_names, station_passes):
    """
    station_passes: { st_name: [(start_t, end_t, dur_min), ...], ...}
    -> self.future_los_array[m, j] = "현재 m분 시점에서 station j의 LoS 남은 시간(분)"
    """
    total_minutes = num_steps_per_cycle  # 23040
    n_st = num_stations
    arr = np.zeros((total_minutes, n_st), dtype=int)
    
    # station_names가 self.station_names (list) 인지, dict.keys() 인지 확인
    # 여기선 self.station_names 가 list 라고 가정
    # 그리고 station_passes.keys()와 동일 순서를 가정
    st_name_to_idx = {}
    for i, stn in enumerate(station_names):
        st_name_to_idx[stn] = i
    
    for stn in station_passes:
        j = st_name_to_idx[stn]  # 몇 번째 스테이션인지
        pass_list = station_passes[stn]
        for (start_t, end_t, dur_min) in pass_list:
            start_offset = int((start_t - start_date).total_seconds()//60)
            end_offset   = int((end_t - start_date).total_seconds()//60)
            if end_offset>total_minutes:
                end_offset=total_minutes
            for m in range(start_offset, end_offset):
                arr[m, j] = end_offset - m
    return arr



def get_transmit_power_watt(dist_km, freq_mhz, required_Pr_dBm, Gt_dBi, Gr_dBi):
    """
    Friis 방정식 기반으로, 원하는 Pr_req_dBm을 만족하기 위한
    Tx power(W) 값을 계산해주는 예시 함수.
    (상황에 맞춰 링크 마진, 시스템 온도, 대역폭 고려 가능)
    """
    # FSPL(dB) = 20 log10(d_km) + 20 log10(freq_mhz) + 32.44 #!!!!!!!!!!!!! 이부분 다시 체크
    fspl_db = 20.0 * math.log10(dist_km) + 20.0 * math.log10(freq_mhz) + 32.44

    # Friis 식 재배열: Pt(dBm) = Pr_req + FSPL - Gt - Gr
    pt_need_dbm = required_Pr_dBm + fspl_db - Gt_dBi - Gr_dBi

    # dBm -> mW -> W
    pt_need_mw = 10 ** (pt_need_dbm / 10.0)
    pt_need_w = pt_need_mw / 1000.0

    # 여유있게 0 이하(너무 가까운 경우)면 최소값으로 보정
    if pt_need_w < 1e-6:
        pt_need_w = 1e-6

    # 상한 값도 추가1!!!!!! (ex: 최대 5W로 제한)
    # 멀리 있는 지상국이라도 "이론적 계산"이 10W, 20W가 나올 수 있는데, 실제로는 "5W 까지 밖에 못 낸다" 라고 시스템 제약을 모사하게 됨.
    # 위성 PA 출력 한계 등.
    max_power_w = 5.0
    if pt_need_w > max_power_w:
        pt_need_w = max_power_w

    return pt_need_w