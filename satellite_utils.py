import numpy as np
from datetime import datetime, timedelta
from skyfield.api import wgs84, utc
from skyfield.framelib import itrs
import math

# Additional methods for coordinate and visibility calculations
def get_oco2_positions_16days(start_date, satellite, ts):
    """
    Calculates the satellite positions in ITRF (ECEF) coordinates for a 16-day cycle.
    """
    print("[DEBUG] get_oco2_positions_16days: Start calculation")
    num_days = 16
    end_date = start_date + timedelta(days=num_days)
    positions = []
    current_date = start_date

    total_minutes = int((end_date - start_date).total_seconds() // 60)  # e.g., 23040 minutes for 16 days
    count = 0 

    while current_date < end_date:
        t = ts.from_datetime(current_date)
        geocentric = satellite.at(t)
        itrf_xyz = geocentric.frame_xyz(itrs)
        x_km, y_km, z_km = itrf_xyz.km
        positions.append((x_km, y_km, z_km))
        current_date += timedelta(minutes=1)
        
        count += 1
        if count % 1440 == 0:
            print(f"[DEBUG] get_oco2_positions_16days: {count}/{total_minutes} steps processed")

    print(f"[DEBUG] get_oco2_positions_16days: Done, generated {len(positions)} positions.")
    return positions


def get_station_itrf_dict(stations):
    """
    Returns a dictionary of ground station ITRF (ECEF) xyz (km) coordinates.
    """
    gs_dict = {}
    for station_name, (lat, lon) in stations.items():
        location = wgs84.latlon(lat, lon, elevation_m=0.0)
        itrs_distance = location.itrs_xyz
        gs_dict[station_name] = itrs_distance.km
    return gs_dict


def compute_visibility_all(satellite, stations, start_date, ts, num_steps_per_cycle):
    """
    Checks line-of-sight visibility (alt > 0) for all ground stations 
    during the 16-day cycle (23,040 minutes).
    Returns 0 for visible and -1 for invisible.
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
                vis_map[minute_idx, j] = 0  # Visible
            else:
                vis_map[minute_idx, j] = -1 # Invisible
        current_date += timedelta(minutes=1)

    return station_names, vis_map


def compute_station_itrf_list(stations, station_names):
    """
    Converts ground station lat/lon to ITRF (x, y, z) coordinates 
    in the specific order provided in station_names.
    Ensures index alignment between names and coordinate lists.
    """
    gs_list = []
    for st_name in station_names:
        lat, lon = stations[st_name] 
        loc = wgs84.latlon(lat, lon, elevation_m=0.0)
        itrs_xyz = loc.itrs_xyz
        x_km, y_km, z_km = itrs_xyz.km
        gs_list.append((x_km, y_km, z_km))
    return gs_list


def compute_distance_all(num_steps_per_cycle, num_stations, oco2_positions, gs_positions):
    """
    Calculates Euclidean distance between the satellite and each ground station 
    for every time step. Result shape: (23040, num_stations).
    """
    dist_now = np.zeros((num_steps_per_cycle, num_stations), dtype=float)
    for m in range(num_steps_per_cycle):
        sx, sy, sz = oco2_positions[m]  # Satellite position (km)
        for j in range(num_stations):
            gx, gy, gz = gs_positions[j]
            dx = sx - gx
            dy = sy - gy
            dz = sz - gz
            dist_km = np.sqrt(dx*dx + dy*dy + dz*dz)
            dist_now[m, j] = dist_km
    return dist_now


def compute_station_passes(num_steps_per_cycle, start_date, satellite, stations):
    """
    Iterates through the 16-day cycle to identify visibility windows for each station.
    Returns: { station_name: [(start_time, end_time, duration_min), ...], ... }
    """
    from collections import defaultdict

    print("[DEBUG] Start compute_station_passes")
    # 1) Check alt > 0 for every minute for 16 days
    results_los = []
    current_date = start_date 
    end_date = start_date + timedelta(days=16)

    while current_date < end_date:
        # Time scale (ts) assumed to be attached to the satellite object
        t = satellite.ts.from_datetime(current_date) 
        for st_name, (lat, lon) in stations.items():
            station = wgs84.latlon(lat, lon, elevation_m=0.0)
            difference = satellite - station
            topocentric = difference.at(t)
            alt, az, dist = topocentric.altaz()
            visible = (alt.degrees > 0.0)
            results_los.append((current_date, st_name, alt.degrees, visible))
        current_date += timedelta(minutes=1)

    # 2) Organize results into station-specific records
    station_los_records = defaultdict(list)
    for (t, st_name, alt_deg, visible) in results_los:
        station_los_records[st_name].append((t, visible))

    for st_name in station_los_records:
        station_los_records[st_name].sort(key=lambda x: x[0])

    # 3) Internal function to extract pass intervals from visibility flags
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
                dur_min = (pass_end - pass_start).total_seconds() / 60
                passes.append((pass_start, pass_end, dur_min))
                in_pass = False
        if in_pass:
            pass_end = records[-1][0]
            dur_min = (pass_end - pass_start).total_seconds() / 60
            passes.append((pass_start, pass_end, dur_min))
        return passes

    station_passes = {}
    for stn in station_los_records:
        station_passes[stn] = find_los_passes(station_los_records[stn])

    return station_passes


def build_future_los_array(start_date, num_steps_per_cycle, num_stations, station_names, station_passes):
    """
    Maps the station passes to an array.
    arr[m, j] contains the "remaining LoS duration in minutes" for station j at time step m.
    """
    total_minutes = num_steps_per_cycle 
    n_st = num_stations
    arr = np.zeros((total_minutes, n_st), dtype=int)
    
    st_name_to_idx = {stn: i for i, stn in enumerate(station_names)}
    
    for stn in station_passes:
        j = st_name_to_idx[stn] # station 'n'
        pass_list = station_passes[stn]
        for (start_t, end_t, dur_min) in pass_list:
            start_offset = int((start_t - start_date).total_seconds() // 60)
            end_offset   = int((end_t - start_date).total_seconds() // 60)
            if end_offset > total_minutes:
                end_offset = total_minutes
            for m in range(start_offset, end_offset):
                arr[m, j] = end_offset - m
    return arr


def get_transmit_power_watt(dist_km, freq_mhz, required_Pr_dBm, Gt_dBi, Gr_dBi):
    """
    Calculates the required Transmission Power (W) based on the Friis Transmission Equation
    to meet the target Received Power (Pr_req_dBm).
    """
    # FSPL(dB) calculation: 20 log10(d_km) + 20 log10(freq_mhz) + 32.44
    fspl_db = 20.0 * math.log10(dist_km) + 20.0 * math.log10(freq_mhz) + 32.44

    # Rearranging Friis Equation: Pt(dBm) = Pr_req + FSPL - Gt - Gr
    pt_need_dbm = required_Pr_dBm + fspl_db - Gt_dBi - Gr_dBi

    # Convert dBm -> mW -> W
    pt_need_mw = 10 ** (pt_need_dbm / 10.0)
    pt_need_w = pt_need_mw / 1000.0

    # Minimum threshold correction (e.g., if very close)
    if pt_need_w < 1e-6:
        pt_need_w = 1e-6

    # Maximum power limitation based on system hardware constraints (e.g., 5W limit)
    # This reflects real-world hardware saturation or Power Amplifier (PA) limits.
    max_power_w = 5.0
    if pt_need_w > max_power_w:
        pt_need_w = max_power_w

    return pt_need_w