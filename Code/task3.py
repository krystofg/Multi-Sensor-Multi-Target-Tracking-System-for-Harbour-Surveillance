import numpy as np
from itertools import groupby
from pathlib import Path

from coordinate_frame_manager import CoordinateFrameManager
from ekf_tracker import EKFTracker
from load_simulation_data import load_simulation_output
from plot_results import plot_tracking_results

# 1. Initialisation
cfm = CoordinateFrameManager()
data = load_simulation_output("A")

# Seed from first true radar detection
first_true = next(m for m in data.measurements if m.sensor_id == "radar" and not m.is_false_alarm)

x0 = np.array([[first_true.range_m * np.cos(first_true.bearing_rad)], 
               [first_true.range_m * np.sin(first_true.bearing_rad)], 
               [-2.0], [-1.0]])
P0 = np.diag([225.0, 225.0, 25.0, 25.0]) # 15^2, 5^2

tracker = EKFTracker(x0=x0, P0=P0, cfm=cfm)
last_t = first_true.time

# Logging
scan_window = []  
est_history = []
nis_history = []
confirmation_time = None

# 2. Processing
radar_meas = [m for m in data.measurements if m.sensor_id == "radar" and m.time >= first_true.time]
gnss_meas = [m for m in data.measurements if m.sensor_id == "gnss"]

# Update vessel position (CFM)
for m in gnss_meas:
    cfm.update_vessel_pos(m)

# Process radar scans
for t, group in groupby(radar_meas, key=lambda m: round(m.time, 1)):

    dt = t - last_t
    if dt > 0:
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        q = tracker.sigma_a**2
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        Q = q * np.array([
            [dt4/4,     0, dt3/2,     0],
            [    0, dt4/4,     0, dt3/2],
            [dt3/2,     0,   dt2,     0],
            [    0, dt3/2,     0,   dt2]
        ])
        tracker.x, tracker.P = tracker.predict(tracker.x, tracker.P, F, Q)
        last_t = t

    # Gating
    best_nis, best_m = np.inf, None
    gate = 5.991       # χ²(2, 95%) = 5.991
    
    for m in group:
        hx, H = cfm.get_h_and_H(tracker.x, m.sensor_id)
        R = cfm.R_specs["radar"]
        z = np.array([[m.range_m], [m.bearing_rad]])
        y = z - hx
        y[1, 0] = (y[1, 0] + np.pi) % (2 * np.pi) - np.pi
        S = H @ tracker.P @ H.T + R
        nis = (y.T @ np.linalg.inv(S) @ y).item()
        
        if nis < best_nis:
            best_nis, best_m = nis, m

    if best_m is not None:
        nis_history.append({'t': t, 'nis': best_nis})

    # Update
    hit = False
    if best_m and best_nis <= gate:
        z = np.array([[best_m.range_m], [best_m.bearing_rad]])
        R = cfm.R_specs["radar"]
        tracker.x, tracker.P, _, _ = tracker.update(tracker.x, tracker.P, z, R)
        hit = True
        
    est_history.append({'t': t, 'N': tracker.x[0,0], 'E': tracker.x[1,0]})

    # Confirmation window (3-of-5)
    scan_window.append(hit)
    if len(scan_window) > 5: 
        scan_window.pop(0)

    if confirmation_time is None and sum(scan_window) >= 3:
        confirmation_time = t

# 3. Metrics & Reporting
gt, gt_t = data.ground_truth[0], data.ground_truth_times
ss_errors = [np.sqrt((e['N'] - gt[np.argmin(np.abs(gt_t - e['t'])), 0])**2 + 
                     (e['E'] - gt[np.argmin(np.abs(gt_t - e['t'])), 1])**2)
             for e in est_history if e['t'] > 12.0] 

rmse = float(np.sqrt(np.mean(np.square(ss_errors)))) if ss_errors else 0.0
pct_nis = float((np.array([n['nis'] for n in nis_history]) < 5.99).mean() * 100) if nis_history else 0.0

# Dashboard
plot_tracking_results(
    data, est_history, nis_history, pct_nis,
    title="Scenario A — Single target, radar only",
    save_path= "figures/task3/scenario_A.png"
)

print("Last est time:", est_history[-1]['t'])
print("Total duration:", est_history[-1]['t'] - est_history[0]['t'])

# Report
confirm_ok = (confirmation_time is not None and confirmation_time <= first_true.time + 5*(1/0.3))
print(f"\n{'='*44}\nSCENARIO A  QUALIFICATION REPORT\n{'='*44}")
print(f"  1. CONFIRMATION : {'PASSED' if confirm_ok else 'FAILED'} ({confirmation_time:.2f}s)")
print(f"  2. ACCURACY     : {'PASSED' if rmse < 12 else 'FAILED'} (RMSE: {rmse:.2f}m)")
print(f"  3. CONSISTENCY  : {'PASSED' if pct_nis >= 90 else 'FAILED'} ({pct_nis:.1f}%)")
print(f"{'='*44}")
