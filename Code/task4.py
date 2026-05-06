"""
Task 4: Radar + Camera Fusion (Scenario B)
Compares sequential vs. joint (centralised) EKF update architectures.
"""
import numpy as np
from itertools import groupby
from pathlib import Path
from collections import defaultdict

from coordinate_frame_manager import CoordinateFrameManager
from ekf_tracker import MultiSensorEKF
from load_simulation_data import load_simulation_output
from plot_results import plot_tracking_results

# ── Setup ─────────────────────────────────────────────────────────────────────
cfm  = CoordinateFrameManager()
data = load_simulation_output("B")

first_true = next(m for m in data.measurements if m.sensor_id == "radar" and not m.is_false_alarm)
iN = first_true.range_m * np.cos(first_true.bearing_rad)
iE = first_true.range_m * np.sin(first_true.bearing_rad)
x0 = np.array([[iN], [iE], [-2.0], [-1.0]])
P0 = np.diag([225.0, 225.0, 25.0, 25.0])

# Two independent tracker instances — one per architecture
seq_tracker   = MultiSensorEKF(x0=x0, P0=P0, cfm=cfm)
joint_tracker = MultiSensorEKF(x0=x0, P0=P0, cfm=cfm)

# Logging
GATE = 13.82  # χ²(2) at 99.9%

def make_state(): return dict(window=[], est=[], nis=[], confirmation=None, last_t=first_true.time)

seq_state   = make_state()
joint_state = make_state()

# ── Separate measurements by sensor ─────────────────────────────────────────
for m in data.measurements:
    if m.sensor_id == "gnss":
        cfm.update_vessel_pos(m)

radar_meas  = [m for m in data.measurements if m.sensor_id == "radar"  and m.time >= first_true.time]
camera_meas = [m for m in data.measurements if m.sensor_id == "camera" and m.time >= first_true.time]

# Build scan-time buckets: {rounded_time: {'radar': [...], 'camera': [...]}}
scan_buckets = defaultdict(lambda: {"radar": [], "camera": []})
for m in radar_meas:
    scan_buckets[round(m.time, 4)]["radar"].append(m)
for m in camera_meas:
    scan_buckets[round(m.time, 4)]["camera"].append(m)

all_scan_times = sorted(scan_buckets)

# ── Processing loop ───────────────────────────────────────────────────────────
for t in all_scan_times:
    r_meas = scan_buckets[t]["radar"]
    c_meas = scan_buckets[t]["camera"]

    # ── Sequential tracker ────────────────────────────────────────────────────
    dt_seq = t - seq_state["last_t"]
    seq_tracker.predict(dt_seq)
    seq_state["last_t"] = t

    r_acc, c_acc, r_nis, c_nis = seq_tracker.update_sequential(r_meas, c_meas, GATE)
    any_hit = r_acc or c_acc

    if r_acc:
        seq_state["est"].append({"t": t, "N": seq_tracker.x[0,0], "E": seq_tracker.x[1,0]})
        seq_state["nis"].append({"t": t, "nis": r_nis, "sensor": "radar"})
    if c_acc:
        seq_state["est"].append({"t": t, "N": seq_tracker.x[0,0], "E": seq_tracker.x[1,0]})
        seq_state["nis"].append({"t": t, "nis": c_nis, "sensor": "camera"})

    seq_state["window"].append(r_acc)  # confirmation driven by radar (primary miss clock)
    if len(seq_state["window"]) > 5: seq_state["window"].pop(0)
    if seq_state["confirmation"] is None and sum(seq_state["window"]) >= 3:
        seq_state["confirmation"] = t

    # ── Joint tracker ─────────────────────────────────────────────────────────
    dt_jnt = t - joint_state["last_t"]
    joint_tracker.predict(dt_jnt)
    joint_state["last_t"] = t

    # Joint update: only possible when BOTH sensors have gated detections at same t
    if r_meas and c_meas and joint_tracker.is_in_camera_fov():
        best_r, _ = joint_tracker._gate_best(r_meas, "radar",  GATE)
        best_c, _ = joint_tracker._gate_best(c_meas, "camera", GATE)
        if best_r is not None and best_c is not None:
            j_acc, j_nis = joint_tracker.update_joint(best_r, best_c, GATE)
            if j_acc:
                joint_state["est"].append({"t": t, "N": joint_tracker.x[0,0], "E": joint_tracker.x[1,0]})
                joint_state["nis"].append({"t": t, "nis": j_nis, "sensor": "joint"})
                joint_state["window"].append(True)
                if len(joint_state["window"]) > 5: joint_state["window"].pop(0)
                if joint_state["confirmation"] is None and sum(joint_state["window"]) >= 3:
                    joint_state["confirmation"] = t
                continue  # skip single-sensor fallback for this time step

    # Joint fallback: if only one sensor fires (or joint gating failed), use sequential logic
    r_acc_j, c_acc_j, r_nis_j, c_nis_j = joint_tracker.update_sequential(r_meas, c_meas, GATE)
    if r_acc_j:
        joint_state["est"].append({"t": t, "N": joint_tracker.x[0,0], "E": joint_tracker.x[1,0]})
        joint_state["nis"].append({"t": t, "nis": r_nis_j, "sensor": "radar"})
    if c_acc_j:
        joint_state["est"].append({"t": t, "N": joint_tracker.x[0,0], "E": joint_tracker.x[1,0]})
        joint_state["nis"].append({"t": t, "nis": c_nis_j, "sensor": "camera"})

    joint_state["window"].append(r_acc_j)
    if len(joint_state["window"]) > 5: joint_state["window"].pop(0)
    if joint_state["confirmation"] is None and sum(joint_state["window"]) >= 3:
        joint_state["confirmation"] = t

# ── Metrics ───────────────────────────────────────────────────────────────────
gt, gt_t = data.ground_truth[0], data.ground_truth_times

def compute_metrics(est_history, nis_history, t_transient=20.0):
    ss_errors = []
    for e in est_history:
        if e["t"] <= t_transient: continue
        idx = np.argmin(np.abs(gt_t - e["t"]))
        ss_errors.append(np.sqrt((e["N"] - gt[idx, 0])**2 + (e["E"] - gt[idx, 1])**2))
    rmse = float(np.sqrt(np.mean(np.square(ss_errors)))) if ss_errors else np.nan
    nis_vals = np.array([n["nis"] for n in nis_history])
    pct_nis  = float((nis_vals < 5.99).mean() * 100) if len(nis_vals) else 0.0
    return rmse, pct_nis, ss_errors

seq_rmse,   seq_pct,   _  = compute_metrics(seq_state["est"],   seq_state["nis"])
joint_rmse, joint_pct, _  = compute_metrics(joint_state["est"], joint_state["nis"])

# ── Plots ─────────────────────────────────────────────────────────────────────
out_dir = Path("figures/task4")
plot_tracking_results(data, seq_state["est"],   seq_state["nis"],
    title="Scenario B — Sequential fusion (radar + camera)",
    save_path=out_dir / "scenario_B_sequential.png")
plot_tracking_results(data, joint_state["est"], joint_state["nis"],
    title="Scenario B — Joint (centralised) fusion (radar + camera)",
    save_path=out_dir / "scenario_B_joint.png")

# ── Qualification Report ──────────────────────────────────────────────────────
radar_dt   = 1 / 0.3
scan_limit = 5 * radar_dt  # 5-scan confirmation deadline

def confirm_ok(state):
    ct = state["confirmation"]
    return ct is not None and ct <= first_true.time + scan_limit

w = 46
print(f"\n{'='*w}")
print(f"  SCENARIO B  QUALIFICATION REPORT")
print(f"{'='*w}")
print(f"  {'Metric':<28}  {'Sequential':>8}  {'Joint':>8}")
print(f"  {'-'*28}  {'-'*8}  {'-'*8}")
print(f"  {'Confirmation (3-of-5)':<28}  "
      f"{'PASS' if confirm_ok(seq_state)   else 'FAIL':>8}  "
      f"{'PASS' if confirm_ok(joint_state) else 'FAIL':>8}")
print(f"  {'RMSE [m]  (limit 12 m)':<28}  "
      f"{seq_rmse:>8.2f}  {joint_rmse:>8.2f}")
print(f"  {'NIS in 95% bounds [%]':<28}  "
      f"{seq_pct:>8.1f}  {joint_pct:>8.1f}")
print(f"{'='*w}")
print()
print("Architecture comparison:")
winner_rmse = "Sequential" if seq_rmse  <= joint_rmse  else "Joint"
winner_nis  = "Sequential" if seq_pct   >= joint_pct   else "Joint"
print(f"  Lower RMSE        -> {winner_rmse}")
print(f"  Better NIS        -> {winner_nis}")
print(f"{'='*w}")
