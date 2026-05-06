import numpy as np
from ekf_tracker import EKFTracker, build_motion_model, build_R_radar, build_R_camera, position_rmse_over_time
from load_simulation_data import load_simulation_output
from coordinate_frame_manager import CoordinateFrameManager
import matplotlib.pyplot as plt
from scipy.stats import chi2

# ── Initialization ───────────────────────────────────────────────────────

data = load_simulation_output("B")
cfm = CoordinateFrameManager()

# --- Data Extraction & Synchronization ---
# Filter measurements for target 0 (Scenario B is single target)
radar_meas = [m for m in data.measurements if m.sensor_id == "radar" and m.target_id == 0]
camera_meas = [m for m in data.measurements if m.sensor_id == "camera" and m.target_id == 0]

# Extract parameters from configs
configs     = data.sensor_configs
sigma_r     = configs["radar"]["sigma_r_m"]
sigma_phi   = np.deg2rad(configs["radar"]["sigma_phi_deg"])
sigma_phi_c = np.deg2rad(configs["camera"]["sigma_phi_deg"])
sigma_a     = 0.1  # Process noise std (tuned parameter)
dt          = 1.0 / configs["radar"]["rate_hz"]

# Synchronize: Use radar times as the primary timeline
radar_times = [m.time for m in radar_meas]
zs          = np.array([[m.range_m, m.bearing_rad] for m in radar_meas])
zs_radar    = zs

# Match each radar scan to a recent camera measurement.
# If there is no camera measurement close enough, fusion stops at the last
# radar scan for which both sensors provide data.
CAMERA_TIME_WINDOW = 1.0  # seconds

def get_camera_measurement(radar_time, camera_meas, time_window=CAMERA_TIME_WINDOW):
    candidates = [m for m in camera_meas if abs(m.time - radar_time) <= time_window]
    if not candidates:
        return None
    return min(candidates, key=lambda m: abs(m.time - radar_time)).bearing_rad

zs_cam = [get_camera_measurement(t, camera_meas) for t in radar_times]

N = len(zs)
valid_flags = [z is not None for z in zs_cam]
if not any(valid_flags):
    raise RuntimeError("No radar-camera pairs found within the camera time window.")
first_common = next(k for k, ok in enumerate(valid_flags) if ok)
last_common = first_common
for k in range(first_common + 1, len(valid_flags)):
    if not valid_flags[k]:
        break
    last_common = k
N_fusion = last_common + 1
print(f"Fusion will start at radar scan {first_common} ({radar_times[first_common]:.1f} s) and stop at {radar_times[last_common]:.1f} s.")

# Extract ground truth for Target 0 and match to radar times
# (include t=0 state for index 0 of xs_true)
gt_all   = data.ground_truth[0]
gt_times = data.ground_truth_times
xs_true  = np.zeros((N + 1, 4))
xs_true[0] = gt_all[0]
for k, t_meas in enumerate(radar_times):
    idx = np.argmin(np.abs(gt_times - t_meas))
    xs_true[k+1] = gt_all[idx]

# --- Initialization ---
# Convert first radar measurement to Cartesian for initialisation
r0, phi0 = zs[0]
x_init = np.array([[r0 * np.cos(phi0)], 
                   [r0 * np.sin(phi0)], 
                   [0.0], 
                   [0.0]])

# Initial covariance: high position uncertainty, high velocity uncertainty
P_init = np.diag([sigma_r**2, sigma_r**2, 50.0**2, 50.0**2])

R = build_R_radar(sigma_r, sigma_phi)
F, Q = build_motion_model(dt, sigma_a)

# ── Storage ───────────────────────────────────────────────────────────────────
x_est      = np.zeros((N, 4))     # filtered state estimates
P_est      = np.zeros((N, 4, 4))  # filtered covariances
innov_hist = np.zeros((N, 2))     # innovation history
S_hist     = np.zeros((N, 2, 2))

# ── Initialize EKF tracker ────────────────────────────────────────────────────
ekf = EKFTracker(x_init, P_init, cfm, sigma_a)

# ── Radar only EKF ─────────────────────────────────────────────────────────────
x = x_init.copy()
P = P_init.copy()

for k in range(N):
    # Step 1: Prediction — propagate state and covariance forward by dt
    x, P = ekf.predict(x, P, F, Q)

    # Step 2: Update with radar measurement zs[k]
    x, P, innov, S = ekf.update_radar(x, P, zs[k], R)

    # Store results
    x_est[k]      = x.flatten()
    P_est[k]      = P
    innov_hist[k] = innov.flatten()
    S_hist[k]     = S

print("Radar only EKF loop complete.")

# ── Sequential Fusion ────────────────────────────────────────────────────────────────
r0, phi0 = zs_radar[0]
x_init   = np.array([[r0 * np.cos(phi0)], [r0 * np.sin(phi0)], [0.0], [0.0]])
P_init   = np.diag([sigma_r**2, sigma_r**2, 50.0**2, 50.0**2])
R_cam    = build_R_camera(sigma_phi_c)

x_seq   = np.zeros((N_fusion, 4))
P_seq   = np.zeros((N_fusion, 4, 4))
innov_seq = np.zeros((N_fusion, 2))
S_seq     = np.zeros((N_fusion, 2, 2))

x = x_init.copy()
P = P_init.copy()

for k in range(N_fusion):
    # Step 1: Predict
    x, P = ekf.predict(x, P, F, Q)

    # Step 2: Radar update
    x, P, innov_radar, S_radar = ekf.update_radar(x, P, zs_radar[k], R)
    innov_seq[k] = innov_radar.flatten()
    S_seq[k]     = S_radar

    # Step 3: Camera update only if a recent camera measurement exists
    if zs_cam[k] is not None:
        x, P, _, _ = ekf.update_camera(x, P, zs_cam[k], R_cam)

    x_seq[k] = x.flatten()
    P_seq[k] = P

print("Sequential fusion EKF complete.")

# ── Centralised Fusion ───────────────────────────────────
x = x_init.copy()
P = P_init.copy()

x_cen   = np.zeros((N_fusion, 4))
P_cen   = np.zeros((N_fusion, 4, 4))
innov_cen = np.zeros((N_fusion, 3))
S_cen     = np.zeros((N_fusion, 3, 3))

for k in range(N_fusion):
    # Step 1: Predict
    x, P = ekf.predict(x, P, F, Q)

    # Step 2: Joint update only when camera data is available
    if zs_cam[k] is not None:
        x, P, innov, S = ekf.update_joint(x, P, zs_radar[k], zs_cam[k], R, R_cam)
        innov_cen[k] = innov.flatten()
        S_cen[k]     = S
    else:
        x, P, innov, S = ekf.update_radar(x, P, zs_radar[k], R)
        innov_cen[k][:2] = innov.flatten()
        innov_cen[k][2] = np.nan
        S_cen[k][:2, :2] = S
        S_cen[k][2, :] = np.nan
        S_cen[k][:, 2] = np.nan

    x_cen[k]     = x.flatten()
    P_cen[k]     = P

print("Centralised fusion EKF complete.")#


# ============================================================================
# PLOT RESULTS
# ============================================================================

err_radar = position_rmse_over_time(x_est, xs_true)
err_radar_matched = position_rmse_over_time(x_est[:N_fusion], xs_true[:N_fusion+1])
err_seq   = position_rmse_over_time(x_seq, xs_true[:N_fusion+1])
err_cen   = position_rmse_over_time(x_cen, xs_true[:N_fusion+1])

t_radar = np.arange(1, N + 1) * dt
t_fusion = np.arange(1, N_fusion + 1) * dt

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(t_radar, err_radar, color='#d62728', lw=2.0, alpha=0.85, label='Radar only')
ax.plot(t_fusion, err_seq, color='#9467bd', lw=2.0, alpha=0.8, label='Sequential fusion')
ax.plot(t_fusion, err_cen, color='#1f77b4', lw=2.0, alpha=0.8, label='Centralised fusion')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Position error [m]')
ax.set_title('Tracking Performance: Radar-Only vs. Radar-Camera Fusion')
ax.set_xlim(0, max(t_radar) + 1)
ax.set_ylim(0, max(np.nanmax(err_radar), np.nanmax(err_seq), np.nanmax(err_cen)) * 1.1)
ax.legend(frameon=True, framealpha=0.85, loc='upper left')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("figures/task4/tracking_performance.png", dpi=300, bbox_inches='tight')
plt.close()

# Summary table
print("\nOverall position RMSE:")
print(f"  Radar only:          {np.sqrt(np.mean(err_radar**2)):.2f} m")
print(f"  Radar only (matched): {np.sqrt(np.mean(err_radar_matched**2)):.2f} m")
print(f"  Sequential fusion:   {np.sqrt(np.mean(err_seq**2)):.2f} m")
print(f"  Centralised fusion:  {np.sqrt(np.mean(err_cen**2)):.2f} m")

# ── Trajectory plot: all three trackers ──────────────────────────────────
meas_x = zs_radar[:, 0] * np.cos(zs_radar[:, 1])
meas_y = zs_radar[:, 0] * np.sin(zs_radar[:, 1])

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(xs_true[1:, 1], xs_true[1:, 0], color='#2ca02c', lw=2.2, alpha=0.9, label='True trajectory')
ax.scatter(meas_y, meas_x, s=12, color='gray', alpha=0.2, label='Radar measurements')
ax.plot(x_est[:, 1], x_est[:, 0], color='#d62728', lw=2, alpha=0.75, label='Radar only')
ax.plot(x_seq[:, 1], x_seq[:, 0], color='#9467bd', lw=2, alpha=0.75, label='Sequential fusion')
ax.plot(x_cen[:, 1], x_cen[:, 0], color='#1f77b4', lw=2, alpha=0.75, label='Centralised fusion')
ax.set_xlabel('East [m]')
ax.set_ylabel('North [m]')
ax.set_title('Trajectory Comparison')
ax.legend(frameon=True, framealpha=0.85, loc='best')
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("figures/task4/trajectory_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

# ── NIS plot ────────────────────────────────────────────────────────────────
nz_radar = 2
lower_radar = chi2.ppf(0.025, df=nz_radar)
upper_radar = chi2.ppf(0.975, df=nz_radar)

nis_radar = np.zeros(N)
for k in range(N):
    yk = innov_hist[k].reshape(-1, 1)
    Sk = S_hist[k]
    try:
        if np.abs(np.linalg.det(Sk)) < 1e-12:
            nis_radar[k] = np.nan
        else:
            nis_radar[k] = (yk.T @ np.linalg.inv(Sk) @ yk).item()
    except:
        nis_radar[k] = np.nan

nis_seq = np.zeros(N_fusion)
for k in range(N_fusion):
    yk = innov_seq[k].reshape(-1, 1)
    Sk = S_seq[k]
    try:
        if np.abs(np.linalg.det(Sk)) < 1e-12:
            nis_seq[k] = np.nan
        else:
            nis_seq[k] = (yk.T @ np.linalg.inv(Sk) @ yk).item()
    except:
        nis_seq[k] = np.nan

nis_cen = np.zeros(N_fusion)
for k in range(N_fusion):
    yk = innov_cen[k].reshape(-1, 1)
    Sk = S_cen[k]
    try:
        if np.abs(np.linalg.det(Sk)) < 1e-12:
            nis_cen[k] = np.nan
        else:
            nis_cen[k] = (yk.T @ np.linalg.inv(Sk) @ yk).item()
    except:
        nis_cen[k] = np.nan

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_radar, nis_radar, marker='o', markersize=4, linestyle='-', color='#d62728', alpha=0.75, label='Radar-only NIS')
ax.plot(t_fusion, nis_seq, marker='s', markersize=4, linestyle='-', color='#9467bd', alpha=0.75, label='Sequential fusion NIS')
ax.plot(t_fusion, nis_cen, marker='^', markersize=4, linestyle='-', color='#1f77b4', alpha=0.75, label='Centralised fusion NIS')
ax.axhline(upper_radar, color='#2ca02c', ls='--', lw=1.2, label=f'Radar 95% upper ({upper_radar:.2f})')
ax.axhline(lower_radar, color='#17becf', ls='--', lw=1.2, label=f'Radar 95% lower ({lower_radar:.2f})')
ax.set_title('NIS Comparison')
ax.set_xlabel('Time [s]')
ax.set_ylabel('NIS')
ax.set_xlim(0, max(t_radar) + 1)
ax.set_ylim(0, max(upper_radar, np.nanmax(nis_radar), np.nanmax(nis_seq), np.nanmax(nis_cen)) * 1.15)
ax.legend(frameon=True, framealpha=0.9, loc='upper right')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("figures/task4/nis_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
