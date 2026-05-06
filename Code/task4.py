import numpy as np
from ekf_tracker import EKFTracker
from load_simulation_data import load_simulation_output
from coordinate_frame_manager import CoordinateFrameManager
import matplotlib.pyplot as plt
from scipy.stats import chi2

def position_rmse_over_time(x_est, xs_true):
    """Per-step position error (not cumulative RMSE)."""
    return np.sqrt((x_est[:, 0] - xs_true[1:, 0])**2 +
                   (x_est[:, 1] - xs_true[1:, 1])**2)

def build_R_camera(sigma_phi_c: float) -> np.ndarray:
    """Measurement noise covariance for camera (1×1 matrix)."""
    return np.array([[sigma_phi_c**2]])

def build_R_radar(sigma_r: float, sigma_phi: float) -> np.ndarray:
    """Measurement noise covariance matrix for radar."""
    # Diagonal: sensors are independent
    return np.diag([sigma_r**2, sigma_phi**2])

def build_motion_model(dt: float, sigma_a: float):
    """
    Build the constant-velocity state transition matrix F and
    process noise covariance Q.

    Parameters
    ----------
    dt      : float  — time step [s]
    sigma_a : float  — std of unmodelled acceleration [m/s^2]

    Returns
    -------
    F : (4, 4) ndarray
    Q : (4, 4) ndarray
    """
    # State transition matrix for constant-velocity model.
    # x_{k+1} = F x_k  =>  p_{k+1} = p_k + dt*v_k,  v_{k+1} = v_k
    F = np.array([[1, 0, dt, 0],
                  [0, 1,  0, dt],
                  [0, 0,  1,  0],
                  [0, 0,  0,  1]], dtype=float)

    # Process noise covariance derived from a piecewise-constant white-noise
    # acceleration model (Singer/DWNA model):
    #   Q = sigma_a^2 * G G^T,  with G = [dt^2/2, dt^2/2, dt, dt]^T (block)
    q  = sigma_a ** 2
    dt2 = dt ** 2
    dt3 = dt ** 3
    dt4 = dt ** 4
    Q = q * np.array([[dt4 / 4,       0, dt3 / 2,       0],
                       [      0, dt4 / 4,       0, dt3 / 2],
                       [dt3 / 2,       0,     dt2,       0],
                       [      0, dt3 / 2,       0,     dt2]])
    return F, Q


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

# Match each radar scan to the nearest camera measurement
zs_cam = np.array([min(camera_meas, key=lambda m: abs(m.time - t)).bearing_rad for t in radar_times])

N = len(zs)

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

x_seq   = np.zeros((N, 4))
P_seq   = np.zeros((N, 4, 4))

x = x_init.copy()
P = P_init.copy()

for k in range(N):
    # Step 1: Predict
    x, P = ekf.predict(x, P, F, Q)

    # Step 2: Radar update
    x, P, _, _ = ekf.update_radar(x, P, zs_radar[k], R)

    # Step 3: Camera update (bearing only)
    x, P, _, _ = ekf.update_camera(x, P, zs_cam[k], R_cam)


    x_seq[k] = x.flatten()
    P_seq[k] = P

print("Sequential fusion EKF complete.")

# ── Centralised Fusion ───────────────────────────────────
x = x_init.copy()
P = P_init.copy()

x_cen   = np.zeros((N, 4))
P_cen   = np.zeros((N, 4, 4))
innov_cen = np.zeros((N, 3))
S_cen     = np.zeros((N, 3, 3))

for k in range(N):
    # Step 1: Predict
    x, P = ekf.predict(x, P, F, Q)

    # Step 2: Joint update
    x, P, innov, S = ekf.update_joint(x, P, zs_radar[k], zs_cam[k], R, R_cam)

    x_cen[k]     = x.flatten()
    P_cen[k]     = P
    innov_cen[k] = innov.flatten()
    S_cen[k]     = S

print("Centralised fusion EKF complete.")#


# ============================================================================
# PLOT RESULTS
# ============================================================================

err_radar = position_rmse_over_time(x_est, xs_true)
err_seq   = position_rmse_over_time(x_seq, xs_true)
err_cen   = position_rmse_over_time(x_cen, xs_true)

t = np.arange(1, N + 1) * dt

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t, err_radar, 'steelblue',  lw=1.5, label='Radar only')
ax.plot(t, err_seq,   'darkorange', lw=1.5, label='Sequential fusion')
ax.plot(t, err_cen,   'green',      lw=1.5, ls='--', label='Centralised fusion')
ax.set_xlabel('Time [s]');  ax.set_ylabel('Position error [m]')
ax.set_title('Tracking Performance: Radar-Only vs. Sensor Fusion')
ax.legend();  ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("figures/task4/tracking_performance.png", dpi=300)
plt.close()

# Summary table
print("\nOverall position RMSE:")
print(f"  Radar only:          {np.sqrt(np.mean(err_radar**2)):.2f} m")
print(f"  Sequential fusion:   {np.sqrt(np.mean(err_seq**2)):.2f} m")
print(f"  Centralised fusion:  {np.sqrt(np.mean(err_cen**2)):.2f} m")

# ── Trajectory plot: all three trackers ──────────────────────────────────
meas_x = zs_radar[:, 0] * np.cos(zs_radar[:, 1])
meas_y = zs_radar[:, 0] * np.sin(zs_radar[:, 1])

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(xs_true[1:, 1], xs_true[1:, 0], 'k-',    lw=2,       label='True trajectory')
ax.scatter(meas_y, meas_x, s=10, c='steelblue',   alpha=0.4,  label='Radar meas.')
ax.plot(x_est[:, 1], x_est[:, 0], 'b-',         label='Radar only')
ax.plot(x_seq[:, 1],  x_seq[:, 0],  'r-',   lw=1.5,          label='Sequential fusion')
ax.plot(x_cen[:, 1],  x_cen[:, 0],  'g--',  lw=1.5,          label='Centralised fusion')
ax.plot(0, 0, 'k^', ms=12, label='Ownship')
ax.set_xlabel('East [m]');  ax.set_ylabel('North [m]')
ax.set_title('Trajectory Comparison')
ax.legend();  ax.set_aspect('equal');  ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("figures/task4/trajectory_comparison.png", dpi=300)
plt.close()

# ── NIS plot ────────────────────────────────────────────────────────────────
nz    = 3
lower = chi2.ppf(0.025, df=nz)
upper = chi2.ppf(0.975, df=nz)

# Compute NIS for each step using innov_cen and S_cen
nis_cen = np.zeros(N)
for k in range(N):
    yk = innov_cen[k].reshape(-1, 1)
    Sk = S_cen[k]
    try:
        # Check if Sk is singular
        if np.abs(np.linalg.det(Sk)) < 1e-12:
            nis_cen[k] = np.nan
        else:
            nis_cen[k] = (yk.T @ np.linalg.inv(Sk) @ yk).item()
    except:
        nis_cen[k] = np.nan

frac = np.nanmean((nis_cen >= lower) & (nis_cen <= upper))

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t, nis_cen, '.', ms=4, alpha=0.7)
ax.axhline(upper, color='r', ls='--', label=f'95% upper ({upper:.2f})')
ax.axhline(lower, color='g', ls='--', label=f'95% lower ({lower:.2f})')
ax.set_title(f'NIS — Centralised Fusion  ({frac*100:.1f}% within bounds)')
ax.set_xlabel('Time [s]');  ax.set_ylabel('NIS')
ax.legend();  ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("figures/task4/nis_centralised.png", dpi=300)
plt.close()
