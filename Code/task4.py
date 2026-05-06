import numpy as np
from ekf_tracker import EKFTracker
from load_simulation_data import load_simulation_output
from coordinate_frame_manager import CoordinateFrameManager

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

print("Centralised fusion EKF complete.")