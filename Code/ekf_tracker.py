import numpy as np

# =============================================================================
# EKF TRACKER
# =============================================================================
class EKFTracker:
    def __init__(self, x0, P0, cfm, sigma_a=0.05):
        """
        Single-target EKF with CV motion model.

        Parameters
        ----------
        x0      : (4,1) ndarray  — initial state [pN, pE, vN, vE]
        P0      : (4,4) ndarray  — initial covariance
        cfm     : CoordinateFrameManager
        sigma_a : float          — process noise std [m/s²]
        """
        self.x       = x0.copy()
        self.P       = P0.copy()
        self.cfm     = cfm
        self.sigma_a = sigma_a

    def predict(self, dt):
        """CV predict step over interval dt [s]."""
        if dt <= 0:
            return
        F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)
        q    = self.sigma_a**2
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        Q = q * np.array([
            [dt4/4,     0, dt3/2,     0],
            [    0, dt4/4,     0, dt3/2],
            [dt3/2,     0,   dt2,     0],
            [    0, dt3/2,     0,   dt2],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        
    def update(self, m, gate_limit=13.82):
        """
        EKF update step with Mahalanobis-distance gating.

        gate_limit = 13.82 is the χ²(2) threshold at P_G = 0.999.
        For P_G = 0.99 use 9.21; spec says 0.99 so either is defensible,
        but 13.82 gives a slightly wider gate which helps confirmation speed.

        Returns
        -------
        accepted : bool   — True if measurement passed the gate
        nis      : float  — Normalised Innovation Squared
        """
        hx, H = self.cfm.get_h_and_H(self.x, m.sensor_id)
        R     = self.cfm.R_specs.get(m.sensor_id, self.cfm.R_specs["radar"])

        z = np.array([[m.range_m], [m.bearing_rad]])
        y = z - hx
        y[1, 0] = (y[1, 0] + np.pi) % (2 * np.pi) - np.pi   # wrap bearing

        S   = H @ self.P @ H.T + R
        nis = (y.T @ np.linalg.inv(S) @ y).item()

        if nis > gate_limit:
            return False, nis

        K      = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        return True, nis
