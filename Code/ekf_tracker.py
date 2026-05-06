import numpy as np
from numpy.linalg import inv

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

    def predict(self, x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray):
        """
        EKF prediction step (linear motion model).

        Parameters
        ----------
        x : (4,) ndarray  — current state estimate
        P : (4,4) ndarray — current error covariance
        F : (4,4) ndarray — state transition matrix
        Q : (4,4) ndarray — process noise covariance

        Returns
        -------
        x_pred : (4,)   predicted state
        P_pred : (4,4)  predicted covariance
        """
        # Propagate the state mean through the linear motion model
        x_pred = F @ x
        # Propagate the covariance and add process noise
        P_pred = F @ P @ F.T + Q
        return x_pred, P_pred
            
    def update(self, x_pred: np.ndarray, P_pred: np.ndarray,
                        z: np.ndarray, R: np.ndarray):
        """
        EKF update step for a radar measurement.

        Parameters
        ----------
        x_pred : (4,)   predicted state
        P_pred : (4,4)  predicted covariance
        z      : (2,)   radar measurement [range, bearing]
        R      : (2,2)  measurement noise covariance

        Returns
        -------
        x_upd : (4,)   updated state estimate
        P_upd : (4,4)  updated error covariance
        innov : (2,)   innovation vector (for diagnostics)
        S     : (2,2)  innovation covariance
        """
        # Step 1: Linearise — Jacobian at the predicted state
        H = self._H_jacobian_radar(x_pred)

        # Step 2: Predicted measurement via the nonlinear function
        z_pred = self._h_radar(x_pred)

        # Step 3: Innovation — difference between actual and predicted measurement.
        #         Wrap the bearing component to keep it in (-pi, pi].
        innov    = z - z_pred
        innov[1, 0] = self._wrap_angle(innov[1, 0])

        # Step 4: Innovation covariance
        S = H @ P_pred @ H.T + R

        # Step 5: Kalman gain
        K = P_pred @ H.T @ inv(S)

        # Step 6: Updated state
        x_upd = x_pred + K @ innov

        # Step 7: Updated covariance — Joseph form for numerical stability.
        #   P = (I - KH) P_pred (I - KH)^T + K R K^T
        #   This guarantees symmetry and positive-semidefiniteness even when
        #   floating-point rounding makes (I - KH) slightly non-symmetric.
        I     = np.eye(len(x_pred))
        IKH   = I - K @ H
        P_upd = IKH @ P_pred @ IKH.T + K @ R @ K.T

        return x_upd, P_upd, innov, S
    
    def _H_jacobian_radar(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of h_radar with respect to the state x.

        Parameters
        ----------
        x : (4,) ndarray  — state [px, py, vx, vy]

        Returns
        -------
        H : (2, 4) ndarray
        """
        px, py = x[0], x[1]
        r2 = px**2 + py**2   # r squared
        r  = np.sqrt(r2)      # r

        H = np.zeros((2, 4))
        # d(range)/d(px), d(range)/d(py)  — velocity components are zero
        H[0, 0] =  px / r
        H[0, 1] =  py / r
        # d(bearing)/d(px), d(bearing)/d(py)
        H[1, 0] = -py / r2
        H[1, 1] =  px / r2
        return H

    def _h_radar(self, x: np.ndarray) -> np.ndarray:
        """
        Nonlinear radar measurement function.

        Parameters
        ----------
        x : (4,) ndarray  — state [px, py, vx, vy]

        Returns
        -------
        z : (2,) ndarray  — predicted measurement [range, bearing]
        """
        px, py = x[0, 0], x[1, 0]
        r   = np.sqrt(px**2 + py**2)   # range
        phi = np.arctan2(py, px)        # bearing
        return np.array([[r], [phi]])

    # ── Camera Jacobian ─────────────────────────────────────────────────────────
    def _h_camera(self, x: np.ndarray) -> float:
        """
        Camera measurement function: bearing angle to the target.

        Parameters
        ----------
        x : (4,) ndarray  — state [px, py, vx, vy]

        Returns
        -------
        phi_c : float  — predicted bearing [rad]
        """
        # TODO
        pass


    def _H_jacobian_camera(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of h_camera with respect to x.

        Returns
        -------
        H_c : (1, 4) ndarray
        """
        # TODO
        H_c = np.zeros((1, 4))
        # --- your code here ---
        return H_c


    def _build_R_camera(self, sigma_phi_c: float) -> np.ndarray:
        """Measurement noise covariance for camera (1×1 matrix)."""
        # TODO
        pass


    def _wrap_angle(self, angle: float) -> np.ndarray:
        """Wrap an angle to the interval (-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi


