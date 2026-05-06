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
            
    def update_radar(self, x_pred: np.ndarray, P_pred: np.ndarray,
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
        # Step 1 & 2: Get predicted measurement and Jacobian from CFM
        z_pred, H = self.cfm.get_h_and_H(x_pred, "radar")

        # Step 3: Innovation — difference between actual and predicted measurement.
        #         Wrap the bearing component to keep it in (-pi, pi].
        z = np.reshape(z, (2, 1))
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

    def update_camera(self, x_pred: np.ndarray, P_pred: np.ndarray,
                      z_c: float, R_c: np.ndarray):
        """
        EKF update step for a camera bearing measurement.

        Parameters
        ----------
        x_pred : (4,)   predicted (or intermediate) state
        P_pred : (4,4)  predicted (or intermediate) covariance
        z_c    : float  camera bearing measurement [rad]
        R_c    : (1,1)  camera measurement noise covariance

        Returns
        -------
        x_upd : (4,)   updated state estimate
        P_upd : (4,4)  updated covariance
        innov : float  bearing innovation [rad]
        S_c   : (1,1)  innovation covariance
        """
        # Step 1 & 2: Get predicted measurement and Jacobian from CFM
        h_full, H_full = self.cfm.get_h_and_H(x_pred, "camera")
        z_pred = h_full[1, 0]     # bearing only
        H_c    = H_full[1:2, :]  # bearing row dh/dx

        # Step 3: Innovation — difference between actual and predicted measurement.
        #         Wrap the bearing to keep it in (-pi, pi].
        innov = float(self._wrap_angle(z_c - z_pred))

        # Step 4: Innovation covariance
        S_c = H_c @ P_pred @ H_c.T + R_c

        # Step 5: Kalman gain
        K = P_pred @ H_c.T @ inv(S_c)

        # Step 6: Updated state
        # K is (4,1), innov is scalar -> K * innov is (4,1)
        x_upd = x_pred + K * innov

        # Step 7: Updated covariance — Joseph form for numerical stability.
        I     = np.eye(len(x_pred))
        IKH   = I - K @ H_c
        P_upd = IKH @ P_pred @ IKH.T + K @ R_c @ K.T

        return x_upd, P_upd, innov, S_c

    def update_joint(self, x_pred: np.ndarray, P_pred: np.ndarray,
                        z_r: np.ndarray, z_c: float,
                        R_radar: np.ndarray, R_cam: np.ndarray):
        """
        EKF joint update using a stacked radar + camera measurement vector.

        Parameters
        ----------
        x_pred  : (4,)   predicted state
        P_pred  : (4,4)  predicted covariance
        z_r     : (2,)   radar measurement [range, bearing_radar]
        z_c     : float  camera bearing measurement
        R_radar : (2,2)  radar noise covariance
        R_cam   : (1,1)  camera noise covariance

        Returns
        -------
        x_upd : (4,)   updated state
        P_upd : (4,4)  updated covariance
        innov : (3,)   joint innovation vector
        S     : (3,3)  joint innovation covariance
        """
        # Step 1: build the joint measurement vector z_joint (3,1)
        z_joint = np.array([[z_r[0]], 
                            [z_r[1]], 
                            [z_c]])

        # Step 2: build the joint predicted measurement h_joint (3,1)
        h_r_full, H_r_full = self.cfm.get_h_and_H(x_pred, "radar")
        h_c_full, H_c_full = self.cfm.get_h_and_H(x_pred, "camera")

        h_joint = np.array([[h_r_full[0, 0]], 
                            [h_r_full[1, 0]], 
                            [h_c_full[1, 0]]])

        # Step 3: build the joint Jacobian H_joint (3×4)
        H_joint = np.vstack([H_r_full, H_c_full[1:2, :]])

        # Step 4: build the block-diagonal joint noise covariance R_joint (3×3)
        R_joint = np.zeros((3, 3))
        R_joint[:2, :2] = R_radar
        R_joint[2, 2]   = R_cam.item()

        # Step 5: compute innovation, wrap BOTH bearing components
        innov = z_joint - h_joint
        innov[1, 0] = self._wrap_angle(innov[1, 0]) # radar bearing
        innov[2, 0] = self._wrap_angle(innov[2, 0]) # camera bearing

        # Step 6: innovation covariance S (3×3)
        S = H_joint @ P_pred @ H_joint.T + R_joint

        # Step 7: Kalman gain K (4×3)
        K = P_pred @ H_joint.T @ inv(S)

        # Step 8: updated state (4,1)
        x_upd = x_pred + K @ innov

        # Step 9: updated covariance — Joseph form
        I     = np.eye(4)
        IKH   = I - K @ H_joint
        P_upd = IKH @ P_pred @ IKH.T + K @ R_joint @ K.T

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
        x = x.flatten()  # handles (4,) and (4,1)

        px, py = x[0], x[1]

        r2 = px**2 + py**2
        if r2 < 1e-8:   # avoid division by zero
            r2 = 1e-8

        r = np.sqrt(r2)

        H = np.zeros((2, 4))

        H[0, 0] = px / r
        H[0, 1] = py / r

        H[1, 0] = -py / r2
        H[1, 1] = px / r2

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
        # Ensure x is 2D for indexing consistency with _h_radar
        if x.ndim == 1:
            px, py = x[0], x[1]
        else:
            px, py = x[0, 0], x[1, 0]
            
        phi_c = np.arctan2(py, px)
        return float(phi_c)


    def _H_jacobian_camera(self, x: np.ndarray) -> np.ndarray:
        """
        Jacobian of h_camera with respect to x.

        Returns
        -------
        H_c : (1, 4) ndarray
        """
        x_flat = x.flatten()
        px, py = x_flat[0], x_flat[1]
        
        r2 = px**2 + py**2
        if r2 < 1e-8:   # avoid division by zero
            r2 = 1e-8
            
        H_c = np.zeros((1, 4))
        H_c[0, 0] = -py / r2
        H_c[0, 1] = px / r2
        # Derivatives wrt vx, vy are zero
        
        return H_c

    def _wrap_angle(self, angle: float) -> np.ndarray:
        """Wrap an angle to the interval (-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi


