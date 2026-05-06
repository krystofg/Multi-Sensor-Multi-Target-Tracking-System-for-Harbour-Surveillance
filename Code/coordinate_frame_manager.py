import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# T2: COORDINATE FRAME MANAGER (CFM)
# =============================================================================
class CoordinateFrameManager:
    def __init__(self):
        """
        Responsible for:
        - Managing sensor positions in the global NED frame
        - Computing the measurement function h_i(x, t)
        - Computing the Jacobian H_i for EKF updates
        - Providing measurement noise covariance R_i for each sensor

        Assumptions:
        - All sensors share the same NED orientation (no rotation needed)
        - Only position offsets are considered
        """

        # ---------------------------------------------------------------------
        # Sensor position offsets in NED frame (meters)
        # FIX 1: Camera offset corrected to [-80, 120] m per spec.
        #         Was wrongly set to [2.0, 0.0] — critical for scenarios B-E.
        # ---------------------------------------------------------------------
        self.offsets = {
            "radar":  np.array([0.0,   0.0]),    # mm-wave radar at NED origin
            "camera": np.array([-80.0, 120.0]),  # stereo camera NED offset
        }

        # ---------------------------------------------------------------------
        # Vessel (ownship) position in NED frame
        # Updated using GNSS measurements
        # Used as dynamic offset for AIS measurements
        # ---------------------------------------------------------------------
        self.vessel_pos = np.array([0.0, 0.0])

        # ---------------------------------------------------------------------
        # Measurement noise covariance matrices R_i for each sensor
        # FIX 2: All R matrices corrected to match the project spec:
        #   Radar:  σ_r = 5 m,  σ_φ = 0.3°
        #   Camera: σ_r = 8 m,  σ_φ = 0.15°   (was 2 m / 0.5° — wrong)
        #   AIS:    σ = 4 m NED position noise  (was mismatched units)
        # Note: AIS outputs NED position, so R_ais is in metres², not (m, rad).
        # ---------------------------------------------------------------------
        self.R_specs = {
            "radar":  np.diag([5.0**2, np.deg2rad(0.3)**2]),
            "camera": np.diag([8.0**2, np.deg2rad(0.15)**2]),
            "ais":    np.diag([4.0**2, 4.0**2]),
        }

    def update_vessel_pos(self, gnss_m):
        """
        Update vessel (ownship) position in NED frame using a GNSS measurement.
        Required for AIS processing — AIS gives absolute NED positions that
        must be resolved relative to the vessel.
        """
        self.vessel_pos = np.array([gnss_m.north_m, gnss_m.east_m])

    def get_h_and_H(self, x, sensor_id):
        """
        Compute measurement function h_i(x) and its Jacobian H_i for a sensor.

        All sensors share NED orientation — only position offsets differ.
        Measurement model: h_i(x) = [range, bearing] relative to sensor i.
          range   = sqrt((pN - sN)^2 + (pE - sE)^2)
          bearing = atan2(pE - sE, pN - sN)   [from North, clockwise]

        Parameters
        ----------
        x         : (4,1) ndarray  — target state [pN, pE, vN, vE]
        sensor_id : str            — 'radar', 'camera', or 'ais'

        Returns
        -------
        h : (2,1) ndarray  — predicted measurement [range, bearing]
        H : (2,4) ndarray  — Jacobian dh/dx
        """
        # Select sensor position offset
        if sensor_id == "ais":
            offset = self.vessel_pos          # time-varying vessel position
        else:
            offset = self.offsets.get(sensor_id, np.array([0.0, 0.0]))

        dN = x[0, 0] - offset[0]
        dE = x[1, 0] - offset[1]

        r_sq = dN**2 + dE**2
        r    = np.sqrt(r_sq)

        h = np.array([
            [r],
            [np.arctan2(dE, dN)],   # bearing: from North, clockwise
        ])

        H = np.zeros((2, 4))
        H[0, 0] =  dN / r       # ∂range/∂pN
        H[0, 1] =  dE / r       # ∂range/∂pE
        H[1, 0] = -dE / r_sq    # ∂bearing/∂pN
        H[1, 1] =  dN / r_sq    # ∂bearing/∂pE
        # velocity columns remain zero

        return h, H

# =========================================================
# Helper: create state vector in correct shape (4x1)
# =========================================================
def make_state(pN, pE, vN=0.0, vE=0.0):
    return np.array([[pN], [pE], [vN], [vE]])


# =========================================================
# (a) Known position → expected measurement
# =========================================================
def test_radar_basic():
    cfm = CoordinateFrameManager()

    x = make_state(100.0, 0.0)  # target north of radar

    h, H = cfm.get_h_and_H(x, "radar")

    assert np.isclose(h[0, 0], 100.0), "Range should be 100"
    assert np.isclose(h[1, 0], 0.0), "Bearing should be 0 rad"


def test_quadrant_east():
    cfm = CoordinateFrameManager()

    x = make_state(0.0, 100.0)  # east

    h, _ = cfm.get_h_and_H(x, "radar")

    assert np.isclose(h[0, 0], 100.0)
    assert np.isclose(h[1, 0], np.pi / 2), "Bearing should be pi/2"


def test_camera_offset():
    cfm = CoordinateFrameManager()

    x = make_state(0.0, 0.0)  # target at origin
    # camera at [-80, 120]

    h, _ = cfm.get_h_and_H(x, "camera")

    dN = 0 - (-80)
    dE = 0 - (120)

    expected_range = np.sqrt(dN**2 + dE**2)
    expected_bearing = np.arctan2(dE, dN)

    assert np.isclose(h[0, 0], expected_range)
    assert np.isclose(h[1, 0], expected_bearing)


# =========================================================
# (b) Measurement consistency (translation invariance)
# =========================================================
def test_translation_invariance():
    cfm = CoordinateFrameManager()

    x1 = make_state(100.0, 50.0)
    x2 = make_state(1100.0, 1050.0)  # shifted by +1000

    # also shift radar manually (simulate offset change)
    cfm.offsets["radar"] = np.array([0.0, 0.0])
    h1, _ = cfm.get_h_and_H(x1, "radar")

    cfm.offsets["radar"] = np.array([1000.0, 1000.0])
    h2, _ = cfm.get_h_and_H(x2, "radar")

    assert np.allclose(h1, h2), "Should be invariant to translation"


# =========================================================
# (c) AIS consistency with radar
# =========================================================
def test_ais_matches_radar_same_origin():
    cfm = CoordinateFrameManager()

    x = make_state(300.0, 400.0)

    # Set vessel at origin → same as radar
    cfm.vessel_pos = np.array([0.0, 0.0])

    h_radar, _ = cfm.get_h_and_H(x, "radar")
    h_ais, _   = cfm.get_h_and_H(x, "ais")

    assert np.allclose(h_radar, h_ais), "AIS should match radar if same reference"


def test_ais_offset():
    cfm = CoordinateFrameManager()

    x = make_state(300.0, 400.0)

    # vessel is NOT at origin
    cfm.vessel_pos = np.array([100.0, 50.0])

    h, _ = cfm.get_h_and_H(x, "ais")

    dN = 300 - 100
    dE = 400 - 50

    expected_range = np.sqrt(dN**2 + dE**2)
    expected_bearing = np.arctan2(dE, dN)

    assert np.isclose(h[0, 0], expected_range)
    assert np.isclose(h[1, 0], expected_bearing)


# =========================================================
# BONUS: Jacobian sanity check (VERY useful)
# =========================================================
def test_jacobian_dimensions():
    cfm = CoordinateFrameManager()

    x = make_state(100.0, 50.0)

    h, H = cfm.get_h_and_H(x, "radar")

    assert H.shape == (2, 4), "Jacobian must be 2x4"


# =========================================================
# Run manually
# =========================================================
if __name__ == "__main__":
    test_radar_basic()
    test_quadrant_east()
    test_camera_offset()
    test_translation_invariance()
    test_ais_matches_radar_same_origin()
    test_ais_offset()
    test_jacobian_dimensions()

    print("Unit Tests for the Coordinate Frame Manager Module were Successful!")