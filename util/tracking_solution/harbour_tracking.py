"""Standalone T2-T7 tracking code for the harbour surveillance project.

This module intentionally does not import or edit the original notebook.  It
uses the JSON scenarios produced by ``harbour_simulation.ipynb`` as input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SENSOR_ORDER = ("radar", "camera", "ais")
CHI2_95 = {2: 5.9915, 4: 9.4877, 6: 12.5916}
CHI2_99 = {2: 9.2103, 4: 13.2767, 6: 16.8119}


def wrap_angle(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rb_to_cart(range_m: float, bearing_rad: float) -> np.ndarray:
    """Convert range/bearing to an NED position delta [dN, dE]."""
    return np.array(
        [range_m * math.cos(bearing_rad), range_m * math.sin(bearing_rad)],
        dtype=float,
    )


def cart_to_rb(delta_ned: np.ndarray) -> np.ndarray:
    """Convert an NED position delta [dN, dE] to [range, bearing]."""
    d_n, d_e = float(delta_ned[0]), float(delta_ned[1])
    return np.array([math.hypot(d_n, d_e), math.atan2(d_e, d_n)], dtype=float)


def block_diag(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Small local block diagonal helper to avoid a scipy dependency."""
    rows = sum(m.shape[0] for m in matrices)
    cols = sum(m.shape[1] for m in matrices)
    out = np.zeros((rows, cols), dtype=float)
    r = 0
    c = 0
    for matrix in matrices:
        rr, cc = matrix.shape
        out[r : r + rr, c : c + cc] = matrix
        r += rr
        c += cc
    return out


@dataclass(frozen=True)
class RawMeasurement:
    sensor_id: str
    time: float
    is_false_alarm: bool
    target_id: int
    range_m: float | None = None
    bearing_rad: float | None = None
    north_m: float | None = None
    east_m: float | None = None

    @classmethod
    def from_json(cls, item: dict) -> "RawMeasurement":
        return cls(
            sensor_id=item["sensor_id"],
            time=float(item["time"]),
            is_false_alarm=bool(item.get("is_false_alarm", False)),
            target_id=int(item.get("target_id", -1)),
            range_m=None if item.get("range_m") is None else float(item["range_m"]),
            bearing_rad=None
            if item.get("bearing_rad") is None
            else float(item["bearing_rad"]),
            north_m=None if item.get("north_m") is None else float(item["north_m"]),
            east_m=None if item.get("east_m") is None else float(item["east_m"]),
        )


@dataclass
class ScenarioData:
    scenario_name: str
    t_end: float
    sensor_configs: dict
    ground_truth: dict[int, np.ndarray]
    measurements: list[RawMeasurement]
    vessel_positions: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioData":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        measurements = [RawMeasurement.from_json(m) for m in raw["measurements"]]
        measurements.sort(key=lambda m: (m.time, m.sensor_id))
        ground_truth = {
            int(tid): np.asarray(rows, dtype=float)
            for tid, rows in raw["ground_truth"].items()
        }
        vessel_positions = np.asarray(raw.get("vessel_positions", []), dtype=float)
        return cls(
            scenario_name=str(raw["scenario_name"]),
            t_end=float(raw["t_end"]),
            sensor_configs=raw["sensor_configs"],
            ground_truth=ground_truth,
            measurements=measurements,
            vessel_positions=vessel_positions,
        )

    @property
    def gnss_measurements(self) -> list[RawMeasurement]:
        return [m for m in self.measurements if m.sensor_id == "gnss"]

    def truth_at(self, time_s: float, target_id: int = 0) -> np.ndarray:
        """Linearly interpolate ground truth [N, E, vN, vE] at time_s."""
        rows = self.ground_truth[target_id]
        times = rows[:, 0]
        return np.array(
            [np.interp(time_s, times, rows[:, col]) for col in range(1, 5)],
            dtype=float,
        )

    def nearest_gnss(self, time_s: float) -> tuple[np.ndarray, float]:
        """Return nearest noisy GNSS ownship position and its timestamp."""
        gnss = self.gnss_measurements
        if not gnss:
            return np.zeros(2, dtype=float), float("nan")
        times = np.array([m.time for m in gnss], dtype=float)
        idx = int(np.searchsorted(times, time_s))
        candidates = []
        if idx < len(gnss):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        best = min(candidates, key=lambda i: abs(gnss[i].time - time_s))
        m = gnss[best]
        return np.array([m.north_m, m.east_m], dtype=float), m.time

    def grouped_measurements(
        self,
        allowed_sensors: Iterable[str],
        start_time: float = -math.inf,
        stop_time: float = math.inf,
    ) -> list[tuple[float, dict[str, list[RawMeasurement]]]]:
        allowed = set(allowed_sensors)
        buckets: dict[float, dict[str, list[RawMeasurement]]] = {}
        for m in self.measurements:
            if m.sensor_id not in allowed:
                continue
            if m.time < start_time or m.time > stop_time:
                continue
            key = round(m.time, 6)
            buckets.setdefault(key, {}).setdefault(m.sensor_id, []).append(m)
        return [(t, buckets[t]) for t in sorted(buckets)]


class CoordinateFrameManager:
    """Coordinate manager and range/bearing measurement model for T2-T5."""

    def __init__(self, sensor_configs: dict | None = None):
        sensor_configs = sensor_configs or {}
        radar_cfg = sensor_configs.get("radar", {})
        camera_cfg = sensor_configs.get("camera", {})
        ais_cfg = sensor_configs.get("ais", {})
        gnss_cfg = sensor_configs.get("gnss", {})

        self.offsets = {
            "radar": np.asarray(radar_cfg.get("pos_ned", [0.0, 0.0]), dtype=float),
            "camera": np.asarray(
                camera_cfg.get("pos_ned", [-80.0, 120.0]), dtype=float
            ),
        }
        self.sigma = {
            "radar_r": float(radar_cfg.get("sigma_r_m", 5.0)),
            "radar_b": math.radians(float(radar_cfg.get("sigma_phi_deg", 0.3))),
            "camera_r": float(camera_cfg.get("sigma_r_m", 8.0)),
            "camera_b": math.radians(float(camera_cfg.get("sigma_phi_deg", 0.15))),
            "ais_pos": float(ais_cfg.get("sigma_pos_m", 4.0)),
            "gnss_pos": float(gnss_cfg.get("sigma_pos_m", 2.0)),
        }
        self.vessel_pos = np.zeros(2, dtype=float)

    def update_vessel_pos(self, north_m: float, east_m: float) -> None:
        self.vessel_pos = np.array([north_m, east_m], dtype=float)

    def sensor_position(
        self, sensor_id: str, vessel_pos: np.ndarray | None = None
    ) -> np.ndarray:
        if sensor_id == "ais":
            return self.vessel_pos if vessel_pos is None else np.asarray(vessel_pos)
        return self.offsets[sensor_id]

    def h_and_H(
        self,
        x: np.ndarray,
        sensor_id: str,
        sensor_pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return predicted [range, bearing] and Jacobian for a sensor."""
        sensor_pos = self.sensor_position(sensor_id, sensor_pos)
        d_n = float(x[0, 0] - sensor_pos[0])
        d_e = float(x[1, 0] - sensor_pos[1])
        r_sq = max(d_n * d_n + d_e * d_e, 1e-9)
        r = math.sqrt(r_sq)

        h = np.array([[r], [math.atan2(d_e, d_n)]], dtype=float)
        H = np.zeros((2, 4), dtype=float)
        H[0, 0] = d_n / r
        H[0, 1] = d_e / r
        H[1, 0] = -d_e / r_sq
        H[1, 1] = d_n / r_sq
        return h, H

    def range_bearing_noise(
        self,
        sensor_id: str,
        delta_ned: np.ndarray | None = None,
        include_gnss_for_ais: bool = True,
    ) -> np.ndarray:
        """Return R in range/bearing units for the requested sensor.

        AIS reports arrive as noisy target NED positions.  T5 uses them as
        implied range/bearing observations relative to the vessel GNSS fix, so
        their Cartesian covariance must be transformed into polar coordinates.
        """
        if sensor_id == "radar":
            return np.diag([self.sigma["radar_r"] ** 2, self.sigma["radar_b"] ** 2])
        if sensor_id == "camera":
            return np.diag([self.sigma["camera_r"] ** 2, self.sigma["camera_b"] ** 2])
        if sensor_id != "ais":
            raise ValueError(f"Unsupported sensor for EKF update: {sensor_id}")

        if delta_ned is None:
            raise ValueError("AIS range/bearing covariance requires a NED delta")
        d_n, d_e = float(delta_ned[0]), float(delta_ned[1])
        r_sq = max(d_n * d_n + d_e * d_e, 1e-9)
        r = math.sqrt(r_sq)
        j = np.array([[d_n / r, d_e / r], [-d_e / r_sq, d_n / r_sq]], dtype=float)
        sigma_sq = self.sigma["ais_pos"] ** 2
        if include_gnss_for_ais:
            sigma_sq += self.sigma["gnss_pos"] ** 2
        return j @ (sigma_sq * np.eye(2)) @ j.T

    def measurement_to_position(
        self, measurement: RawMeasurement, scenario: ScenarioData
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a raw measurement to an initial global position estimate.

        Returns (position_ned, covariance_position_ned).  This helper is used
        for measurement-only track initiation; it never uses target_id labels.
        """
        if measurement.sensor_id in ("radar", "camera"):
            assert measurement.range_m is not None
            assert measurement.bearing_rad is not None
            sensor_pos = self.sensor_position(measurement.sensor_id)
            z_delta = rb_to_cart(measurement.range_m, measurement.bearing_rad)
            r_cov = self.range_bearing_noise(measurement.sensor_id)
            c = math.cos(measurement.bearing_rad)
            s = math.sin(measurement.bearing_rad)
            j = np.array([[c, -measurement.range_m * s], [s, measurement.range_m * c]])
            return sensor_pos + z_delta, j @ r_cov @ j.T

        if measurement.sensor_id == "ais":
            if measurement.north_m is None or measurement.east_m is None:
                raise ValueError("AIS measurement has no NED position")
            sigma_sq = self.sigma["ais_pos"] ** 2
            return (
                np.array([measurement.north_m, measurement.east_m], dtype=float),
                sigma_sq * np.eye(2),
            )

        raise ValueError(f"Cannot initialize from {measurement.sensor_id}")

    def measurement_to_rb(
        self, measurement: RawMeasurement, scenario: ScenarioData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert raw JSON measurement to (z, sensor_pos, R) for EKF update."""
        if measurement.sensor_id in ("radar", "camera"):
            assert measurement.range_m is not None
            assert measurement.bearing_rad is not None
            z = np.array([[measurement.range_m], [measurement.bearing_rad]], dtype=float)
            sensor_pos = self.sensor_position(measurement.sensor_id)
            R = self.range_bearing_noise(measurement.sensor_id)
            return z, sensor_pos, R

        if measurement.sensor_id == "ais":
            if measurement.north_m is None or measurement.east_m is None:
                raise ValueError("AIS measurement has no NED position")
            vessel_pos, _ = scenario.nearest_gnss(measurement.time)
            target_pos = np.array([measurement.north_m, measurement.east_m], dtype=float)
            delta = target_pos - vessel_pos
            rb = cart_to_rb(delta)
            self.update_vessel_pos(vessel_pos[0], vessel_pos[1])
            z = np.array([[rb[0]], [rb[1]]], dtype=float)
            R = self.range_bearing_noise("ais", delta)
            return z, vessel_pos, R

        raise ValueError(f"Cannot update from {measurement.sensor_id}")


@dataclass
class EKFTracker:
    x: np.ndarray
    P: np.ndarray
    cfm: CoordinateFrameManager
    time: float
    sigma_a: float = 0.05

    def copy(self) -> "EKFTracker":
        return EKFTracker(self.x.copy(), self.P.copy(), self.cfm, self.time, self.sigma_a)

    def predict_to(self, time_s: float) -> None:
        dt = float(time_s - self.time)
        if dt <= 0.0:
            self.time = max(self.time, time_s)
            return
        F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        q = self.sigma_a**2
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        Q = q * np.array(
            [
                [dt4 / 4, 0, dt3 / 2, 0],
                [0, dt4 / 4, 0, dt3 / 2],
                [dt3 / 2, 0, dt2, 0],
                [0, dt3 / 2, 0, dt2],
            ],
            dtype=float,
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.time = time_s

    def innovation(
        self,
        sensor_id: str,
        z: np.ndarray,
        sensor_pos: np.ndarray,
        R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        h, H = self.cfm.h_and_H(self.x, sensor_id, sensor_pos=sensor_pos)
        y = z - h
        y[1, 0] = wrap_angle(float(y[1, 0]))
        S = H @ self.P @ H.T + R
        nis = float(y.T @ np.linalg.solve(S, y))
        return y, H, S, nis

    def update_rb(
        self,
        sensor_id: str,
        z: np.ndarray,
        sensor_pos: np.ndarray,
        R: np.ndarray,
        gate_limit: float = CHI2_99[2],
    ) -> tuple[bool, float]:
        y, H, S, nis = self.innovation(sensor_id, z, sensor_pos, R)
        if nis > gate_limit:
            return False, nis
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        return True, nis

    def joint_update_rb(
        self,
        observations: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    ) -> tuple[bool, float]:
        if not observations:
            return False, float("nan")
        ys = []
        hs = []
        Hs = []
        Rs = []
        for sensor_id, z, sensor_pos, R in observations:
            h, H = self.cfm.h_and_H(self.x, sensor_id, sensor_pos=sensor_pos)
            y = z - h
            y[1, 0] = wrap_angle(float(y[1, 0]))
            ys.append(y)
            hs.append(h)
            Hs.append(H)
            Rs.append(R)
        y_joint = np.vstack(ys)
        H_joint = np.vstack(Hs)
        R_joint = block_diag(Rs)
        S = H_joint @ self.P @ H_joint.T + R_joint
        nis = float(y_joint.T @ np.linalg.solve(S, y_joint))
        gate_limit = CHI2_99.get(len(y_joint), max(CHI2_99.values()))
        if nis > gate_limit:
            return False, nis
        K = self.P @ H_joint.T @ np.linalg.inv(S)
        self.x = self.x + K @ y_joint
        I = np.eye(4)
        self.P = (I - K @ H_joint) @ self.P @ (I - K @ H_joint).T + K @ R_joint @ K.T
        return True, nis


@dataclass
class HistoryRow:
    time: float
    north: float
    east: float
    v_north: float
    v_east: float
    accepted: bool
    sensors: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class TrackingResult:
    name: str
    fusion_mode: str
    allowed_sensors: tuple[str, ...]
    bootstrap_sensor: str
    bootstrap_time: float
    confirmation_time: float | None
    history: list[HistoryRow]
    nis: list[dict]

    @property
    def accepted_count(self) -> int:
        return sum(1 for row in self.history if row.accepted)

    def accepted_sensor_count(self, sensor_id: str) -> int:
        return sum(sensor_id in row.sensors for row in self.history)


@dataclass
class ManagedTrack:
    """One EKF track with T7 lifecycle bookkeeping."""

    track_id: int
    tracker: EKFTracker
    status: str = "tentative"
    hit_window: list[bool] = field(default_factory=list)
    total_hits: int = 1
    consecutive_misses: int = 0
    first_detection_time: float | None = None
    first_detection_pos: np.ndarray | None = None
    last_update_time: float | None = None
    last_detection_pos: np.ndarray | None = None
    history: list[dict] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.status != "deleted"

    @property
    def is_confirmed(self) -> bool:
        return self.status in ("confirmed", "coasting")

    def mark_hit(
        self,
        time_s: float,
        detection_pos: np.ndarray,
        sensor_id: str,
        confirmation_m: int,
        confirmation_n: int,
    ) -> None:
        if self.first_detection_time is None:
            self.first_detection_time = time_s
            self.first_detection_pos = detection_pos.copy()
        elif self.total_hits == 1 and self.first_detection_pos is not None:
            dt = max(time_s - self.first_detection_time, 1e-6)
            velocity = (detection_pos - self.first_detection_pos) / dt
            self.tracker.x[2, 0] = velocity[0]
            self.tracker.x[3, 0] = velocity[1]
            self.tracker.P[2, 2] = min(self.tracker.P[2, 2], 25.0)
            self.tracker.P[3, 3] = min(self.tracker.P[3, 3], 25.0)

        self.total_hits += 1
        self.consecutive_misses = 0
        self.hit_window.append(True)
        if len(self.hit_window) > confirmation_n:
            self.hit_window.pop(0)
        if self.status == "tentative" and sum(self.hit_window) >= confirmation_m:
            self.status = "confirmed"
        elif self.status == "coasting":
            self.status = "confirmed"
        self.last_update_time = time_s
        self.last_detection_pos = detection_pos.copy()
        self.history.append(
            {
                "time": time_s,
                "north": float(self.tracker.x[0, 0]),
                "east": float(self.tracker.x[1, 0]),
                "status": self.status,
                "sensor": sensor_id,
                "event": "hit",
            }
        )

    def mark_miss(
        self,
        time_s: float,
        confirmation_n: int,
        tentative_delete_after: int,
        confirmed_delete_after: int,
    ) -> None:
        self.consecutive_misses += 1
        self.hit_window.append(False)
        if len(self.hit_window) > confirmation_n:
            self.hit_window.pop(0)

        if self.status == "tentative":
            if self.consecutive_misses >= tentative_delete_after:
                self.status = "deleted"
        elif self.status in ("confirmed", "coasting"):
            if self.consecutive_misses >= confirmed_delete_after:
                self.status = "deleted"
            else:
                self.status = "coasting"

        self.history.append(
            {
                "time": time_s,
                "north": float(self.tracker.x[0, 0]),
                "east": float(self.tracker.x[1, 0]),
                "status": self.status,
                "sensor": None,
                "event": "miss",
            }
        )


@dataclass
class MultiTargetResult:
    name: str
    allowed_sensors: tuple[str, ...]
    snapshots: list[dict]
    tracks: list[ManagedTrack]
    associations: list[dict]
    initiated_tracks: int
    deleted_tracks: int


def make_tracker_from_measurement(
    measurement: RawMeasurement,
    scenario: ScenarioData,
    cfm: CoordinateFrameManager,
    velocity_std: float = 15.0,
) -> EKFTracker:
    pos, pos_cov = cfm.measurement_to_position(measurement, scenario)
    x0 = np.array([[pos[0]], [pos[1]], [0.0], [0.0]], dtype=float)
    P0 = np.zeros((4, 4), dtype=float)
    P0[:2, :2] = pos_cov + 25.0 * np.eye(2)
    P0[2, 2] = velocity_std**2
    P0[3, 3] = velocity_std**2
    return EKFTracker(x0, P0, cfm, time=measurement.time)


def _candidate_measurements(
    scenario: ScenarioData,
    bootstrap_sensors: Sequence[str],
    window_s: float = 4.0,
) -> list[RawMeasurement]:
    allowed = set(bootstrap_sensors)
    usable = [
        m
        for m in scenario.measurements
        if m.sensor_id in allowed and (m.range_m is not None or m.north_m is not None)
    ]
    if not usable:
        raise ValueError(f"No bootstrap measurements for sensors {bootstrap_sensors}")
    first_t = min(m.time for m in usable)
    return [m for m in usable if m.time <= first_t + window_s]


def choose_bootstrap_measurement(
    scenario: ScenarioData,
    cfm: CoordinateFrameManager,
    allowed_sensors: Sequence[str],
    bootstrap_sensors: Sequence[str] | None = None,
    score_window_s: float = 24.0,
) -> RawMeasurement:
    """Choose an initial measurement using only measurement consistency.

    No target_id or is_false_alarm flags are used.  Each early detection starts
    a tentative EKF; the one with the most gated follow-up detections wins.
    """
    bootstrap_sensors = tuple(bootstrap_sensors or allowed_sensors)
    candidates = _candidate_measurements(scenario, bootstrap_sensors)
    best_score: tuple[int, float, float] | None = None
    best_measurement: RawMeasurement | None = None

    for candidate in candidates:
        local_cfm = CoordinateFrameManager(scenario.sensor_configs)
        tracker = make_tracker_from_measurement(candidate, scenario, local_cfm)
        result = run_tracker(
            scenario,
            tracker,
            allowed_sensors=allowed_sensors,
            fusion_mode="sequential",
            start_time=candidate.time,
            stop_time=candidate.time + score_window_s,
            gate_limit=30.0,
            confirmation_m=99,
            confirmation_n=99,
            name="bootstrap_score",
        )
        nis_vals = [item["nis"] for item in result.nis if item["accepted"]]
        mean_nis = float(np.mean(nis_vals)) if nis_vals else 1e6
        score = (result.accepted_count, -mean_nis, -candidate.time)
        if best_score is None or score > best_score:
            best_score = score
            best_measurement = candidate

    assert best_measurement is not None
    return best_measurement


def _select_best_measurement(
    tracker: EKFTracker,
    scenario: ScenarioData,
    sensor_id: str,
    measurements: Sequence[RawMeasurement],
    gate_limit: float,
) -> tuple[RawMeasurement, np.ndarray, np.ndarray, np.ndarray, float] | None:
    best = None
    for measurement in measurements:
        try:
            z, sensor_pos, R = tracker.cfm.measurement_to_rb(measurement, scenario)
            _, _, _, nis = tracker.innovation(sensor_id, z, sensor_pos, R)
        except (AssertionError, ValueError, np.linalg.LinAlgError):
            continue
        if best is None or nis < best[-1]:
            best = (measurement, z, sensor_pos, R, nis)
    if best is None or best[-1] > gate_limit:
        return None
    return best


def run_tracker(
    scenario: ScenarioData,
    tracker: EKFTracker,
    allowed_sensors: Sequence[str],
    fusion_mode: str = "sequential",
    start_time: float | None = None,
    stop_time: float | None = None,
    gate_limit: float = CHI2_99[2],
    early_gate_limit: float = 30.0,
    early_gate_s: float = 10.0,
    confirmation_m: int = 3,
    confirmation_n: int = 5,
    name: str = "run",
) -> TrackingResult:
    if fusion_mode not in ("sequential", "joint"):
        raise ValueError("fusion_mode must be 'sequential' or 'joint'")
    allowed_sensors = tuple(s for s in SENSOR_ORDER if s in set(allowed_sensors))
    start_time = tracker.time if start_time is None else start_time
    stop_time = scenario.t_end if stop_time is None else stop_time

    history: list[HistoryRow] = []
    nis_history: list[dict] = []
    confirm_window: list[bool] = []
    confirmation_time: float | None = None

    groups = scenario.grouped_measurements(allowed_sensors, start_time, stop_time)
    for scan_t, sensor_groups in groups:
        tracker.predict_to(scan_t)
        scan_success = False
        accepted_sensors: list[str] = []
        current_gate = early_gate_limit if scan_t <= start_time + early_gate_s else gate_limit

        if fusion_mode == "sequential":
            for sensor_id in allowed_sensors:
                measurements = sensor_groups.get(sensor_id, [])
                if not measurements:
                    continue
                selected = _select_best_measurement(
                    tracker, scenario, sensor_id, measurements, current_gate
                )
                if selected is None:
                    continue
                measurement, z, sensor_pos, R, selected_nis = selected
                accepted, nis = tracker.update_rb(sensor_id, z, sensor_pos, R, current_gate)
                nis_history.append(
                    {
                        "time": scan_t,
                        "sensor": sensor_id,
                        "nis": float(nis),
                        "accepted": bool(accepted),
                    }
                )
                if accepted:
                    scan_success = True
                    accepted_sensors.append(sensor_id)

        else:
            observations = []
            selected_sensors = []
            for sensor_id in allowed_sensors:
                measurements = sensor_groups.get(sensor_id, [])
                if not measurements:
                    continue
                selected = _select_best_measurement(
                    tracker, scenario, sensor_id, measurements, current_gate
                )
                if selected is None:
                    continue
                _, z, sensor_pos, R, _ = selected
                observations.append((sensor_id, z, sensor_pos, R))
                selected_sensors.append(sensor_id)
            if observations:
                accepted, nis = tracker.joint_update_rb(observations)
                nis_history.append(
                    {
                        "time": scan_t,
                        "sensor": "+".join(selected_sensors),
                        "nis": float(nis),
                        "accepted": bool(accepted),
                    }
                )
                if accepted:
                    scan_success = True
                    accepted_sensors.extend(selected_sensors)

        history.append(
            HistoryRow(
                time=scan_t,
                north=float(tracker.x[0, 0]),
                east=float(tracker.x[1, 0]),
                v_north=float(tracker.x[2, 0]),
                v_east=float(tracker.x[3, 0]),
                accepted=scan_success,
                sensors=tuple(accepted_sensors),
            )
        )
        confirm_window.append(scan_success)
        if len(confirm_window) > confirmation_n:
            confirm_window.pop(0)
        if confirmation_time is None and sum(confirm_window) >= confirmation_m:
            confirmation_time = scan_t

    return TrackingResult(
        name=name,
        fusion_mode=fusion_mode,
        allowed_sensors=allowed_sensors,
        bootstrap_sensor="",
        bootstrap_time=start_time,
        confirmation_time=confirmation_time,
        history=history,
        nis=nis_history,
    )


def run_tracking(
    scenario: ScenarioData,
    allowed_sensors: Sequence[str],
    fusion_mode: str = "sequential",
    bootstrap_sensors: Sequence[str] | None = None,
    name: str = "run",
) -> TrackingResult:
    cfm = CoordinateFrameManager(scenario.sensor_configs)
    boot = choose_bootstrap_measurement(
        scenario,
        cfm,
        allowed_sensors=allowed_sensors,
        bootstrap_sensors=bootstrap_sensors,
    )
    tracker = make_tracker_from_measurement(boot, scenario, cfm)
    result = run_tracker(
        scenario,
        tracker,
        allowed_sensors=allowed_sensors,
        fusion_mode=fusion_mode,
        start_time=boot.time,
        name=name,
    )
    result.bootstrap_sensor = boot.sensor_id
    result.bootstrap_time = boot.time
    return result


def _make_managed_track(
    track_id: int,
    measurement: RawMeasurement,
    scenario: ScenarioData,
    cfm: CoordinateFrameManager,
    confirmation_n: int,
) -> ManagedTrack:
    tracker = make_tracker_from_measurement(measurement, scenario, cfm)
    pos, _ = cfm.measurement_to_position(measurement, scenario)
    track = ManagedTrack(
        track_id=track_id,
        tracker=tracker,
        hit_window=[True],
        total_hits=1,
        consecutive_misses=0,
        first_detection_time=measurement.time,
        first_detection_pos=pos.copy(),
        last_update_time=measurement.time,
        last_detection_pos=pos.copy(),
    )
    if len(track.hit_window) > confirmation_n:
        track.hit_window = track.hit_window[-confirmation_n:]
    track.history.append(
        {
            "time": measurement.time,
            "north": float(tracker.x[0, 0]),
            "east": float(tracker.x[1, 0]),
            "status": track.status,
            "sensor": measurement.sensor_id,
            "event": "init",
        }
    )
    return track


def _gnn_assign_sensor(
    tracks: Sequence[ManagedTrack],
    measurements: Sequence[RawMeasurement],
    scenario: ScenarioData,
    sensor_id: str,
    gate_limit: float,
    state_limit: int = 120_000,
) -> tuple[list[tuple[int, int, float]], set[int]]:
    """Global nearest-neighbour assignment for one sensor stream.

    Returns ``[(track_index, measurement_index, nis), ...]`` and the set of
    measurement indices that had no gated assignment.  The recursion optimises
    first for the number of assigned detections, then for total NIS.
    """
    if not tracks or not measurements:
        return [], set(range(len(measurements)))

    candidate_by_det: list[list[tuple[int, float]]] = []
    for det_idx, measurement in enumerate(measurements):
        candidates: list[tuple[int, float]] = []
        try:
            z, sensor_pos, R = tracks[0].tracker.cfm.measurement_to_rb(measurement, scenario)
        except (AssertionError, ValueError):
            candidate_by_det.append(candidates)
            continue
        for track_idx, track in enumerate(tracks):
            if not track.is_active:
                continue
            try:
                _, _, _, nis = track.tracker.innovation(sensor_id, z, sensor_pos, R)
            except np.linalg.LinAlgError:
                continue
            if nis <= gate_limit:
                candidates.append((track_idx, nis))
        candidates.sort(key=lambda item: item[1])
        candidate_by_det.append(candidates)

    det_order = [
        idx
        for idx in sorted(
            range(len(measurements)),
            key=lambda i: (len(candidate_by_det[i]) == 0, len(candidate_by_det[i])),
        )
        if candidate_by_det[idx]
    ]
    if not det_order:
        return [], set(range(len(measurements)))

    best_count = -1
    best_cost = float("inf")
    best_assignment: dict[int, tuple[int, float]] = {}
    calls = 0

    def greedy_fallback() -> dict[int, tuple[int, float]]:
        pairs = []
        for det_idx, candidates in enumerate(candidate_by_det):
            for track_idx, nis in candidates:
                pairs.append((nis, det_idx, track_idx))
        pairs.sort()
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        assignment: dict[int, tuple[int, float]] = {}
        for nis, det_idx, track_idx in pairs:
            if det_idx in used_dets or track_idx in used_tracks:
                continue
            used_dets.add(det_idx)
            used_tracks.add(track_idx)
            assignment[det_idx] = (track_idx, nis)
        return assignment

    def recurse(
        pos: int,
        used_tracks: set[int],
        assignment: dict[int, tuple[int, float]],
        count: int,
        cost: float,
    ) -> None:
        nonlocal best_count, best_cost, best_assignment, calls
        calls += 1
        if calls > state_limit:
            return
        remaining = len(det_order) - pos
        if count + remaining < best_count:
            return
        if pos == len(det_order):
            if count > best_count or (count == best_count and cost < best_cost):
                best_count = count
                best_cost = cost
                best_assignment = dict(assignment)
            return

        det_idx = det_order[pos]
        # Leave this detection unmatched.
        recurse(pos + 1, used_tracks, assignment, count, cost)
        for track_idx, nis in candidate_by_det[det_idx]:
            if track_idx in used_tracks:
                continue
            used_tracks.add(track_idx)
            assignment[det_idx] = (track_idx, nis)
            recurse(pos + 1, used_tracks, assignment, count + 1, cost + nis)
            assignment.pop(det_idx, None)
            used_tracks.remove(track_idx)

    recurse(0, set(), {}, 0, 0.0)
    if calls > state_limit:
        best_assignment = greedy_fallback()

    assignments = [
        (track_idx, det_idx, nis)
        for det_idx, (track_idx, nis) in sorted(best_assignment.items())
    ]
    used_dets = {det_idx for _, det_idx, _ in assignments}
    unmatched = set(range(len(measurements))) - used_dets
    return assignments, unmatched


def _merge_duplicate_tracks(
    tracks: list[ManagedTrack],
    threshold: float = CHI2_99[2],
) -> None:
    active = [track for track in tracks if track.is_active]
    for i, track_a in enumerate(active):
        if not track_a.is_active:
            continue
        for track_b in active[i + 1 :]:
            if not track_b.is_active:
                continue
            # Avoid merging two already-confirmed tracks during close crossings.
            if track_a.is_confirmed and track_b.is_confirmed:
                continue
            dx = track_a.tracker.x[:2] - track_b.tracker.x[:2]
            P = track_a.tracker.P[:2, :2] + track_b.tracker.P[:2, :2]
            try:
                d2 = float(dx.T @ np.linalg.solve(P, dx))
            except np.linalg.LinAlgError:
                continue
            if d2 > threshold:
                continue
            keep, drop = sorted(
                (track_a, track_b),
                key=lambda tr: (tr.is_confirmed, tr.total_hits, -tr.consecutive_misses),
                reverse=True,
            )
            drop.status = "deleted"
            keep.total_hits = max(keep.total_hits, drop.total_hits)


def run_multi_target_tracking(
    scenario: ScenarioData,
    allowed_sensors: Sequence[str],
    name: str = "multi-target",
    confirmation_m: int = 3,
    confirmation_n: int = 5,
    tentative_delete_after: int = 2,
    confirmed_delete_after: int = 5,
    gate_limit: float = CHI2_99[2],
) -> MultiTargetResult:
    """Run T6/T7 multi-target tracking with GNN and track management."""
    cfm = CoordinateFrameManager(scenario.sensor_configs)
    allowed_sensors = tuple(s for s in SENSOR_ORDER if s in set(allowed_sensors))
    groups = scenario.grouped_measurements(allowed_sensors)

    tracks: list[ManagedTrack] = []
    snapshots: list[dict] = []
    associations: list[dict] = []
    next_track_id = 1
    initiated_tracks = 0

    for scan_t, sensor_groups in groups:
        active_tracks = [track for track in tracks if track.is_active]
        for track in active_tracks:
            track.tracker.predict_to(scan_t)

        hit_positions: dict[int, list[np.ndarray]] = {}
        hit_sensors: dict[int, list[str]] = {}
        unmatched_by_sensor: dict[str, set[int]] = {
            sensor_id: set(range(len(sensor_groups.get(sensor_id, []))))
            for sensor_id in allowed_sensors
        }

        for sensor_id in allowed_sensors:
            detections = sensor_groups.get(sensor_id, [])
            if not detections:
                continue
            active_tracks = [track for track in tracks if track.is_active]
            assignments, unmatched = _gnn_assign_sensor(
                active_tracks, detections, scenario, sensor_id, gate_limit
            )
            unmatched_by_sensor[sensor_id] = unmatched
            for local_track_idx, det_idx, nis in assignments:
                track = active_tracks[local_track_idx]
                measurement = detections[det_idx]
                z, sensor_pos, R = cfm.measurement_to_rb(measurement, scenario)
                accepted, used_nis = track.tracker.update_rb(
                    sensor_id, z, sensor_pos, R, gate_limit=gate_limit
                )
                if not accepted:
                    unmatched_by_sensor[sensor_id].add(det_idx)
                    continue
                detection_pos, _ = cfm.measurement_to_position(measurement, scenario)
                hit_positions.setdefault(track.track_id, []).append(detection_pos)
                hit_sensors.setdefault(track.track_id, []).append(sensor_id)
                associations.append(
                    {
                        "time": scan_t,
                        "track_id": track.track_id,
                        "sensor": sensor_id,
                        "nis": float(used_nis),
                    }
                )

        hit_track_ids = set(hit_positions)
        miss_due_this_cycle = "radar" in sensor_groups or (
            "radar" not in allowed_sensors and any(sensor_groups.values())
        )
        for track in [track for track in tracks if track.is_active]:
            if track.track_id in hit_track_ids:
                pos = np.mean(hit_positions[track.track_id], axis=0)
                sensor_label = "+".join(sorted(set(hit_sensors[track.track_id])))
                track.mark_hit(
                    scan_t,
                    pos,
                    sensor_label,
                    confirmation_m=confirmation_m,
                    confirmation_n=confirmation_n,
                )
            elif miss_due_this_cycle:
                track.mark_miss(
                    scan_t,
                    confirmation_n=confirmation_n,
                    tentative_delete_after=tentative_delete_after,
                    confirmed_delete_after=confirmed_delete_after,
                )

        # T7 initiation: every unassigned detection starts a tentative track.
        for sensor_id in allowed_sensors:
            detections = sensor_groups.get(sensor_id, [])
            for det_idx in sorted(unmatched_by_sensor.get(sensor_id, set())):
                measurement = detections[det_idx]
                try:
                    new_track = _make_managed_track(
                        next_track_id,
                        measurement,
                        scenario,
                        cfm,
                        confirmation_n=confirmation_n,
                    )
                except (AssertionError, ValueError):
                    continue
                tracks.append(new_track)
                next_track_id += 1
                initiated_tracks += 1

        _merge_duplicate_tracks(tracks)
        confirmed = []
        for track in tracks:
            if not track.is_confirmed:
                continue
            confirmed.append(
                {
                    "track_id": track.track_id,
                    "status": track.status,
                    "north": float(track.tracker.x[0, 0]),
                    "east": float(track.tracker.x[1, 0]),
                    "v_north": float(track.tracker.x[2, 0]),
                    "v_east": float(track.tracker.x[3, 0]),
                    "misses": track.consecutive_misses,
                }
            )
        snapshots.append({"time": scan_t, "tracks": confirmed})

    deleted_tracks = sum(1 for track in tracks if track.status == "deleted")
    return MultiTargetResult(
        name=name,
        allowed_sensors=allowed_sensors,
        snapshots=snapshots,
        tracks=tracks,
        associations=associations,
        initiated_tracks=initiated_tracks,
        deleted_tracks=deleted_tracks,
    )


def active_truth_positions(
    scenario: ScenarioData, time_s: float
) -> dict[int, np.ndarray]:
    truth: dict[int, np.ndarray] = {}
    for target_id, rows in scenario.ground_truth.items():
        times = rows[:, 0]
        if time_s < times[0] or time_s > times[-1]:
            continue
        truth[target_id] = np.array(
            [np.interp(time_s, times, rows[:, col]) for col in (1, 2)],
            dtype=float,
        )
    return truth


def _minimum_distance_assignment(
    track_positions: dict[int, np.ndarray],
    truth_positions: dict[int, np.ndarray],
    max_distance: float = 75.0,
) -> list[tuple[int, int, float]]:
    track_ids = list(track_positions)
    truth_ids = list(truth_positions)
    if not track_ids or not truth_ids:
        return []

    pairs = []
    for t_idx, track_id in enumerate(track_ids):
        for g_idx, truth_id in enumerate(truth_ids):
            dist = float(np.linalg.norm(track_positions[track_id] - truth_positions[truth_id]))
            if dist <= max_distance:
                pairs.append((dist, t_idx, g_idx))
    pairs.sort()

    used_tracks: set[int] = set()
    used_truth: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for dist, t_idx, g_idx in pairs:
        if t_idx in used_tracks or g_idx in used_truth:
            continue
        used_tracks.add(t_idx)
        used_truth.add(g_idx)
        matches.append((track_ids[t_idx], truth_ids[g_idx], dist))
    return matches


def multi_target_metrics(
    result: MultiTargetResult,
    scenario: ScenarioData,
    start_time: float = 0.0,
    max_match_distance: float = 75.0,
) -> dict:
    motp_series = []
    ce_series = []
    id_switches = 0
    last_match_by_track: dict[int, int] = {}

    for snapshot in result.snapshots:
        time_s = float(snapshot["time"])
        if time_s < start_time:
            continue
        truth = active_truth_positions(scenario, time_s)
        tracks = {
            int(track["track_id"]): np.array([track["north"], track["east"]], dtype=float)
            for track in snapshot["tracks"]
        }
        matches = _minimum_distance_assignment(tracks, truth, max_match_distance)
        if matches:
            motp = float(np.mean([dist for _, _, dist in matches]))
            motp_series.append({"time": time_s, "motp": motp, "matches": len(matches)})
        else:
            motp_series.append({"time": time_s, "motp": float("nan"), "matches": 0})
        ce = abs(len(tracks) - len(truth))
        ce_series.append(
            {
                "time": time_s,
                "ce": int(ce),
                "confirmed_tracks": len(tracks),
                "active_truth": len(truth),
            }
        )
        for track_id, truth_id, _ in matches:
            if track_id in last_match_by_track and last_match_by_track[track_id] != truth_id:
                id_switches += 1
            last_match_by_track[track_id] = truth_id

    motp_values = [row["motp"] for row in motp_series if math.isfinite(row["motp"])]
    ce_values = [row["ce"] for row in ce_series]
    return {
        "avg_motp_m": float(np.mean(motp_values)) if motp_values else float("nan"),
        "avg_ce": float(np.mean(ce_values)) if ce_values else float("nan"),
        "id_switches": int(id_switches),
        "motp_series": motp_series,
        "ce_series": ce_series,
        "initiated_tracks": result.initiated_tracks,
        "deleted_tracks": result.deleted_tracks,
        "final_confirmed_tracks": len(result.snapshots[-1]["tracks"]) if result.snapshots else 0,
    }


def multi_target_summary(
    result: MultiTargetResult,
    scenario: ScenarioData,
    start_time: float = 0.0,
) -> dict:
    metrics = multi_target_metrics(result, scenario, start_time=start_time)
    return {
        "avg_motp_m": round(metrics["avg_motp_m"], 3)
        if math.isfinite(metrics["avg_motp_m"])
        else None,
        "avg_ce": round(metrics["avg_ce"], 3)
        if math.isfinite(metrics["avg_ce"])
        else None,
        "id_switches": metrics["id_switches"],
        "initiated_tracks": metrics["initiated_tracks"],
        "deleted_tracks": metrics["deleted_tracks"],
        "final_confirmed_tracks": metrics["final_confirmed_tracks"],
    }


def rmse(
    result: TrackingResult,
    scenario: ScenarioData,
    target_id: int = 0,
    start_time: float = 0.0,
    end_time: float | None = None,
) -> float:
    errors = []
    for row in result.history:
        if row.time < start_time:
            continue
        if end_time is not None and row.time > end_time:
            continue
        truth = scenario.truth_at(row.time, target_id)
        errors.append((row.north - truth[0]) ** 2 + (row.east - truth[1]) ** 2)
    return float(math.sqrt(np.mean(errors))) if errors else float("nan")


def nis_fraction_inside_95(result: TrackingResult) -> float:
    values = [item["nis"] for item in result.nis if item["accepted"] and math.isfinite(item["nis"])]
    if not values:
        return 0.0
    values_arr = np.asarray(values, dtype=float)
    return float(np.mean((values_arr >= 0.103) & (values_arr <= CHI2_95[2])) * 100.0)


def first_accepted_after(
    result: TrackingResult, sensor_id: str, time_s: float
) -> float | None:
    for row in result.history:
        if row.time > time_s and sensor_id in row.sensors:
            return row.time
    return None


def max_history_gap(result: TrackingResult, start_time: float, end_time: float) -> float:
    times = [row.time for row in result.history if start_time <= row.time <= end_time]
    if len(times) < 2:
        return float("nan")
    return float(max(b - a for a, b in zip(times, times[1:])))
