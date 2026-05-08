"""Task 7: track management for Scenarios D and E.

The file keeps T7 separate from earlier tasks.  It uses the existing loader,
coordinate frame code, motion model helpers, AIS/GNSS conversion idea from T5,
and the GNN association style from T6.

Usage from the repository root:

    python Code/task7.py
    python Code/task7.py --scenario E
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from coordinate_frame_manager import CoordinateFrameManager
from ekf_tracker import build_motion_model, build_R_camera, build_R_radar
from load_simulation_data import Measurement, SimulationOutput, load_simulation_output


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "task7"
PLOT_DIR = OUT_DIR / "plots"

SENSOR_ORDER = ("radar", "camera", "ais")
TENTATIVE = "tentative"
CONFIRMED = "confirmed"
COASTING = "coasting"
DELETED = "deleted"
LARGE_COST = 1e9


def wrap(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def rb_to_cart(range_m: float, bearing_rad: float) -> np.ndarray:
    """Convert range/bearing to NED [N, E] displacement."""
    return np.array(
        [range_m * math.cos(bearing_rad), range_m * math.sin(bearing_rad)],
        dtype=float,
    )


def cart_to_rb(delta_ned: np.ndarray) -> np.ndarray:
    """Convert NED [N, E] displacement to [range, bearing]."""
    d_n, d_e = float(delta_ned[0]), float(delta_ned[1])
    return np.array([math.hypot(d_n, d_e), math.atan2(d_e, d_n)], dtype=float)


def rb_jacobian_position(delta_ned: np.ndarray) -> np.ndarray:
    """Jacobian of range/bearing wrt Cartesian NED position."""
    d_n, d_e = float(delta_ned[0]), float(delta_ned[1])
    r2 = max(d_n * d_n + d_e * d_e, 1e-9)
    r = math.sqrt(r2)
    return np.array([[d_n / r, d_e / r], [-d_e / r2, d_n / r2]], dtype=float)


def raw_truth_rows(scenario: str) -> Dict[int, np.ndarray]:
    """Load per-target truth rows [t, N, E, vN, vE] directly from JSON.

    The existing loader exposes one shared ground_truth_times vector.  Scenario E
    contains targets with different active intervals, so T7 metrics need each
    target's own time vector.
    """
    path = ROOT / "harbour_sim_output" / f"scenario_{scenario}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(target_id): np.asarray(rows, dtype=float)
        for target_id, rows in raw["ground_truth"].items()
    }


def gnss_timeline(data: SimulationOutput) -> Tuple[List[Measurement], np.ndarray]:
    fixes = sorted(
        [m for m in data.measurements if m.sensor_id == "gnss"],
        key=lambda m: m.time,
    )
    return fixes, np.asarray([m.time for m in fixes], dtype=float)


def nearest_gnss(fixes: List[Measurement], times: np.ndarray, time_s: float) -> Measurement:
    if not fixes:
        raise ValueError("AIS processing needs GNSS fixes")
    idx = int(np.searchsorted(times, time_s))
    if idx <= 0:
        return fixes[0]
    if idx >= len(fixes):
        return fixes[-1]
    before, after = fixes[idx - 1], fixes[idx]
    return after if abs(after.time - time_s) < abs(time_s - before.time) else before


@dataclass
class SensorAdapter:
    """Adapter for radar/camera/AIS measurements used by T7."""

    data: SimulationOutput
    cfm: CoordinateFrameManager = field(default_factory=CoordinateFrameManager)

    def __post_init__(self) -> None:
        self.gnss_fixes, self.gnss_times = gnss_timeline(self.data)
        cfg = self.data.sensor_configs
        self.R_radar = build_R_radar(
            cfg["radar"]["sigma_r_m"],
            math.radians(cfg["radar"]["sigma_phi_deg"]),
        )
        self.R_camera = build_R_camera(math.radians(cfg["camera"]["sigma_phi_deg"]))
        self.R_camera_rb = np.diag(
            [
                cfg["camera"]["sigma_r_m"] ** 2,
                math.radians(cfg["camera"]["sigma_phi_deg"]) ** 2,
            ]
        )
        self.ais_sigma2 = (
            cfg["ais"]["sigma_pos_m"] ** 2 + cfg["gnss"]["sigma_pos_m"] ** 2
        )

    def update_ownship_for_time(self, time_s: float) -> Optional[Measurement]:
        if not self.gnss_fixes:
            return None
        fix = nearest_gnss(self.gnss_fixes, self.gnss_times, time_s)
        self.cfm.update_vessel_pos(fix)
        return fix

    def measurement_position(self, meas: Measurement) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate Cartesian position and covariance for track initiation."""
        if meas.sensor_id in ("radar", "camera"):
            origin = self.cfm.offsets.get(meas.sensor_id, np.zeros(2))
            pos = origin + rb_to_cart(float(meas.range_m), float(meas.bearing_rad))
            R_rb = self.R_radar if meas.sensor_id == "radar" else np.diag(
                [self.data.sensor_configs["camera"]["sigma_r_m"] ** 2, self.R_camera[0, 0]]
            )
            r = max(float(meas.range_m), 1e-9)
            b = float(meas.bearing_rad)
            J = np.array(
                [[math.cos(b), -r * math.sin(b)], [math.sin(b), r * math.cos(b)]],
                dtype=float,
            )
            return pos, J @ R_rb @ J.T

        if meas.sensor_id == "ais":
            pos = np.array([meas.north_m, meas.east_m], dtype=float)
            return pos, self.ais_sigma2 * np.eye(2)

        raise ValueError(f"Unsupported initiation sensor: {meas.sensor_id}")

    def measurement_info(self, track: "Track", meas: Measurement) -> Optional[Dict]:
        """Return measurement model pieces for one track/detection pair."""
        try:
            if meas.sensor_id == "radar":
                z = np.array([[meas.range_m], [meas.bearing_rad]], dtype=float)
                h, H = self.cfm.get_h_and_H(track.x, "radar")
                return {
                    "z": z,
                    "h": h,
                    "H": H,
                    "R": self.R_radar,
                    "angle_rows": (1,),
                    "dof": 2,
                    "gate": float(chi2.ppf(0.99, df=2)),
                }

            if meas.sensor_id == "camera":
                z = np.array([[meas.range_m], [meas.bearing_rad]], dtype=float)
                h_full, H_full = self.cfm.get_h_and_H(track.x, "camera")
                return {
                    "z": z,
                    "h": h_full,
                    "H": H_full,
                    "R": self.R_camera_rb,
                    "angle_rows": (1,),
                    "dof": 2,
                    "gate": float(chi2.ppf(0.99, df=2)),
                }

            if meas.sensor_id == "ais":
                fix = self.update_ownship_for_time(meas.time)
                if fix is None:
                    return None
                delta = np.array([meas.north_m - fix.north_m, meas.east_m - fix.east_m])
                z = cart_to_rb(delta).reshape(2, 1)
                J = rb_jacobian_position(delta)
                R = J @ (self.ais_sigma2 * np.eye(2)) @ J.T
                h, H = self.cfm.get_h_and_H(track.x, "ais")
                return {
                    "z": z,
                    "h": h,
                    "H": H,
                    "R": R,
                    "angle_rows": (1,),
                    "dof": 2,
                    "gate": float(chi2.ppf(0.99, df=2)),
                }

        except (TypeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            return None

        return None


@dataclass
class Track:
    """Single EKF track with the full T7 lifecycle."""

    id: int
    x: np.ndarray
    P: np.ndarray
    sigma_a: float
    born_at: float
    state: str = TENTATIVE
    hit_window: List[bool] = field(default_factory=lambda: [True])
    miss_streak: int = 0
    total_hits: int = 1
    first_detection_time: float = 0.0
    first_detection_pos: Optional[np.ndarray] = None
    history: List[Dict] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.state != DELETED

    @property
    def is_reportable(self) -> bool:
        return self.state in (CONFIRMED, COASTING)

    def predict(self, dt: float) -> None:
        F, Q = build_motion_model(dt, self.sigma_a)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def gate_and_nis(self, info: Dict) -> Tuple[float, np.ndarray, np.ndarray]:
        innov = info["z"] - info["h"]
        for row_i in info["angle_rows"]:
            innov[row_i, 0] = wrap(float(innov[row_i, 0]))
        S = info["H"] @ self.P @ info["H"].T + info["R"]
        nis = float((innov.T @ np.linalg.solve(S, innov)).item())
        return nis, innov, S

    def update(self, innov: np.ndarray, H: np.ndarray, R: np.ndarray, S: np.ndarray) -> None:
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        I = np.eye(4)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    def record_hit(
        self,
        time_s: float,
        detection_pos: np.ndarray,
        sensor: str,
        confirmation_m: int,
        confirmation_n: int,
    ) -> None:
        if self.total_hits == 1 and self.first_detection_pos is not None:
            dt = max(time_s - self.first_detection_time, 1e-6)
            velocity = (detection_pos - self.first_detection_pos) / dt
            self.x[2, 0] = velocity[0]
            self.x[3, 0] = velocity[1]
            self.P[2, 2] = min(self.P[2, 2], 25.0)
            self.P[3, 3] = min(self.P[3, 3], 25.0)

        self.total_hits += 1
        self.miss_streak = 0
        self.hit_window.append(True)
        self.hit_window = self.hit_window[-confirmation_n:]

        if self.state == TENTATIVE and sum(self.hit_window) >= confirmation_m:
            self.state = CONFIRMED
        elif self.state == COASTING:
            self.state = CONFIRMED

        self.log(time_s, event="hit", sensor=sensor)

    def record_miss(
        self,
        time_s: float,
        confirmation_n: int,
        tentative_delete_after: int,
        confirmed_delete_after: int,
    ) -> None:
        self.miss_streak += 1
        self.hit_window.append(False)
        self.hit_window = self.hit_window[-confirmation_n:]

        if self.state == TENTATIVE:
            if self.miss_streak >= tentative_delete_after:
                self.state = DELETED
        elif self.state in (CONFIRMED, COASTING):
            if self.miss_streak >= confirmed_delete_after:
                self.state = DELETED
            else:
                self.state = COASTING

        self.log(time_s, event="miss", sensor=None)

    def log(self, time_s: float, event: str, sensor: Optional[str]) -> None:
        self.history.append(
            {
                "t": float(time_s),
                "track_id": int(self.id),
                "N": float(self.x[0, 0]),
                "E": float(self.x[1, 0]),
                "vN": float(self.x[2, 0]),
                "vE": float(self.x[3, 0]),
                "state": self.state,
                "event": event,
                "sensor": sensor,
                "miss_streak": int(self.miss_streak),
            }
        )


def grouped_measurements(
    data: SimulationOutput, active_sensors: Iterable[str]
) -> List[Tuple[float, Dict[str, List[Measurement]]]]:
    allowed = set(active_sensors)
    groups: Dict[float, Dict[str, List[Measurement]]] = {}
    for meas in data.measurements:
        if meas.sensor_id in allowed:
            key = round(float(meas.time), 6)
            groups.setdefault(key, {}).setdefault(meas.sensor_id, []).append(meas)
    return [(time_s, groups[time_s]) for time_s in sorted(groups)]


def init_track_from_meas(
    track_id: int,
    meas: Measurement,
    adapter: SensorAdapter,
    sigma_a: float,
) -> Optional[Track]:
    try:
        pos, pos_cov = adapter.measurement_position(meas)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None

    x0 = np.array([[pos[0]], [pos[1]], [0.0], [0.0]], dtype=float)
    P0 = np.zeros((4, 4), dtype=float)
    P0[:2, :2] = pos_cov + 25.0 * np.eye(2)
    P0[2, 2] = 15.0**2
    P0[3, 3] = 15.0**2

    track = Track(
        id=track_id,
        x=x0,
        P=P0,
        sigma_a=sigma_a,
        born_at=float(meas.time),
        first_detection_time=float(meas.time),
        first_detection_pos=pos.copy(),
    )
    track.log(float(meas.time), event="init", sensor=meas.sensor_id)
    return track


def gnn_associate(
    tracks: List[Track],
    meas_list: List[Measurement],
    adapter: SensorAdapter,
) -> Tuple[Dict[int, int], List[int]]:
    """GNN association using NIS costs and chi-square gates."""
    if not tracks or not meas_list:
        return {}, list(range(len(meas_list)))

    cost = np.full((len(tracks), len(meas_list)), LARGE_COST)
    for ti, track in enumerate(tracks):
        for mi, meas in enumerate(meas_list):
            info = adapter.measurement_info(track, meas)
            if info is None:
                continue
            try:
                nis, _, _ = track.gate_and_nis(info)
            except np.linalg.LinAlgError:
                continue
            if nis <= info["gate"]:
                cost[ti, mi] = nis

    row_ind, col_ind = linear_sum_assignment(cost)
    assignment: Dict[int, int] = {}
    assigned_meas = set()
    for ti, mi in zip(row_ind, col_ind):
        if cost[ti, mi] < LARGE_COST:
            assignment[int(ti)] = int(mi)
            assigned_meas.add(int(mi))

    unmatched_meas = [mi for mi in range(len(meas_list)) if mi not in assigned_meas]
    return assignment, unmatched_meas


def merge_duplicate_tracks(tracks: List[Track], threshold: float) -> None:
    """Delete duplicate tentative/coasting tracks using Mahalanobis distance."""
    active = [track for track in tracks if track.is_active]
    for i, a in enumerate(active):
        if not a.is_active:
            continue
        for b in active[i + 1 :]:
            if not b.is_active:
                continue
            if a.is_reportable and b.is_reportable:
                continue
            dx = a.x[:2] - b.x[:2]
            P = a.P[:2, :2] + b.P[:2, :2]
            try:
                d2 = float((dx.T @ np.linalg.solve(P, dx)).item())
            except np.linalg.LinAlgError:
                continue
            if d2 <= threshold:
                keep, drop = sorted(
                    (a, b),
                    key=lambda tr: (tr.is_reportable, tr.total_hits, -tr.miss_streak),
                    reverse=True,
                )
                keep.total_hits = max(keep.total_hits, drop.total_hits)
                drop.state = DELETED


def run_tracker(
    data: SimulationOutput,
    active_sensors: Tuple[str, ...],
    sigma_a: float = 0.1,
    confirmation_m: int = 3,
    confirmation_n: int = 5,
    tentative_delete_after: int = 2,
    confirmed_delete_after: int = 5,
) -> Dict:
    adapter = SensorAdapter(data)
    scans = grouped_measurements(data, active_sensors)
    tracks: List[Track] = []
    nis_log: List[Dict] = []
    snapshots: List[Dict] = []
    next_track_id = 1
    last_t = 0.0

    for t, sensor_groups in scans:
        dt = t - last_t if last_t > 0.0 else 0.0
        last_t = t
        if dt > 0.0:
            for track in tracks:
                if track.is_active:
                    track.predict(dt)

        hit_positions: Dict[int, List[np.ndarray]] = {}
        hit_sensors: Dict[int, List[str]] = {}
        unmatched_by_sensor: Dict[str, List[int]] = {
            sensor: list(range(len(sensor_groups.get(sensor, []))))
            for sensor in active_sensors
        }

        for sensor in SENSOR_ORDER:
            if sensor not in active_sensors:
                continue
            meas_list = sensor_groups.get(sensor, [])
            if not meas_list:
                continue

            active_tracks = [track for track in tracks if track.is_active]
            assignment, unmatched = gnn_associate(active_tracks, meas_list, adapter)
            unmatched_by_sensor[sensor] = unmatched

            for ti, mi in assignment.items():
                track = active_tracks[ti]
                meas = meas_list[mi]
                info = adapter.measurement_info(track, meas)
                if info is None:
                    continue
                nis, innov, S = track.gate_and_nis(info)
                if nis > info["gate"]:
                    unmatched_by_sensor[sensor].append(mi)
                    continue
                track.update(innov, info["H"], info["R"], S)
                pos, _ = adapter.measurement_position(meas)
                hit_positions.setdefault(track.id, []).append(pos)
                hit_sensors.setdefault(track.id, []).append(sensor)
                nis_log.append(
                    {
                        "t": float(t),
                        "track_id": int(track.id),
                        "sensor": sensor,
                        "nis": float(nis),
                        "dof": int(info["dof"]),
                    }
                )

        hit_ids = set(hit_positions)
        miss_due_this_scan = "radar" in sensor_groups or (
            "radar" not in active_sensors and any(sensor_groups.values())
        )

        for track in [track for track in tracks if track.is_active]:
            if track.id in hit_ids:
                mean_pos = np.mean(hit_positions[track.id], axis=0)
                sensor_label = "+".join(sorted(set(hit_sensors[track.id])))
                track.record_hit(
                    t,
                    mean_pos,
                    sensor_label,
                    confirmation_m=confirmation_m,
                    confirmation_n=confirmation_n,
                )
            elif miss_due_this_scan:
                track.record_miss(
                    t,
                    confirmation_n=confirmation_n,
                    tentative_delete_after=tentative_delete_after,
                    confirmed_delete_after=confirmed_delete_after,
                )

        for sensor in active_sensors:
            meas_list = sensor_groups.get(sensor, [])
            for mi in sorted(set(unmatched_by_sensor.get(sensor, []))):
                track = init_track_from_meas(next_track_id, meas_list[mi], adapter, sigma_a)
                if track is not None:
                    tracks.append(track)
                    next_track_id += 1

        merge_duplicate_tracks(tracks, threshold=float(chi2.ppf(0.99, df=2)))

        confirmed = []
        for track in tracks:
            if track.is_reportable:
                confirmed.append(
                    {
                        "track_id": int(track.id),
                        "state": track.state,
                        "N": float(track.x[0, 0]),
                        "E": float(track.x[1, 0]),
                        "vN": float(track.x[2, 0]),
                        "vE": float(track.x[3, 0]),
                    }
                )
        snapshots.append({"t": float(t), "tracks": confirmed})

    return {
        "tracks": tracks,
        "snapshots": snapshots,
        "nis_log": nis_log,
        "num_spawned_tracks": next_track_id - 1,
        "num_deleted_tracks": sum(1 for track in tracks if track.state == DELETED),
    }


def active_truth_positions(
    truth_rows: Dict[int, np.ndarray], time_s: float
) -> Dict[int, np.ndarray]:
    truth = {}
    for target_id, rows in truth_rows.items():
        times = rows[:, 0]
        if time_s < times[0] or time_s > times[-1]:
            continue
        truth[target_id] = np.array(
            [
                np.interp(time_s, times, rows[:, 1]),
                np.interp(time_s, times, rows[:, 2]),
            ],
            dtype=float,
        )
    return truth


def min_distance_matches(
    tracks: Dict[int, np.ndarray],
    truth: Dict[int, np.ndarray],
    max_distance: float = 75.0,
) -> List[Tuple[int, int, float]]:
    track_ids = list(tracks)
    truth_ids = list(truth)
    pairs = []
    for ti, track_id in enumerate(track_ids):
        for gi, truth_id in enumerate(truth_ids):
            d = float(np.linalg.norm(tracks[track_id] - truth[truth_id]))
            if d <= max_distance:
                pairs.append((d, ti, gi))
    pairs.sort()

    used_tracks, used_truth = set(), set()
    matches = []
    for d, ti, gi in pairs:
        if ti in used_tracks or gi in used_truth:
            continue
        used_tracks.add(ti)
        used_truth.add(gi)
        matches.append((track_ids[ti], truth_ids[gi], d))
    return matches


def compute_metrics(data: SimulationOutput, result: Dict, scenario: str) -> Dict:
    truth_rows = raw_truth_rows(scenario)
    motp_series = []
    ce_series = []
    id_switches = 0
    last_truth_by_track: Dict[int, int] = {}

    for snapshot in result["snapshots"]:
        t = float(snapshot["t"])
        truth = active_truth_positions(truth_rows, t)
        tracks = {
            int(row["track_id"]): np.array([row["N"], row["E"]], dtype=float)
            for row in snapshot["tracks"]
        }
        matches = min_distance_matches(tracks, truth)
        motp = float(np.mean([d for _, _, d in matches])) if matches else None
        motp_series.append({"t": t, "motp": motp, "matches": len(matches)})
        ce_series.append(
            {
                "t": t,
                "ce": abs(len(tracks) - len(truth)),
                "confirmed_tracks": len(tracks),
                "active_truth": len(truth),
            }
        )
        for track_id, truth_id, _ in matches:
            old = last_truth_by_track.get(track_id)
            if old is not None and old != truth_id:
                id_switches += 1
            last_truth_by_track[track_id] = truth_id

    motp_values = [row["motp"] for row in motp_series if row["motp"] is not None]
    ce_values = [row["ce"] for row in ce_series]
    summary = {
        "scenario": data.scenario_name,
        "avg_motp_m": round(float(np.mean(motp_values)), 3) if motp_values else None,
        "avg_ce": round(float(np.mean(ce_values)), 3) if ce_values else None,
        "id_switches": int(id_switches),
        "num_spawned_tracks": int(result["num_spawned_tracks"]),
        "num_deleted_tracks": int(result["num_deleted_tracks"]),
        "final_confirmed_tracks": len(result["snapshots"][-1]["tracks"])
        if result["snapshots"]
        else 0,
    }
    return {"summary": summary, "motp_series": motp_series, "ce_series": ce_series}


def plot_results(
    data: SimulationOutput,
    result: Dict,
    metrics: Dict,
    scenario: str,
    plot_dir: Path,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    truth_rows = raw_truth_rows(scenario)

    fig, ax = plt.subplots(figsize=(9, 8))
    truth_e = []
    truth_n = []
    for target_id, rows in truth_rows.items():
        ax.plot(rows[:, 2], rows[:, 1], "k--", lw=1.2, alpha=0.55)
        ax.plot(rows[0, 2], rows[0, 1], "ks", ms=4)
        truth_e.extend(rows[:, 2].tolist())
        truth_n.extend(rows[:, 1].tolist())

    e_min, e_max = min(truth_e), max(truth_e)
    n_min, n_max = min(truth_n), max(truth_n)
    e_margin = max(100.0, 0.15 * (e_max - e_min))
    n_margin = max(100.0, 0.15 * (n_max - n_min))
    e_bounds = (e_min - e_margin, e_max + e_margin)
    n_bounds = (n_min - n_margin, n_max + n_margin)

    track_points: Dict[int, List[Tuple[float, float]]] = {}
    for snapshot in result["snapshots"]:
        for row in snapshot["tracks"]:
            track_points.setdefault(int(row["track_id"]), []).append((row["E"], row["N"]))

    cmap = plt.get_cmap("tab20")
    for i, (track_id, points) in enumerate(sorted(track_points.items())):
        pts = np.asarray(points, dtype=float)
        in_view = (
            (pts[:, 0] >= e_bounds[0])
            & (pts[:, 0] <= e_bounds[1])
            & (pts[:, 1] >= n_bounds[0])
            & (pts[:, 1] <= n_bounds[1])
        )
        pts = pts[in_view]
        if len(pts) < 2:
            continue
        ax.plot(pts[:, 0], pts[:, 1], lw=1.5, color=cmap(i % 20), label=f"Track {track_id}")

    ax.set_title(f"Scenario {scenario} - T7 confirmed/coasting tracks")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_xlim(*e_bounds)
    ax.set_ylim(*n_bounds)
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plot_dir / f"scenario_{scenario}_tracks.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    motp_t = [row["t"] for row in metrics["motp_series"]]
    motp = [np.nan if row["motp"] is None else row["motp"] for row in metrics["motp_series"]]
    ce_t = [row["t"] for row in metrics["ce_series"]]
    ce = [row["ce"] for row in metrics["ce_series"]]

    axes[0].plot(motp_t, motp, "b-", lw=1.5)
    axes[0].set_ylabel("MOTP [m]")
    axes[0].grid(True, alpha=0.25)
    axes[1].step(ce_t, ce, where="post", color="green")
    axes[1].set_ylabel("CE")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle(f"Scenario {scenario} - T7 MOTP and Cardinality Error")
    fig.tight_layout()
    fig.savefig(plot_dir / f"scenario_{scenario}_motp_ce.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def scenario_sensors(scenario: str) -> Tuple[str, ...]:
    if scenario == "D":
        return ("radar", "camera")
    if scenario == "E":
        return ("radar", "camera", "ais")
    raise ValueError("T7 validation is defined for scenarios D and E")


def run_scenario(args: argparse.Namespace, scenario: str) -> Dict:
    data = load_simulation_output(scenario)
    sensors = scenario_sensors(scenario)
    result = run_tracker(
        data,
        active_sensors=sensors,
        sigma_a=args.sigma_a,
        confirmation_m=args.confirmation_m,
        confirmation_n=args.confirmation_n,
        tentative_delete_after=args.tentative_delete_after,
        confirmed_delete_after=args.confirmed_delete_after,
    )
    metrics = compute_metrics(data, result, scenario)
    report = {
        "task": "T7 - Track management",
        "scenario": scenario,
        "sensors": list(sensors),
        "lifecycle": {
            "confirmation_m": args.confirmation_m,
            "confirmation_n": args.confirmation_n,
            "tentative_delete_after": args.tentative_delete_after,
            "confirmed_delete_after": args.confirmed_delete_after,
        },
        **metrics,
    }

    if not args.no_files:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"scenario_{scenario}_task7_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
    if not args.no_plots:
        plot_results(data, result, metrics, scenario, args.figure_dir)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 7 - Track management.")
    parser.add_argument(
        "--scenario",
        choices=("D", "E", "all"),
        default="all",
        help="Scenario to run. Default runs D and E.",
    )
    parser.add_argument("--sigma-a", type=float, default=0.1)
    parser.add_argument("--confirmation-m", type=int, default=3)
    parser.add_argument("--confirmation-n", type=int, default=5)
    parser.add_argument("--tentative-delete-after", type=int, default=2)
    parser.add_argument("--confirmed-delete-after", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=PLOT_DIR)
    parser.add_argument("--no-files", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = ("D", "E") if args.scenario == "all" else (args.scenario,)
    combined = {}
    for scenario in scenarios:
        print(f"\nRunning T7 on Scenario {scenario}...")
        report = run_scenario(args, scenario)
        combined[scenario] = report
        print(json.dumps(report["summary"], indent=2))

    if not args.no_files and len(combined) > 1:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "task7_summary.json").write_text(
            json.dumps(combined, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
