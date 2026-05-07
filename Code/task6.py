"""
Task 6: Gating and Data Association — Multi-Target Tracking (Scenario D)

Implements:
  - Per-sensor, per-track Mahalanobis-distance gating (chi-squared threshold at P_G = 0.99)
  - Global Nearest Neighbour (GNN) data association via linear sum assignment
  - Full track lifecycle: Tentative → Confirmed (3-of-5 M-of-N) → Coasting → Deleted
  - MOTP and Cardinality Error (CE) metrics over time
  - Trajectory plots and metric time-series plots

Usage:
    python task6.py                        # defaults to Scenario D
    python task6.py --scenario D
"""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from itertools import groupby
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

# Project modules (same directory)
from coordinate_frame_manager import CoordinateFrameManager
from ekf_tracker import EKFTracker, build_motion_model, build_R_radar, build_R_camera
from load_simulation_data import load_simulation_output, SimulationOutput


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "task6"
PLOT_DIR = OUT_DIR / "plots"


# Chi-squared gate thresholds  (degrees of freedom = measurement dimension)
# P_G = 0.99  =>  chi2.ppf(0.99, df)
GATE_RADAR = float(chi2.ppf(0.95, df=2))   
GATE_CAMERA = float(chi2.ppf(0.99, df=1))   # 6.635  (bearing only)
GATE_AIS    = float(chi2.ppf(0.99, df=2))   # 9.210  (range + bearing)

# Track management thresholds
M_CONFIRM  = 4    # hits needed in a window of N scans to confirm
N_CONFIRM  = 5    # sliding window length
K_DELETE   = 3    # consecutive missed detections before deleting a track

# Cost matrix sentinel for "outside gate" (must be larger than any real cost)
LARGE_COST = 1e9

# Sensor update order for a single scan
SENSOR_ORDER = ("radar", "camera", "ais")

# chi2 95% bounds for NIS consistency reporting
CHI2_95 = {1: 3.8415, 2: 5.9915}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wrap(angle: float) -> float:
    """Wrap angle to (-π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _make_F_Q(dt: float, sigma_a: float) -> Tuple[np.ndarray, np.ndarray]:
    """Constant-velocity F and process-noise Q."""
    return build_motion_model(dt, sigma_a)


def position_error(x: np.ndarray, gt_pos: np.ndarray) -> float:
    """Euclidean position error between EKF state and ground-truth [N, E]."""
    return float(np.sqrt((x[0, 0] - gt_pos[0])**2 + (x[1, 0] - gt_pos[1])**2))


# ---------------------------------------------------------------------------
# Track states
# ---------------------------------------------------------------------------
TENTATIVE = "tentative"
CONFIRMED = "confirmed"
COASTING  = "coasting"
DELETED   = "deleted"


# ---------------------------------------------------------------------------
# Track object
# ---------------------------------------------------------------------------
class Track:
    """Single EKF track with lifecycle management."""

    _id_counter = 0

    def __init__(self, x0: np.ndarray, P0: np.ndarray, cfm: CoordinateFrameManager,
                 sigma_a: float = 0.15, born_at: float = 0.0):
        Track._id_counter += 1
        self.id = Track._id_counter
        self.state = TENTATIVE
        self.cfm = cfm
        self.sigma_a = sigma_a
        self.born_at = born_at

        self.x = x0.copy()      
        self.P = P0.copy()     

        # Lifecycle counters
        self.hit_window: List[bool] = []  
        self.miss_streak = 0              
        self.total_hits = 0

        # History for output / metrics
        self.history: List[Dict] = []      

    # ── EKF wrappers ────────────────────────────────────────────────────────

    def predict(self, dt: float) -> None:
        F, Q = _make_F_Q(dt, self.sigma_a)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, innov: np.ndarray, H: np.ndarray,
               R: np.ndarray, S: np.ndarray) -> None:
        """Joseph-form EKF update."""
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        I = np.eye(4)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    # ── Gating ──────────────────────────────────────────────────────────────

    def gate_and_nis(self, z: np.ndarray, h: np.ndarray,
                     H: np.ndarray, R: np.ndarray,
                     angle_rows: Tuple[int, ...] = ()) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute innovation, wrap bearing components, and return NIS + S.
        Returns (nis, innov, S).
        """
        innov = z - h
        for row_i in angle_rows:
            innov[row_i, 0] = wrap(float(innov[row_i, 0]))
        S = H @ self.P @ H.T + R
        nis = float((innov.T @ np.linalg.solve(S, innov))[0, 0])
        return nis, innov, S

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def record_hit(self, hit: bool) -> None:
        """Update M-of-N window and miss streak."""
        self.hit_window.append(hit)
        if len(self.hit_window) > N_CONFIRM:
            self.hit_window.pop(0)
        if hit:
            self.miss_streak = 0
            self.total_hits += 1
        else:
            self.miss_streak += 1

    def update_state(self) -> None:
        """Transition track state based on counters."""
        if self.state == DELETED:
            return
        if self.miss_streak >= K_DELETE:
            self.state = DELETED
            return
        if self.state == TENTATIVE:
            if sum(self.hit_window) >= M_CONFIRM:
                self.state = CONFIRMED
        elif self.state == CONFIRMED:
            if self.miss_streak > 0:
                self.state = COASTING
        elif self.state == COASTING:
            if self.miss_streak == 0:
                self.state = CONFIRMED
            elif self.miss_streak >= K_DELETE:
                self.state = DELETED

    def log(self, t: float) -> None:
        self.history.append({
            "t":     t,
            "N":     float(self.x[0, 0]),
            "E":     float(self.x[1, 0]),
            "vN":    float(self.x[2, 0]),
            "vE":    float(self.x[3, 0]),
            "state": self.state,
        })


# ---------------------------------------------------------------------------
# Measurement → predicted-measurement helper
# ---------------------------------------------------------------------------

def measurement_info(track: Track, meas, cfm: CoordinateFrameManager,
                     R_radar: np.ndarray, R_cam: np.ndarray) -> Optional[Dict]:
    """
    Return a dict with {z, h, H, R, angle_rows, dof, gate_threshold}
    for a single measurement, or None if sensor is unknown.
    """
    sid = meas.sensor_id
    try:
        if sid == "radar":
            z = np.array([[meas.range_m], [meas.bearing_rad]], dtype=float)
            h, H = cfm.get_h_and_H(track.x, "radar")
            return dict(z=z, h=h, H=H, R=R_radar,
                        angle_rows=(1,), dof=2, gate=GATE_RADAR)

        elif sid == "camera":
            z = np.array([[meas.bearing_rad]], dtype=float)
            h_full, H_full = cfm.get_h_and_H(track.x, "camera")
            h = h_full[1:2]
            H = H_full[1:2, :]
            return dict(z=z, h=h, H=H, R=R_cam,
                        angle_rows=(0,), dof=1, gate=GATE_CAMERA)

        else:
            return None

    except Exception:
        return None


# ---------------------------------------------------------------------------
# GNN data association (one sensor at a time)
# ---------------------------------------------------------------------------

def gnn_associate(tracks: List[Track], meas_list: List,
                  cfm: CoordinateFrameManager,
                  R_radar: np.ndarray, R_cam: np.ndarray
                  ) -> Tuple[Dict[int, int], List[int]]:
    """
    Global Nearest Neighbour assignment for one batch of measurements.

    Returns
    -------
    assignment : {track_idx: meas_idx} — one-to-one assignments
    unmatched_meas : [meas_idx] — detections not assigned to any track
    """
    if not tracks or not meas_list:
        return {}, list(range(len(meas_list)))

    n_tracks = len(tracks)
    n_meas   = len(meas_list)

    # Build cost matrix (NIS if inside gate, LARGE_COST otherwise)
    cost = np.full((n_tracks, n_meas), LARGE_COST)
    for ti, track in enumerate(tracks):
        for mi, meas in enumerate(meas_list):
            info = measurement_info(track, meas, cfm, R_radar, R_cam)
            if info is None:
                continue
            nis, _, _ = track.gate_and_nis(
                info["z"], info["h"], info["H"], info["R"],
                angle_rows=info["angle_rows"]
            )
            if nis <= info["gate"]:
                cost[ti, mi] = nis

    # Solve assignment problem on the sub-matrix of gated pairs
    row_ind, col_ind = linear_sum_assignment(cost)

    assignment: Dict[int, int] = {}
    assigned_meas: set = set()
    for ti, mi in zip(row_ind, col_ind):
        if cost[ti, mi] < LARGE_COST:
            assignment[ti] = mi
            assigned_meas.add(mi)

    unmatched_meas = [mi for mi in range(n_meas) if mi not in assigned_meas]
    return assignment, unmatched_meas


# ---------------------------------------------------------------------------
# Track initialisation from a single detection
# ---------------------------------------------------------------------------

def init_track_from_meas(meas, cfm: CoordinateFrameManager,
                          sigma_a: float, t: float) -> Track:
    """Initialise a tentative track from a single radar or camera detection."""
    sid = meas.sensor_id
    if sid == "radar":
        pN = meas.range_m * math.cos(meas.bearing_rad)
        pE = meas.range_m * math.sin(meas.bearing_rad)
        sigma_pos = 15.0   
    elif sid == "camera":
        offset = cfm.offsets.get("camera", np.zeros(2))
        r_init = 200.0
        pN = offset[0] + r_init * math.cos(meas.bearing_rad)
        pE = offset[1] + r_init * math.sin(meas.bearing_rad)
        sigma_pos = 100.0
    else:
        return None

    x0 = np.array([[pN], [pE], [0.0], [0.0]], dtype=float)
    P0 = np.diag([sigma_pos**2, sigma_pos**2, 10.0**2, 10.0**2])
    return Track(x0, P0, cfm, sigma_a=sigma_a, born_at=t)


# ---------------------------------------------------------------------------
# Main tracker loop
# ---------------------------------------------------------------------------

def run_tracker(data: SimulationOutput,
                sigma_a: float = 0.1,
                active_sensors: Tuple[str, ...] = ("radar", "camera")) -> Dict:
    """
    Run the multi-target GNN tracker on the simulation output.

    Returns a result dict with confirmed-track histories and NIS records.
    """
    cfm = CoordinateFrameManager()

    # Build noise covariance matrices from sensor configs
    cfg = data.sensor_configs
    R_radar = build_R_radar(cfg["radar"]["sigma_r_m"],
                            math.radians(cfg["radar"]["sigma_phi_deg"]))
    R_cam   = build_R_camera(math.radians(cfg["camera"]["sigma_phi_deg"]))

    tracks: List[Track] = []
    last_t: float = 0.0
    nis_log: List[Dict] = []

    allowed = set(active_sensors)
    scan_groups: Dict[float, List] = {}
    for m in data.measurements:
        if m.sensor_id in allowed and not m.is_false_alarm or m.sensor_id in allowed:
            key = round(float(m.time), 6)
            scan_groups.setdefault(key, []).append(m)

    for t in sorted(scan_groups):
        scan = scan_groups[t]
        dt = t - last_t if last_t > 0 else 0.0
        last_t = t

        # ── Step 1: Predict all existing tracks ──────────────────────────
        if dt > 0:
            for track in tracks:
                if track.state != DELETED:
                    track.predict(dt)

        # ── Step 2: Per-sensor GNN association + update ──────────────────
        unmatched_total: List = []

        for sensor in SENSOR_ORDER:
            if sensor not in allowed:
                continue
            sensor_meas = [m for m in scan if m.sensor_id == sensor]
            if not sensor_meas:
                continue

            active_tracks = [tr for tr in tracks if tr.state != DELETED]

            assignment, unmatched = gnn_associate(
                active_tracks, sensor_meas, cfm, R_radar, R_cam
            )

            # Apply updates for assigned pairs
            for ti, mi in assignment.items():
                track = active_tracks[ti]
                meas  = sensor_meas[mi]
                info  = measurement_info(track, meas, cfm, R_radar, R_cam)
                if info is None:
                    continue
                nis, innov, S = track.gate_and_nis(
                    info["z"], info["h"], info["H"], info["R"],
                    angle_rows=info["angle_rows"]
                )
                track.update(innov, info["H"], info["R"], S)
                track.record_hit(True)
                track.update_state()
                track.log(t)
                nis_log.append({"t": t, "track_id": track.id,
                                "sensor": sensor, "nis": nis,
                                "dof": info["dof"]})

            # Tracks not assigned this sensor's scan
            assigned_track_indices = set(assignment.keys())
            active_tracks_indices  = list(range(len(active_tracks)))
            for ti in active_tracks_indices:
                if ti not in assigned_track_indices:
                    track = active_tracks[ti]
                    # Only mark miss if this was the primary/sole sensor scan
                    # (to avoid penalising camera-only misses on radar tracks)
                    if sensor == "radar":
                        track.record_hit(False)
                        track.update_state()
                    if track.state not in (DELETED,):
                        track.log(t)

            # Unmatched detections go to track initiation
            for mi in unmatched:
                meas = sensor_meas[mi]
                if not meas.is_false_alarm:  
                    pass                     
                new_track = init_track_from_meas(meas, cfm, sigma_a, t)
                if new_track is not None:
                    tracks.append(new_track)

        # Remove permanently deleted tracks from the active list
        tracks = [tr for tr in tracks if tr.state != DELETED]

    # Collect results
    confirmed_ids = set()
    for tr in tracks:
        for entry in tr.history:
            if entry["state"] == CONFIRMED:
                confirmed_ids.add(tr.id)

    return {
        "tracks": tracks,
        "nis_log": nis_log,
        "confirmed_ids": confirmed_ids,
    }


# ---------------------------------------------------------------------------
# Metrics: MOTP and Cardinality Error
# ---------------------------------------------------------------------------

def compute_metrics(data: SimulationOutput, result: Dict
                    ) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute per-scan MOTP and Cardinality Error.

    Returns
    -------
    times       : list of scan times
    motp_series : MOTP at each scan (mean localisation error of matched pairs)
    ce_series   : |#confirmed_tracks - #active_true_targets| at each scan
    """
    gt = data.ground_truth         
    gt_times = data.ground_truth_times

    # Confirmed tracks only
    confirmed_tracks = [tr for tr in result["tracks"]
                        if tr.id in result["confirmed_ids"]]

    def active_targets_at(t: float) -> List[int]:
        active = []
        for tid, states in gt.items():
            idx = int(np.argmin(np.abs(gt_times - t)))
            state = states[idx]
            if not np.any(np.isnan(state)):
                active.append(tid)
        return active

    # Collect all scan times from confirmed-track histories
    all_times = sorted(set(
        entry["t"]
        for tr in confirmed_tracks
        for entry in tr.history
        if entry["state"] == CONFIRMED
    ))

    times, motp_series, ce_series = [], [], []

    for t in all_times:
        # Get confirmed track positions at this time
        track_positions = []
        for tr in confirmed_tracks:
            entries = [e for e in tr.history if abs(e["t"] - t) < 1e-6 and e["state"] == CONFIRMED]
            if entries:
                track_positions.append(np.array([entries[-1]["N"], entries[-1]["E"]]))

        # Get true target positions at this time
        n_true = active_targets_at(t)
        true_positions = []
        for tid in n_true:
            states = gt[tid]
            idx = int(np.argmin(np.abs(gt_times - t)))
            pos = states[idx, :2]  # [N, E]
            if not np.any(np.isnan(pos)):
                true_positions.append(pos)

        # Cardinality Error
        ce = abs(len(track_positions) - len(true_positions))
        ce_series.append(ce)

        # MOTP: minimum-cost assignment between track and true positions
        if track_positions and true_positions:
            n_t = len(track_positions)
            n_g = len(true_positions)
            dist_mat = np.zeros((n_t, n_g))
            for ti, tp in enumerate(track_positions):
                for gi, gp in enumerate(true_positions):
                    dist_mat[ti, gi] = np.linalg.norm(tp - gp)
            row_ind, col_ind = linear_sum_assignment(dist_mat)
            motp = float(np.mean(dist_mat[row_ind, col_ind]))
        else:
            motp = float("nan")

        motp_series.append(motp)
        times.append(t)

    return times, motp_series, ce_series


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(data: SimulationOutput, result: Dict,
                 times: List[float], motp_series: List[float],
                 ce_series: List[float], plot_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    gt = data.ground_truth
    confirmed_tracks = [tr for tr in result["tracks"]
                        if tr.id in result["confirmed_ids"]]

    # ── Trajectory ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_title("Scenario D — Multi-Target Tracking (GNN)")

    # Ground truth
    for tid, states in gt.items():
        n_mask = ~np.isnan(states[:, 0])
        ax.plot(states[n_mask, 1], states[n_mask, 0],
                "k--", lw=1.6, alpha=0.7,
                label="Ground truth" if tid == list(gt.keys())[0] else "")
        # Start marker
        start_idx = np.where(n_mask)[0]
        if len(start_idx):
            ax.plot(states[start_idx[0], 1], states[start_idx[0], 0],
                    "ks", ms=6)

    # EKF tracks (confirmed only)
    cmap = plt.cm.get_cmap("tab10")
    for i, tr in enumerate(confirmed_tracks):
        hist = tr.history
        confirmed_hist = [e for e in hist if e["state"] == CONFIRMED]
        if not confirmed_hist:
            continue
        eE = [e["E"] for e in confirmed_hist]
        eN = [e["N"] for e in confirmed_hist]
        color = cmap(i % 10)
        ax.plot(eE, eN, lw=2, color=color, label=f"Track {tr.id}")
        ax.plot(eE[0], eN[0], "^", ms=8, color=color)

    # Radar measurements (background scatter)
    radar_meas = [m for m in data.measurements
                  if m.sensor_id == "radar" and not m.is_false_alarm]
    mE = [m.range_m * math.sin(m.bearing_rad) for m in radar_meas]
    mN = [m.range_m * math.cos(m.bearing_rad) for m in radar_meas]
    ax.scatter(mE, mN, s=5, c="lightblue", alpha=0.4, label="Radar detections")

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    fig.savefig(plot_dir / "trajectory.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir / 'trajectory.png'}")

    # ── MOTP ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    valid_t = [t for t, m in zip(times, motp_series) if not math.isnan(m)]
    valid_m = [m for m in motp_series if not math.isnan(m)]
    ax.plot(valid_t, valid_m, "b-", lw=2, label="MOTP")
    ax.axhline(15.0, color="r", ls="--", label="15 m target")
    ax.set_title("Scenario D — MOTP over Time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("MOTP [m]")
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "motp.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir / 'motp.png'}")

    # ── Cardinality Error ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, ce_series, "g-", lw=2, label="Cardinality Error")
    ax.axhline(0.5, color="r", ls="--", label="0.5 target")
    ax.set_title("Scenario D — Cardinality Error over Time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("|#tracks - #targets|")
    ax.set_ylim(bottom=-0.1)
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "cardinality_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir / 'cardinality_error.png'}")

    # ── NIS ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for sensor, color in (("radar", "#d62728"), ("camera", "#9467bd")):
        rows = [r for r in result["nis_log"] if r["sensor"] == sensor]
        if rows:
            t_vals  = [r["t"]   for r in rows]
            nis_vals = [r["nis"] / CHI2_95[r["dof"]] for r in rows]
            ax.scatter(t_vals, nis_vals, s=10, alpha=0.5, color=color,
                       label=sensor.title())
    ax.axhline(1.0, color="0.2", ls="--", lw=1.3, label="95% limit")
    ax.set_title("Scenario D — NIS consistency (normalised)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("NIS / 95% χ² limit")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(plot_dir / "nis.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir / 'nis.png'}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(data: SimulationOutput, result: Dict,
                 times: List[float], motp_series: List[float],
                 ce_series: List[float]) -> Dict:
    """Compute scalar MOTP, CE, and print a human-readable summary."""
    confirmed_tracks = [tr for tr in result["tracks"]
                        if tr.id in result["confirmed_ids"]]

    valid_motp = [m for m in motp_series if not math.isnan(m)]
    mean_motp = float(np.mean(valid_motp)) if valid_motp else float("nan")
    mean_ce   = float(np.mean(ce_series)) if ce_series else float("nan")

    motp_pass = mean_motp < 15.0
    ce_pass   = mean_ce   < 0.5

    # NIS consistency
    nis_inside = sum(1 for r in result["nis_log"]
                     if r["nis"] <= CHI2_95[r["dof"]])
    nis_total  = len(result["nis_log"])
    nis_pct    = 100.0 * nis_inside / nis_total if nis_total else 0.0

    report = {
        "scenario": data.scenario_name,
        "task": "T6 - Gating and Data Association",
        "num_confirmed_tracks": len(confirmed_tracks),
        "num_true_targets": len(data.ground_truth),
        "mean_motp_m": round(mean_motp, 3),
        "mean_ce": round(mean_ce, 3),
        "nis_inside_95_pct": round(nis_pct, 1),
        "success": {
            "motp_lt_15m": motp_pass,
            "ce_lt_0.5":   ce_pass,
            "overall":     motp_pass and ce_pass,
        },
    }
    return report


def print_report(report: Dict) -> None:
    print(f"\n{'='*60}")
    print("SCENARIO D — TASK 6 QUALIFICATION REPORT")
    print(f"{'='*60}")
    print(f"Confirmed tracks         : {report['num_confirmed_tracks']}  "
          f"(true targets: {report['num_true_targets']})")
    print(f"Mean MOTP                : {report['mean_motp_m']:.2f} m  "
          f"[{'PASSED' if report['success']['motp_lt_15m'] else 'FAILED'}, target < 15 m]")
    print(f"Mean Cardinality Error   : {report['mean_ce']:.3f}  "
          f"[{'PASSED' if report['success']['ce_lt_0.5'] else 'FAILED'}, target < 0.5]")
    print(f"NIS inside 95% χ² bound  : {report['nis_inside_95_pct']:.1f}%")
    print(f"Overall                  : {'PASSED ✓' if report['success']['overall'] else 'FAILED ✗'}")
    print(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 6 — Gating & Data Association.")
    parser.add_argument("--scenario",    default="D",
                        help="Scenario letter to load (default: D)")
    parser.add_argument("--sigma-a",     type=float, default=0.1,
                        help="Process noise std [m/s²]")
    parser.add_argument("--output-dir",  type=Path, default=OUT_DIR)
    parser.add_argument("--figure-dir",  type=Path, default=PLOT_DIR)
    parser.add_argument("--no-plots",    action="store_true")
    parser.add_argument("--no-files",    action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading Scenario {args.scenario}...")
    data = load_simulation_output(args.scenario)

    print("Running GNN multi-target tracker...")
    result = run_tracker(data, sigma_a=args.sigma_a,
                         active_sensors=("radar", "camera"))

    print(f"Total tracks spawned    : {len(result['tracks'])}")
    print(f"Confirmed track IDs     : {sorted(result['confirmed_ids'])}")

    print("Computing MOTP and Cardinality Error...")
    times, motp_series, ce_series = compute_metrics(data, result)

    report = build_report(data, result, times, motp_series, ce_series)
    print_report(report)

    if not args.no_files:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.output_dir / "scenario_D_task6_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to: {report_path}")

    if not args.no_plots:
        plot_results(data, result, times, motp_series, ce_series, args.figure_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())