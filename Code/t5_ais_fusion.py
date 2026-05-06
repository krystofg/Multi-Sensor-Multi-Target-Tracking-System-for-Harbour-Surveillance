"""Task 5: asynchronous AIS fusion layer.

This module is prepared for the future T4 output. It does not read simulator
scenario files by default. Once T4 is ready, pass the T4 radar/camera track file
plus AIS and GNSS measurement files to the CLI and this layer will add
asynchronous AIS EKF updates and write a fused track history for T6.

Expected future data flow:

    T4 radar+camera tracks + AIS measurements + GNSS fixes
        -> T5 asynchronous AIS fusion
        -> T5 fused track CSV/JSON for T6
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

CHI2_95_LOWER_2D = 0.103
CHI2_95_UPPER_2D = 5.9915
CHI2_99_2D = 9.2103


def wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def cart_to_rb(delta_ned: np.ndarray) -> np.ndarray:
    """Convert NED displacement [dN, dE] into [range, bearing]."""
    d_n, d_e = float(delta_ned[0]), float(delta_ned[1])
    return np.array([math.hypot(d_n, d_e), math.atan2(d_e, d_n)], dtype=float)


def polar_covariance_from_cartesian(
    delta_ned: np.ndarray,
    sigma_position_m: float,
) -> np.ndarray:
    """Transform isotropic NED position covariance into range/bearing units."""
    d_n, d_e = float(delta_ned[0]), float(delta_ned[1])
    r_sq = max(d_n * d_n + d_e * d_e, 1e-9)
    r = math.sqrt(r_sq)
    jacobian = np.array(
        [[d_n / r, d_e / r], [-d_e / r_sq, d_n / r_sq]],
        dtype=float,
    )
    return jacobian @ (sigma_position_m**2 * np.eye(2)) @ jacobian.T


@dataclass(frozen=True)
class T4TrackState:
    """One radar/camera state estimate produced by the future T4 module."""

    time_s: float
    track_id: int
    north_m: float
    east_m: float
    v_north_mps: float
    v_east_mps: float
    covariance: np.ndarray
    accepted: bool = True
    sensors: str = "radar+camera"


@dataclass(frozen=True)
class AISMeasurement:
    """One AIS target position report in NED coordinates."""

    time_s: float
    north_m: float
    east_m: float
    target_hint: str | None = None


@dataclass(frozen=True)
class GNSSFix:
    """One ownship GNSS fix in NED coordinates."""

    time_s: float
    north_m: float
    east_m: float


@dataclass
class FusedTrackState:
    """One output row for downstream tasks."""

    time_s: float
    track_id: int
    north_m: float
    east_m: float
    v_north_mps: float
    v_east_mps: float
    accepted: bool
    sensors: tuple[str, ...] = field(default_factory=tuple)
    source_update: str = "predict"
    nis: float | None = None
    nearest_gnss_time_s: float | None = None


@dataclass
class T5Config:
    """Noise and gating settings for the AIS fusion layer."""

    sigma_process_accel_mps2: float = 0.05
    sigma_ais_position_m: float = 4.0
    sigma_gnss_position_m: float = 2.0
    gate_limit: float = CHI2_99_2D
    default_position_variance_m2: float = 25.0
    default_velocity_variance_m2ps2: float = 225.0

    @property
    def combined_ais_gnss_sigma_m(self) -> float:
        return math.sqrt(self.sigma_ais_position_m**2 + self.sigma_gnss_position_m**2)


class AISFusionEKF:
    """Small EKF wrapper that adds AIS updates to T4 track states."""

    def __init__(self, initial: T4TrackState, config: T5Config):
        self.config = config
        self.track_id = initial.track_id
        self.time_s = initial.time_s
        self.x = np.array(
            [
                initial.north_m,
                initial.east_m,
                initial.v_north_mps,
                initial.v_east_mps,
            ],
            dtype=float,
        )
        self.P = np.asarray(initial.covariance, dtype=float)

    def predict_to(self, time_s: float) -> None:
        dt = float(time_s - self.time_s)
        if dt <= 0.0:
            self.time_s = max(self.time_s, time_s)
            return

        F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        q = self.config.sigma_process_accel_mps2**2
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
        self.time_s = time_s

    def assimilate_t4_state(self, state: T4TrackState) -> None:
        """Use a T4 radar/camera state as the current base estimate."""
        self.predict_to(state.time_s)
        self.x = np.array(
            [state.north_m, state.east_m, state.v_north_mps, state.v_east_mps],
            dtype=float,
        )
        self.P = np.asarray(state.covariance, dtype=float)
        self.time_s = state.time_s

    def h_and_H(self, sensor_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d_n = float(self.x[0] - sensor_pos[0])
        d_e = float(self.x[1] - sensor_pos[1])
        r_sq = max(d_n * d_n + d_e * d_e, 1e-9)
        r = math.sqrt(r_sq)

        h = np.array([r, math.atan2(d_e, d_n)], dtype=float)
        H = np.zeros((2, 4), dtype=float)
        H[0, 0] = d_n / r
        H[0, 1] = d_e / r
        H[1, 0] = -d_e / r_sq
        H[1, 1] = d_n / r_sq
        return h, H

    def update_ais(
        self,
        measurement: AISMeasurement,
        gnss_fix: GNSSFix,
    ) -> tuple[bool, float]:
        """Predict to AIS time and update with AIS converted to range/bearing."""
        self.predict_to(measurement.time_s)
        vessel_pos = np.array([gnss_fix.north_m, gnss_fix.east_m], dtype=float)
        target_pos = np.array([measurement.north_m, measurement.east_m], dtype=float)
        delta = target_pos - vessel_pos

        z = cart_to_rb(delta)
        R = polar_covariance_from_cartesian(
            delta,
            sigma_position_m=self.config.combined_ais_gnss_sigma_m,
        )
        h, H = self.h_and_H(vessel_pos)
        innovation = z - h
        innovation[1] = wrap_angle(float(innovation[1]))
        S = H @ self.P @ H.T + R
        nis = float(innovation.T @ np.linalg.solve(S, innovation))

        if nis > self.config.gate_limit:
            return False, nis

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        return True, nis

    def output_row(
        self,
        accepted: bool,
        sensors: Sequence[str],
        source_update: str,
        nis: float | None = None,
        nearest_gnss_time_s: float | None = None,
    ) -> FusedTrackState:
        return FusedTrackState(
            time_s=self.time_s,
            track_id=self.track_id,
            north_m=float(self.x[0]),
            east_m=float(self.x[1]),
            v_north_mps=float(self.x[2]),
            v_east_mps=float(self.x[3]),
            accepted=accepted,
            sensors=tuple(sensors),
            source_update=source_update,
            nis=nis,
            nearest_gnss_time_s=nearest_gnss_time_s,
        )


def nearest_gnss_fix(gnss_fixes: Sequence[GNSSFix], time_s: float) -> GNSSFix:
    if not gnss_fixes:
        raise ValueError("AIS fusion requires at least one GNSS fix")
    return min(gnss_fixes, key=lambda fix: abs(fix.time_s - time_s))


def fuse_t4_with_ais(
    t4_states: Sequence[T4TrackState],
    ais_measurements: Sequence[AISMeasurement],
    gnss_fixes: Sequence[GNSSFix],
    config: T5Config | None = None,
) -> list[FusedTrackState]:
    """Fuse future T4 radar/camera states with asynchronous AIS messages."""
    config = config or T5Config()
    if not t4_states:
        raise ValueError("T5 needs T4 states once T4 is ready")

    states_by_time = sorted(t4_states, key=lambda row: row.time_s)
    ais_by_time = sorted(ais_measurements, key=lambda row: row.time_s)
    events = [(row.time_s, "t4", row) for row in states_by_time] + [
        (row.time_s, "ais", row) for row in ais_by_time
    ]
    events.sort(key=lambda item: (item[0], 0 if item[1] == "t4" else 1))

    ekf = AISFusionEKF(states_by_time[0], config)
    output: list[FusedTrackState] = []
    started = False

    for _, event_type, payload in events:
        if event_type == "t4":
            state = payload
            assert isinstance(state, T4TrackState)
            ekf.assimilate_t4_state(state)
            started = True
            sensors = tuple(s for s in state.sensors.split("+") if s)
            output.append(
                ekf.output_row(
                    accepted=state.accepted,
                    sensors=sensors,
                    source_update="t4",
                )
            )
            continue

        if not started:
            continue
        measurement = payload
        assert isinstance(measurement, AISMeasurement)
        fix = nearest_gnss_fix(gnss_fixes, measurement.time_s)
        accepted, nis = ekf.update_ais(measurement, fix)
        output.append(
            ekf.output_row(
                accepted=accepted,
                sensors=("ais",) if accepted else (),
                source_update="ais",
                nis=nis,
                nearest_gnss_time_s=fix.time_s,
            )
        )

    return output


def covariance_from_row(row: dict, config: T5Config) -> np.ndarray:
    """Read a 4x4 covariance from CSV/JSON row fields, or make a conservative one."""
    keys = [f"cov_{i}{j}" for i in range(4) for j in range(4)]
    if all(key in row and row[key] not in ("", None) for key in keys):
        values = [float(row[key]) for key in keys]
        return np.asarray(values, dtype=float).reshape(4, 4)

    return np.diag(
        [
            config.default_position_variance_m2,
            config.default_position_variance_m2,
            config.default_velocity_variance_m2ps2,
            config.default_velocity_variance_m2ps2,
        ]
    )


def load_rows(path: Path) -> list[dict]:
    """Load CSV or JSON list-of-dicts input."""
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for key in ("rows", "tracks", "measurements", "gnss"):
                if key in raw and isinstance(raw[key], list):
                    return list(raw[key])
        if isinstance(raw, list):
            return list(raw)
        raise ValueError(f"JSON file {path} must contain a list of rows")

    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_t4_states(path: Path, config: T5Config) -> list[T4TrackState]:
    states = []
    for row in load_rows(path):
        states.append(
            T4TrackState(
                time_s=float(row["time_s"]),
                track_id=int(row.get("track_id", 1)),
                north_m=float(row["north_m"]),
                east_m=float(row["east_m"]),
                v_north_mps=float(row.get("v_north_mps", 0.0)),
                v_east_mps=float(row.get("v_east_mps", 0.0)),
                covariance=covariance_from_row(row, config),
                accepted=str(row.get("accepted", "1")).lower() not in ("0", "false"),
                sensors=str(row.get("sensors", "radar+camera")),
            )
        )
    return states


def load_ais_measurements(path: Path) -> list[AISMeasurement]:
    measurements = []
    for row in load_rows(path):
        measurements.append(
            AISMeasurement(
                time_s=float(row["time_s"]),
                north_m=float(row["north_m"]),
                east_m=float(row["east_m"]),
                target_hint=row.get("target_hint") or row.get("mmsi") or None,
            )
        )
    return measurements


def load_gnss_fixes(path: Path) -> list[GNSSFix]:
    fixes = []
    for row in load_rows(path):
        fixes.append(
            GNSSFix(
                time_s=float(row["time_s"]),
                north_m=float(row["north_m"]),
                east_m=float(row["east_m"]),
            )
        )
    return fixes


def write_fused_tracks(rows: Iterable[FusedTrackState], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "time_s",
        "track_id",
        "north_m",
        "east_m",
        "v_north_mps",
        "v_east_mps",
        "accepted",
        "sensors",
        "source_update",
        "nis",
        "nearest_gnss_time_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "time_s": f"{row.time_s:.6f}",
                    "track_id": row.track_id,
                    "north_m": f"{row.north_m:.6f}",
                    "east_m": f"{row.east_m:.6f}",
                    "v_north_mps": f"{row.v_north_mps:.6f}",
                    "v_east_mps": f"{row.v_east_mps:.6f}",
                    "accepted": int(row.accepted),
                    "sensors": "+".join(row.sensors),
                    "source_update": row.source_update,
                    "nis": "" if row.nis is None else f"{row.nis:.6f}",
                    "nearest_gnss_time_s": ""
                    if row.nearest_gnss_time_s is None
                    else f"{row.nearest_gnss_time_s:.6f}",
                }
            )


def write_report(rows: Sequence[FusedTrackState], path: Path) -> None:
    accepted = [row for row in rows if row.accepted]
    ais_updates = [row for row in accepted if "ais" in row.sensors]
    nis_values = [row.nis for row in ais_updates if row.nis is not None]
    nis_inside = [
        value
        for value in nis_values
        if CHI2_95_LOWER_2D <= value <= CHI2_95_UPPER_2D
    ]
    report = {
        "task": "T5 - asynchronous AIS fusion",
        "status": "completed_with_supplied_inputs",
        "rows": len(rows),
        "accepted_rows": len(accepted),
        "accepted_ais_updates": len(ais_updates),
        "ais_nis_inside_95_pct": None
        if not nis_values
        else round(100.0 * len(nis_inside) / len(nis_values), 2),
        "output_schema": [
            "time_s",
            "track_id",
            "north_m",
            "east_m",
            "v_north_mps",
            "v_east_mps",
            "accepted",
            "sensors",
            "source_update",
            "nis",
            "nearest_gnss_time_s",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def print_waiting_for_t4() -> None:
    print("T5 AIS fusion is ready, but no input files were supplied.")
    print("This is expected until T4 is finished.")
    print("")
    print("Later run it like:")
    print(
        "python Code/T5/t5_ais_fusion.py "
        "--t4-input Code/T4/outputs/t4_tracks.csv "
        "--ais-input <ais_measurements.csv> "
        "--gnss-input <gnss_fixes.csv>"
    )
    print("")
    print("No simulator scenario file is read by default.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add asynchronous AIS updates to the future T4 track output."
    )
    parser.add_argument("--t4-input", type=Path, help="Future T4 track CSV/JSON.")
    parser.add_argument("--ais-input", type=Path, help="AIS NED measurement CSV/JSON.")
    parser.add_argument("--gnss-input", type=Path, help="GNSS ownship CSV/JSON.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for T5 fused output files.",
    )
    parser.add_argument("--sigma-ais", type=float, default=4.0)
    parser.add_argument("--sigma-gnss", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = (args.t4_input, args.ais_input, args.gnss_input)
    if not any(required):
        print_waiting_for_t4()
        return 0
    if not all(required):
        raise SystemExit(
            "Provide all three inputs together: --t4-input, --ais-input, --gnss-input"
        )

    config = T5Config(
        sigma_ais_position_m=args.sigma_ais,
        sigma_gnss_position_m=args.sigma_gnss,
    )
    t4_states = load_t4_states(args.t4_input, config)
    ais_measurements = load_ais_measurements(args.ais_input)
    gnss_fixes = load_gnss_fixes(args.gnss_input)

    fused_rows = fuse_t4_with_ais(t4_states, ais_measurements, gnss_fixes, config)

    output_dir = args.output_dir.resolve()
    track_path = output_dir / "t5_fused_tracks.csv"
    report_path = output_dir / "t5_report.json"
    write_fused_tracks(fused_rows, track_path)
    write_report(fused_rows, report_path)

    print("T5 AIS fusion complete")
    print(f"Fused tracks: {track_path}")
    print(f"Report:       {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
