import csv
import contextlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# coordinate_frame_manager imports matplotlib for old plotting experiments, but
# T5 does not need it. Keep the shared file untouched and provide a tiny import
# stub when matplotlib is not installed in the current runtime.
try:
    import matplotlib.pyplot  # noqa: F401
except ModuleNotFoundError:
    sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
    sys.modules.setdefault("matplotlib.pyplot", types.ModuleType("matplotlib.pyplot"))

with contextlib.redirect_stdout(io.StringIO()):
    from coordinate_frame_manager import CoordinateFrameManager
from ekf_tracker import EKFTracker
from harbour_sim_output.load_simulation_data import Measurement, load_simulation_output


# =============================================================================
# T5 PROTOTYPE: FUSION STEP 2 - ADD AIS
# =============================================================================
#
# This file is intentionally a prototype that mixes the previous task pieces.
# Radar/camera EKF logic is reused from EKFTracker and CoordinateFrameManager.
# The new T5-specific logic is:
#   1. time-ordered asynchronous update loop,
#   2. AIS NED position -> implied range/bearing relative to nearest GNSS,
#   3. AIS-only initialisation,
#   4. dropout/reacquisition reporting for Scenario C.
#
# Later, once T4 is stable, the bootstrap and radar/camera parts can be replaced
# by the real T4 output while keeping the AIS helper functions below.


CHI2_GATE_99 = 9.2103
WIDE_BOOTSTRAP_GATE = 50.0
AIS_DROPOUT_START = 60.0
AIS_DROPOUT_END = 90.0


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def range_bearing_to_ned(range_m, bearing_rad, sensor_pos):
    delta = np.array(
        [
            range_m * np.cos(bearing_rad),
            range_m * np.sin(bearing_rad),
        ]
    )
    return sensor_pos + delta


def measurement_to_ned_position(measurement, cfm):
    """Convert one radar/camera/AIS measurement into a rough NED position."""
    if measurement.sensor_id in ("radar", "camera"):
        sensor_pos = cfm.offsets[measurement.sensor_id]
        return range_bearing_to_ned(
            measurement.range_m,
            measurement.bearing_rad,
            sensor_pos,
        )

    if measurement.sensor_id == "ais":
        return np.array([measurement.north_m, measurement.east_m], dtype=float)

    raise ValueError(f"Cannot initialise from sensor {measurement.sensor_id}")


def make_tracker_from_measurement(measurement, cfm):
    """Prototype track initialisation from a single measurement."""
    pos = measurement_to_ned_position(measurement, cfm)
    x0 = np.array([[pos[0]], [pos[1]], [0.0], [0.0]], dtype=float)
    P0 = np.diag([225.0, 225.0, 225.0, 225.0])
    return EKFTracker(x0=x0, P0=P0, cfm=cfm)


def build_gnss_timeline(measurements):
    gnss = sorted(
        [m for m in measurements if m.sensor_id == "gnss"],
        key=lambda m: m.time,
    )
    times = np.array([m.time for m in gnss], dtype=float)
    return gnss, times


def nearest_gnss(gnss_measurements, gnss_times, time_s):
    """Nearest GNSS lookup for AIS reference position."""
    if len(gnss_measurements) == 0:
        raise ValueError("AIS update requires GNSS measurements")

    idx = int(np.searchsorted(gnss_times, time_s))
    if idx <= 0:
        return gnss_measurements[0]
    if idx >= len(gnss_measurements):
        return gnss_measurements[-1]

    before = gnss_measurements[idx - 1]
    after = gnss_measurements[idx]
    if abs(after.time - time_s) < abs(time_s - before.time):
        return after
    return before


def ais_cartesian_to_polar_covariance(ais_measurement, gnss_measurement):
    """Convert AIS/GNSS Cartesian position noise into range/bearing noise."""
    dN = ais_measurement.north_m - gnss_measurement.north_m
    dE = ais_measurement.east_m - gnss_measurement.east_m
    r_sq = max(dN**2 + dE**2, 1e-9)
    r = np.sqrt(r_sq)

    J = np.array(
        [
            [dN / r, dE / r],
            [-dE / r_sq, dN / r_sq],
        ]
    )

    # Project spec: AIS sigma = 4 m, GNSS sigma = 2 m.
    R_ne = (4.0**2 + 2.0**2) * np.eye(2)
    return J @ R_ne @ J.T


def ais_to_range_bearing_measurement(ais_measurement, gnss_measurement):
    """Create a synthetic range/bearing measurement for EKFTracker.update()."""
    dN = ais_measurement.north_m - gnss_measurement.north_m
    dE = ais_measurement.east_m - gnss_measurement.east_m
    range_m = float(np.sqrt(dN**2 + dE**2))
    bearing_rad = float(np.arctan2(dE, dN))

    return Measurement(
        sensor_id="ais",
        time=ais_measurement.time,
        is_false_alarm=ais_measurement.is_false_alarm,
        target_id=ais_measurement.target_id,
        range_m=range_m,
        bearing_rad=bearing_rad,
        north_m=ais_measurement.north_m,
        east_m=ais_measurement.east_m,
    )


def nis_for_measurement(tracker, measurement, R_override=None):
    """Compute NIS without changing the tracker."""
    hx, H = tracker.cfm.get_h_and_H(tracker.x, measurement.sensor_id)
    R = R_override
    if R is None:
        R = tracker.cfm.R_specs.get(measurement.sensor_id, tracker.cfm.R_specs["radar"])

    z = np.array([[measurement.range_m], [measurement.bearing_rad]])
    y = z - hx
    y[1, 0] = wrap_angle(y[1, 0])

    S = H @ tracker.P @ H.T + R
    return float((y.T @ np.linalg.solve(S, y))[0, 0])


def update_with_optional_R(tracker, measurement, gate_limit, R_override=None):
    """Use EKFTracker.update(), optionally with a temporary sensor covariance."""
    if R_override is None:
        return tracker.update(measurement, gate_limit=gate_limit)

    old_R = tracker.cfm.R_specs.get(measurement.sensor_id)
    tracker.cfm.R_specs[measurement.sensor_id] = R_override
    try:
        return tracker.update(measurement, gate_limit=gate_limit)
    finally:
        if old_R is None:
            tracker.cfm.R_specs.pop(measurement.sensor_id, None)
        else:
            tracker.cfm.R_specs[measurement.sensor_id] = old_R


def prepare_measurement_for_update(measurement, tracker, gnss_measurements, gnss_times):
    """Return EKF-ready measurement plus covariance override and helper metadata."""
    if measurement.sensor_id != "ais":
        return measurement, None, None

    gnss = nearest_gnss(gnss_measurements, gnss_times, measurement.time)
    tracker.cfm.update_vessel_pos(gnss)

    ais_rb = ais_to_range_bearing_measurement(measurement, gnss)
    R_ais_rb = ais_cartesian_to_polar_covariance(measurement, gnss)
    return ais_rb, R_ais_rb, gnss


def grouped_measurements_by_time(data, allowed_sensors):
    """Create one asynchronous update queue from selected sensor streams."""
    allowed = set(allowed_sensors)
    buckets = {}
    for measurement in data.measurements:
        if measurement.sensor_id not in allowed:
            continue
        key = round(measurement.time, 6)
        buckets.setdefault(key, []).append(measurement)

    for time_s in sorted(buckets):
        order = {"radar": 0, "camera": 1, "ais": 2}
        yield time_s, sorted(buckets[time_s], key=lambda m: order.get(m.sensor_id, 99))


def choose_bootstrap_measurement(data, cfm, bootstrap_sensor):
    """Prototype bootstrap: use first non-false-alarm measurement from a sensor."""
    for measurement in data.measurements:
        if measurement.sensor_id != bootstrap_sensor:
            continue
        if measurement.is_false_alarm:
            continue
        return measurement
    raise RuntimeError(f"No bootstrap measurement found for {bootstrap_sensor}")


def select_best_measurement(tracker, measurements, gnss_measurements, gnss_times, gate):
    """Nearest-neighbour selection for this single-target T5 prototype."""
    best = None

    for raw_m in measurements:
        try:
            m, R_override, gnss = prepare_measurement_for_update(
                raw_m,
                tracker,
                gnss_measurements,
                gnss_times,
            )
            nis = nis_for_measurement(tracker, m, R_override)
        except (ValueError, np.linalg.LinAlgError):
            continue

        if best is None or nis < best.nis:
            best = SimpleNamespace(
                raw=raw_m,
                measurement=m,
                R_override=R_override,
                gnss=gnss,
                nis=nis,
            )

    if best is None or best.nis > gate:
        return None
    return best


def run_t5_prototype(data, allowed_sensors, bootstrap_sensor, run_name):
    """Run the single-target asynchronous T5 prototype."""
    cfm = CoordinateFrameManager()
    gnss_measurements, gnss_times = build_gnss_timeline(data.measurements)

    bootstrap = choose_bootstrap_measurement(data, cfm, bootstrap_sensor)
    if bootstrap.sensor_id == "ais":
        gnss = nearest_gnss(gnss_measurements, gnss_times, bootstrap.time)
        cfm.update_vessel_pos(gnss)

    tracker = make_tracker_from_measurement(bootstrap, cfm)
    last_t = bootstrap.time
    history = []
    nis_history = []
    accepted_by_sensor = {"radar": 0, "camera": 0, "ais": 0}

    for time_s, scan_measurements in grouped_measurements_by_time(data, allowed_sensors):
        if time_s < bootstrap.time:
            continue

        tracker.predict(time_s - last_t)
        last_t = time_s

        accepted_sensors = []
        for sensor_id in ("radar", "camera", "ais"):
            sensor_measurements = [
                m for m in scan_measurements if m.sensor_id == sensor_id
            ]
            if not sensor_measurements:
                continue

            gate = WIDE_BOOTSTRAP_GATE if time_s <= bootstrap.time + 10.0 else CHI2_GATE_99
            selected = select_best_measurement(
                tracker,
                sensor_measurements,
                gnss_measurements,
                gnss_times,
                gate,
            )
            if selected is None:
                continue

            accepted, nis = update_with_optional_R(
                tracker,
                selected.measurement,
                gate,
                selected.R_override,
            )
            nis_history.append(
                {
                    "t": time_s,
                    "sensor": sensor_id,
                    "nis": nis,
                    "accepted": accepted,
                }
            )

            if accepted:
                accepted_sensors.append(sensor_id)
                accepted_by_sensor[sensor_id] += 1

        history.append(
            {
                "t": time_s,
                "N": float(tracker.x[0, 0]),
                "E": float(tracker.x[1, 0]),
                "vN": float(tracker.x[2, 0]),
                "vE": float(tracker.x[3, 0]),
                "accepted": bool(accepted_sensors),
                "sensors": "+".join(accepted_sensors),
                "run": run_name,
            }
        )

    return {
        "name": run_name,
        "allowed_sensors": list(allowed_sensors),
        "bootstrap_sensor": bootstrap.sensor_id,
        "bootstrap_time": bootstrap.time,
        "history": history,
        "nis": nis_history,
        "accepted_by_sensor": accepted_by_sensor,
    }


def rmse_after(data, result, start_time=20.0, end_time=None, target_id=0):
    gt = data.ground_truth[target_id]
    gt_t = data.ground_truth_times

    errors = []
    for row in result["history"]:
        if row["t"] < start_time:
            continue
        if end_time is not None and row["t"] > end_time:
            continue

        idx = int(np.argmin(np.abs(gt_t - row["t"])))
        errors.append((row["N"] - gt[idx, 0])**2 + (row["E"] - gt[idx, 1])**2)

    return float(np.sqrt(np.mean(errors))) if errors else float("nan")


def first_accepted_ais_after(result, time_s):
    for row in result["history"]:
        if row["t"] > time_s and "ais" in row["sensors"].split("+"):
            return row["t"]
    return None


def count_updates_in_window(result, sensor_id, start_s, end_s):
    count = 0
    for row in result["history"]:
        if start_s <= row["t"] <= end_s and sensor_id in row["sensors"].split("+"):
            count += 1
    return count


def write_history_csv(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "t",
                "N",
                "E",
                "vN",
                "vE",
                "accepted",
                "sensors",
                "run",
            ],
        )
        writer.writeheader()
        writer.writerows(result["history"])


def main():
    data = load_simulation_output("C")

    without_ais = run_t5_prototype(
        data,
        allowed_sensors=("radar", "camera"),
        bootstrap_sensor="radar",
        run_name="radar_camera_without_ais",
    )

    with_ais = run_t5_prototype(
        data,
        allowed_sensors=("radar", "camera", "ais"),
        bootstrap_sensor="radar",
        run_name="radar_camera_with_ais",
    )

    ais_only = run_t5_prototype(
        data,
        allowed_sensors=("ais",),
        bootstrap_sensor="ais",
        run_name="ais_only_initialisation",
    )

    out_dir = REPO_ROOT / "outputs" / "task5"
    write_history_csv(without_ais, out_dir / "scenario_C_without_ais.csv")
    write_history_csv(with_ais, out_dir / "scenario_C_with_ais.csv")
    write_history_csv(ais_only, out_dir / "scenario_C_ais_only.csv")

    report = {
        "scenario": "C",
        "status": "prototype",
        "without_ais": {
            "rmse_after_20_m": rmse_after(data, without_ais, start_time=20.0),
            "accepted_by_sensor": without_ais["accepted_by_sensor"],
        },
        "with_ais": {
            "rmse_after_20_m": rmse_after(data, with_ais, start_time=20.0),
            "rmse_dropout_60_90_m": rmse_after(
                data,
                with_ais,
                start_time=AIS_DROPOUT_START,
                end_time=AIS_DROPOUT_END,
            ),
            "accepted_by_sensor": with_ais["accepted_by_sensor"],
            "ais_updates_during_dropout": count_updates_in_window(
                with_ais,
                "ais",
                AIS_DROPOUT_START,
                AIS_DROPOUT_END,
            ),
            "radar_camera_updates_during_dropout": count_updates_in_window(
                with_ais,
                "radar",
                AIS_DROPOUT_START,
                AIS_DROPOUT_END,
            )
            + count_updates_in_window(
                with_ais,
                "camera",
                AIS_DROPOUT_START,
                AIS_DROPOUT_END,
            ),
            "first_ais_reacquisition_after_90_s": first_accepted_ais_after(
                with_ais,
                AIS_DROPOUT_END,
            ),
        },
        "ais_only": {
            "bootstrap_sensor": ais_only["bootstrap_sensor"],
            "rmse_after_20_m": rmse_after(data, ais_only, start_time=20.0),
            "accepted_by_sensor": ais_only["accepted_by_sensor"],
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scenario_C_task5_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 48)
    print("SCENARIO C - TASK 5 AIS FUSION PROTOTYPE")
    print("=" * 48)
    print(f"RMSE without AIS  : {report['without_ais']['rmse_after_20_m']:.2f} m")
    print(f"RMSE with AIS     : {report['with_ais']['rmse_after_20_m']:.2f} m")
    print(
        "AIS dropout 60-90 : "
        f"{report['with_ais']['ais_updates_during_dropout']} AIS updates, "
        f"{report['with_ais']['radar_camera_updates_during_dropout']} radar/camera updates"
    )
    print(
        "AIS reacquisition : "
        f"{report['with_ais']['first_ais_reacquisition_after_90_s']} s"
    )
    print(f"Outputs written to {out_dir}")
    print("=" * 48)


if __name__ == "__main__":
    main()
