"""Task 5: asynchronous AIS fusion for Scenario C."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from coordinate_frame_manager import CoordinateFrameManager
from ekf_tracker import EKFTracker, build_motion_model, build_R_radar
from load_simulation_data import load_simulation_output


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "task5"
PLOT_DIR = OUT_DIR / "plots"

DROP_START = 60.0
DROP_END = 90.0
SENSOR_ORDER = {"radar": 0, "camera": 1, "ais": 2}
CHI2_95 = {1: 3.8415, 2: 5.9915}
CHI2_99 = {1: 6.6349, 2: 9.2103}


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def initial_state(meas, cfm):
    if meas.sensor_id == "ais":
        pos = np.array([meas.north_m, meas.east_m], dtype=float)
    else:
        origin = cfm.offsets.get(meas.sensor_id, np.zeros(2))
        pos = origin + np.array([meas.range_m * math.cos(meas.bearing_rad), meas.range_m * math.sin(meas.bearing_rad)])
    x0 = np.array([[pos[0]], [pos[1]], [0.0], [0.0]], dtype=float)
    P0 = np.diag([15.0**2, 15.0**2, 50.0**2, 50.0**2])
    return x0, P0


def first_true(data, sensor_id, target_id=0):
    for meas in data.measurements:
        if meas.sensor_id == sensor_id and not meas.is_false_alarm and meas.target_id in (target_id, -1):
            return meas
    raise RuntimeError(f"No usable {sensor_id} bootstrap measurement found")


def gnss_timeline(data):
    fixes = sorted([m for m in data.measurements if m.sensor_id == "gnss"], key=lambda m: m.time)
    return fixes, np.array([m.time for m in fixes], dtype=float)


def nearest_gnss(fixes, times, time_s):
    if not fixes:
        raise ValueError("AIS fusion needs GNSS fixes")
    idx = int(np.searchsorted(times, time_s))
    if idx <= 0:
        return fixes[0]
    if idx >= len(fixes):
        return fixes[-1]
    before, after = fixes[idx - 1], fixes[idx]
    return after if abs(after.time - time_s) < abs(time_s - before.time) else before


def sensor_noise(data):
    cfg = data.sensor_configs["radar"]
    return {
        "radar": build_R_radar(cfg["sigma_r_m"], math.radians(cfg["sigma_phi_deg"])),
        "camera": np.array([[math.radians(data.sensor_configs["camera"]["sigma_phi_deg"]) ** 2]]),
        "ais_sigma2": data.sensor_configs["ais"]["sigma_pos_m"] ** 2 + data.sensor_configs["gnss"]["sigma_pos_m"] ** 2,
    }


def ais_z_R(ais, gnss, sigma2):
    dN, dE = ais.north_m - gnss.north_m, ais.east_m - gnss.east_m
    r2 = max(dN * dN + dE * dE, 1e-9)
    r = math.sqrt(r2)
    z = np.array([[r], [math.atan2(dE, dN)]], dtype=float)
    J = np.array([[dN / r, dE / r], [-dE / r2, dN / r2]], dtype=float)
    return z, J @ (sigma2 * np.eye(2)) @ J.T


def prepare_update(tracker, meas, cfm, noise, gnss, gnss_times):
    try:
        if meas.sensor_id == "radar":
            z = np.array([[meas.range_m], [meas.bearing_rad]], dtype=float)
            h, H, R, dof, angle_i = *cfm.get_h_and_H(tracker.x, "radar"), noise["radar"], 2, 1
        elif meas.sensor_id == "camera":
            h_full, H_full = cfm.get_h_and_H(tracker.x, "camera")
            z = np.array([[meas.bearing_rad]], dtype=float)
            h, H, R, dof, angle_i = h_full[1:2], H_full[1:2], noise["camera"], 1, 0
        elif meas.sensor_id == "ais":
            fix = nearest_gnss(gnss, gnss_times, meas.time)
            cfm.update_vessel_pos(fix)
            z, R = ais_z_R(meas, fix, noise["ais_sigma2"])
            h, H, dof, angle_i = *cfm.get_h_and_H(tracker.x, "ais"), 2, 1
        else:
            return None

        innov = z - h
        innov[angle_i, 0] = wrap(float(innov[angle_i, 0]))
        S = H @ tracker.P @ H.T + R
        nis = float((innov.T @ np.linalg.solve(S, innov))[0, 0])
        return {"meas": meas, "innov": innov, "H": H, "R": R, "S": S, "nis": nis, "dof": dof}
    except (TypeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None


def apply_update(tracker, update):
    K = tracker.P @ update["H"].T @ np.linalg.inv(update["S"])
    tracker.x = tracker.x + K @ update["innov"]
    I = np.eye(4)
    IKH = I - K @ update["H"]
    tracker.P = IKH @ tracker.P @ IKH.T + K @ update["R"] @ K.T


def grouped_measurements(data, allowed_sensors):
    allowed_sensors = set(allowed_sensors)
    groups = {}
    for meas in data.measurements:
        if meas.sensor_id in allowed_sensors:
            groups.setdefault(round(float(meas.time), 6), []).append(meas)
    for time_s in sorted(groups):
        yield time_s, sorted(groups[time_s], key=lambda m: (SENSOR_ORDER.get(m.sensor_id, 99), m.is_false_alarm))


def best_update(tracker, measurements, cfm, noise, gnss, gnss_times):
    updates = [u for m in measurements if (u := prepare_update(tracker, m, cfm, noise, gnss, gnss_times))]
    return min(updates, key=lambda u: u["nis"], default=None)


def run_tracker(data, sensors, bootstrap_sensor, name, sigma_a=0.1):
    cfm = CoordinateFrameManager()
    gnss, gnss_times = gnss_timeline(data)
    noise = sensor_noise(data)
    bootstrap = first_true(data, bootstrap_sensor)
    tracker = EKFTracker(*initial_state(bootstrap, cfm), cfm=cfm, sigma_a=sigma_a)
    last_t = float(bootstrap.time)
    result = {
        "name": name,
        "history": [],
        "nis": [],
        "accepted_by_sensor": {sensor: 0 for sensor in sensors},
        "bootstrap_sensor": bootstrap.sensor_id,
        "bootstrap_time": float(bootstrap.time),
    }

    for time_s, scan in grouped_measurements(data, sensors):
        if time_s < bootstrap.time:
            continue
        dt = time_s - last_t
        if dt > 0.0:
            F, Q = build_motion_model(dt, sigma_a)
            tracker.x, tracker.P = tracker.predict(tracker.x, tracker.P, F, Q)
            last_t = time_s

        accepted = []
        for sensor in ("radar", "camera", "ais"):
            candidates = [m for m in scan if m.sensor_id == sensor and not (time_s == bootstrap.time and m is bootstrap)]
            if not candidates:
                continue
            update = best_update(tracker, candidates, cfm, noise, gnss, gnss_times)
            if update is None:
                continue
            ok = update["nis"] <= CHI2_99[update["dof"]]
            result["nis"].append({"t": time_s, "sensor": sensor, "nis": update["nis"], "dof": update["dof"], "accepted": ok})
            if ok:
                apply_update(tracker, update)
                accepted.append(sensor)
                result["accepted_by_sensor"][sensor] += 1

        result["history"].append(
            {
                "t": time_s,
                "N": float(tracker.x[0, 0]),
                "E": float(tracker.x[1, 0]),
                "vN": float(tracker.x[2, 0]),
                "vE": float(tracker.x[3, 0]),
                "accepted": bool(accepted),
                "sensors": "+".join(accepted),
                "run": name,
            }
        )
    return result


def run_all(data):
    return {
        "without_ais": run_tracker(data, ("radar", "camera"), "radar", "radar_camera_without_ais"),
        "with_ais": run_tracker(data, ("radar", "camera", "ais"), "radar", "radar_camera_with_ais"),
        "ais_only": run_tracker(data, ("ais",), "ais", "ais_only_initialisation"),
    }


def errors(data, result, start=0.0, end=None):
    gt, gt_t = data.ground_truth[0], data.ground_truth_times
    out = []
    for row in result["history"]:
        if row["t"] < start or (end is not None and row["t"] > end):
            continue
        truth = gt[int(np.argmin(np.abs(gt_t - row["t"])))]
        out.append((row["t"], math.hypot(row["N"] - truth[0], row["E"] - truth[1])))
    return out


def rmse(data, result, start=20.0, end=None):
    vals = [err for _, err in errors(data, result, start, end)]
    return float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")


def rmse_windows(data, result, windows):
    vals = [err for start, end in windows for _, err in errors(data, result, start, end)]
    return float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")


def nis_pct(result):
    accepted = [row for row in result["nis"] if row["accepted"]]
    if not accepted:
        return 0.0
    return 100.0 * sum(row["nis"] <= CHI2_95[row["dof"]] for row in accepted) / len(accepted)


def count_updates(result, sensor, start, end):
    return sum(1 for row in result["history"] if start <= row["t"] <= end and sensor in row["sensors"].split("+"))


def first_after(result, sensor, time_s):
    return next((row["t"] for row in result["history"] if row["t"] > time_s and sensor in row["sensors"].split("+")), None)


def summary(data, result):
    return {
        "rmse_after_20_m": rmse(data, result),
        "nis_inside_95_pct": nis_pct(result),
        "accepted_by_sensor": result["accepted_by_sensor"],
        "bootstrap_sensor": result["bootstrap_sensor"],
        "bootstrap_time_s": result["bootstrap_time"],
    }


def build_report(data, results):
    no_ais, with_ais, ais_only = results["without_ais"], results["with_ais"], results["ais_only"]
    ais_windows = [(20.0, DROP_START), (DROP_END, data.t_end)]
    no_ais_available = rmse_windows(data, no_ais, ais_windows)
    with_ais_available = rmse_windows(data, with_ais, ais_windows)
    dropout_rmse = rmse(data, with_ais, DROP_START, DROP_END)
    reacq_time = first_after(with_ais, "ais", DROP_END)
    ais_dropout = count_updates(with_ais, "ais", DROP_START, DROP_END)
    radar_camera_dropout = count_updates(with_ais, "radar", DROP_START, DROP_END) + count_updates(with_ais, "camera", DROP_START, DROP_END)
    ais_interval = data.sensor_configs["ais"]["interval_s"]
    survives_dropout = radar_camera_dropout > 0 and not math.isnan(dropout_rmse)
    improves_with_ais = with_ais_available < no_ais_available
    smooth_reacq = reacq_time is not None and reacq_time <= DROP_END + ais_interval + 1e-6
    return {
        "scenario": data.scenario_name,
        "task": "T5 - asynchronous AIS fusion",
        "dropout_window_s": [DROP_START, DROP_END],
        "ais_available_windows_s": ais_windows,
        "scenario_C_validation": {
            "track_survives_dropout": survives_dropout,
            "ais_improves_rmse_when_available": improves_with_ais,
            "smooth_reacquisition": smooth_reacq,
            "overall_pass": survives_dropout and improves_with_ais and smooth_reacq,
        },
        "without_ais": summary(data, no_ais) | {
            "rmse_ais_available_windows_m": no_ais_available,
        },
        "with_ais": summary(data, with_ais) | {
            "rmse_ais_available_windows_m": with_ais_available,
            "rmse_dropout_60_90_m": dropout_rmse,
            "ais_updates_during_dropout": ais_dropout,
            "radar_camera_updates_during_dropout": radar_camera_dropout,
            "first_ais_reacquisition_after_90_s": reacq_time,
        },
        "ais_only": summary(data, ais_only),
    }


def write_csv(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["t", "N", "E", "vN", "vE", "accepted", "sensors", "run"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["history"]:
            writer.writerow({key: row[key] if key in ("accepted", "sensors", "run") else f"{row[key]:.6f}" for key in fields})


def write_outputs(results, report, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in (
        ("without_ais", "scenario_C_without_ais.csv"),
        ("with_ais", "scenario_C_with_ais.csv"),
        ("ais_only", "scenario_C_ais_only.csv"),
    ):
        write_csv(results[key], out_dir / filename)
    (out_dir / "scenario_C_task5_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def plot_results(data, results, plot_dir):
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"without_ais": "#d62728", "with_ais": "#1f77b4", "ais_only": "#2ca02c"}
    labels = {"without_ais": "Radar + camera", "with_ais": "Radar + camera + AIS", "ais_only": "AIS only"}

    def finish(fig, ax, name, legend_size=None):
        ax.grid(True, alpha=0.2)
        legend_args = {"frameon": True, "framealpha": 0.9}
        if legend_size:
            legend_args["fontsize"] = legend_size
        ax.legend(**legend_args)
        fig.tight_layout()
        fig.savefig(plot_dir / name, dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    for key, result in results.items():
        err = errors(data, result)
        ax.plot([t for t, _ in err], [e for _, e in err], lw=2, color=colors[key], label=labels[key])
    ax.axvspan(DROP_START, DROP_END, color="0.85", alpha=0.35, label="AIS dropout")
    ax.set(title="Scenario C position error", xlabel="Time [s]", ylabel="Position error [m]")
    finish(fig, ax, "position_error.png")

    fig, ax = plt.subplots(figsize=(7, 7))
    gt = data.ground_truth[0]
    ax.plot(gt[:, 1], gt[:, 0], color="black", lw=2.2, label="Ground truth")
    for key, result in results.items():
        ax.plot([r["E"] for r in result["history"]], [r["N"] for r in result["history"]], lw=1.8, color=colors[key], label=labels[key])
    ax.set(title="Scenario C trajectory", xlabel="East [m]", ylabel="North [m]", aspect="equal")
    finish(fig, ax, "trajectory.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    sensor_colors = {"radar": "#d62728", "camera": "#9467bd", "ais": "#1f77b4"}
    for sensor, color in sensor_colors.items():
        rows = [r for result in results.values() for r in result["nis"] if r["sensor"] == sensor and r["accepted"]]
        if rows:
            ax.scatter(
                [r["t"] for r in rows],
                [r["nis"] / CHI2_95[r["dof"]] for r in rows],
                s=20,
                alpha=0.75,
                color=color,
                label=sensor.upper() if sensor == "ais" else sensor.title(),
            )
    ax.axhline(1.0, color="0.2", ls="--", lw=1.3, label="95% consistency limit")
    ax.set(title="Accepted update consistency", xlabel="Time [s]", ylabel="NIS / 95% limit")
    ax.set_ylim(bottom=0)
    finish(fig, ax, "nis.png", legend_size=8)


def print_report(report):
    a, b, c = report["without_ais"], report["with_ais"], report["ais_only"]
    v = report["scenario_C_validation"]
    print("\n" + "=" * 56)
    print("SCENARIO C - TASK 5 AIS FUSION")
    print("=" * 56)
    print(f"RMSE radar/camera      : {a['rmse_after_20_m']:.2f} m")
    print(f"RMSE radar/camera+AIS  : {b['rmse_after_20_m']:.2f} m")
    print(f"RMSE AIS-only          : {c['rmse_after_20_m']:.2f} m\n")
    print(f"AIS available RMSE     : without AIS {a['rmse_ais_available_windows_m']:.2f} m, with AIS {b['rmse_ais_available_windows_m']:.2f} m")
    print(f"AIS dropout 60-90 s    : {b['ais_updates_during_dropout']} AIS updates, {b['radar_camera_updates_during_dropout']} radar/camera updates")
    print(f"AIS reacquisition      : {b['first_ais_reacquisition_after_90_s']} s\n")
    print(f"NIS inside 95%         : without AIS {a['nis_inside_95_pct']:.1f}%, with AIS {b['nis_inside_95_pct']:.1f}%, AIS-only {c['nis_inside_95_pct']:.1f}%")
    print(f"Scenario C validation  : {'PASSED' if v['overall_pass'] else 'FAILED'}")
    print("=" * 56)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Task 5 AIS fusion.")
    parser.add_argument("--scenario", default="C")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=PLOT_DIR)
    parser.add_argument("--no-files", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_simulation_output(args.scenario)
    results = run_all(data)
    report = build_report(data, results)
    print_report(report)
    if not args.no_files:
        write_outputs(results, report, args.output_dir)
        print(f"Outputs written to {args.output_dir}")
    if not args.no_plots:
        plot_results(data, results, args.figure_dir)
        print(f"Figures written to {args.figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
