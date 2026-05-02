"""Generate result figures for the T2-T7 notebook/poster.

This script requires matplotlib.  It writes PNG files to
``tracking_solution/figures``.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

from harbour_tracking import (
    CoordinateFrameManager,
    ScenarioData,
    active_truth_positions,
    multi_target_metrics,
    rmse,
    run_multi_target_tracking,
    run_tracking,
)


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "harbour_sim_output"
OUT_DIR = ROOT / "tracking_solution" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def truth_xy(scenario: ScenarioData, target_id: int = 0):
    rows = scenario.ground_truth[target_id]
    return rows[:, 2], rows[:, 1]


def result_xy(result):
    east = [row.east for row in result.history]
    north = [row.north for row in result.history]
    return east, north


def raw_sensor_xy(scenario: ScenarioData, sensor_id: str, max_points: int = 1000):
    cfm = CoordinateFrameManager(scenario.sensor_configs)
    xs = []
    ys = []
    for measurement in scenario.measurements:
        if measurement.sensor_id != sensor_id:
            continue
        try:
            pos, _ = cfm.measurement_to_position(measurement, scenario)
        except (AssertionError, ValueError):
            continue
        ys.append(pos[0])
        xs.append(pos[1])
    if len(xs) > max_points:
        step = max(1, len(xs) // max_points)
        xs = xs[::step]
        ys = ys[::step]
    return xs, ys


def save_current(name: str) -> None:
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"wrote {path}")


def plot_t3_trajectory() -> None:
    scenario = ScenarioData.load(SCENARIO_DIR / "scenario_A.json")
    result = run_tracking(
        scenario,
        allowed_sensors=("radar",),
        fusion_mode="sequential",
        bootstrap_sensors=("radar",),
        name="T3 radar only",
    )
    plt.figure(figsize=(7, 6))
    raw_e, raw_n = raw_sensor_xy(scenario, "radar")
    plt.scatter(raw_e, raw_n, s=8, c="#8aa6c1", alpha=0.35, label="Raw radar")
    gt_e, gt_n = truth_xy(scenario)
    plt.plot(gt_e, gt_n, "k--", lw=2, label="Ground truth")
    est_e, est_n = result_xy(result)
    plt.plot(est_e, est_n, color="#d62728", lw=2, label="EKF track")
    plt.scatter([0], [0], marker="*", s=160, c="#f2c14e", edgecolor="k", label="Radar")
    plt.title("T3 Scenario A - Radar-only EKF")
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_current("t3_scenario_A_trajectory.png")


def plot_t4_comparison() -> None:
    scenario = ScenarioData.load(SCENARIO_DIR / "scenario_B.json")
    runs = {
        "Radar only": run_tracking(
            scenario,
            allowed_sensors=("radar",),
            fusion_mode="sequential",
            bootstrap_sensors=("radar",),
        ),
        "Sequential": run_tracking(
            scenario,
            allowed_sensors=("radar", "camera"),
            fusion_mode="sequential",
            bootstrap_sensors=("radar",),
        ),
        "Joint": run_tracking(
            scenario,
            allowed_sensors=("radar", "camera"),
            fusion_mode="joint",
            bootstrap_sensors=("radar",),
        ),
    }

    labels = list(runs)
    rmse_all = [rmse(runs[label], scenario, start_time=5.0) for label in labels]
    rmse_early = [
        rmse(runs[label], scenario, start_time=0.0, end_time=20.0) for label in labels
    ]
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(7, 4.6))
    plt.bar(x - width / 2, rmse_all, width, label="RMSE after 5 s", color="#4c78a8")
    plt.bar(x + width / 2, rmse_early, width, label="RMSE 0-20 s", color="#59a14f")
    plt.xticks(x, labels)
    plt.ylabel("Position RMSE [m]")
    plt.title("T4 Scenario B - Camera Fusion Benefit")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    save_current("t4_scenario_B_rmse_comparison.png")


def _time_error(result, scenario: ScenarioData, target_id: int = 0):
    times = []
    errors = []
    for row in result.history:
        truth = scenario.truth_at(row.time, target_id)
        times.append(row.time)
        errors.append(((row.north - truth[0]) ** 2 + (row.east - truth[1]) ** 2) ** 0.5)
    return np.asarray(times), np.asarray(errors)


def plot_t5_dropout() -> None:
    scenario = ScenarioData.load(SCENARIO_DIR / "scenario_C.json")
    no_ais = run_tracking(
        scenario,
        allowed_sensors=("radar", "camera"),
        fusion_mode="sequential",
        bootstrap_sensors=("radar",),
    )
    with_ais = run_tracking(
        scenario,
        allowed_sensors=("radar", "camera", "ais"),
        fusion_mode="sequential",
        bootstrap_sensors=("radar",),
    )
    t0, e0 = _time_error(no_ais, scenario)
    t1, e1 = _time_error(with_ais, scenario)
    plt.figure(figsize=(8, 4.8))
    plt.plot(t0, e0, color="#e15759", lw=1.8, label="Radar + camera")
    plt.plot(t1, e1, color="#4c78a8", lw=1.8, label="Radar + camera + AIS")
    plt.axvspan(60, 90, color="#f2c14e", alpha=0.25, label="AIS dropout")
    plt.xlabel("Time [s]")
    plt.ylabel("Position error [m]")
    plt.title("T5 Scenario C - AIS Dropout and Re-acquisition")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_current("t5_scenario_C_dropout_error.png")


def _plot_metric_pair(metrics: dict, title: str, filename: str) -> None:
    motp_t = [row["time"] for row in metrics["motp_series"]]
    motp = [row["motp"] for row in metrics["motp_series"]]
    ce_t = [row["time"] for row in metrics["ce_series"]]
    ce = [row["ce"] for row in metrics["ce_series"]]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(motp_t, motp, color="#4c78a8", lw=1.5)
    axes[0].set_ylabel("MOTP [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)
    axes[1].step(ce_t, ce, where="post", color="#e15759", lw=1.5)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("CE")
    axes[1].grid(True, alpha=0.3)
    save_current(filename)


def _plot_multi_tracks(
    scenario: ScenarioData,
    result,
    title: str,
    filename: str,
) -> None:
    plt.figure(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(scenario.ground_truth))))
    for idx, target_id in enumerate(sorted(scenario.ground_truth)):
        gt_e, gt_n = truth_xy(scenario, target_id)
        plt.plot(gt_e, gt_n, "--", lw=2, color=colors[idx], label=f"Truth {target_id}")

    confirmed_tracks = [track for track in result.tracks if track.total_hits >= 4]
    for track in confirmed_tracks:
        rows = [
            item
            for item in track.history
            if item["status"] in ("confirmed", "coasting") and item["event"] != "miss"
        ]
        if len(rows) < 2:
            continue
        east = [item["east"] for item in rows]
        north = [item["north"] for item in rows]
        plt.plot(east, north, lw=1.4, alpha=0.9)

    plt.title(title)
    plt.xlabel("East [m]")
    plt.ylabel("North [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    save_current(filename)


def plot_t6_t7() -> None:
    lifecycle = {
        "confirmation_m": 4,
        "confirmation_n": 6,
        "tentative_delete_after": 2,
        "confirmed_delete_after": 5,
    }
    scenario_d = ScenarioData.load(SCENARIO_DIR / "scenario_D.json")
    result_d = run_multi_target_tracking(
        scenario_d,
        allowed_sensors=("radar", "camera"),
        name="T6/T7 Scenario D",
        **lifecycle,
    )
    metrics_d = multi_target_metrics(result_d, scenario_d, start_time=0.0)
    _plot_metric_pair(metrics_d, "T6 Scenario D - MOTP and Cardinality Error", "t6_scenario_D_motp_ce.png")
    _plot_multi_tracks(
        scenario_d,
        result_d,
        "T6 Scenario D - Confirmed Tracks and Ground Truth",
        "t6_scenario_D_tracks.png",
    )

    scenario_e = ScenarioData.load(SCENARIO_DIR / "scenario_E.json")
    result_e = run_multi_target_tracking(
        scenario_e,
        allowed_sensors=("radar", "camera", "ais"),
        name="T6/T7 Scenario E",
        **lifecycle,
    )
    metrics_e = multi_target_metrics(result_e, scenario_e, start_time=0.0)
    _plot_metric_pair(metrics_e, "T7 Scenario E - MOTP and Cardinality Error", "t7_scenario_E_motp_ce.png")
    _plot_multi_tracks(
        scenario_e,
        result_e,
        "T7 Scenario E - Confirmed Tracks and Ground Truth",
        "t7_scenario_E_tracks.png",
    )


def main() -> int:
    plot_t3_trajectory()
    plot_t4_comparison()
    plot_t5_dropout()
    plot_t6_t7()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
