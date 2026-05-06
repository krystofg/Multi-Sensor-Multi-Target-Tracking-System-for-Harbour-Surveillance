"""Run the standalone T2-T5 validation workflow.

Usage from the repository root:

    python tracking_solution/run_t2_t5.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from harbour_tracking import (
    CHI2_99,
    CoordinateFrameManager,
    ScenarioData,
    first_accepted_after,
    max_history_gap,
    nis_fraction_inside_95,
    rmse,
    run_tracking,
)


def _state(north: float, east: float, v_north: float = 0.0, v_east: float = 0.0):
    return np.array([[north], [east], [v_north], [v_east]], dtype=float)


def run_t2_unit_tests() -> dict:
    cfm = CoordinateFrameManager()

    h, H = cfm.h_and_H(_state(100.0, 0.0), "radar")
    assert np.isclose(h[0, 0], 100.0)
    assert np.isclose(h[1, 0], 0.0)
    assert H.shape == (2, 4)

    h, _ = cfm.h_and_H(_state(0.0, 100.0), "radar")
    assert np.isclose(h[0, 0], 100.0)
    assert np.isclose(h[1, 0], math.pi / 2)

    h, _ = cfm.h_and_H(_state(0.0, 0.0), "camera")
    expected_delta = np.array([80.0, -120.0])
    assert np.isclose(h[0, 0], np.linalg.norm(expected_delta))
    assert np.isclose(h[1, 0], math.atan2(expected_delta[1], expected_delta[0]))

    cfm.update_vessel_pos(0.0, 0.0)
    h_radar, _ = cfm.h_and_H(_state(300.0, 400.0), "radar")
    h_ais, _ = cfm.h_and_H(_state(300.0, 400.0), "ais")
    assert np.allclose(h_radar, h_ais)

    # Finite-difference Jacobian check for the camera model.
    x = _state(320.0, 260.0, 1.5, -0.5)
    _, H = cfm.h_and_H(x, "camera")
    eps = 1e-5
    for col in [0, 1]:
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[col, 0] += eps
        x_minus[col, 0] -= eps
        h_plus, _ = cfm.h_and_H(x_plus, "camera")
        h_minus, _ = cfm.h_and_H(x_minus, "camera")
        numeric = ((h_plus - h_minus) / (2 * eps)).ravel()
        assert np.allclose(numeric, H[:, col], atol=1e-5)

    # AIS covariance is transformed from NED metres into range/bearing units.
    R_ais = cfm.range_bearing_noise("ais", np.array([200.0, 350.0]))
    assert R_ais.shape == (2, 2)
    assert R_ais[0, 0] > 0.0
    assert 0.0 < R_ais[1, 1] < 1.0
    assert not np.allclose(R_ais, np.diag([16.0, 16.0]))

    return {
        "radar_basic": "passed",
        "camera_offset": "passed",
        "ais_same_origin": "passed",
        "jacobian_finite_difference": "passed",
        "ais_covariance_transform": "passed",
    }


def result_summary(result, scenario, start_time=20.0):
    return {
        "bootstrap_sensor": result.bootstrap_sensor,
        "bootstrap_time_s": round(result.bootstrap_time, 4),
        "confirmation_time_s": None
        if result.confirmation_time is None
        else round(result.confirmation_time, 4),
        "accepted_updates": result.accepted_count,
        "rmse_after_start_m": round(rmse(result, scenario, start_time=start_time), 3),
        "nis_inside_95_pct": round(nis_fraction_inside_95(result), 2),
        "accepted_by_sensor": {
            sensor: result.accepted_sensor_count(sensor)
            for sensor in ("radar", "camera", "ais")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario-dir",
        default="harbour_sim_output",
        help="Directory containing scenario_A/B/C JSON files.",
    )
    parser.add_argument(
        "--json-out",
        default="tracking_solution/results_t2_t5.json",
        help="Path for the machine-readable report.",
    )
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir)
    scenario_a = ScenarioData.load(scenario_dir / "scenario_A.json")
    scenario_b = ScenarioData.load(scenario_dir / "scenario_B.json")
    scenario_c = ScenarioData.load(scenario_dir / "scenario_C.json")

    print("\nT2 unit tests")
    print("-------------")
    t2 = run_t2_unit_tests()
    for name, status in t2.items():
        print(f"{name:28s} {status}")

    print("\nT3 Scenario A: radar-only EKF")
    print("-----------------------------")
    t3 = run_tracking(
        scenario_a,
        allowed_sensors=("radar",),
        fusion_mode="sequential",
        bootstrap_sensors=("radar",),
        name="T3 radar only",
    )
    t3_summary = result_summary(t3, scenario_a, start_time=20.0)
    t3_confirm_limit = t3.bootstrap_time + 5.0 * (1.0 / 0.3)
    t3_pass = {
        "confirmation": t3.confirmation_time is not None
        and t3.confirmation_time <= t3_confirm_limit,
        "rmse": t3_summary["rmse_after_start_m"] < 12.0,
        "nis": t3_summary["nis_inside_95_pct"] >= 90.0,
    }
    print(json.dumps(t3_summary, indent=2))
    print("pass:", t3_pass)

    print("\nT4 Scenario B: radar + camera")
    print("-----------------------------")
    t4_radar = run_tracking(
        scenario_b,
        allowed_sensors=("radar",),
        fusion_mode="sequential",
        bootstrap_sensors=("radar",),
        name="T4 radar baseline",
    )
    t4_seq = run_tracking(
        scenario_b,
        allowed_sensors=("radar", "camera"),
        fusion_mode="sequential",
        bootstrap_sensors=("radar",),
        name="T4 sequential",
    )
    t4_joint = run_tracking(
        scenario_b,
        allowed_sensors=("radar", "camera"),
        fusion_mode="joint",
        bootstrap_sensors=("radar",),
        name="T4 joint",
    )
    t4_summary = {
        "radar_only": result_summary(t4_radar, scenario_b, start_time=5.0),
        "sequential": result_summary(t4_seq, scenario_b, start_time=5.0),
        "joint": result_summary(t4_joint, scenario_b, start_time=5.0),
        "rmse_window_0_20_m": {
            "radar_only": round(rmse(t4_radar, scenario_b, start_time=0.0, end_time=20.0), 3),
            "sequential": round(rmse(t4_seq, scenario_b, start_time=0.0, end_time=20.0), 3),
            "joint": round(rmse(t4_joint, scenario_b, start_time=0.0, end_time=20.0), 3),
        },
    }
    print(json.dumps(t4_summary, indent=2))

    print("\nT5 Scenario C: add AIS and handle dropout")
    print("-----------------------------------------")
    t5_no_ais = run_tracking(
        scenario_c,
        allowed_sensors=("radar", "camera"),
        fusion_mode="sequential",
        bootstrap_sensors=("radar",),
        name="T5 no AIS",
    )
    t5_with_ais = run_tracking(
        scenario_c,
        allowed_sensors=("radar", "camera", "ais"),
        fusion_mode="sequential",
        bootstrap_sensors=("radar",),
        name="T5 with AIS",
    )
    t5_ais_only = run_tracking(
        scenario_c,
        allowed_sensors=("ais",),
        fusion_mode="sequential",
        bootstrap_sensors=("ais",),
        name="T5 AIS-only initiation",
    )
    reacq_time = first_accepted_after(t5_with_ais, "ais", 90.0)
    t5_summary = {
        "without_ais": result_summary(t5_no_ais, scenario_c, start_time=20.0),
        "with_ais": result_summary(t5_with_ais, scenario_c, start_time=20.0),
        "ais_only_initiation": result_summary(t5_ais_only, scenario_c, start_time=20.0),
        "rmse_available_windows_m": {
            "without_ais": round(
                np.mean(
                    [
                        rmse(t5_no_ais, scenario_c, start_time=20.0, end_time=60.0),
                        rmse(t5_no_ais, scenario_c, start_time=90.0),
                    ]
                ),
                3,
            ),
            "with_ais": round(
                np.mean(
                    [
                        rmse(t5_with_ais, scenario_c, start_time=20.0, end_time=60.0),
                        rmse(t5_with_ais, scenario_c, start_time=90.0),
                    ]
                ),
                3,
            ),
        },
        "dropout_60_90": {
            "rmse_without_ais_m": round(
                rmse(t5_no_ais, scenario_c, start_time=60.0, end_time=90.0), 3
            ),
            "rmse_with_ais_m": round(
                rmse(t5_with_ais, scenario_c, start_time=60.0, end_time=90.0), 3
            ),
            "max_history_gap_s": round(max_history_gap(t5_with_ais, 60.0, 90.0), 3),
            "accepted_radar_camera_updates": sum(
                1
                for row in t5_with_ais.history
                if 60.0 <= row.time <= 90.0
                and ("radar" in row.sensors or "camera" in row.sensors)
            ),
        },
        "first_ais_reacquisition_after_90_s": reacq_time,
    }
    print(json.dumps(t5_summary, indent=2))

    report = {"T2": t2, "T3": t3_summary, "T3_pass": t3_pass, "T4": t4_summary, "T5": t5_summary}
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
