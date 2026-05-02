"""Run the standalone T6-T7 multi-target validation workflow.

Usage from the repository root:

    python tracking_solution/run_t6_t7.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harbour_tracking import (
    ScenarioData,
    multi_target_metrics,
    multi_target_summary,
    run_multi_target_tracking,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario-dir",
        default="harbour_sim_output",
        help="Directory containing scenario_D/E JSON files.",
    )
    parser.add_argument(
        "--json-out",
        default="tracking_solution/results_t6_t7.json",
        help="Path for the machine-readable report.",
    )
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir)
    scenario_d = ScenarioData.load(scenario_dir / "scenario_D.json")
    scenario_e = ScenarioData.load(scenario_dir / "scenario_E.json")

    # M and N are configurable in the T7 lifecycle.  The project text mentions
    # 3-of-5 as the default; this run uses 4-of-6 to suppress false confirmed
    # tracks in the provided high-clutter scenarios.
    lifecycle = {
        "confirmation_m": 4,
        "confirmation_n": 6,
        "tentative_delete_after": 2,
        "confirmed_delete_after": 5,
    }

    print("\nT6 Scenario D: gating + GNN data association")
    print("---------------------------------------------")
    t6_d = run_multi_target_tracking(
        scenario_d,
        allowed_sensors=("radar", "camera"),
        name="T6/T7 Scenario D",
        **lifecycle,
    )
    metrics_d = multi_target_metrics(t6_d, scenario_d, start_time=0.0)
    summary_d = multi_target_summary(t6_d, scenario_d, start_time=0.0)
    pass_d = {
        "motp": summary_d["avg_motp_m"] < 15.0,
        "ce": summary_d["avg_ce"] < 0.5,
        "identity": summary_d["id_switches"] == 0,
        "final_cardinality": summary_d["final_confirmed_tracks"] == 4,
    }
    print(json.dumps(summary_d, indent=2))
    print("pass:", pass_d)

    print("\nT7 Scenario E: full lifecycle with mixed AIS / non-AIS traffic")
    print("--------------------------------------------------------------")
    t7_e = run_multi_target_tracking(
        scenario_e,
        allowed_sensors=("radar", "camera", "ais"),
        name="T6/T7 Scenario E",
        **lifecycle,
    )
    metrics_e = multi_target_metrics(t7_e, scenario_e, start_time=0.0)
    summary_e = multi_target_summary(t7_e, scenario_e, start_time=0.0)
    pass_e = {
        "motp": summary_e["avg_motp_m"] < 20.0,
        "ce": summary_e["avg_ce"] < 1.0,
        "identity": summary_e["id_switches"] == 0,
        # At t=180 target 4 has left the scene, so 5 active tracks is correct.
        "final_cardinality": summary_e["final_confirmed_tracks"] == 5,
    }
    print(json.dumps(summary_e, indent=2))
    print("pass:", pass_e)

    report = {
        "lifecycle": lifecycle,
        "T6_scenario_D": {
            "summary": summary_d,
            "pass": pass_d,
            "motp_series": metrics_d["motp_series"],
            "ce_series": metrics_d["ce_series"],
        },
        "T7_scenario_E": {
            "summary": summary_e,
            "pass": pass_e,
            "motp_series": metrics_e["motp_series"],
            "ce_series": metrics_e["ce_series"],
        },
    }
    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
