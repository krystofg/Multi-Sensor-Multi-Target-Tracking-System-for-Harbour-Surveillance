# Standalone T2-T7 Solution

This folder contains our separate implementation for the harbour surveillance
project.  The original `harbour_simulation.ipynb` is left untouched and is used
only as the source of scenario JSON files in `harbour_sim_output/`.

## What is implemented

- T2 coordinate frame manager with radar, camera, and AIS offsets.
- Range/bearing Jacobians for EKF updates.
- AIS covariance conversion from noisy NED position into range/bearing units.
- Measurement-only track initiation; it does not use `target_id` or
  `is_false_alarm` labels.
- T3 radar-only EKF validation on Scenario A.
- T4 radar + camera fusion on Scenario B with sequential and joint updates.
- T5 radar + camera + AIS asynchronous fusion on Scenario C, including AIS
  dropout and AIS-only initiation support.
- T6 Mahalanobis gating and Global Nearest Neighbour-style data association on
  Scenario D.
- T7 tentative/confirmed/coasting/deleted lifecycle, duplicate merging, and
  MOTP/CE metrics on Scenarios D and E.
- T1 state-of-the-art survey in `T1_state_of_the_art.md`.
- Implementation explanation in `IMPLEMENTATION_NOTES.md`.
- Result figure generation in `generate_result_plots.py`.

## Run

From the repository root:

```powershell
jupyter notebook tracking_solution/T2_T7_tracking_solution.ipynb
```

or run the same workflow as a script:

```powershell
python tracking_solution/run_t2_t5.py
python tracking_solution/run_t6_t7.py
```

Generate figures for the notebook/poster:

```powershell
python tracking_solution/generate_result_plots.py
```

The script prints the validation report and writes:

```text
tracking_solution/results_t2_t5.json
tracking_solution/results_t6_t7.json
tracking_solution/results_t2_t7.json
tracking_solution/figures/*.png
```

## Notes

The AIS update follows the project wording that AIS NED reports are converted
to implied range/bearing relative to the vessel GNSS fix.  Because that changes
measurement units, the NED covariance is transformed through the polar Jacobian
before the EKF update.
