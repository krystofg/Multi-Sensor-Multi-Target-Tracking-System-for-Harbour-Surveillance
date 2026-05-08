# Task 7 - Track Management

This file explains what `Code/task7.py` does and how to check the result.

## Run

From the repository root:

```bash
python Code/task7.py
```

By default the script runs both scenarios required for T7:

- Scenario D: radar + camera
- Scenario E: radar + camera + AIS

Run only one scenario:

```bash
python Code/task7.py --scenario D
python Code/task7.py --scenario E
```

Default confirmation setting from the task description:

```text
M = 3
N = 5
```

The values are configurable, as required by the task.

## Output files

A full run writes:

```text
outputs/task7/scenario_D_task7_report.json
outputs/task7/scenario_E_task7_report.json
outputs/task7/task7_summary.json
outputs/task7/plots/scenario_D_tracks.png
outputs/task7/plots/scenario_E_tracks.png
outputs/task7/plots/scenario_D_motp_ce.png
outputs/task7/plots/scenario_E_motp_ce.png
```

The `outputs/` directory is ignored by Git, so these are local result files.

## What the code does

### 1. Load data

The script uses the existing `load_simulation_output()` function to load the
simulation data.

For T7 metrics it also reads the ground truth directly from the scenario JSON.
This is needed because in Scenario E targets can enter and leave the scene at
different times.

### 2. Prepare measurements

`SensorAdapter` converts sensor measurements into the form used by the EKF:

- radar: range + bearing,
- camera: range + bearing,
- AIS: NED position converted to range + bearing relative to the nearest GNSS fix.

For each measurement it prepares:

- `z`: the measurement,
- `h`: predicted measurement,
- `H`: Jacobian,
- `R`: measurement covariance,
- chi-square gate threshold.

### 3. Predict tracks

Each active track is predicted to the current scan time with a constant velocity
model:

```text
position = position + dt * velocity
velocity = velocity
```

If a track has no measurement, it is still predicted. Its covariance grows, so
the gate becomes wider and the track has a better chance of being re-acquired.

### 4. Gating and GNN association

For each track-measurement pair the code computes NIS:

```text
NIS = y^T S^-1 y
```

If NIS is below the chi-square threshold, the measurement is inside the gate.
Then GNN chooses the best one-to-one assignment between tracks and measurements.

### 5. EKF update

If a measurement is assigned to a track, the track gets an EKF update and a hit.

If a measurement is not assigned to any track, it starts a new tentative track.

### 6. Track lifecycle

A track can be in four states:

```text
tentative -> confirmed -> coasting -> deleted
```

- `tentative`: a new track from an unassigned detection.
- `confirmed`: the track has at least M hits in the last N scans.
- `coasting`: a confirmed track is missed, but is still predicted.
- `deleted`: the track has too many consecutive misses.

Initial velocity is estimated from the first two associated detections:

```text
v = (pos2 - pos1) / (t2 - t1)
```

### 7. Duplicate merging

After each scan, active tracks are compared. If two tracks are very close by
Mahalanobis distance, the weaker one is marked as deleted.

## Metrics

The script reports the two T7 metrics.

### MOTP

MOTP is the mean localization error over matched pairs:

```text
confirmed track <-> true target
```

Matching is done with minimum-distance assignment.

### Cardinality Error

CE checks if the number of confirmed tracks matches the number of active true
targets:

```text
CE(t) = |confirmed_tracks(t) - active_targets(t)|
```

Both metrics are saved as time series and scalar averages.

## How to check it

Quick run without writing files:

```bash
python Code/task7.py --no-files --no-plots
```

Full run:

```bash
python Code/task7.py
```

Validation run used for the final T7 results:

```bash
python Code/task7.py --confirmation-m 4 --confirmation-n 6
```

Check the JSON output:

```bash
python -m json.tool outputs/task7/task7_summary.json
```

If the commands run and print results for Scenario D and Scenario E, T7 is
working.

## Current results

The task requires M-of-N confirmation with default `M = 3`, `N = 5`, and both
values must be configurable. The code keeps that default. For the final
validation results below, `M = 4`, `N = 6` was used because Scenarios D and E
contain heavy clutter and the stricter confirmation window reduces false
confirmed tracks.

| Scenario | Sensors | Avg. MOTP [m] | Avg. CE | ID switches | Final confirmed tracks |
|---|---|---:|---:|---:|---:|
| D | radar + camera | 3.405 | 0.361 | 0 | 4 |
| E | radar + camera + AIS | 3.032 | 0.442 | 0 | 5 |

These values satisfy the scenario success criteria from the project text:

- Scenario D: MOTP `< 15 m`, CE `< 0.5`, and no identity switches.
- Scenario E: MOTP `< 20 m` and CE `< 1.0`.

MOTP is low in both scenarios, so the confirmed tracks that match real targets
are spatially accurate. The identity switch count is zero, so matched tracks
keep their target identity during the run. The CE values are below the required
limits, which means the number of reported confirmed tracks stays close to the
number of active true targets.
