# Implementation Notes - How the Tracker Works

## File Layout

- `harbour_tracking.py`: reusable tracking code.
- `run_t2_t5.py`: script validation for T2-T5.
- `run_t6_t7.py`: script validation for T6-T7.
- `T2_T7_tracking_solution.ipynb`: notebook version for presentation and review.
- `results_t2_t7.json`: combined machine-readable report from the notebook.

The original `harbour_simulation.ipynb` is not modified.  It is treated as the
school-provided simulator that generates `harbour_sim_output/scenario_*.json`.

## Data Flow

1. `ScenarioData.load(...)` reads one scenario JSON file.
2. Ground truth is stored as interpolatable arrays per target.
3. Raw sensor detections are converted into `RawMeasurement` objects.
4. The coordinate frame manager provides sensor offsets, measurement functions,
   Jacobians, and covariances.
5. EKF tracks are predicted to each measurement timestamp.
6. Measurements are gated and associated.
7. Associated tracks are updated; unmatched detections initialise new tentative
   tracks; unmatched tracks coast or are deleted.
8. Metrics are computed from confirmed track snapshots.

## Coordinate Frame Manager

The state vector is always in NED:

```text
x = [p_N, p_E, v_N, v_E]^T
```

Radar is at `[0, 0]`.  The simulated camera is at `[-80, 120]`.  AIS is dynamic:
the effective sensor position is the vessel GNSS position closest to the AIS
timestamp.

The common range/bearing model is:

```text
d_N = p_N - s_N
d_E = p_E - s_E
r   = sqrt(d_N^2 + d_E^2)
phi = atan2(d_E, d_N)
```

The Jacobian is:

```text
dr/dp_N     = d_N / r
dr/dp_E     = d_E / r
dphi/dp_N   = -d_E / r^2
dphi/dp_E   =  d_N / r^2
```

Velocity columns are zero because range/bearing measure position only.

## AIS Covariance Fix

AIS arrives as noisy NED position, but T5 asks for the implied range/bearing
observation relative to the moving vessel.  That means the covariance must be
transformed from Cartesian metres into polar units:

```text
R_rb = J_polar R_ne J_polar^T
```

where `R_ne = (sigma_AIS^2 + sigma_GNSS^2) I` when the GNSS ownship uncertainty
is included.  This is why the implementation does not use `diag([16, 16])` as a
range/bearing covariance.

## EKF Prediction and Update

The motion model is constant velocity with white acceleration noise:

```text
p_N(k+1) = p_N(k) + dt v_N(k)
p_E(k+1) = p_E(k) + dt v_E(k)
```

The update uses the standard EKF equations:

```text
y = z - h(x)
S = H P H^T + R
K = P H^T S^-1
x = x + K y
P = (I - K H) P (I - K H)^T + K R K^T
```

Bearing innovations are wrapped to `(-pi, pi]`.

## T3-T5 Single-Target Runs

T3 uses radar only on Scenario A.  Track initiation is measurement-only: early
detections are scored by how many future measurements gate consistently.  It
does not use `target_id` or `is_false_alarm`.

T4 compares:

- radar-only baseline;
- sequential radar/camera updates;
- joint stacked radar/camera updates.

T5 adds AIS:

- AIS is asynchronous;
- the EKF predicts to the AIS timestamp;
- the AIS NED report is converted to range/bearing relative to nearest GNSS;
- AIS-only initiation is supported;
- during the `60-90 s` dropout, radar/camera keep the track alive.

## T6 Data Association

For each sensor scan:

1. Predict all active tracks to the scan time.
2. Compute the EKF innovation for every track/detection pair.
3. Reject pairs with Mahalanobis distance above the chi-square gate.
4. Solve a Global Nearest Neighbour assignment:
   - maximise number of assigned detections;
   - minimise total NIS among assignments.
5. Update assigned tracks.
6. Pass unmatched detections to track initiation.

The code uses an exact recursive assignment for the small project scenarios and
a greedy fallback if the candidate set becomes too large.

## T7 Track Lifecycle

Each track can be:

- `tentative`: spawned from an unmatched detection;
- `confirmed`: enough hits in an M-of-N window;
- `coasting`: confirmed but recently missed;
- `deleted`: too many consecutive misses or duplicate.

The lifecycle is configurable.  The scripts use `4-of-6` confirmation for the
high-clutter D/E scenarios.  The project text gives `3-of-5` as a default, but
explicitly says M and N should be configurable; `4-of-6` reduces false confirmed
tracks from clutter.

Radar scans drive missed-detection deletion in simulation.  Camera and AIS are
not used alone as a deletion clock because they have limited/asynchronous
availability.

## Metrics

Single-target:

- RMSE against interpolated ground truth.
- NIS consistency using both 95% chi-square bounds.

Multi-target:

- MOTP: mean localisation error over matched confirmed track/target pairs.
- CE: absolute difference between number of confirmed tracks and active truth
  targets.
- Identity switches: change in nearest matched truth target per confirmed track.

For Scenario E, final cardinality is 5, not 6, because one target leaves the
scene before the end.
