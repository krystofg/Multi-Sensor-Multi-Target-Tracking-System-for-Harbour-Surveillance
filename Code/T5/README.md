# T5 - Fusion Step 2: Add AIS

This folder contains the Task 5 implementation scaffold. It is ready for the
future T4 output, but it intentionally does **not** read any project input file
yet. When T4 is finished, we will connect its radar/camera track output here.

## Current Status

T5 currently provides:

- the asynchronous AIS EKF update logic;
- the expected input contract from T4, AIS, and GNSS;
- the output CSV/JSON contract for T6;
- a CLI that exits cleanly when no input is supplied.

It does not use `harbour_sim_output/scenario_C.json` or any other simulator file
by default. The input will be added later from the previous task folders once T4
is available.

## Files

- `t5_ais_fusion.py` - AIS fusion module and future runner.
- `README.md` - this English explanation.
- `outputs/` - created later only when real T4/AIS/GNSS inputs are supplied.

## How To Run Now

From the repository root:

```powershell
python Code/T5/t5_ais_fusion.py
```

Expected current behaviour:

```text
T5 AIS fusion is ready, but no input files were supplied.
This is expected until T4 is finished.
No simulator scenario file is read by default.
```

## How To Run Later

Once T4 produces a track file:

```powershell
python Code/T5/t5_ais_fusion.py `
  --t4-input Code/T4/outputs/t4_tracks.csv `
  --ais-input <ais_measurements.csv> `
  --gnss-input <gnss_fixes.csv>
```

The default output directory will be:

```text
Code/T5/outputs/
```

## Future Input Contract

### T4 Track Input

T4 should provide one radar/camera track state per timestamp. CSV or JSON is
supported.

Required columns:

```text
time_s,track_id,north_m,east_m,v_north_mps,v_east_mps
```

Optional columns:

```text
accepted,sensors,cov_00,cov_01,...,cov_33
```

If covariance fields are missing, T5 uses a conservative diagonal covariance so
the interface can still be tested early.

### AIS Input

AIS reports should be NED target positions:

```text
time_s,north_m,east_m
```

Optional identification hints such as `mmsi` or `target_hint` may be present,
but the current T5 layer does not require them.

### GNSS Input

GNSS reports should be ownship NED positions:

```text
time_s,north_m,east_m
```

For every AIS update, T5 selects the nearest GNSS fix in time.

## Output Contract For T6

When connected, T5 writes:

```text
Code/T5/outputs/t5_fused_tracks.csv
Code/T5/outputs/t5_report.json
```

The fused track CSV schema is:

```text
time_s,track_id,north_m,east_m,v_north_mps,v_east_mps,accepted,sensors,source_update,nis,nearest_gnss_time_s
```

- `source_update = t4` means the row came from the radar/camera T4 estimate.
- `source_update = ais` means an asynchronous AIS update was attempted.
- `sensors = ais` means the AIS update passed the gate and was accepted.
- `nis` stores the AIS normalised innovation squared for consistency checks.

## How The Code Works

The EKF state is:

```text
x = [p_N, p_E, v_N, v_E]^T
```

Prediction uses a constant-velocity model:

```text
p_N(k+1) = p_N(k) + dt * v_N(k)
p_E(k+1) = p_E(k) + dt * v_E(k)
```

The future T4 estimate is treated as the current radar/camera fused base state.
AIS messages arrive asynchronously between those T4 states. For each AIS
message, T5:

1. predicts the EKF to the AIS timestamp;
2. finds the GNSS ownship fix closest in time;
3. subtracts the GNSS vessel position from the AIS target NED position;
4. converts that relative vector into range/bearing;
5. transforms AIS/GNSS Cartesian covariance into range/bearing covariance;
6. applies a gated EKF update using a 2D chi-square gate at `P_G = 0.99`;
7. writes the resulting fused state for the next task.

The AIS covariance conversion is:

```text
R_rb = J_polar * R_ne * J_polar^T
R_ne = (sigma_AIS^2 + sigma_GNSS^2) * I
```

This matters because AIS and GNSS noise start in metres, while the EKF update is
performed in mixed range/bearing units.

## Notes For Integration

T5 is single-track friendly because Task 5 is still before the full multi-target
association step. T6 should keep the measurement conversion and EKF update logic
from this folder, then replace the single-track flow with multi-target gating
and data association.
