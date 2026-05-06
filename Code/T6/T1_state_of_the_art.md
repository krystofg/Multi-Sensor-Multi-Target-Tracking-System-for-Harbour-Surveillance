# T1 - State-of-the-Art Survey and Association Choice

## Scope

The harbour surveillance task is a multi-sensor, multi-target tracking problem
with heterogeneous measurements:

- land-based mm-wave radar: range/bearing, high clutter, low update rate;
- land-based stereo camera: range/bearing, limited field of view;
- AIS: asynchronous NED position reports for cooperative vessels;
- GNSS: ownship position for the fusion centre and AIS-relative geometry.

The implemented tracker estimates each target state in the NED frame as
`x = [p_N, p_E, v_N, v_E]^T`, uses an EKF measurement model for sensor-dependent
range/bearing observations, and solves multi-target conflicts with gated global
nearest-neighbour association.

## Tracking Architectures

### Measurement-Level Centralised Fusion

In measurement-level fusion, raw detections are brought into one common
coordinate/state model and are associated directly to central tracks. This is
the architecture used in this project. Its main advantage is that the EKF update
can use sensor-specific Jacobians and covariances before track-level decisions
are made. It is also natural for asynchronous AIS: predict the EKF to the AIS
timestamp, convert AIS NED position to an implied observation relative to the
vessel GNSS fix, and update immediately.

This approach matches maritime AIS/radar fusion literature, where AIS is
accurate and intermittent while radar is persistent but cluttered. Habtemariam
et al. explicitly study measurement-level AIS/radar fusion for maritime
surveillance and note the importance of handling variable AIS revisit intervals
and multi-target ambiguity [1].

### Track-to-Track Fusion

Track-to-track fusion runs local trackers per sensor and combines their track
estimates. It is attractive when sensors are geographically distributed, have
limited bandwidth, or cannot expose raw detections. The cost is that local-track
cross-correlation becomes difficult: two sensors may track the same target using
partly shared information, and a naive weighted average can become overconfident.

For this course project, track-to-track fusion would add complexity without a
clear benefit because all scenario JSON measurements are available centrally.

### Decentralised / Distributed Fusion

Distributed fusion is useful for networks of autonomous surface vehicles or
shore stations. Recent maritime work uses Bayesian distributed models such as
belief propagation to combine radar, AIS, and camera observations across agents
[2]. This is powerful but outside the project scale: the provided architecture
has one fusion centre, not a peer-to-peer sensor network.

## Data Association Methods

| Method | Idea | Accuracy | Computational cost | Strength | Weakness |
| --- | --- | --- | --- | --- | --- |
| Nearest Neighbour (NN) | Each track takes the closest gated detection. | Low to moderate. | Very low. | Simple and fast. | Can assign the same detection to multiple tracks or fail at crossings. |
| Global Nearest Neighbour (GNN) | Solve a global one-to-one assignment that minimises total association cost. | Good for moderate target counts and low/medium ambiguity. | Polynomial with Hungarian-style assignment; small scenarios can also use recursion. | Real-time, deterministic, interpretable. | Commits to one association; no explicit uncertainty over alternatives. |
| PDA | One track is updated by a probability-weighted mixture of gated detections. | Good for a single target in clutter. | Moderate per track. | Handles clutter probabilistically. | Standard PDA assumes one target; not enough for close multi-target conflicts. |
| JPDA | Joint association probabilities across multiple tracks and detections. | High in ambiguous clutter. | High; grows quickly with tracks/detections. | Reduces coalescence risk compared with hard decisions. | More complex and heavier than needed for 4-6 targets. |
| MHT | Maintains multiple competing association histories over time. | Very high when ambiguity is severe. | Very high unless heavily pruned. | Best identity preservation under long ambiguity. | Implementation and tuning cost are large for this project. |

NN is too weak for Scenario D because crossing trajectories create simultaneous
gating conflicts. PDA is useful but is not the correct baseline for multiple
targets without the JPDA extension. JPDA, originally demonstrated in sonar
multi-target tracking by Fortmann, Bar-Shalom, and Scheffe [3], is a strong
probabilistic option but would require more bookkeeping and probability
normalisation than the project needs. MHT, introduced by Reid [4], is the most
complete classical approach for persistent ambiguity, but hypothesis growth and
pruning are disproportionate for the provided scenarios.

GNN is therefore the chosen implementation. It uses the Mahalanobis distance
from the EKF innovation,

`d^2 = y^T S^{-1} y`,

with a chi-square gate at `P_G = 0.99`, then chooses a one-to-one assignment
between all active tracks and all gated detections for each sensor stream. This
is directly aligned with T6: unmatched detections initialise tentative tracks,
and unmatched tracks coast through predict-only steps. The underlying assignment
problem is classically solved by Hungarian/Kuhn-Munkres methods [5]. Our small
scenario sizes allow an exact recursive GNN search with a greedy fallback.

## Multi-Sensor EKF Fusion

### Sequential Update

Sequential fusion applies one EKF update per measurement, predicting to each
measurement time when necessary. With independent sensor noise and measurements
at the same timestamp, sequential updates are equivalent to a stacked central
update up to numerical ordering effects. Sequential fusion is convenient for
asynchronous radar, camera, and AIS because it avoids waiting for a complete
sensor packet.

This is the main implementation path for T3, T4, and T5.

### Joint / Centralised Update

Joint fusion stacks simultaneous sensor observations into one larger vector and
performs one EKF update. It is clean when radar and camera detections are known
to correspond to the same target at the same time. In practice, the project data
are asynchronous and cluttered, so joint update is mainly used as a comparison
in T4 rather than as the main architecture.

### Out-of-Sequence Measurements

Out-of-sequence measurements occur when delayed sensor packets arrive after the
filter has already moved past their timestamps. Bar-Shalom and collaborators
show that this is a distinct filtering problem in multi-sensor target tracking
[6]. The provided JSON scenarios are processed in timestamp order, so the
implementation does not require OOSM correction. For a live system, the tracker
would need a fixed-lag buffer or a dedicated OOSM update.

## Maritime-Specific Considerations

### AIS Integration

AIS is highly informative because it carries target identity and absolute
position, but it is cooperative and intermittent. Small craft may not broadcast
AIS at all, and cooperative vessels can have large or irregular revisit
intervals. AIS/radar fusion work in maritime surveillance emphasises this
tradeoff: AIS is accurate and low-clutter, while radar is continuous but noisy
and cluttered [1,7].

In this project, AIS is treated as an asynchronous measurement. Since the EKF
measurement model is range/bearing relative to a sensor position, AIS NED
reports are converted to implied range/bearing relative to the nearest GNSS
ownship fix. The AIS Cartesian covariance is transformed through the polar
Jacobian before update, avoiding the mixed-units covariance issue.

### Sea Clutter and Multipath

Harbours are cluttered: waves, wakes, quay structures, moored vessels, cranes,
and multipath can all produce false or biased returns. High-resolution marine
radar work often treats vessels as extended targets and uses specialised
trackers such as MEM-EKF or PAKF-JPDA variants for orientation and extent
estimation [8,9]. Our project uses point-target EKF tracking, so false alarms
are handled primarily by gating, GNN assignment, M-of-N confirmation, coasting,
and deletion.

### Camera Limitations

The camera has a limited field of view, can be occluded, and degrades in poor
visibility. It provides strong bearing information when the vessel is visible,
but it should not drive deletion when a target is simply outside the camera
sector. For that reason, our lifecycle treats radar scans as the primary miss
clock in simulation; camera and AIS supplement updates but do not alone delete
tracks outside their coverage.

## Deep Learning Approaches

Modern visual multi-object tracking often follows a tracking-by-detection
pipeline. ByteTrack is a representative recent method: it improves visual MOT by
associating low-confidence detections as well as high-confidence detections,
recovering occluded objects and reducing fragmented tracks [10]. This is
powerful for camera bounding boxes and can run in real time on visual benchmarks.

For this harbour project, a deep MOT method is not the primary tracker because:

- the dominant measurements are metric radar/AIS/GNSS observations, not only
  camera boxes;
- the project requires physically interpretable NED position, velocity, heading,
  NIS, RMSE, MOTP, and CE metrics;
- no labelled training set is provided for end-to-end deep identity learning;
- AIS and radar are already structured metric sensors where EKF/GNN is data
  efficient and transparent.

A practical future extension would be to use deep learning in the camera
front-end only: vessel detection, visual ID/re-identification, or occlusion
handling. The resulting camera detections would still be fused by the EKF/GNN
back-end.

## Selected Algorithm

The final implemented association algorithm is:

1. Predict every active EKF track to the current scan time.
2. For each active sensor stream, compute innovations and Mahalanobis distances
   to all detections.
3. Gate detections with `chi2(df=2, P_G=0.99)`.
4. Solve a one-to-one GNN assignment that maximises the number of assigned
   detections and minimises total NIS.
5. Update assigned tracks.
6. Initiate tentative tracks from unmatched detections.
7. Mark unmatched tracks as missed on radar scan cycles.
8. Manage tentative, confirmed, coasting, and deleted tracks with configurable
   M-of-N confirmation and missed-detection deletion.

This choice is justified because it is accurate enough for Scenarios D/E,
computationally light for real-time use, deterministic for debugging, and easy
to explain on the poster. The final simulation metrics support the decision:
Scenario D achieves `MOTP = 2.864 m`, `CE = 0.442`, and `0` identity switches;
Scenario E achieves `MOTP = 2.386 m`, `CE = 0.345`, and `0` identity switches.

## References

[1] B. K. Habtemariam, R. Tharmarasa, M. McDonald, T. Kirubarajan, "Measurement level AIS/radar fusion", *Signal Processing*, 2015. https://doi.org/10.1016/j.sigpro.2014.07.029

[2] A. La Grappe, E. le Flecher, G. De Cubber, "Multi-Sensor Multi-Target Tracking for Maritime Surveillance with Autonomous Surface Vehicles Using Belief Propagation", 2025. https://researchportal.rma.ac.be/en/publications/multi-sensor-multi-target-tracking-for-maritime-surveillance-with

[3] T. Fortmann, Y. Bar-Shalom, M. Scheffe, "Sonar tracking of multiple targets using joint probabilistic data association", *IEEE Journal of Oceanic Engineering*, 1983. https://doi.org/10.1109/JOE.1983.1145560

[4] D. B. Reid, "An algorithm for tracking multiple targets", *IEEE Transactions on Automatic Control*, 1979. https://doi.org/10.1109/TAC.1979.1102177

[5] H. W. Kuhn, "The Hungarian Method for the Assignment Problem", *Naval Research Logistics Quarterly*, 1955. https://doi.org/10.1002/nav.3800020109

[6] M. Mallick, J. Krant, Y. Bar-Shalom, "Multi-sensor multi-target tracking using out-of-sequence measurements", *International Conference on Information Fusion*, 2002. https://doi.org/10.1109/ICIF.2002.1021142

[7] B. K. Habtemariam, R. Tharmarasa, E. Meger, T. Kirubarajan, "Measurement level AIS/Radar Fusion for Maritime Surveillance", *SPIE*, 2012. https://doi.org/10.1117/12.920156

[8] J. S. Fowdur, M. Baum, F. Heymann, "A Marine Radar Dataset for Multiple Extended Target Tracking", MSAW, 2019. https://www.researchgate.net/publication/336666186_A_Marine_Radar_Dataset_for_Multiple_Extended_Target_Tracking

[9] J. S. Fowdur, M. Baum, F. Heymann, "An Overview of the PAKF-JPDA Approach for Elliptical Multiple Extended Target Tracking Using High-Resolution Marine Radar Data", *Remote Sensing*, 2023. https://www.mdpi.com/2072-4292/15/10/2503

[10] Y. Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV, 2022. https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/315_ECCV_2022_paper.php
