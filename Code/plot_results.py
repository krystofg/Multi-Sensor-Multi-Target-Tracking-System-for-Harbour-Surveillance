import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_tracking_results(simulation_output, est_history, nis_history, pct_nis,
                          title="Tracking Results", target_id=0, save_path=None):

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1])

    fig.suptitle(title, fontsize=14)

    # =========================================================
    # 1. ZOOMED TRAJECTORY (LEFT SIDE, SPANS BOTH ROWS)
    # =========================================================
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_traj.set_title("2D NED Trajectory of Single Target Using EKF (Radar Measurements Only)")

    # Ground truth
    for tid, gt in simulation_output.ground_truth.items():
        color = 'k' if tid == target_id else 'gray'
        ax_traj.plot(gt[:, 1], gt[:, 0], '--', color=color,
                     label=f"Ground Truth Trajectory (Target {tid})")

    # Radar points
    raw_radar = [m for m in simulation_output.measurements if m.sensor_id == "radar"]
    raw_e = [m.range_m * np.sin(m.bearing_rad) for m in raw_radar]
    raw_n = [m.range_m * np.cos(m.bearing_rad) for m in raw_radar]
    ax_traj.scatter(raw_e, raw_n, c='steelblue', s=6, label="Raw Radar Measurements")

    # EKF estimate
    est_e = [e['E'] for e in est_history]
    est_n = [e['N'] for e in est_history]
    ax_traj.plot(est_e, est_n, 'r-', label="EKF Track")

    ax_traj.set_xlabel("East [m]")
    ax_traj.set_ylabel("North [m]")
    ax_traj.grid(True)
    ax_traj.legend(fontsize=8)

    # 🔥 ZOOM automatically around trajectory
    margin = 10
    ax_traj.set_xlim(min(est_e) - margin, max(est_e) + margin)
    ax_traj.set_ylim(min(est_n) - margin, max(est_n) + margin)

    # =========================================================
    # 2. STEADY-STATE POSITION ERROR (TOP RIGHT)
    # =========================================================
    ax_err = fig.add_subplot(gs[0, 1])
    ax_err.set_title("Steady-State Position Error (t > 20 s)")

    if target_id in simulation_output.ground_truth:

        gt_t = simulation_output.ground_truth_times
        gt = simulation_output.ground_truth[target_id]

        ss_t = []
        ss_errors = []

        for e in est_history:
            if e['t'] <= 20.0:
                continue

            idx = np.argmin(np.abs(gt_t - e['t']))
            err = np.sqrt((e['N'] - gt[idx, 0])**2 +
                          (e['E'] - gt[idx, 1])**2)

            ss_t.append(e['t'])
            ss_errors.append(err)

        if ss_errors:
            ax_err.plot(ss_t, ss_errors, 'g', lw=2, label="Position Error")
            ax_err.axhline(12.0, color='r', ls='--', label="Limit 12 m")

        ax_err.set_xlabel("Time [s]")
        ax_err.set_ylabel("Error [m]")
        ax_err.grid(True)
        ax_err.legend(fontsize=8)

    # =========================================================
    # 3. NIS (BOTTOM RIGHT)
    # =========================================================
    ax_nis = fig.add_subplot(gs[1, 1])
    ax_nis.set_title("NIS Consistency")

    nis_t = [n['t'] for n in nis_history]
    nis_vals = [n['nis'] for n in nis_history]

    ax_nis.plot(nis_t, nis_vals, 'b.', ms=4,
                label=f"NIS ({pct_nis:.1f}% in bounds)")

    chi2_lo, chi2_hi = 0.103, 5.991

    t = np.array(nis_t)
    ax_nis.fill_between(t, chi2_lo, chi2_hi, alpha=0.08, color='green')

    ax_nis.axhline(chi2_hi, color='r', ls='--', label="χ² upper")
    ax_nis.axhline(chi2_lo, color='orange', ls='--', label="χ² lower")

    ax_nis.set_xlabel("Time [s]")
    ax_nis.set_ylabel("NIS")
    ax_nis.set_ylim(0, 10)
    ax_nis.grid(True)
    ax_nis.legend(fontsize=8)

    # =========================================================
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()