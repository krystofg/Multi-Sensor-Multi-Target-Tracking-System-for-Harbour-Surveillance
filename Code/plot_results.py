import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_tracking_results(simulation_output, est_history, nis_history, title="Tracking Results", target_id=0, save_path=None):
    """
    Generates a 4-panel dashboard for tracking performance.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=13)

    # 1. Trajectory (GT, Raw Radar, EKF Track)
    ax = axes[0, 0]
    ax.set_title("2-D NED Trajectory")
    
    # GT Trajectory
    for tid, gt in simulation_output.ground_truth.items():
        color = 'k' if tid == target_id else 'gray'
        alpha = 0.7 if tid == target_id else 0.3
        ax.plot(gt[:, 1], gt[:, 0], color=color, ls='--', alpha=alpha, label=f"GT (Target {tid})")
    
    # Raw Radar
    raw_radar = [m for m in simulation_output.measurements if m.sensor_id == "radar"]
    raw_e = [m.range_m * np.sin(m.bearing_rad) for m in raw_radar]
    raw_n = [m.range_m * np.cos(m.bearing_rad) for m in raw_radar]
    ax.scatter(raw_e, raw_n, c='steelblue', s=6, alpha=0.15, label="Raw radar")
    
    # EKF Track
    est_e = [e['E'] for e in est_history]
    est_n = [e['N'] for e in est_history]
    ax.plot(est_e, est_n, 'r-', lw=1.5, label="EKF track")
    
    ax.plot(0, 0, 'y*', ms=12, label="Radar (origin)")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.axis('equal'); ax.grid(True); ax.legend(fontsize=8)

    # 2. Steady-State RMSE
    ax = axes[0, 1]
    ax.set_title(f"Steady-State Position Error (Target {target_id}, t > 20s)")
    
    if target_id in simulation_output.ground_truth:
        gt_t = simulation_output.ground_truth_times
        gt = simulation_output.ground_truth[target_id]
        
        ss_errors = []
        for e in est_history:
            if e['t'] <= 20.0: continue
            idx = np.argmin(np.abs(gt_t - e['t']))
            err = np.sqrt((e['N'] - gt[idx, 0])**2 + (e['E'] - gt[idx, 1])**2)
            ss_errors.append(err)
        
        if ss_errors:
            ax.plot(ss_errors, 'g', lw=1.2)
            ax.axhline(12.0, color='r', ls='--', label="Limit 12 m")
            ax.set_xlabel("Scan index"); ax.set_ylabel("Error [m]")
            ax.legend(); ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No steady-state data", ha='center')
    else:
        ax.text(0.5, 0.5, f"Target {target_id} not in GT", ha='center')


    # 3. NIS Consistency
    ax = axes[1, 0]
    ax.set_title("NIS Consistency")
    nis_t = [n['t'] for n in nis_history]
    nis_vals = [n['nis'] for n in nis_history]
    ax.plot(nis_t, nis_vals, 'b.', ms=4)
    
    chi2_lo, chi2_hi = 0.103, 5.991
    ax.axhline(chi2_hi, color='r', ls='--', label=f"χ²(2) 95% upper = {chi2_hi}")
    ax.axhline(chi2_lo, color='orange', ls='--', label=f"χ²(2) 95% lower = {chi2_lo}")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("NIS")
    ax.legend(fontsize=8); ax.grid(True)

    # 4. NIS Histogram
    ax = axes[1, 1]
    ax.set_title("NIS Histogram")
    if nis_vals:
        ax.hist(nis_vals, bins=20, color='steelblue', edgecolor='white')
        ax.axvline(chi2_hi, color='r', ls='--', label=f"95% upper = {chi2_hi:.2f}")
        ax.set_xlabel("NIS"); ax.set_ylabel("Count")
        ax.legend(fontsize=8); ax.grid(True)
    else:
        ax.text(0.5, 0.5, "No NIS data", ha='center')

    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
