import numpy as np
from itertools import groupby
from pathlib import Path

from coordinate_frame_manager import CoordinateFrameManager
from ekf_tracker import EKFTracker
from harbour_sim_output.load_simulation_data import load_simulation_output
from plot_results import plot_tracking_results

# 1. Initialisation
cfm = CoordinateFrameManager()
data = load_simulation_output("A")

# Seed from first true radar detection
first_true = next(m for m in data.measurements if m.sensor_id == "radar" and not m.is_false_alarm)
