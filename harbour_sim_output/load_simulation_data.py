import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

@dataclass
class TargetConfig:
    """Initial configuration for one simulated marine target."""
    target_id      : int
    initial_north  : float          # m, NED
    initial_east   : float          # m, NED
    velocity_north : float          # m/s
    velocity_east  : float          # m/s
    has_ais        : bool  = False  # True → target broadcasts AIS
    active_from    : float = 0.0   # s — target enters the scene at this time
    active_until   : Optional[float] = None  # s — None means stays forever

@dataclass
class Measurement:
    """One sensor return (true detection or false alarm)."""
    sensor_id     : str    # 'radar' | 'camera' | 'ais' | 'gnss'
    time          : float  # seconds since simulation start
    is_false_alarm: bool
    target_id     : int    # true target ID; -1 for false alarms / GNSS
    # Range-bearing (radar, camera) — metres / radians
    range_m       : Optional[float] = None
    bearing_rad   : Optional[float] = None
    # NED position (AIS, GNSS) — metres
    north_m       : Optional[float] = None
    east_m        : Optional[float] = None

@dataclass
class SimulationOutput:
    """Complete output of one simulation run."""
    scenario_name     : str
    dt_true           : float                    # GT propagation step [s]
    t_end             : float                    # simulation duration [s]
    ground_truth      : Dict[int, np.ndarray]    # {target_id: (T,4) states}
    ground_truth_times: np.ndarray               # (T,) time axis for GT
    measurements      : List[Measurement]        # sorted by time
    vessel_positions  : np.ndarray               # (T_gnss, 2) NED vessel pos
    vessel_times      : np.ndarray               # (T_gnss,) GNSS times
    sensor_configs    : Dict                     # parameter summary

def load_simulation_output(scenario_name: str, base_dir: str = None) -> SimulationOutput:
    """Loads a SimulationOutput object from a saved JSON file."""
    if base_dir is None:
        # Default to the directory where this script is located
        base_dir = Path(__file__).parent
    
    filepath = Path(base_dir) / f"scenario_{scenario_name}.json"
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Reconstruct Measurements
    measurements = [Measurement(**m) for m in data['measurements']]
    
    # Reconstruct Ground Truth (time is index 0, state is index 1:5)
    first_tid = next(iter(data['ground_truth']))
    ground_truth_times = np.array([row[0] for row in data['ground_truth'][first_tid]])
    
    ground_truth = {
        int(tid): np.array([row[1:] for row in states]) 
        for tid, states in data['ground_truth'].items()
    }
    
    # Reconstruct Vessel Data (time is index 0, position is index 1:3)
    vessel_raw = np.array(data['vessel_positions'])
    if vessel_raw.size > 0:
        vessel_times = vessel_raw[:, 0]
        vessel_positions = vessel_raw[:, 1:]
    else:
        vessel_times = np.array([])
        vessel_positions = np.zeros((0, 2))

    return SimulationOutput(
        scenario_name=data['scenario_name'],
        dt_true=data['dt_true'],
        t_end=data['t_end'],
        ground_truth=ground_truth,
        ground_truth_times=ground_truth_times,
        measurements=measurements,
        vessel_positions=vessel_positions,
        vessel_times=vessel_times,
        sensor_configs=data['sensor_configs']
    )
