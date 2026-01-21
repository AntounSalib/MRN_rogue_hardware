from typing import Set, Tuple
from constants import SENSING_RANGE, EPS
import numpy as np

def sensed_neighbors(ego_info: dict, neighbors_dict: dict) -> Set[str]:
    """
    Given a dictionary of neighbors, return a set of names of neighbors that are currently sensed.
    A neighbor is considered sensed if its within sensing range.

    Args:
        neighbors_dict (dict): A dictionary where keys are neighbor names and values are dictionaries
                              containing neighbor information, including 'position'.
    Returns:
        Set[str]: A set of names of sensed neighbors."""
    
    sensed = set()
    ego_pos = ego_info['position']
    for name, info in neighbors_dict.items():
        neighbor_pos = info['position']
        distance = ((ego_pos[0] - neighbor_pos[0]) ** 2 + (ego_pos[1] - neighbor_pos[1]) ** 2) ** 0.5
        if distance <= SENSING_RANGE:
            sensed.add(name)

    return sensed

#todo: figure out which conflicting neighbors script to use
def conflicting_neighbors(ego_info: dict, neighbors_dict: dict) -> Set[str]:
    return sensed_neighbors(ego_info, neighbors_dict)

def tca_and_rmin(ego_info: dict, neighbor_info: dict) -> Tuple[float, float]:
    """
    Compute time to closest approach (t_star) and distance at closest approach (d_star) between ego and neighbor
    """
    ego_pos = np.array(ego_info['position'])
    ego_vel = np.array(ego_info['velocity'])
    neighbor_pos = np.array(neighbor_info['position'])
    neighbor_vel = np.array(neighbor_info['velocity'])

    r0 = neighbor_pos - ego_pos  # relative position
    w = neighbor_vel - ego_vel    # relative velocity

    a = float(np.dot(w, w))
    b = float(np.dot(r0, w))
    c = float(np.dot(r0, r0))

    if np.linalg.norm(w) <= EPS: # relative velocity negligible
        return np.inf, np.sqrt(max(0.0, c))

    # time to closest appraoch
    tca = -b / a 
    if tca < 0.0:
        t_star = np.inf
    else:
        t_star = tca

    # distance at closest approach
    rmin_sq = max(0.0, c - ((b * b) / a))
    d_star = np.sqrt(rmin_sq)

   
    return t_star, d_star