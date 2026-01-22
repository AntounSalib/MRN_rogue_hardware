from typing import Set, Tuple
from constants import NodConfig, EPS
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
        if distance <= NodConfig.neighbors.SENSING_RANGE:
            sensed.add(name)

    return sensed

def conflicting_neighbors(ego_info: dict, neighbors_dict: dict) -> Set[str]:
     
    conflicting = set()
    ego_pos = ego_info['position']
    for name, neighbor_info in neighbors_dict.items():
        # check if within sensing range
        neighbor_pos = neighbor_info['position']
        distance = ((ego_pos[0] - neighbor_pos[0]) ** 2 + (ego_pos[1] - neighbor_pos[1]) ** 2) ** 0.5
        if distance > NodConfig.neighbors.SENSING_RANGE:
            continue

        # check if will intersect in future
        pij = np.array(ego_pos) -  np.array(neighbor_pos)
        ei = np.array(ego_info['velocity']) / (np.linalg.norm(np.array(ego_info['velocity'])))
        ej = np.array(neighbor_info['velocity']) / (np.linalg.norm(np.array(neighbor_info['velocity'])))
        if not will_intersect_in_future(pij, ei, ej):
            continue

        # check if will come within R_PRED
        r0 = np.array(neighbor_pos) - np.array(ego_pos)  # relative position
        w = np.array(neighbor_info['velocity'])-np.array(ego_info['velocity'])    # relative velocity
        a = float(np.dot(w, w))
        b = 2*float(np.dot(r0, w))
        c = float(np.dot(r0, r0)) - 1.*(NodConfig.neighbors.R_PRED)**2
        kept = (b**2 - 4*a*c >= 0)
        print(f"robot: {ego_info['name']}, neighbor: {name}, a: {a:.3f}, b: {b:.3f}, c: {c:.3f}, discriminant: {b**2 - 4*a*c:.3f}, kept: {kept}")

        if kept: 
            conflicting.add(name)
     

    return conflicting

def will_intersect_in_future(pij: np.ndarray, ei: np.ndarray, ej: np.ndarray):
    """Solve s*ei = pij - t*ej. Return (s,t) or None if parallel."""
    A = np.array([[ei[0], -ej[0]], [ei[1], -ej[1]]], float)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-9:
        return False
    inv = (1.0 / det) * np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]], float)
    s, t = inv @ pij

    if s < 0.0 or abs(s) > NodConfig.neighbors.R_OCC:
        return False
    else:
        return True

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