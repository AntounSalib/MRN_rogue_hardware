from typing import Optional, Set, Tuple
from constants import NodConfig, EPS
import numpy as np
import math

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

        # ray intersection check
        ray_sol = solve_ray_intersection(ego_info, neighbor_info)
        if ray_sol is None:
            continue
        s,t,ei,ej = ray_sol
        if (s < 0.0 and abs(s) > NodConfig.neighbors.R_OCC):
            continue

        ti_occ, tj_occ, ti_rogue, _, _ = arrival_times_to_disk(ego_info, neighbor_info, ray_sol)
        # print(f"robot: {ego_info['name']}, neighbor: {name}, ti_occ: {ti_occ}, tj_occ: {tj_occ}, ti_rogue: {ti_rogue}, s: {s}, t: {t}")

        if (ti_occ is None or tj_occ is None or ti_rogue is None):
            continue
        if (ti_occ > 100 and tj_occ > 100):
            continue


        conflicting.add(name)

    return conflicting

def tca_and_rmin(ego_info: dict, neighbor_info: dict, inside_i: bool, inside_j: bool) -> Tuple[float, float, float, float]:
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

    if a <= EPS: # match the simulation's squared-relative-speed threshold
        # print("EPS Skip")
        return a, b, 0.0, np.sqrt(max(0.0, c))

    # time to closest appraoch
    t_star = max(0.0, -b / a)

    # distance at closest approach
    d_star = float(np.linalg.norm(r0 + t_star * w))
    # t_star = max(0.0, tca)  # only consider future approach
    # r_star = r0 + t_star * w
    # d_star = np.linalg.norm(r_star)

    # print(f"robot: {ego_info['name']}, neighbor: {neighbor_info['name']}, t_star: {t_star:.3f}, d_star: {d_star:.3f}")

   
    return a, b, t_star, d_star

def solve_ray_intersection(ego_info: dict, neighbor_info: dict) -> Optional[Tuple[float, float, np.ndarray, np.ndarray]]:
    """Return metric ray distances and unit directions, or None if parallel."""
    ego_pos = np.array(ego_info['position'])
    ego_vel = np.array(ego_info['velocity'])
    neighbor_pos = np.array(neighbor_info['position'])
    neighbor_vel = np.array(neighbor_info['velocity'])
    pij = neighbor_pos - ego_pos
    if 'heading' in ego_info:
        h = ego_info['heading']
        ei = np.array([math.cos(h), math.sin(h)])
    else:
        speed = float(np.linalg.norm(ego_vel))
        if speed <= 1e-9:
            return None
        ei = ego_vel / speed
    if 'heading' in neighbor_info:
        h = neighbor_info['heading']
        ej = np.array([math.cos(h), math.sin(h)])
    else:
        speed = float(np.linalg.norm(neighbor_vel))
        if speed <= 1e-9:
            return None
        ej = neighbor_vel / speed

    A = np.array([[ei[0], -ej[0]], [ei[1], -ej[1]]], float)
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-9:
        return None
    inv = (1.0 / det) * np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]], float)
    s, t = inv @ pij
    return float(s), float(t), ei, ej

def arrival_times_to_disk(ego_info: dict, neighbor_info: dict, ray_sol=None) -> tuple:
    if ray_sol is None:
        ray_sol = solve_ray_intersection(ego_info, neighbor_info)
    if ray_sol is None:
        return None, None, None, False, False
    s, t, ei, ej = ray_sol

    inside_i, inside_j = False, False
    vi = np.linalg.norm(np.array(ego_info['velocity']))
    vj = np.linalg.norm(np.array(neighbor_info['velocity']))
    sin_alpha = abs(float(ei[0] * ej[1] - ei[1] * ej[0]))
    r = NodConfig.neighbors.R_OCC / max(sin_alpha, EPS)
    
    # Neighbor arrival
    if t < 0.0 and abs(t) > r:
        tj = None
    elif t > r:
        tj = (t - r) / max(vj, EPS)
    else:
        tj = 0.0
        inside_j = True

    # Ego arrival
    if s < 0.0 and abs(s) > r:
        ti = None
        ti_rogue = None
    elif s > r:
        ti = (s - r) / max(vi, EPS)
        # if tj is None: 
        #     ti_rogue = ti
        # else:
        #     # ti_entry = ti
        #     # ti_exit = (s + r*1) / max(vi, EPS)
        #     # tj_entry = tj
        #     # tj_exit = (tj_entry + r/max(vj, 1e-9))
        #     # if (ti_exit < tj_entry) or (tj_exit < ti_entry):
        #     #     ti_rogue = ti
        #     # else:   
        ti_rogue = (s + r*1.) / max(vi,EPS)
    else:
        ti = 0.0  # already inside or in [0, r)
        ti_rogue = 0.0
        inside_i = True

    # print(f"solve_ray_intersection: s={s}, t={t}, ti={ti}, tj={tj}, ti_rogue={ti_rogue}")
    return ti, tj, ti_rogue, inside_i, inside_j
