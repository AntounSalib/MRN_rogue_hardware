from typing import Set
from constants import sensing_range

def sensed_neighbors(ego_info: dict, neighbor_dict: dict) -> Set[str]:
    """
    Given a dictionary of neighbors, return a set of names of neighbors that are currently sensed.
    A neighbor is considered sensed if its within sensing range.

    Args:
        neighbor_dict (dict): A dictionary where keys are neighbor names and values are dictionaries
                              containing neighbor information, including 'position'.
    Returns:
        Set[str]: A set of names of sensed neighbors."""
    
    sensed = set()
    ego_pos = ego_info['position']
    for name, info in neighbor_dict.items():
        neighbor_pos = info['position']
        distance = ((ego_pos[0] - neighbor_pos[0]) ** 2 + (ego_pos[1] - neighbor_pos[1]) ** 2) ** 0.5
        if distance <= sensing_range:
            sensed.add(name)

    return sensed