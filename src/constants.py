import numpy as np

ROBOT_NAMES = ("tb1", "tb2", "tb3", "tb4", "tb5", "tb6", "tb7", "tb8", "tb9", "tb10", "tb11", "tb12")
HUMAN_NAMES = {"crnr_x0_y0", "crnr_x0_y1", "crnr_x1_y0", "hat2"}
ROGUE_AGENTS = {}
ROGUE_SPEEDS = {
    "tb2": 0.3,
    "tb4": 0.28,
    "tb8": 0.55,
}
ORCA_AGENTS = {}
ORCA_DD_AGENTS = {}
MPC_CBF_AGENTS = {}
ACTIVE_ROBOTS = {"tb1", "tb2", "tb3", "tb5", "tb6", "tb9"}

def get_agent_type(name):
    if name in ROGUE_AGENTS:
        return "ROGUE"
    if name in ORCA_DD_AGENTS:
        return "ORCA_DD"
    if name in MPC_CBF_AGENTS:
        return "MPC_CBF"
    if name in ORCA_AGENTS:
        return "ORCA"
    return "NOD"


TRIAL_ID = "robot_symposium"
TRIAL_SEED = "agent_stopped_in_path"

RESET_TO_START = 0
# START_POSITIONS = {
#     "tb1": (-2.6,  0.5,   0.0),      # NOD,   from west, heading east
#     "tb9": ( 2.6,  1.2,  -3.1416),  # NOD,   from east, heading west
#     "tb6": (-2.6, -0.5,   0.0),     # ROGUE, from west, heading east
#     "tb3": (-0.5,  2.6,  -1.5708),  # NOD,   from north, heading south
#     "tb2": ( 0.5,  2.4,  -1.5708),  # NOD,   from north, heading south
#     "tb5": ( 1.2, -1.5,   1.5708),  # ROGUE, from south, heading north
# }

START_POSITIONS = {
    "tb1": (-1.8,  0.5,   0.0),      # NOD,   from west, heading east
    "tb9": ( 2.0,  1.2,  -3.1416),  # NOD,   from east, heading west
    "tb6": (-1.7, -0.8,   0.0),     # ROGUE, from west, heading east
    "tb3": (-0.5,  2.3,  -1.5708),  # NOD,   from north, heading south
    "tb2": ( 0.5,  2.4,  -1.5708),  # NOD,   from north, heading south
    "tb5": ( 1.2, -1.5,   1.5708),  # ROGUE, from south, heading north
}


EPS = 1e-2
D_SAFE = 0.4

class NodConfig:
    class pressure:
        T_COLL = 5.
        KAPPA_SAME = 5.0
        KAPPA_TCA = KAPPA_SAME
        KAPPA_DMIN = KAPPA_SAME
        DMIN_CLEAR = np.sqrt(5)*D_SAFE # 0.84
        PHI_TILT = 0.05
        TEMP_SM = 0.1

    class dynamics:
        U_0 = 2
        K_U = 0
        K_U_S = 0
        OPINION_DECAY = 1
        ATTENTION_DECAY = 1
        TAU_Z = 1
        TAU_COOPERATION = 1
        TIMING_TAU_U_RELAX = TAU_Z
        TAU_Z_RELAX = TAU_Z
        ITERATIONS_OD = 50

    class neighbors:
        SENSING_RANGE = 15
        R_PRED = np.sqrt(5)*D_SAFE # 0.84
        R_OCC = 1.*D_SAFE


    class kin:
        V_NOMINAL = 0.35
        V_ROGUE = 0.35
        KAPPA_Z = 2.0
        KAPPA_V = 5.0
        KAPPA_ANG = 3.0
        KAPPA_ANG_I = 0.5
        V_MAX = 0.5

    class cooperation:
        COOPERATION_LAYER_ON = True
        COOPERATION_THRESHOLD = 0.0

    class orca_dd:
        E = 0.05          # tracking error bound (m)
        T = 0.1           # time step (s)
        OMEGA_MAX = 1.82  # TurtleBot3 Waffle max angular velocity (rad/s)
        WHEEL_SEP = 0.287 # TurtleBot3 Waffle wheel separation (m)
        FORWARD_ONLY = True
        PAHV_ANGLES = 65

    class mpc_cbf:
        N = 5             # prediction horizon (steps)
        DT = 0.1          # time step (s)
        GAMMA = 0.3       # CBF decay rate in (0,1]; smaller = more conservative / earlier avoidance
        OMEGA_MAX = 1.82  # TurtleBot3 Waffle max angular velocity (rad/s)
        A_MAX = 0.5       # max acceleration magnitude (m/s^2)
        Q_V = 1.0         # stage cost: velocity deviation weight
        Q_W = 0.1         # stage cost: acceleration weight
        Q_THETA = 5.0     # terminal cost: velocity alignment weight
