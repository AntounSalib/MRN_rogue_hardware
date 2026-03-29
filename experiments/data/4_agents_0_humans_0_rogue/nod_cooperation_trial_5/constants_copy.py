import numpy as np

ROBOT_NAMES = ("tb1", "tb2", "tb3", "tb4", "tb5", "tb6", "tb7", "tb8", "tb9", "tb10", "tb11", "tb12")
HUMAN_NAMES = {"crnr_x0_y0", "crnr_x0_y1", "crnr_x1_y0", "crnr_x1_y1"}
ROGUE_AGENTS = set()
ORCA_AGENTS = set()
ORCA_DD_AGENTS = {}
MPC_CBF_AGENTS = {}

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


TRIAL_ID = "4_agents_0_humans_0_rogue"
TRIAL_SEED = "nod_cooperation_trial_5"


EPS = 1e-5
D_SAFE = 0.4

class NodConfig:
    class pressure:
        T_COLL = 6.
        KAPPA_SAME = 1.0
        KAPPA_TCA = KAPPA_SAME
        KAPPA_DMIN = KAPPA_SAME
        DMIN_CLEAR = 2.5*D_SAFE
        PHI_TILT = 0.02
        TEMP_SM = 0.1

    class dynamics:
        U_0 = 2
        K_U = 0
        K_U_S = 0
        OPINION_DECAY = 1
        ATTENTION_DECAY = 1
        TAU_Z = 0.1
        TAU_COOPERATION = 0.01
        TIMING_TAU_U_RELAX = TAU_Z
        TAU_Z_RELAX = TAU_Z
        ITERATIONS_OD = 50

    class neighbors:
        SENSING_RANGE = 15
        R_PRED = 1.5*np.sqrt(2)*D_SAFE # 0.84
        R_OCC = 1.*D_SAFE


    class kin:
        V_NOMINAL = 0.35
        V_ROGUE = 0.2
        KAPPA_Z = 5.0
        KAPPA_V = 5.0
        KAPPA_ANG = 2.0
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
