import numpy as np

ROBOT_NAMES = ("tb1", "tb2", "tb3", "tb4", "tb5", "tb6", "tb7", "tb8", "tb9", "tb10", "tb11", "tb12")
HUMAN_NAMES = {"crnr_x0_y0", "crnr_x0_y1", "crnr_x1_y0", "crnr_x1_y1"}
ROGUE_AGENTS = {}

TRIAL_ID = "ID_3_4H"
TRIAL_SEED = "1"


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
        TAU_Z = 0.01
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
        V_MAX = 0.5

    class cooperation:
        COOPERATION_LAYER_ON = True
        COOPERATION_THRESHOLD = 0.0
