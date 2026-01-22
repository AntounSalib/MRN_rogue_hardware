import numpy as np

ROBOT_NAMES = ("tb1", "tb2", "tb3", "tb4", "tb5", "tb6", "tb7", "tb8", "tb9")

EPS = 1e-5
D_SAFE = .35

# Nod parameters
T_COLL = 4.
KAPPA_SAME = 1.0
KAPPA_TCA = KAPPA_SAME
KAPPA_DMIN = KAPPA_SAME
DMIN_CLEAR = 3*D_SAFE
PHI_TILT = 0.02
TEMP_SM = 0.1
U_0 = 10
K_U = 5
OPINION_DECAY = 1
ATTENTION_DECAY = 1
TAU_Z = 0.1
TIMING_TAU_U_RELAX = TAU_Z
K_U_S = 0
TAU_Z_RELAX = TAU_Z
ITERATIONS_OD = 50

class NodConfig:
    class neighbors:
        SENSING_RANGE = 10
        R_PRED = 1.5*np.sqrt(2)*D_SAFE


    class kin:
        V_NOMINAL = 0.1
        KAPPA_Z = 1.0
        V_MAX = 1.0