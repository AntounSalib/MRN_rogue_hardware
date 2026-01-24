import numpy as np

ROBOT_NAMES = ("tb1", "tb2", "tb3", "tb4", "tb5", "tb6", "tb7", "tb8", "tb9", "tb10", "tb11", "tb12")
HUMAN_NAMES = {"hat2"}
ROGUE_AGENTS = {'tb10'}

TRIAL_ID = 1
TRIAL_SEED = 2


EPS = 1e-5
D_SAFE = 0.4

# Nod parameters
T_COLL = 2.
KAPPA_SAME = 1.0
KAPPA_TCA = KAPPA_SAME
KAPPA_DMIN = KAPPA_SAME
DMIN_CLEAR = 2*D_SAFE
PHI_TILT = 0.02
TEMP_SM = 0.1
U_0 = 1
K_U = 0
OPINION_DECAY = 1
ATTENTION_DECAY = 1
TAU_Z = 0.01
TIMING_TAU_U_RELAX = TAU_Z
K_U_S = 0
TAU_Z_RELAX = TAU_Z
ITERATIONS_OD = 50

class NodConfig:
    class neighbors:
        SENSING_RANGE = 2
        R_PRED = 1.5*np.sqrt(2)*D_SAFE
        R_OCC = 1.*D_SAFE


    class kin:
        V_NOMINAL = 0.35
        KAPPA_Z = 3.0
        KAPPA_V = 3.0
        V_MAX = 1

    class cooperation:
        COOPERATION_LAYER_ON = True
        COOPERATION_THRESHOLD = 0.3
    
import numpy as np

# ROBOT_NAMES = ("tb1", "tb2", "tb3", "tb4", "tb5", "tb6", "tb7", "tb8", "tb9")

# EPS = 1e-5
# D_SAFE = 0.3

# # Nod parameters
# T_COLL = 2.
# KAPPA_SAME = 1.0
# KAPPA_TCA = KAPPA_SAME
# KAPPA_DMIN = KAPPA_SAME
# KAPPA_ATT = 3.0 # make it 1.0 in congested environemnts
# DMIN_CLEAR = 2*D_SAFE
# PHI_TILT = 0.02
# TEMP_SM = 0.1
# U_0 = 10
# K_U = 5
# OPINION_DECAY = 1
# ATTENTION_DECAY = 1
# TAU_Z = 0.01
# TIMING_TAU_U_RELAX = TAU_Z
# K_U_S = 0
# TAU_Z_RELAX = TAU_Z
# ITERATIONS_OD = 50

# class NodConfig:
#     class neighbors:
#         SENSING_RANGE = 10
#         R_PRED = 1.5*np.sqrt(2)*D_SAFE
#         R_OCC = 1.*D_SAFE#/np.sqrt(2)

#     class kin:
#         V_NOMINAL = 0.35
#         KAPPA_Z = 3.0
#         V_MAX = 1
#         KAPPA_V = 3.0
        
    
