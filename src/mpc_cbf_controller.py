import math
import numpy as np
from scipy.optimize import minimize
from constants import NodConfig, D_SAFE


class MPCCBFController:
    """
    Safety-Critical MPC with Discrete-Time Control Barrier Functions.

    Implements MPC-CBF from: Zeng, Zhang, Sreenath (2021).

    At each timestep, solves a finite-horizon optimal control problem for a
    unicycle robot subject to discrete-time CBF safety constraints:

        h(x_{k+1|t}) >= (1 - gamma) * h(x_{k|t})   for all k = 0..N-1

    where h_ij(x) = ||p_i - p_j||^2 - D_SAFE^2  (Eq. 17 of paper).

    Smaller gamma  => stricter set invariance (more conservative, earlier avoidance).
    Larger  gamma  => looser (robot avoids only when close, like MPC-DC).

    Robot model: unicycle  x = [px, py, theta],  u = [v, omega]
    Solver:      SLSQP via scipy.optimize.minimize
    """

    def __init__(self):
        cfg = NodConfig.mpc_cbf

        self._N = int(cfg.N)
        self._dt = float(cfg.DT)
        self._gamma = float(cfg.GAMMA)
        self._v_max = float(NodConfig.kin.V_MAX)
        self._v_nom = float(NodConfig.kin.V_NOMINAL)
        self._omega_max = float(cfg.OMEGA_MAX)
        self._Q_v = float(cfg.Q_V)
        self._Q_w = float(cfg.Q_W)
        self._Q_theta = float(cfg.Q_THETA)

        # Warm-start storage: flattened [v0,w0, v1,w1, ..., v_{N-1},w_{N-1}]
        self._prev_u = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_velocity(self, ego_info: dict, neighbors_dict: dict, goal_heading: float):
        """
        Solve MPC-CBF and return (v_lin, omega) for this timestep.

        ego_info:       {'position': [x,y], 'velocity': [vx,vy], 'heading': theta}
        neighbors_dict: {name: ego_info-like dict, ...}
        goal_heading:   desired global heading (fixed goal direction, radians)
        """
        x0 = np.array([
            ego_info['position'][0],
            ego_info['position'][1],
            ego_info['heading'],
        ], dtype=float)

        # Predict neighbor positions over horizon with constant-velocity model
        nb_trajs = []
        for nb in neighbors_dict.values():
            p0 = np.array(nb['position'], dtype=float)
            vel = np.array(nb['velocity'], dtype=float)
            # traj[k] = position at horizon step k
            traj = np.array([p0 + k * self._dt * vel for k in range(self._N + 1)])
            nb_trajs.append(traj)

        # Warm start: shift previous solution and append nominal
        if self._prev_u is not None:
            u0 = np.concatenate([self._prev_u[2:], [self._v_nom, 0.0]])
        else:
            u0 = np.tile([self._v_nom, 0.0], self._N).astype(float)

        # Bounds: v in [0, v_max], omega in [-omega_max, omega_max]
        bounds = [(0.0, self._v_max), (-self._omega_max, self._omega_max)] * self._N

        N = self._N
        dt = self._dt
        gamma = self._gamma
        Q_v = self._Q_v
        Q_w = self._Q_w
        Q_theta = self._Q_theta
        v_nom = self._v_nom

        def cost(u_flat):
            xs = _rollout(x0, u_flat, N, dt)
            u_seq = u_flat.reshape(N, 2)
            total = 0.0
            for k in range(N):
                v, w = u_seq[k]
                total += Q_v * (v - v_nom) ** 2 + Q_w * w ** 2
            # Terminal cost: penalize heading misalignment
            h_err = math.atan2(math.sin(xs[-1][2] - goal_heading),
                               math.cos(xs[-1][2] - goal_heading))
            total += Q_theta * h_err ** 2
            return total

        def cbf_constraints(u_flat):
            """
            Returns array where each element must be >= 0.
            One constraint per (neighbor, horizon_step):
                h(x_{k+1}, p_j_{k+1}) - (1-gamma)*h(x_k, p_j_k) >= 0
            """
            if not nb_trajs:
                return np.array([1.0])
            xs = _rollout(x0, u_flat, N, dt)
            vals = []
            for traj in nb_trajs:
                for k in range(N):
                    h_k = _h_cbf(xs[k][0], xs[k][1], traj[k, 0], traj[k, 1])
                    h_kp1 = _h_cbf(xs[k + 1][0], xs[k + 1][1], traj[k + 1, 0], traj[k + 1, 1])
                    vals.append(h_kp1 - (1.0 - gamma) * h_k)
            return np.array(vals)

        constraints = [{'type': 'ineq', 'fun': cbf_constraints}]

        try:
            result = minimize(
                cost, u0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 150, 'ftol': 1e-4, 'disp': False},
            )
            u_opt = result.x
        except Exception:
            u_opt = u0

        self._prev_u = u_opt.copy()

        v_cmd = float(np.clip(u_opt[0], 0.0, self._v_max))
        omega_cmd = float(np.clip(u_opt[1], -self._omega_max, self._omega_max))
        return v_cmd, omega_cmd


# ------------------------------------------------------------------
# Module-level helpers (no self overhead, called many times by solver)
# ------------------------------------------------------------------

def _rollout(x0, u_flat, N, dt):
    """Simulate unicycle dynamics over N steps. Returns list of N+1 states."""
    xs = [x0]
    u_seq = u_flat.reshape(N, 2)
    for k in range(N):
        v, w = u_seq[k]
        px, py, th = xs[-1]
        xs.append(np.array([
            px + v * math.cos(th) * dt,
            py + v * math.sin(th) * dt,
            th + w * dt,
        ]))
    return xs


def _h_cbf(ex, ey, ox, oy):
    """CBF value: h = ||p_ego - p_obs||^2 - D_SAFE^2  (Eq. 17)"""
    dx = ex - ox
    dy = ey - oy
    return dx * dx + dy * dy - D_SAFE * D_SAFE
