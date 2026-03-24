import math
import numpy as np
from scipy.optimize import minimize
from constants import NodConfig, D_SAFE


class MPCCBFController:
    """
    Safety-Critical MPC with Discrete-Time Control Barrier Functions.

    Implements MPC-CBF from: Zeng, Zhang, Sreenath (2021).

    At each timestep, solves a finite-horizon optimal control problem for a
    2D double integrator (Eq. 15 of paper) subject to discrete-time CBF
    safety constraints:

        h(x_{k+1|t}) >= (1 - gamma) * h(x_{k|t})   for all k = 0..N-1

    where h_ij(x) = ||p_i - p_j||^2 - D_SAFE^2  (Eq. 17 of paper).

    State:    x = [px, py, vx, vy]
    Control:  u = [ax, ay]
    Dynamics: x_{k+1} = A*x_k + B*u_k  (linear, Eq. 15)

    Output (v_lin, omega) is obtained by converting the optimal
    world-frame velocity x_1[2:4] to unicycle controls.

    Solver: SLSQP via scipy.optimize.minimize
    """

    def __init__(self):
        cfg = NodConfig.mpc_cbf

        self._N         = int(cfg.N)
        self._dt        = float(cfg.DT)
        self._gamma     = float(cfg.GAMMA)
        self._v_max     = float(NodConfig.kin.V_MAX)
        self._v_nom     = float(NodConfig.kin.V_NOMINAL)
        self._omega_max = float(cfg.OMEGA_MAX)
        self._a_max     = float(cfg.A_MAX)
        self._Q_v       = float(cfg.Q_V)
        self._Q_a       = float(cfg.Q_W)      # stage cost: acceleration weight
        self._Q_term    = float(cfg.Q_THETA)  # terminal cost: velocity alignment weight

        dt = self._dt
        # Double integrator system matrices (Eq. 15)
        self._A = np.array([[1, 0, dt,       0],
                             [0, 1,  0,      dt],
                             [0, 0,  1,       0],
                             [0, 0,  0,       1]], dtype=float)
        self._B = np.array([[0.5*dt*dt,        0],
                             [0,        0.5*dt*dt],
                             [dt,               0],
                             [0,               dt]], dtype=float)

        # Warm-start storage: flattened [ax0,ay0, ax1,ay1, ..., ax_{N-1},ay_{N-1}]
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
        pos = np.array(ego_info['position'], dtype=float)
        vel = np.array(ego_info['velocity'], dtype=float)
        x0  = np.array([pos[0], pos[1], vel[0], vel[1]], dtype=float)

        # Desired world-frame velocity (goal direction at nominal speed)
        v_des = self._v_nom * np.array([math.cos(goal_heading),
                                         math.sin(goal_heading)], dtype=float)

        # Predict neighbor positions over horizon (constant-velocity model)
        nb_trajs = []
        for nb in neighbors_dict.values():
            p0   = np.array(nb['position'], dtype=float)
            v_nb = np.array(nb['velocity'], dtype=float)
            traj = np.array([p0 + k * self._dt * v_nb
                              for k in range(self._N + 1)])
            nb_trajs.append(traj)

        # Warm start: shift previous solution, append zero acceleration
        if self._prev_u is not None:
            u0 = np.concatenate([self._prev_u[2:], [0.0, 0.0]])
        else:
            u0 = np.zeros(2 * self._N, dtype=float)

        # Bounds: ax, ay in [-a_max, a_max]
        bounds = [(-self._a_max, self._a_max),
                  (-self._a_max, self._a_max)] * self._N

        N      = self._N
        A      = self._A
        B      = self._B
        gamma  = self._gamma
        Q_v    = self._Q_v
        Q_a    = self._Q_a
        Q_term = self._Q_term
        v_max  = self._v_max

        def cost(u_flat):
            xs = _rollout(x0, u_flat, N, A, B)
            u_seq = u_flat.reshape(N, 2)
            total = 0.0
            for k in range(N):
                dvx = xs[k][2] - v_des[0]
                dvy = xs[k][3] - v_des[1]
                total += Q_v * (dvx*dvx + dvy*dvy)
                total += Q_a * float(np.dot(u_seq[k], u_seq[k]))
            # Terminal cost: penalize velocity misalignment
            dvx = xs[-1][2] - v_des[0]
            dvy = xs[-1][3] - v_des[1]
            total += Q_term * (dvx*dvx + dvy*dvy)
            return total

        def cbf_constraints(u_flat):
            """
            Returns array where each element must be >= 0.
            One constraint per (neighbor, horizon_step):
                h(x_{k+1}, p_j_{k+1}) - (1-gamma)*h(x_k, p_j_k) >= 0
            """
            if not nb_trajs:
                return np.array([1.0])
            xs = _rollout(x0, u_flat, N, A, B)
            vals = []
            for traj in nb_trajs:
                for k in range(N):
                    h_k   = _h_cbf(xs[k][0],   xs[k][1],
                                   traj[k,   0], traj[k,   1])
                    h_kp1 = _h_cbf(xs[k+1][0], xs[k+1][1],
                                   traj[k+1, 0], traj[k+1, 1])
                    vals.append(h_kp1 - (1.0 - gamma) * h_k)
            return np.array(vals)

        def vel_constraints(u_flat):
            """Keep each velocity component within [-v_max, v_max]."""
            xs = _rollout(x0, u_flat, N, A, B)
            vals = []
            for k in range(1, N + 1):
                vx, vy = xs[k][2], xs[k][3]
                vals.extend([v_max - vx, v_max + vx,
                              v_max - vy, v_max + vy])
            return np.array(vals)

        constraints = [
            {'type': 'ineq', 'fun': cbf_constraints},
            {'type': 'ineq', 'fun': vel_constraints},
        ]

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

        # Extract commanded velocity from optimal next state x_1
        xs_opt = _rollout(x0, u_opt, N, A, B)
        vx_cmd = float(xs_opt[1][2])
        vy_cmd = float(xs_opt[1][3])

        v_lin, omega = _world_vel_to_unicycle(
            vx_cmd, vy_cmd, float(ego_info['heading']),
            self._omega_max)
        return v_lin, omega


# ------------------------------------------------------------------
# Module-level helpers (no self overhead, called many times by solver)
# ------------------------------------------------------------------

def _rollout(x0, u_flat, N, A, B):
    """Simulate double integrator dynamics over N steps. Returns list of N+1 states."""
    xs = [x0]
    u_seq = u_flat.reshape(N, 2)
    for k in range(N):
        xs.append(A @ xs[-1] + B @ u_seq[k])
    return xs


def _h_cbf(ex, ey, ox, oy):
    """CBF value: h = ||p_ego - p_obs||^2 - D_SAFE^2  (Eq. 17)"""
    dx = ex - ox
    dy = ey - oy
    return dx*dx + dy*dy - D_SAFE*D_SAFE


def _world_vel_to_unicycle(vx, vy, heading, omega_max):
    """
    Convert optimal world-frame velocity to unicycle (v_lin, omega).

    omega: proportional to heading error (gain = KAPPA_ANG), not divided by dt —
           dividing by dt=0.1 turns any small misalignment into max angular speed.
    v_lin: scaled by cos(heading_err) so the robot doesn't drive sideways at full
           speed while still correcting its heading.
    """
    v_lin = math.sqrt(vx*vx + vy*vy)
    if v_lin < 1e-6:
        return 0.0, 0.0

    desired_heading = math.atan2(vy, vx)
    heading_err = math.atan2(math.sin(desired_heading - heading),
                              math.cos(desired_heading - heading))

    omega = float(np.clip(NodConfig.kin.KAPPA_ANG * heading_err,
                          -omega_max, omega_max))
    v_lin = float(np.clip(v_lin * max(0.0, math.cos(heading_err)),
                          0.0, NodConfig.kin.V_MAX))
    return v_lin, omega
