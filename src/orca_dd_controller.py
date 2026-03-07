import math
import numpy as np
import rvo2
from constants import NodConfig, D_SAFE, EPS


class NHORCAController:
    """
    NH-ORCA Differential Drive controller (Alonso-Mora et al.).

    Adapted from ORCADDAgent for use with ROS turtlebot info dicts.

    Key ideas:
      - ORCA is run on inflated disks (radius = D_SAFE/2 + E_eff).
      - ORCA produces a holonomic velocity v_H* in world frame.
      - v_H* is projected into the admissible holonomic set P_AHV.
      - P_AHV is mapped to (v, omega) via Eq. (9) with RA/RB branches.
      - Remark 4: E_eff is shrunk so inflated disks do not overlap.
    """

    _EPS = 1e-9

    def __init__(self):
        cfg = NodConfig.orca_dd

        self._E_nom = float(cfg.E)
        self._E_eff = float(self._E_nom)
        self._T = float(cfg.T)
        self._vmax = float(NodConfig.kin.V_MAX)
        self._omega_max = float(cfg.OMEGA_MAX)
        self._lw = float(cfg.WHEEL_SEP)
        self._forward_only = bool(cfg.FORWARD_ONLY)
        self._pahv_angles = int(cfg.PAHV_ANGLES)

        # Last commanded linear speed (used to report ego velocity to ORCA)
        self._v_cmd = 0.0

        # P_AHV polygon in body frame (CCW vertices)
        self._pahv_poly_body = np.zeros((0, 2), dtype=float)
        self._Vx_max0, self._Vy_max0 = self._compute_pahv_bounds()
        self._rebuild_pahv_polygon(self._pahv_angles)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def compute_velocity(self, ego_info: dict, neighbors_dict: dict, goal_heading: float):
        """
        Compute (v_lin, omega) for one control tick.

        ego_info:       {'position': [x,y], 'velocity': [vx,vy], 'heading': theta}
        neighbors_dict: {name: ego_info-like dict, ...}
        goal_heading:   fixed goal direction in world frame (radians)
        """
        # Remark 4: update E_eff based on neighbor distances
        self._update_E_eff(ego_info, neighbors_dict)

        ego_heading = float(ego_info['heading'])
        ego_radius = D_SAFE / 2.0 + self._E_eff
        max_speed = max(0.0, self._Vx_max0)

        sim = rvo2.PyRVOSimulator(
            self._T,                             # time step
            NodConfig.neighbors.SENSING_RANGE,   # neighbor distance
            10,                                  # max neighbors
            2.0,                                 # time horizon
            2.0,                                 # time horizon obstacles
            ego_radius,                          # agent radius (inflated)
            max_speed,                           # max speed
        )

        ego_id = sim.addAgent(tuple(ego_info['position']))

        # Report ego velocity as v_cmd projected onto current heading
        e = np.array([math.cos(ego_heading), math.sin(ego_heading)], dtype=float)
        sim.setAgentVelocity(ego_id, (float(self._v_cmd * e[0]), float(self._v_cmd * e[1])))
        sim.setAgentMaxSpeed(ego_id, max_speed)

        # Preferred velocity: goal direction clipped to P_AHV
        v_pref_world = np.array([NodConfig.kin.V_NOMINAL * math.cos(goal_heading),
                                  NodConfig.kin.V_NOMINAL * math.sin(goal_heading)], dtype=float)
        v_pref_clipped = self._clip_to_pahv(v_pref_world, ego_heading)
        sim.setAgentPrefVelocity(ego_id, tuple(v_pref_clipped))

        for neighbor_info in neighbors_dict.values():
            n_id = sim.addAgent(tuple(neighbor_info['position']))
            sim.setAgentVelocity(n_id, tuple(neighbor_info['velocity']))
            h = neighbor_info['heading']
            sim.setAgentPrefVelocity(n_id, (NodConfig.kin.V_NOMINAL * math.cos(h),
                                             NodConfig.kin.V_NOMINAL * math.sin(h)))

        sim.doStep()
        v_H_star = np.array(sim.getAgentVelocity(ego_id), dtype=float)

        v_H_star = self._clip_to_pahv(v_H_star, ego_heading)
        v_cmd, omega_cmd = self._map_holonomic_to_controls(v_H_star, ego_heading)

        self._v_cmd = v_cmd
        return v_cmd, omega_cmd

    # ------------------------------------------------------------------
    # Remark 4: shrink E_eff
    # ------------------------------------------------------------------

    def _update_E_eff(self, ego_info: dict, neighbors_dict: dict) -> None:
        """
        Shrink E_eff so that (base_r + E_eff) + (base_r + E_nom_j) <= d_ij.
        Rebuilds the P_AHV polygon if E_eff changes.
        """
        base_r = D_SAFE / 2.0
        E_nom = float(self._E_nom)
        E_eff = E_nom

        pos_i = np.array(ego_info['position'], dtype=float)

        for neighbor_info in neighbors_dict.values():
            p_j = np.array(neighbor_info['position'], dtype=float)
            d = float(np.linalg.norm(p_j - pos_i))
            if d <= 1e-12:
                E_eff = 0.0
                break
            max_E_i = d - (2.0 * base_r + E_nom) - 1e-6
            if max_E_i < E_eff:
                E_eff = max(0.0, max_E_i)

        new_E = float(min(E_nom, E_eff))
        if abs(new_E - self._E_eff) > 1e-9:
            self._E_eff = new_E
            self._Vx_max0, self._Vy_max0 = self._compute_pahv_bounds()
            self._rebuild_pahv_polygon(self._pahv_angles)

    # ------------------------------------------------------------------
    # Paper math (Alonso-Mora et al.)
    # ------------------------------------------------------------------

    def _vmax_omega(self, omega: float) -> float:
        """vmax,omega = vmax - |omega| * l_w / 2"""
        return float(max(0.0, self._vmax - 0.5 * abs(float(omega)) * self._lw))

    def _VH_max(self, thetaH: float) -> float:
        """
        Eq. (13): maximum trackable holonomic speed in body-frame direction thetaH.
        Uses stable numerics (identity: (1-cos)^2 = 2(1-cos) - sin^2 avoided via direct form).
        """
        E = float(self._E_eff)
        T = float(self._T)
        vmax = float(self._vmax)
        omega_max = float(self._omega_max)

        th = abs(float(thetaH))
        if th < 1e-6:
            return vmax

        c = math.cos(th)
        s = math.sin(th)
        one_c = 1.0 - c
        if one_c <= 1e-9:
            return vmax

        denom = one_c * one_c  # avoids cancellation in sqrt

        # RA1 candidate (Eq. 10 -> Eq. 13 top)
        VH_ra1 = (E / max(T, self._EPS)) * math.sqrt((2.0 * one_c) / max(denom, 1e-18))

        # v*_E (Eq. 14): threshold to choose RA1 vs RA2
        vE_star = (E / max(T, self._EPS)) * (th * s) / (2.0 * max(one_c, 1e-12)) * math.sqrt(
            (2.0 * one_c) / max(denom, 1e-18)
        )

        if (th / max(T, self._EPS)) <= omega_max + 1e-12:
            omega = th / max(T, self._EPS)
            vmax_w = self._vmax_omega(omega)

            if vE_star <= vmax_w + 1e-12:
                return float(min(VH_ra1, vmax))

            # RA2 quadratic (Eq. 11, 15)
            alpha = T * T
            beta = -(2.0 * T * T) * (s / th) * vmax_w
            gamma = (2.0 * T * T) * (one_c / (th * th)) * (vmax_w * vmax_w) - (E * E)

            disc = beta * beta - 4.0 * alpha * gamma
            if disc <= 0.0:
                return 0.0
            VH_ra2 = (-beta + math.sqrt(disc)) / (2.0 * alpha)
            return float(min(VH_ra2, vmax))

        # RB (Eq. 13 bottom)
        VH_rb = (E * omega_max) / th
        return float(min(VH_rb, vmax))

    def _compute_pahv_bounds(self):
        """Conservative scalar bounds: Vx_max = VH_max(0), Vy_max = VH_max(pi/2)."""
        Vx = float(min(self._vmax, self._VH_max(0.0)))
        Vy = float(min(self._vmax, self._VH_max(0.5 * math.pi)))
        return (max(0.0, 0.99 * Vx), max(0.0, 0.99 * Vy))

    def _rebuild_pahv_polygon(self, n_angles: int = 65) -> None:
        """Convex polygon inner approximation of S_AHV in body frame."""
        if n_angles < 9:
            n_angles = 9

        if self._forward_only:
            angles = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_angles)
        else:
            angles = np.linspace(-np.pi, np.pi, n_angles)

        pts = []
        for ang in angles:
            vh = float(self._VH_max(float(ang)))
            if vh <= 1e-9:
                continue
            x = vh * math.cos(ang)
            y = vh * math.sin(ang)
            if self._forward_only and x < 0.0:
                x = 0.0
            pts.append([x, y])

        pts.append([0.0, 0.0])
        P = np.asarray(pts, dtype=float)
        if P.shape[0] < 3:
            self._pahv_poly_body = P
            return
        self._pahv_poly_body = self._convex_hull_ccw(P)

    @staticmethod
    def _convex_hull_ccw(points: np.ndarray) -> np.ndarray:
        """Andrew monotone chain convex hull. Returns CCW hull vertices (no repeat)."""
        pts = np.unique(points, axis=0)
        if pts.shape[0] <= 2:
            return pts

        pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
                lower.pop()
            lower.append(p)

        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
                upper.pop()
            upper.append(p)

        return np.vstack((lower[:-1], upper[:-1]))

    # ------------------------------------------------------------------
    # Feasible-set projection
    # ------------------------------------------------------------------

    def _clip_to_pahv(self, v_world: np.ndarray, heading: float) -> np.ndarray:
        """Project world-frame velocity into P_AHV (body-frame polygon), rotate back."""
        v_world = np.asarray(v_world, dtype=float).reshape(2)
        c = math.cos(heading)
        s = math.sin(heading)

        # world -> body
        v_body = np.array([c * v_world[0] + s * v_world[1],
                           -s * v_world[0] + c * v_world[1]], dtype=float)

        if self._forward_only:
            v_body[0] = max(0.0, float(v_body[0]))

        poly = self._pahv_poly_body
        if poly.shape[0] >= 3:
            v_body = self._project_onto_convex_polygon(v_body, poly)
        else:
            v_body[0] = float(np.clip(v_body[0],
                                      0.0 if self._forward_only else -self._Vx_max0,
                                      self._Vx_max0))
            v_body[1] = float(np.clip(v_body[1], -self._Vy_max0, self._Vy_max0))

        # body -> world
        return np.array([c * v_body[0] - s * v_body[1],
                         s * v_body[0] + c * v_body[1]], dtype=float)

    @staticmethod
    def _project_onto_convex_polygon(p: np.ndarray, poly: np.ndarray) -> np.ndarray:
        """
        Euclidean projection of p onto convex polygon (CCW vertices).
        Returns p if inside; otherwise returns closest point on boundary.
        """
        p = np.asarray(p, dtype=float).reshape(2)
        V = np.asarray(poly, dtype=float)
        n = V.shape[0]

        inside = True
        for i in range(n):
            a = V[i]
            b = V[(i + 1) % n]
            if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < -1e-12:
                inside = False
                break
        if inside:
            return p

        best = None
        best_d2 = float("inf")
        for i in range(n):
            a = V[i]
            b = V[(i + 1) % n]
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom <= 1e-15:
                q = a
            else:
                t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
                q = a + t * ab
            d2 = float(np.dot(p - q, p - q))
            if d2 < best_d2:
                best_d2 = d2
                best = q
        return best if best is not None else V[0]

    # ------------------------------------------------------------------
    # Eq. (9) mapping: holonomic -> (v, omega)
    # ------------------------------------------------------------------

    def _map_holonomic_to_controls(self, vH_world: np.ndarray, heading: float):
        """
        Eq. (9) RA/RB mapping:
          - thetaH = body-frame angle of v_H*
          - RB: if |thetaH|/T > omega_max  =>  v=0, omega=sign(thetaH)*omega_max
          - RA: omega = thetaH/T; v* = VH * thetaH*sin(thetaH) / (2*(1-cos(thetaH)))
        """
        vH_world = np.asarray(vH_world, dtype=float).reshape(2)
        VH = float(np.linalg.norm(vH_world))
        if VH <= 1e-12:
            return (0.0, 0.0)

        c = math.cos(heading)
        s = math.sin(heading)
        v_body = np.array([c * vH_world[0] + s * vH_world[1],
                           -s * vH_world[0] + c * vH_world[1]], dtype=float)

        thetaH = float(math.atan2(v_body[1], v_body[0]))
        T = float(self._T)
        omega_max = float(self._omega_max)

        # RB branch
        if abs(thetaH) / max(T, self._EPS) > omega_max + 1e-12:
            return (0.0, float(np.sign(thetaH) * omega_max))

        # RA branch
        omega = float(np.clip(thetaH / max(T, self._EPS), -omega_max, omega_max))

        th = abs(thetaH)
        if th <= 1e-6:
            v_star = VH
        else:
            one_c = 1.0 - math.cos(th)
            v_star = VH * (th * math.sin(th)) / (2.0 * max(one_c, 1e-12))

        vmax_w = self._vmax_omega(omega)
        v_cmd = float(np.clip(v_star, 0.0, min(self._vmax, vmax_w)))
        return (v_cmd, omega)
