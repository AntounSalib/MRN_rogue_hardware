import math
from collections import defaultdict
import numpy as np
from typing import List, Tuple
from neighbors import conflicting_neighbors, tca_and_rmin, arrival_times_to_disk, sensed_neighbors, solve_ray_intersection
from scipy.special import expit
from scipy.integrate import solve_ivp
from constants import NodConfig, EPS, D_SAFE, HUMAN_NAMES

class NodController:
    def __init__(self, robot_name: str, time: float):
        # state variables
        self.robot_name = robot_name
        self.current_time = time
        self.time_step = 0.0

        # nod variables
        self.z = 0
        self.u = 0
        self.pairwise_u = defaultdict(float)

        # cooperation variables
        self.pairwise_cooperation = defaultdict(float)
        self.pairwise_cooperation_attention = defaultdict(float)
        self.previous_pairwise_phi = defaultdict(float)
        self.previous_vj = defaultdict(float)
        self.neighbor_ever_moved = defaultdict(bool)

    @staticmethod
    def _softmax(x: float, y: float, tau: float) -> float:
        """Smooth, differentiable approximation of max(x, y) with temperature tau."""
        tau = max(tau, EPS)
        M = max(x, y)
        return M + tau * math.log(math.exp(-(M - x) / tau) + math.exp(-(M - y) / tau))

    def _urgency_from_tstar(self, s: float, t_pt: float, vi: float, vj: float,
                            a: float, b: float, sin_alpha: float,) -> float:

        r_eff = NodConfig.neighbors.R_OCC / max(abs(sin_alpha), EPS)
        t_exit_i = (s + r_eff) / max(vi, EPS)
        t_exit_j = (t_pt + r_eff) / max(vj, EPS)


        t_star_raw = self._softmax(0.0, -b / (a + EPS), NodConfig.pressure.TAU_SOFT_URGENCY)



        U_exit = float(expit(NodConfig.pressure.KAPPA_URGENCY * (t_exit_i - t_star_raw - NodConfig.pressure.DELTA_T_BUFFER))) * float(expit(NodConfig.pressure.KAPPA_URGENCY * (t_exit_j - t_star_raw - NodConfig.pressure.DELTA_T_BUFFER)))

        # margin = (2 * r_eff)/cfg.kin.v_nom  - min(abs(t_exit_j - t_ent_i), abs(t_exit_i - t_ent_j))
        return U_exit

    def update_opinion(self, ego_info: dict, neighbors_dict: dict, current_time: float):
        # print(f"robot: {self.robot_name}, conflicting neighbors: {c_neighbors}")

        # Use elapsed control time, bounding startup waits and long stalls.
        dt = max(0.0, min(current_time - self.current_time, 0.1))
        self.current_time = current_time
        self.time_step = dt
        if dt == 0.0:
            return float(np.clip((1.0 + np.tanh(NodConfig.kin.KAPPA_Z * self.z))
                                 * NodConfig.kin.V_NOMINAL, 0.0, NodConfig.kin.V_MAX))

        sens_neighbors = sensed_neighbors(ego_info, neighbors_dict)

        Pis, Gis, Uis = self._compute_pressure_and_gates(ego_info, neighbors_dict, sens_neighbors)

        a_sum,sumP = self._aggregate(Pis, Gis, Uis)

        # update nod variables

        # for _ in range(n_fast):
        self.z, self.u = self._integrate_fast(self.z, self.u, a_sum, sumP, dt)

        # compute target velocity
        v0 = NodConfig.kin.V_NOMINAL
        v_tar = np.clip((1.0 + np.tanh(NodConfig.kin.KAPPA_Z* self.z)) * v0, 0.0, NodConfig.kin.V_MAX)

        # if a_sum is None:
        #     print(f"robot: {self.robot_name}, conf neighbors: {conf_neighbors}, Pis: {Pis}, Gis: {Gis}, Uis: {Uis}, z: {self.z:.3f}, u: {self.u:.3f}, v_tar: {v_tar:.3f}")
        # else:
        #     print(f"robot: {self.robot_name}, conf neighbors: {conf_neighbors}, Pis: {Pis}, Gis: {Gis}, Uis: {Uis}, a_sum: {a_sum:.3f}, z: {self.z:.3f}, u: {self.u:.3f}, v_tar: {v_tar:.3f}")

        return v_tar

    def relax_opinion(self, current_time: float):
        """Apply free-flow decay during a terminal stop, without a speed target."""
        dt = max(0.0, min(current_time - self.current_time, 0.1))
        self.current_time = current_time
        self.time_step = dt
        self.z, self.u = self._integrate_fast(self.z, self.u, None, None, dt)
        self.pairwise_u.clear()

    def _update_cooperation_for_neighbor(self, ego_info: dict, neighbor_info: dict, neighbor: str) -> None:
        time_step = self.time_step

        # relative vectors
        vij_vec = np.array(neighbor_info['velocity']) - np.array(ego_info['velocity'])
        pij = np.array(neighbor_info['position']) - np.array(ego_info['position'])
        pij_norm = float(np.linalg.norm(pij))
        pij_norm = max(pij_norm, 1e-6)
        vij_vec_norm = float(np.linalg.norm(vij_vec))
        vij_vec_norm = max(vij_vec_norm, 1e-6)
        theta = np.arccos(max(min(np.dot(vij_vec, pij)/(vij_vec_norm*pij_norm), 1), -1))

        safe_ratio = np.clip(D_SAFE / pij_norm, -1.0, 1.0)
        theta_prime = np.arcsin(safe_ratio)
        Phi_prime = np.cos(theta_prime)
        Phi_geom = np.cos(np.pi - theta)

        # phi calculations
        last_Phi = self.previous_pairwise_phi[neighbor]
        Phi = np.cos(theta)
        self.previous_pairwise_phi[neighbor] = Phi
        delta_Phi = Phi - last_Phi

        # neighbor velocity calculations
        prev_vj = self.previous_vj[neighbor]
        vj = np.array(neighbor_info['velocity'], dtype=float)
        self.previous_vj[neighbor] = vj
        delta_vj = np.linalg.norm(vj) - np.linalg.norm(prev_vj)

        pij_hat = pij/pij_norm
        vij_hat = vij_vec/vij_vec_norm
        vj_norm = float(np.linalg.norm(vj))
        vj_unit = vj / max(vj_norm, EPS)
        e_t = (1 / vij_vec_norm) * (vj_unit - vij_hat * (np.dot(vij_hat, vj)))

        Phi_dot_vj = np.dot(pij_hat, e_t)

        latest_cooperation_score = self.pairwise_cooperation[neighbor]
        vj_scalar = float(np.linalg.norm(neighbor_info['velocity']))
        # x = math.tanh(10*vj_scalar)  # old
        x = float(expit(10*vj_scalar))
        y = math.tanh(abs(delta_Phi))
        # delta_vj as acceleration (reference uses aj, not speed difference)
        dt_coop = time_step
        # g = math.tanh(1*delta_vj)  # old: speed difference
        g = math.tanh(1*(delta_vj / max(dt_coop, EPS)))

        # bj = x*(delta_vj*(-Phi_dot_vj))/abs(delta_Phi)+(1-x)*((-1*y*delta_Phi)+(1-y))
        bj = abs(g)*g*(math.tanh(10*Phi_dot_vj)) \
                        + (1-abs(g))* math.tanh(10*(Phi_prime-Phi_geom)) + 1-x

        _,_,_, d_min = tca_and_rmin(ego_info, neighbor_info, False, False)
        d = 1
        u_prev = self.pairwise_cooperation_attention[neighbor]
        cooperation_prev = latest_cooperation_score

        def _cooperation_coupled_rhs(_t, y):
            u_val, score_val = y
            u_dot = -u_val + expit((D_SAFE - d_min))
            score_dot = -d * score_val + math.tanh(u_val * score_val + bj)
            return [u_dot/NodConfig.dynamics.TAU_COOPERATION, score_dot/NodConfig.dynamics.TAU_COOPERATION]

        sol = solve_ivp(
            _cooperation_coupled_rhs,
            [0.0, float(time_step)],
            [u_prev, cooperation_prev],
            rtol=1e-4,
            atol=1e-4,
        )
        u = float(sol.y[0, -1])
        cooperation_score = float(sol.y[1, -1])
        self.pairwise_cooperation_attention[neighbor] = u
        self.pairwise_cooperation[neighbor] = cooperation_score

    def _compute_pressure_and_gates(self, ego_info: dict, neighbor_dict: dict, conflicting_neighbors: set):
        Pis = []
        Gis = []
        Uis = []
        infos = []
        for neighbor in conflicting_neighbors:
            # get neighbor info
            neighbor_info = neighbor_dict[neighbor]
            # print(f"robot: {self.robot_name}, neighbor: {neighbor}, ti: {ti:.3f}, tj: {tj:.3f}, delta_t: {delta_t:.3f}, t_star: {t_star:.3f}, d_min: {d_min:.3f}")


            # neighbor pruning
            ray_sol = solve_ray_intersection(ego_info, neighbor_info)
            if not ray_sol:
                # print(f"[NOD] prune ray_none ego={ego_info['name']} neigh={neighbor}")
                continue
            s, t, ei, ej = ray_sol

            if (s < 0.0 and abs(s) > NodConfig.neighbors.R_OCC):
                # print(f"nhbr: {neighbor_info['name']}, not conflicting, {s=}, {t=}")
                continue

            # Sine of crossing angle: |ei x ej| (2-D cross product magnitude)
            sin_alpha = abs(float(ei[0] * ej[1] - ei[1] * ej[0]))

            ti, tj, ti_cooperation, inside_i, inside_j= arrival_times_to_disk(ego_info, neighbor_info, ray_sol)
            # print(f"robot: {ego_info['name']}, neighbor: {neighbor_info['name']}, ti: {ti}, tj: {tj}, ti_rogue: {ti_cooperation}, s: {s}, t: {t}")

            if (ti is None or tj is None or ti_cooperation is None):
                # print(f"nhbr: {neighbor_info['name']}, NONE")
                continue
            # print("still in the loop though")
            if ti > 100.0 and tj > 100.0:
                continue
            a, b, t_star, d_min = tca_and_rmin(ego_info, neighbor_info, inside_i, inside_j)

            # update cooperation for this conflicting neighbor only
            if NodConfig.cooperation.COOPERATION_LAYER_ON:
                self._update_cooperation_for_neighbor(ego_info, neighbor_info, neighbor)

            # compute pressure
            # P_time = 2*float(expit(float(NodConfig.pressure.KAPPA_TCA) * (float(NodConfig.pressure.T_COLL) - t_star)))
            # P_distance = 2*float(expit(float(NodConfig.pressure.KAPPA_DMIN) * (NodConfig.pressure.DMIN_CLEAR - d_min)))
            thresh = NodConfig.pressure.DMIN_CLEAR/max(abs(sin_alpha),NodConfig.pressure.MIN_SIN_ALPHA)
            P_distance = float(expit(float(NodConfig.pressure.KAPPA_DMIN) * (thresh - d_min)))
            P_urgency = self._urgency_from_tstar(s, t, float(np.linalg.norm(ego_info['velocity'])), float(np.linalg.norm(neighbor_info['velocity'])), a, b, sin_alpha)

            P = P_urgency * P_distance

            # compute gate

            if NodConfig.cooperation.COOPERATION_LAYER_ON and self.pairwise_cooperation[neighbor] < NodConfig.cooperation.COOPERATION_THRESHOLD:
                delta_t = tj - ti_cooperation
                # print(f"robot: {ego_info['name']}, neighbor: {neighbor_info['name']}, cooperation: {self.pairwise_cooperation[neighbor]} using cooperation delta_t: {delta_t}")
            else:
                delta_t = tj - ti
            G = self._compute_gate(delta_t, s, t, inside_i, inside_j)


            U = self._compute_pairwise_attention(ego_info, neighbor_info)

            Pis.append(P)
            Gis.append(G)
            Uis.append(U)
            info = {}
            info['neighbor'] = neighbor
            # info['time_to_closest_approach'] = t_star
            # info['distance_at_closest_approach'] = d_min
            info['s'] = s
            info['t'] = t
            info['ti'] = ti
            info['tj'] = tj
            info['delta_t'] = delta_t
            info['P'] = P
            info['G'] = G
            info['U'] = U
            infos.append(info)

        # print(f"robot: {self.robot_name}, neighbor infos: {infos}")
        return Pis, Gis, Uis

    def _compute_pairwise_attention(self, ego_info, neighbor_info: float) -> float:
        ego_pos = ego_info['position']

        neighbor_pos = neighbor_info['position']

        r0 = np.array(neighbor_pos) - np.array(ego_pos)  # relative position
        w = np.array(neighbor_info['velocity'])-np.array(ego_info['velocity'])    # relative velocity
        a = float(np.dot(w, w))
        b = 2*float(np.dot(r0, w))
        c = float(np.dot(r0, r0)) - 1.*(NodConfig.neighbors.R_PRED)**2
        # When w≈0 (both nearly stopped), b≈0 and the formula reduces to
        # expit(-c) = expit(-(‖r0‖²-R_PRED²)), a position-only conflict check.
        # max(a, EPS) handles the degenerate division safely.
        conflict_intensity = 2*expit(1*(b**2/(4*max(a, EPS)) - c))

        if neighbor_info['name'] in HUMAN_NAMES:
            conflict_intensity = max(conflict_intensity, 1.0)

        u_ij = conflict_intensity
        # att_prec_ij = self.pairwise_u[neighbor_info['name']]

        # def _pairwise_attn_rhs(att: float) -> float:
        #     return  (-att + u_ij )

        # # Iterate RK4 updates on the cooperation score until it stabilizes
        # att_score = float(att_prec_ij)
        # time_step = 0.1
        # for _ in range(100):
        #     k1 = _pairwise_attn_rhs(att_score)
        #     k2 = _pairwise_attn_rhs(att_score + 0.5 * time_step * k1)
        #     k3 = _pairwise_attn_rhs(att_score + 0.5 * time_step * k2)
        #     k4 = _pairwise_attn_rhs(att_score + time_step * k3)

        #     next_att_score = att_score + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        #     if abs(next_att_score - att_score) < 1e-4:
        #         att_score = next_att_score
        #         break
        #     att_score = next_att_score
        att_score = u_ij

        self.pairwise_u[neighbor_info['name']] = att_score
        # print(f"ego: {ego_info['name']}, neighbor: {neighbor_info['name']}, cooperation level: {self.pairwise_u[neighbor_info['name']]}")
        return att_score

    def _compute_gate(self, delta_t: float,
                      s: float, t: float,
                      inside_i: bool = False, inside_j: bool = False) -> float:
        # if delta_t >= 0.0:
        #     G = -1
        # else:
        #     G =  1
        if NodConfig.pressure.USE_S_T_GATE and inside_i and inside_j:
            if s >= 0.0 and t >= 0.0:
                return float(np.sign(t - s))
            if s < 0.0 and t >= 0.0:
                return 1.0
            if s >= 0.0 and t < 0.0:
                return -1.0
            return 1.0
        G = np.tanh(NodConfig.pressure.KAPPA_G * delta_t)

        return G

    def _gate_induced_pressure(self, Pis, Gis) -> List[float]:
        gated_Pis = []
        for P, G in zip(Pis, Gis):
            gated_Pis.append(P * (1 - G * (NodConfig.pressure.PHI_TILT) / max(1-P, EPS)))

        return gated_Pis

    def _aggregate(self, Pis, Gis, Uis) -> float:
        if len(Pis) == 0:
            return None, None

        P = np.asarray(Pis, float)
        G = np.asarray(Gis, float)
        # Tilt affects attention weights only, not the signed opinion drive.
        scores = np.asarray(self._gate_induced_pressure(P, G), float)
        # max_idx = int(np.argmax(P_abs))
        # max_sign = -1.0 if P[max_idx] < 0.0 else 1.0
        w = np.exp((scores - np.max(scores))/max(NodConfig.pressure.TEMP_SM, EPS))
        w /= np.sum(w)
        # a_sum = float(max_sign * np.sum(Uis * (w * G)))
        a_sum = float(np.sum(P * w * G))
        return a_sum, np.sum(P)

    def _nod_update(self, z, u, a_sum, sumP) -> Tuple[float, float, float]:
        if a_sum is None:
            return self._free_flow(z, u)

        u_eff = NodConfig.dynamics.U_0 + NodConfig.dynamics.K_U * (z**2)
        u_active = u if NodConfig.dynamics.USE_ATT_DYNAMICS else u_eff
        z_dot = (float(-NodConfig.dynamics.OPINION_DECAY* z + np.tanh(u_active * a_sum)))/NodConfig.dynamics.TAU_Z
        u_dot = (0.0 if not NodConfig.dynamics.USE_ATT_DYNAMICS
                else float(-NodConfig.dynamics.ATTENTION_DECAY * u + u_eff )/NodConfig.dynamics.TIMING_TAU_U_RELAX)

        return z_dot, u_dot, u_eff

    def _free_flow(self, z: float, u: float) -> Tuple[float, float, float]:
            # u_eff = 0
            z_dot = float(-NodConfig.dynamics.OPINION_DECAY * z)/NodConfig.dynamics.TAU_Z_RELAX
            u_dot = (float(-NodConfig.dynamics.ATTENTION_DECAY * u)/NodConfig.dynamics.TIMING_TAU_U_RELAX
                     if NodConfig.dynamics.USE_ATT_DYNAMICS else 0.0)
            return z_dot, u_dot, 0

    def _integrate_fast(self, z0: float, u0: float,
                       a_sum: float, sumP: float, horizon_s: float) -> Tuple[float, float]:

        if horizon_s <= 0.0:
            return float(z0), float(u0)
        def fast_rhs(_t, y):
            dz, du, u_eff = self._nod_update(y[0], y[1], a_sum, sumP)
            return [dz, du]

        sol = solve_ivp(fast_rhs, [0.0, horizon_s], [z0, u0], rtol=1e-4, atol=1e-4)
        if not sol.success:
            raise RuntimeError("NOD integration failed: " + sol.message)
        z_end = float(sol.y[0, -1])
        u_end = (float(sol.y[1, -1]) if NodConfig.dynamics.USE_ATT_DYNAMICS
                 else float(self._nod_update(z_end, sol.y[1, -1], a_sum, sumP)[2]))
        return z_end, u_end
