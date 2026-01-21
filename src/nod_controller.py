import math
import numpy as np
from typing import List, Tuple
from neighbors import conflicting_neighbors, tca_and_rmin
from scipy.special import expit
from scipy.integrate import solve_ivp
from constants import EPS, T_COLL, KAPPA_TCA, KAPPA_DMIN, DMIN_CLEAR, PHI_TILT, TEMP_SM, U_0, K_U, OPINION_DECAY, ATTENTION_DECAY, TAU_Z, TIMING_TAU_U_RELAX, K_U_S, TAU_Z_RELAX, ITERATIONS_OD

class NodController:
    def __init__(self, robot_name: str, time: float):
        # state variables
        self.robot_name = robot_name
        self.current_time = time

        # nod variables
        self.z = 0
        self.u = 0


    def update_opinion(self, ego_info: dict, neighbors_dict: dict, current_time: float):
        c_neighbors = conflicting_neighbors(ego_info, neighbors_dict)

        Pis, Gis = self._compute_pressurs_and_gates(ego_info, neighbors_dict, c_neighbors)

        a_sum = self._aggregate(Pis, Gis)

        # update nod variables
        dt = current_time - self.current_time 
        self.current_time = current_time
        dt = 0.1#min(dt, 0.2)

        # for _ in range(n_fast):
        self.z, self.u = self._integrate_fast(
        self.z, self.u, a_sum, dt)
        
        # for _ in range(ITERATIONS_OD):
        # # for i in range(cfg.timing.iterations_OD):
        #     k1_z, k1_u = self._nod_update(z, u, a_sum)
        #     # print(f"{k1_z=}, {k1_u=}")
        #     k2_z, k2_u = self._nod_update(z + 0.5 * dt * k1_z, u + 0.5 * dt * k1_u,a_sum)
        #     # print(f"{k2_z=}, {k2_u=}")
        #     k3_z, k3_u = self._nod_update(z + 0.5 * dt * k2_z, u + 0.5 * dt * k2_u, a_sum)
        #     # print(f"{k3_z=}, {k3_u=}")
        #     k4_z, k4_u = self._nod_update(z + dt * k3_z, u + dt * k3_u, a_sum)
        #     # print(f"{k4_z=}, {k4_u=}")

        #     # RK4 update
        #     z += (dt / 6) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)
        #     u += (dt / 6) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)

        #     # Check for convergence
        #     if abs(z - zprev) < 1e-4 and abs(u-uprev) < 1e-4:
        #         break
        #     zprev = z
        #     uprev = u
    
        # print(f"{self.robot_name}, Pis: {Pis}, a_sum: {a_sum}, z: {self.z:.3f}, u: {self.u:.3f}, dt: {dt:.3f}")
       
        return self.z

        

    def _compute_pressurs_and_gates(self, ego_info: dict, neighbor_dict: dict, conflicting_neighbors: set):
        Pis = []
        Gis = []
        for neighbor in conflicting_neighbors:
            neighbor_info = neighbor_dict[neighbor]
            t_star, d_min = tca_and_rmin(ego_info, neighbor_info)
            delta_t = self._compute_delta_t(ego_info, neighbor_info)

            # if delta_t is None or delta_t < 0:
            #     continue

            # compute pressure
            P_time = 2*float(expit(float(KAPPA_TCA) * (float(T_COLL) - t_star)))
            P_distance = 2*float(expit(float(KAPPA_DMIN) * (DMIN_CLEAR - d_min)))
            P = P_time * P_distance

            # compute gate
            G = self._compute_gate(delta_t)

            Pis.append(P)
            Gis.append(G)


        return Pis, Gis
    
    def _compute_delta_t(self, ego_info: dict, neighbor_info: dict) -> float:
        
        # compute s and t
        ego_pos = np.array(ego_info['position'])
        neighbor_pos = np.array(neighbor_info['position'])
        pij = ego_pos - neighbor_pos

        A = np.array([[ego_pos[0], -neighbor_pos[0]], [ego_pos[1], -neighbor_pos[1]]], float)
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < EPS:
            return None
        inv = (1.0 / det) * np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]], float)
        ego_distance_to_int, neighbor_dist_to_int = inv @ pij
        

        # compute delta_t
        ego_speed = np.linalg.norm(np.array(ego_info['velocity']))
        neighbor_speed = np.linalg.norm(np.array(neighbor_info['velocity']))
        ego_time_to_int = ego_distance_to_int / (ego_speed + EPS)
        neighbor_time_to_int = neighbor_dist_to_int / (neighbor_speed + EPS)
        delta_t = ego_time_to_int - neighbor_time_to_int

        # print(f"robot: {ego_info['name']}, delta_t: {delta_t:.3f}, ego_time_to_int: {ego_time_to_int:.3f}, neighbor_time_to_int: {neighbor_time_to_int:.3f}")

        return delta_t
    
    def _compute_gate(self, delta_t: float) -> float:
        if delta_t > 0.0:
            G = -1
        else:
            G =  1
        return G

    def _gate_induced_pressure(self, Pis, Gis) -> List[float]:
        gated_Pis = []
        for P, G in zip(Pis, Gis):
            gated_Pis.append(P * 0.5*(1 - (G) * (PHI_TILT)))
        
        return gated_Pis
    
    def _aggregate(self, Pis, Gis) -> float:
        if len(Pis) == 0:
            return None

        Pis = self._gate_induced_pressure(Pis, Gis)
        P = np.asarray(Pis, float)
        G = np.asarray(Gis, float)

        w = np.exp((P - np.max(P))/TEMP_SM)  # avoid overflow
        w /= (np.sum(w) + 1e-9)  # normalize to sum to 1
        a_sum = float(np.sum((w * G)))
        return a_sum
    
    def _nod_update(self, z, u, a_sum) -> Tuple[float, float, float]:        
        if a_sum is None:
            return self._free_flow(z, u)

        u_eff = U_0 + K_U * (z**2)
        z_dot = (float(-OPINION_DECAY* z + np.tanh(u * a_sum)))/TAU_Z
        u_dot = float(-ATTENTION_DECAY * u + u_eff )/TIMING_TAU_U_RELAX
        
        return z_dot, u_dot, u_eff
    
    def _free_flow(self, z: float, u: float) -> Tuple[float, float, float]:
            # u_eff = 0
            z_dot = float(-OPINION_DECAY * z)/TAU_Z_RELAX
            u_dot = float(-ATTENTION_DECAY * u)/TIMING_TAU_U_RELAX
            return z_dot, u_dot, 0

    def _integrate_fast(self, z0: float, u0: float,
                       a_sum: float, horizon_s: float) -> Tuple[float, float]:
        
        last_u_eff = [u0]
        def fast_rhs(_t, y):
            dz, du, u_eff = self._nod_update(y[0], y[1], a_sum)
            last_u_eff[0] = u_eff
            return [dz, du]

        sol = solve_ivp(fast_rhs, [0.0, float(horizon_s)], [z0, u0], rtol=1e-4, atol=1e-4)
        z_end = float(sol.y[0, -1])
        u_end = float(last_u_eff[0])
        return z_end, u_end