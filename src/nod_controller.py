import math
from collections import defaultdict
import numpy as np
from typing import List, Tuple
from neighbors import conflicting_neighbors, tca_and_rmin, arrival_times_to_disk, sensed_neighbors, solve_ray_intersection
from scipy.special import expit
from scipy.integrate import solve_ivp
from constants import NodConfig, EPS, T_COLL, KAPPA_TCA, KAPPA_DMIN, DMIN_CLEAR, PHI_TILT, TEMP_SM, U_0, K_U, OPINION_DECAY, ATTENTION_DECAY, TAU_Z, TIMING_TAU_U_RELAX, K_U_S, TAU_Z_RELAX, ITERATIONS_OD, D_SAFE

class NodController:
    def __init__(self, robot_name: str, time: float):
        # state variables
        self.robot_name = robot_name
        self.current_time = time

        # nod variables
        self.z = 0
        self.u = 0
        self.pairwise_u = defaultdict(float)

        # cooperation variables
        self.pairwise_cooperation = defaultdict(float)
        self.pairwise_cooperation_attention = defaultdict(float)
        self.previous_pairwise_phi = defaultdict(float)
        self.previous_vj = defaultdict(float)


    def update_opinion(self, ego_info: dict, neighbors_dict: dict, current_time: float):
        # print(f"robot: {self.robot_name}, conflicting neighbors: {c_neighbors}")

        sens_neighbors = sensed_neighbors(ego_info, neighbors_dict)

        if NodConfig.cooperation.COOPERATION_LAYER_ON:
            self._update_cooperation(ego_info, neighbors_dict, sens_neighbors)

        Pis, Gis, Uis = self._compute_pressure_and_gates(ego_info, neighbors_dict, sens_neighbors)

        a_sum,sumP = self._aggregate(Pis, Gis, Uis)

        # update nod variables
        dt = current_time - self.current_time 
        self.current_time = current_time
        dt = 0.1#min(dt, 0.2)

        # for _ in range(n_fast):
        self.z, self.u = self._integrate_fast(
        self.z, self.u, a_sum, sumP, dt)

        # compute target velocity
        v0 = NodConfig.kin.V_NOMINAL
        v_tar = np.clip((1.0 + np.tanh(NodConfig.kin.KAPPA_Z* self.z)) * v0, 0.0, NodConfig.kin.V_MAX)

        # if a_sum is None:
        #     print(f"robot: {self.robot_name}, conf neighbors: {conf_neighbors}, Pis: {Pis}, Gis: {Gis}, Uis: {Uis}, z: {self.z:.3f}, u: {self.u:.3f}, v_tar: {v_tar:.3f}")
        # else:
        #     print(f"robot: {self.robot_name}, conf neighbors: {conf_neighbors}, Pis: {Pis}, Gis: {Gis}, Uis: {Uis}, a_sum: {a_sum:.3f}, z: {self.z:.3f}, u: {self.u:.3f}, v_tar: {v_tar:.3f}")
      
        return v_tar
    
    def _update_cooperation(self, ego_info: dict, neighbors_dict: dict, sensed_neighbors: set) -> None:

        for neighbor in sensed_neighbors:
            neighbor_info = neighbors_dict[neighbor]
            time_step = 0.1

            # relative vectors
            vij_vec = np.array(neighbor_info['velocity']) - np.array(ego_info['velocity'])
            pij = np.array(neighbor_info['position']) - np.array(ego_info['position'])
            pij_norm = float(np.linalg.norm(pij))
            pij_norm = max(pij_norm, 1e-6)
            vij_vec_norm = float(np.linalg.norm(vij_vec))
            vij_vec_norm = max(vij_vec_norm, 1e-6)
            theta = np.arccos(max(min(np.dot(vij_vec, pij)/(vij_vec_norm*pij_norm), 1), -1))

            # phi calculations
            ej = neighbor_info['position'] / np.linalg.norm(neighbor_info['position'])
            last_Phi = self.previous_pairwise_phi[neighbor]
            Phi = np.cos(theta)
            self.previous_pairwise_phi[neighbor] = Phi
            delta_Phi = Phi - last_Phi
            Phi_dot_vj = -1*np.sin(theta)*np.dot(pij, ej)

            # neighbor velocity calculations 
            prev_vj = self.previous_vj[neighbor]
            vj = neighbor_info['velocity']
            self.previous_vj[neighbor] = vj
            delta_vj = np.linalg.norm(vj) - np.linalg.norm(prev_vj)
            
            
            latest_cooperation_score = self.pairwise_cooperation[neighbor]
            x = math.tanh(10*np.linalg.norm(neighbor_info['velocity']))
            y = math.tanh(abs(delta_Phi))     
            if abs(delta_Phi) < np.deg2rad(5):   
                (delta_Phi) = -np.deg2rad(50)
            
            bj = x*(delta_vj*(-Phi_dot_vj))/abs(delta_Phi)+(1-x)*((-1*y*delta_Phi)+(1-y)) 
            
            _, d_min = tca_and_rmin(ego_info, neighbor_info)
            d = 1
            u_prev = self.pairwise_cooperation_attention[neighbor]

            def _cooperation_u_rhs(val: float) -> float:
                return -val+expit((D_SAFE-d_min))

            k1_u = _cooperation_u_rhs(u_prev)
            k2_u = _cooperation_u_rhs(u_prev + 0.5 * time_step * k1_u)
            k3_u = _cooperation_u_rhs(u_prev + 0.5 * time_step * k2_u)
            k4_u = _cooperation_u_rhs(u_prev + time_step * k3_u)
            u = u_prev + (time_step / 6.0) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
            self.pairwise_cooperation_attention[neighbor] = u

            def _cooperation_rhs(score: float) -> float:
                # return beta_input
                # u=expit((cfg.d_safe-d_min))
                return  -d * score + math.tanh(u*score ) + bj
            # Iterate RK4 updates on the cooperation score until it stabilizes
            cooperation_score = latest_cooperation_score
            for _ in range(1):
                k1 = _cooperation_rhs(cooperation_score)
                k2 = _cooperation_rhs(cooperation_score + 0.5 * time_step * k1)
                k3 = _cooperation_rhs(cooperation_score + 0.5 * time_step * k2)
                k4 = _cooperation_rhs(cooperation_score + time_step * k3)

                next_cooperation = cooperation_score + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                if abs(next_cooperation - cooperation_score) < 1e-4:
                    cooperation_score = next_cooperation
                    break
                cooperation_score = next_cooperation
            self.pairwise_cooperation[neighbor] = cooperation_score

    def _compute_pressure_and_gates(self, ego_info: dict, neighbor_dict: dict, conflicting_neighbors: set):
        Pis = []
        Gis = []
        Uis = []
        infos = []
        for neighbor in conflicting_neighbors:
            # get neighbor info
            neighbor_info = neighbor_dict[neighbor]
            t_star, d_min = tca_and_rmin(ego_info, neighbor_info)
            # print(f"robot: {self.robot_name}, neighbor: {neighbor}, ti: {ti:.3f}, tj: {tj:.3f}, delta_t: {delta_t:.3f}, t_star: {t_star:.3f}, d_min: {d_min:.3f}")
            

            # neighbor pruning
            if not (solve_ray_intersection(ego_info, neighbor_info)):
                continue
            s, t = solve_ray_intersection(ego_info, neighbor_info)
            if (s < 0.0 and abs(s) > NodConfig.neighbors.R_OCC):
                continue
            ti, tj, ti_cooperation= arrival_times_to_disk(ego_info, neighbor_info)
            # print(f"robot: {ego_info['name']}, neighbor: {neighbor_info['name']}, ti: {ti}, tj: {tj}, ti_rogue: {ti_cooperation}, s: {s}, t: {t}")

            if (ti is None or tj is None or ti_cooperation is None):
                continue
            if (ti > 40 or tj > 40):
                continue
       
            
            # compute pressure
            P_time = 2*float(expit(float(KAPPA_TCA) * (float(T_COLL) - t_star)))
            P_distance = 2*float(expit(float(KAPPA_DMIN) * (DMIN_CLEAR - d_min)))
            P = P_time * P_distance

            # compute gate
            if NodConfig.cooperation.COOPERATION_LAYER_ON and self.pairwise_cooperation[neighbor] < NodConfig.cooperation.COOPERATION_THRESHOLD:
                delta_t = ti_cooperation - tj
                # print(f"robot: {ego_info['name']}, neighbor: {neighbor_info['name']}, cooperation: {self.pairwise_cooperation[neighbor]} using cooperation delta_t: {delta_t}")
            else:
                delta_t = ti - tj
            G = self._compute_gate(delta_t)

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
        conflict_intensity = 2*expit(1*(b**2/(4*max(a,EPS)) - c))

        u_ij = conflict_intensity
        att_prec_ij = self.pairwise_u[neighbor_info['name']]

        def _pairwise_attn_rhs(att: float) -> float:
            return  (-att + u_ij )
        
        # Iterate RK4 updates on the cooperation score until it stabilizes
        att_score = float(att_prec_ij)
        time_step = 0.1
        for _ in range(100):
            k1 = _pairwise_attn_rhs(att_score)
            k2 = _pairwise_attn_rhs(att_score + 0.5 * time_step * k1)
            k3 = _pairwise_attn_rhs(att_score + 0.5 * time_step * k2)
            k4 = _pairwise_attn_rhs(att_score + time_step * k3)

            next_att_score = att_score + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if abs(next_att_score - att_score) < 1e-4:
                att_score = next_att_score
                break
            att_score = next_att_score

        self.pairwise_u[neighbor_info['name']] = att_score
        # print(f"ego: {ego_info['name']}, neighbor: {neighbor_info['name']}, cooperation level: {self.pairwise_u[neighbor_info['name']]}")
        return att_score
    
    def _compute_gate(self, delta_t: float) -> float:
        if delta_t >= 0.0:
            G = -1
        else:
            G =  1
        return G

    def _gate_induced_pressure(self, Pis, Gis) -> List[float]:
        gated_Pis = []
        for P, G in zip(Pis, Gis):
            gated_Pis.append(P * 0.5*(1 - (G) * (PHI_TILT)))
        
        return gated_Pis
    
    def _aggregate(self, Pis, Gis, Uis) -> float:
        if len(Pis) == 0:
            return None, None

        P = np.asarray(Pis, float)
        G = np.asarray(Gis, float)
        P_abs = abs(P)
        max_idx = int(np.argmax(P_abs))
        max_sign = -1.0 if P[max_idx] < 0.0 else 1.0
        w = np.exp((P_abs - np.max(P_abs))/TEMP_SM)  # avoid overflow
        w /= (np.sum(w) + 1e-9)  # normalize to sum to 1
        a_sum = float(max_sign * np.sum(Uis * (w * G)))
        return a_sum,np.sum(P)
    
    def _nod_update(self, z, u, a_sum, sumP) -> Tuple[float, float, float]:        
        if a_sum is None:
            return self._free_flow(z, u)

        u_eff = sumP + K_U * (z**2)
        z_dot = (float(-OPINION_DECAY* z + np.tanh(u * a_sum)))/TAU_Z
        u_dot = float(-ATTENTION_DECAY * u + u_eff )/TIMING_TAU_U_RELAX
        
        return z_dot, u_dot, u_eff
    
    def _free_flow(self, z: float, u: float) -> Tuple[float, float, float]:
            # u_eff = 0
            z_dot = float(-OPINION_DECAY * z)/TAU_Z_RELAX
            u_dot = float(-ATTENTION_DECAY * u)/TIMING_TAU_U_RELAX
            return z_dot, u_dot, 0

    def _integrate_fast(self, z0: float, u0: float,
                       a_sum: float, sumP: float, horizon_s: float) -> Tuple[float, float]:
        
        last_u_eff = [u0]
        def fast_rhs(_t, y):
            dz, du, u_eff = self._nod_update(y[0], y[1], a_sum, sumP)
            last_u_eff[0] = u_eff
            return [dz, du]

        sol = solve_ivp(fast_rhs, [0.0, float(horizon_s)], [z0, u0], rtol=1e-4, atol=1e-4)
        z_end = float(sol.y[0, -1])
        u_end = float(last_u_eff[0])
        return z_end, u_end