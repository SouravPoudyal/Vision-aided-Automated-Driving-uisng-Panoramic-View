'''
Universität Siegen
Naturwissenschaftlich-Technische Fakultät
Department Elektrotechnik und Informatik
Lehrstuhl für Regelungs- und Steuerungstechnik

Master Thesis: Vision-aided Automated Driving using Panoramic View
Done By: Sourav Poudyal, 1607167
First Examiner: Prof. Dr.-Ing. habil. Michael Gerke
Supervisior: Dr.-Ing. Nasser Gyagenda
'''



import numpy as np
import math
from numpy.linalg import inv

class EKF:
    def __init__(self, max_iteration, dt, x_est, P_est, L, l_r, W, pixel_to_meter, v, c_f='l'):
    
        self.x_est = x_est
        self.P_est = P_est
        
        self.v_std = 0.002
        self.sdot_std = 0.2
        self.tdot_std = 0.2
        self.x_std = 0.01
        self.y_std = 0.01
        self.th_std = 0.001
        self.lr_std = 0.002
        self.ll_std = 0.002
        self.yaw_std = 0.002
        
        self.v_noise_dist = np.random.normal(0, self.v_std, max_iteration)
        self.sdot_noise_dist = np.random.normal(0, self.sdot_std, max_iteration)
        self.tdot_noise_dist = np.random.normal(0, self.tdot_std, max_iteration)
        self.x_noise_dist = np.random.normal(0, self.x_std, max_iteration)
        self.y_noise_dist = np.random.normal(0, self.y_std, max_iteration)
        self.th_noise_dist = np.random.normal(0, self.th_std, max_iteration)
        self.lr_noise_dist = np.random.normal(0, self.lr_std, max_iteration)
        self.ll_noise_dist = np.random.normal(0, self.ll_std, max_iteration)
        self.yaw_noise_dist = np.random.normal(0, self.yaw_std, max_iteration)
        
        self.Q_km = np.diag([self.v_std**2, self.sdot_std**2, self.tdot_std**2])
        self.cov_y = np.diag([self.lr_std**2, self.ll_std**2, self.yaw_std**2])
        
        self.dt = dt
        self.L = L
        self.l_r = l_r
        self.W = W
        self.pixel_to_meter = pixel_to_meter
        self.v = v
        self.c_f = c_f
        
        self.st_prev = 0
        self.th_prev = 0
        self.phi_prev = 0
        
    def normalize_angle(self, x):
        x = x % (2 * np.pi)
        if x > np.pi:
            x -= 2 * np.pi
        return x

    def measurement_update(self, lr_m, ll_m, phi_m, P_check, x_check, cov_y, W, c_f):
        x_k, y_k, th_k, _, yo_k, phi_k = x_check[0, 0], x_check[0, 1], x_check[0, 2], x_check[0, 3], x_check[0, 4], x_check[0, 5]
        x_check_copy = np.copy(x_check)
        
        if c_f == 'l':
            p_m = phi_k
            l_r = W * 3/4 + yo_k
            l_l = -W * 1/4 + yo_k
        else:
            p_m = phi_k
            l_r = W * 1/4 + yo_k
            l_l = -W * 3/4 + yo_k

        H_k = np.array([[0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1, 0]])
        M_k = np.eye(3)
        
        K_k = np.dot(np.dot(P_check, H_k.T), inv(np.dot(np.dot(H_k, P_check), H_k.T) + np.dot(np.dot(M_k, cov_y), M_k.T)))
        
        z_k = np.array([self.normalize_angle(p_m), l_r, l_l])
        z_m = np.array([phi_m, lr_m, ll_m])
        x_check_copy += np.dot(K_k, (z_m - z_k).reshape(-1, 1)).T
        x_check_copy[0,2] = self.normalize_angle(x_check_copy[0,2])
        
        P_check = np.dot((np.eye(6) - np.dot(K_k, H_k)), P_check)
        
        return x_check_copy, P_check
    
    def ekf_step(self, zm):
    
        th_dot = self.normalize_angle(self.x_est[2] - self.th_prev) / self.dt
        st_dot = self.normalize_angle(self.x_est[3] - self.st_prev) / self.dt
        

        x_check = self.x_est.reshape((1, 6))
        P_check = self.P_est

        th = self.normalize_angle(self.th_prev)
        st = self.normalize_angle(self.st_prev)
        phi = self.normalize_angle(self.phi_prev)

        L_m = np.array([[np.cos(th + np.arctan(self.l_r * np.tan(st) / self.L)), 0, 0], 
                        [np.sin(th + np.arctan(self.l_r * np.tan(st) / self.L)), 0, 0], 
                        [np.tan(st) / (self.L * np.sqrt(1 + self.l_r ** 2 * np.tan(st) ** 2 / self.L ** 2)), 0, 0], 
                        [0, 1, 0], 
                        [np.sin(phi), 0, 0], 
                        [0, 0, -1]])
        U_k = np.mat([self.v + np.random.choice(self.v_noise_dist), th_dot + np.random.choice(self.tdot_noise_dist), st_dot + np.random.choice(self.sdot_noise_dist)])
        x_check += self.dt * (np.dot(L_m, U_k.T)).T

        x_check[0, 2] = self.normalize_angle(x_check[0, 2])

        F_km = np.array([[1, 0, -self.dt * self.v * np.sin(th + np.arctan(self.l_r * np.tan(st) / self.L)), 
                          -self.dt * self.l_r * self.v * (np.tan(st) ** 2 + 1) * np.sin(th + np.arctan(self.l_r * np.tan(st) / self.L)) / (self.L * (1 + self.l_r ** 2 * np.tan(st)  ** 2 / self.L ** 2)), 
                          0, 0], 
                         [0, 1, self.dt * self.v * np.cos(th + np.arctan(self.l_r * np.tan(st) / self.L)), 
                          self.dt * self.l_r * self.v * (np.tan(st) ** 2 + 1) * np.cos(th + np.arctan(self.l_r * np.tan(st) / self.L)) / (self.L * (1 + self.l_r ** 2 * np.tan(st) ** 2 / self.L ** 2)), 
                          0, 0], 
                         [0, 0, 1, self.dt * self.v * (np.tan(st) ** 2 + 1) / (self.L * np.sqrt(1 + self.l_r ** 2 * np.tan(st) ** 2 / self.L ** 2)) - self.dt * self.l_r ** 2 * self.v * (2 * np.tan(st) ** 2 + 2) * np.tan(st) ** 2 / (2 * self.L ** 3 * (1 + self.l_r ** 2 * np.tan(st) ** 2 / self.L ** 2) ** (3 / 2)), 
                          0, 0], 
                         [0, 0, 0, 1, 0, 0], 
                         [0, 0, 0, 0, 1, self.dt * self.v * np.cos(phi)], 
                         [0, 0, 0, 0, 0, 1]])

        L_km = self.dt * L_m
        P_check = np.dot(np.dot(F_km, P_check), F_km.T) + np.dot(np.dot(L_km, self.Q_km), L_km.T)

        no_data = min(len(zm), 7)
        for i in range(no_data):
            x_check, P_check = self.measurement_update(zm[i, 1] + np.random.choice(self.lr_noise_dist), 
                                                       zm[i, 0] + np.random.choice(self.ll_noise_dist), 
                                                       zm[i, 2] + np.random.choice(self.yaw_noise_dist), 
                                                       P_check, x_check, self.cov_y, self.W * self.pixel_to_meter, self.c_f)

        self.x_est[:6] = x_check.flatten()
        self.P_est = P_check
        
        self.st_prev = self.x_est[3]
        self.th_prev = self.x_est[2]
        self.phi_prev = self.x_est[5]

