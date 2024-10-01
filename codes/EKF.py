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
    def __init__(self, dt, L, l_r, W, pixel_to_meter, v, th_dot, st_dot, cp, initial_state, c_f='l', dim_x = 6):
    
        self.initial_state = initial_state
        self.x_est = np.array([self.initial_state['x_i'], self.initial_state['y_i'], math.radians(self.initial_state['yaw_i']) , 0 , 0, 0])
        self.P_est = np.diag([0.01, 0.01, 0.01, 0.5, 0.5 , 0.5])
        
        self.v_std = 0.02
        self.sdot_std = 0.02
        self.tdot_std = 0.02
        self.x_std = 0.001
        self.y_std = 0.001
        self.th_std = 0.0001
        self.lr_std = 0.001
        self.ll_std = 0.001
        self.yaw_std = 0.0001
        
        self.Q_km = np.diag([self.v_std**2, self.sdot_std**2, self.tdot_std**2])
        self.cov_y = np.diag([self.x_std**2, self.y_std**2, self.th_std**2, self.lr_std**2, self.ll_std**2, self.yaw_std**2])
        
        self.dt = dt
        self.L = L
        self.l_r = l_r
        self.W = W
        self.pixel_to_meter = pixel_to_meter
        self.v = v
        self.th_dot = th_dot
        self.st_dot = st_dot
        self.c_f = c_f
        self.cp = cp
        
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
        x_g, y_g, z_g = self.cp[0] + np.random.normal(0, scale=self.x_std), self.cp[1] + np.random.normal(0, scale=self.y_std), self.cp[2] + np.random.normal(0, scale=self.th_std)
        
        if c_f == 'l':
            p_m = phi_k
            l_r = W * 3/4 + yo_k
            l_l = -W * 1/4 + yo_k
        else:
            p_m = phi_k
            l_r = W * 1/4 + yo_k
            l_l = -W * 3/4 + yo_k

        #Computing measurement Jacobian
        H_k = np.array([[1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1, 0]])
        M_k = np.eye(6)
        
        K_k = np.dot(np.dot(P_check, H_k.T), inv(np.dot(np.dot(H_k, P_check), H_k.T) + np.dot(np.dot(M_k, cov_y), M_k.T)))
        
        z_k = np.array([x_k, y_k, th_k, self.normalize_angle(p_m), l_r, l_l])
        z_m = np.array([x_g, y_g, z_g, phi_m, lr_m, ll_m])
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
        U_k = np.mat([self.v , self.st_dot , self.th_dot])
        x_check += self.dt * (np.dot(L_m, U_k.T)).T

        x_check[0, 2] = self.normalize_angle(x_check[0, 2])
        #x_check[0, 3] = self.normalize_angle(x_check[0, 3])
        #x_check[0, 5] = self.normalize_angle(x_check[0, 5])
        

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
            x_check, P_check = self.measurement_update(zm[i, 1] + np.random.normal( 0, self.lr_std),
                                                       zm[i, 0] + np.random.normal(0, self.ll_std), 
                                                       zm[i, 2] + np.random.normal(0, self.yaw_std), 
                                                       P_check, x_check, self.cov_y, self.W * self.pixel_to_meter, self.c_f)

        self.x_est[:6] = x_check.flatten()
        self.P_est = P_check
        
        self.st_prev = self.x_est[3]
        self.th_prev = self.x_est[2]
        self.phi_prev = self.x_est[5]

