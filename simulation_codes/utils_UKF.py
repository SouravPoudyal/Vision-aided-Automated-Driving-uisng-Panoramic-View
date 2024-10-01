import numpy as np
import scipy.linalg as lin_alg

class Bicycle:
    def __init__(self, x, y, theta, delta, beta, L, lr, w_max , dt, c = 's'):
        self.x = x
        self.y = y
        self.theta = theta
        self.delta = delta
        self.beta = beta
        self.L = L
        self.lr = lr
        self.w_max = w_max
        self.dt = dt
        self.c = c

    def reset(self):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0
        self.L = 0
        self.lr = 0 

    def step(self, v, w):
        if abs(w) < self.w_max:
            self.delta = self.delta + w * self.dt
        else:
            self.delta = self.delta + self.w_max * self.dt        
        self.beta = np.arctan2((self.lr * np.tan(self.delta)),self.L)
        if self.c =='ns':
            self.beta = 0
        self.theta = self.theta + (v * np.cos(self.beta) * np.tan(self.delta)/ self.L) * self.dt
        self.x = self.x + v * np.cos(self.theta + self.beta) * self.dt
        self.y = self.y + v * np.sin(self.theta + self.beta) * self.dt
        return np.array([self.x, self.y, normalize_angle(self.theta), normalize_angle(self.delta)])
    
class clothoid_lane:
    def __init__(self, W, y_off, y_p, c_0, c_1, dt, d = 'l'):
        self.W = W
        self.y_off = y_off
        self.y_p = y_p
        self.c_0 = c_0
        self.c_1 = c_1
        self.dt = dt
        self.d = d


    def fx_cl(self, v, yr_abs):
        self.y_off = self.y_off + v*self.dt*self.y_p - v**2 * self.dt **2 * self.c_0 / 2 - v**3 * self.dt**3 * self.c_1 / 6 + v * self.dt **2 * yr_abs / 2
        self.y_p = self.y_p - v * self.dt * self.c_0 - v**2 * self.dt **2 * self.c_1/ 2 + self.dt * yr_abs
        self.c_0 = self.c_0 + v * self.dt * self.c_1

        return np.array([self.W, self.y_off, normalize_angle(self.y_p), self.c_0, self.c_1])

    def hx_cl(self, sig_pts_prior):
        if self.d == 'l':
            L_m = -sig_pts_prior[4] * 1/4 + sig_pts_prior[5]
            R_m = sig_pts_prior[4] * 3/4 + sig_pts_prior[5]
        else:
            L_m = -sig_pts_prior[4] * 3/4 + sig_pts_prior[5]
            R_m = sig_pts_prior[4] * 1/4 + sig_pts_prior[5]

        yp_m = sig_pts_prior[6]
        c0_m = sig_pts_prior[7]

        return np.array([L_m, R_m, yp_m, c0_m])
        
class object:
    def __init__(self, x_i, v_i, y_i, dt):
        self.x_i = x_i
        self.v_i = v_i
        self.y_i = y_i
        self.dt = dt

    def fx_o(self, v):
        self.x_i = self.x_i + v * self.dt
        return np.array([self.x_i, self.v_i, self.y_i])
    
    def hx_o(self, sig_points):
        xm_i = sig_points[9]
        ym_i = sig_points[11] - sig_points[5] + sig_points[6] * sig_points[9] - (sig_points[7]/2) * sig_points[9]**2 - (sig_points[8]/6) * sig_points[9]**3

        return np.array([xm_i, ym_i])
    
def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x

def Hx(x):
    return x[:3]

class MerweScaledSigmaPoints:
    def __init__(self, n, alpha=0.0001, beta=2., kappa = 0):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = 3 - self.n
        self.lmbda_c = alpha ** 2 * (n + self.kappa) - n
        self.last_valid_mean = None
        self.last_valid_covariance = None
        self._compute_weights()

    def sigma_points_(self, mean, covariance):
        self.sig_pts = np.zeros((2 * self.n + 1, self.n))
        #print('m, cov', mean, covariance)
        # Compute sigma points
        sqrt_cov = np.linalg.cholesky((self.n + self.lmbda_c) * covariance)
        #print('sqrt_cov', sqrt_cov)
        self.sig_pts[0] = mean
        for i in range(self.n):
            self.sig_pts[i + 1] = mean + sqrt_cov[i]
            self.sig_pts[self.n + i + 1] = mean - sqrt_cov[i]
        #print('sieg_points', self.sig_pts)

    def sigma_points(self, mean, covariance):
        self.sig_pts = np.zeros((2 * len(mean) + 1, len(mean)))
        try:
            sqrt_cov = lin_alg.cholesky((len(mean) + self.lmbda_c) * covariance)
            self.sig_pts[0] = mean
            for i in range(len(mean)):
                self.sig_pts[i + 1] = mean + sqrt_cov[i]
                self.sig_pts[len(mean) + i + 1] = mean - sqrt_cov[i]
            self.last_valid_mean = mean
            self.last_valid_covariance = covariance
        except lin_alg.LinAlgError:
            if self.last_valid_mean is None or self.last_valid_covariance is None:
                raise ValueError("No valid mean and covariance available.")
            mean = self.last_valid_mean
            covariance = self.last_valid_covariance
            sqrt_cov = lin_alg.cholesky((len(mean) + self.lmbda_c) * covariance)
            self.sig_pts[0] = mean
            for i in range(len(mean)):
                self.sig_pts[i + 1] = mean + sqrt_cov[i]
                self.sig_pts[len(mean) + i + 1] = mean - sqrt_cov[i]
        return self.sig_pts

    def _compute_weights(self):
        n = self.n
        c = 1. /(2 * (n + self.lmbda_c))
        self.Wc = np.full((2*n + 1,), c)
        self.Wm = np.full((2*n + 1,), c)
        self.Wc[0] = self.lmbda_c / (n + self.lmbda_c) + (1. - self.alpha**2 + self.beta)
        self.Wm[0] = self.lmbda_c / (n + self.lmbda_c)

class UKF:
    def __init__(self, dim_x, dim_z, fx, hx, dt, points, x_mean_fn, z_mean_fn, residual_x, residual_z, Q, R, c = 's'):

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.fx = fx
        self.c = c
        self.hx = hx
        if self.c == 'ns':
            self.fx[0].c = 'ns'
            self.hx.c = 'ns'
        self.dt = dt
        self.points = points
        self.x_mean_fn = x_mean_fn
        self.z_mean_fn = z_mean_fn
        self.residual_x = residual_x
        self.residual_z = residual_z
        self.Q = Q 
        self.R = R
        self._num_sigmas = 2 * dim_x + 1
        self.x = np.zeros(dim_x)
        self.P = np.diag([0.01, 0.01, 0.00005, 0.00005, 0.00001, 0.0001, 0.00005, 0.00005, 0.0001, 0.0001, 0.00005, 0.00005])
        self.Wm, self.Wc = self.points.Wm, self.points.Wc
        self.sigmas_f = np.zeros((self._num_sigmas, dim_x))
        self.sigmas_h = np.zeros((self._num_sigmas, dim_z))

        self.sigma_h_list = []

    def predict(self, u):
        for i in range(self._num_sigmas):
            self.fx[0].x, self.fx[0].y, self.fx[0].theta, self.fx[0].delta = self.points.sig_pts[i, 0], self.points.sig_pts[i, 1], self.points.sig_pts[i, 2], self.points.sig_pts[i, 3]
            ar_1 = self.fx[0].step(u[0], u[1])
            self.fx[1].W, self.fx[1].y_off, self.fx[1].y_p, self.fx[1].c_0, self.fx[1].c_1  = self.points.sig_pts[i, 4], self.points.sig_pts[i, 5], self.points.sig_pts[i, 6], self.points.sig_pts[i, 7],self.points.sig_pts[i, 8] 
            ar_2 = self.fx[1].fx_cl(u[0], u[2])
            self.fx[2].x_i,self.fx[2].v_i, self.fx[2].y_i  = self.points.sig_pts[i, 9], self.points.sig_pts[i, 10], self.points.sig_pts[i, 11]
            ar_3 = self.fx[2].fx_o(u[0])
            self.sigmas_f[i] = np.concatenate((ar_1, ar_2, ar_3))
        self.x, self.P = uncented_transform(self.sigmas_f, self.Wm, self.Wc, self.Q, self.x_mean_fn, self.residual_x,self.dt, 'p', self.fx, self.c)

    def update(self, z): 
        for _ in range(0, len(z), self.dim_z):
            for i in range(self._num_sigmas):
                ha_1 = self.hx(self.sigmas_f[i])
                ha_2 = self.fx[1].hx_cl(self.sigmas_f[i])
                ha_3 = self.fx[2].hx_o(self.sigmas_f[i])
                self.sigmas_h[i] = np.concatenate((ha_1, ha_2, ha_3))
                
            self.sigma_h_list.append(self.sigmas_h) 
        sig_h = np.concatenate(self.sigma_h_list, axis = 1)
        self.sigma_h_list = []

        C_ = np.kron(np.eye(int(len(z)/self.dim_z)), self.R)
        # Create R_ using broadcasting and tiling
        # C_ = np.tile(self.R, (int(len(z)/self.dim_z), int(len(z)/self.dim_z)))

        zp, Pz = uncented_transform(sig_h, self.Wm, self.Wc, C_, self.z_mean_fn, self.residual_z, self.dt, 'u', self.hx)
        Pxz = np.zeros((self.dim_x, sig_h.shape[1]))
        for i in range(self._num_sigmas):
            Pxz += self.Wc[i] * np.outer(self.residual_x(self.sigmas_f[i], self.x), self.residual_z(sig_h[i], zp))
        Pz_inv = lin_alg.inv(Pz)
        K = np.dot(Pxz, Pz_inv)
        # Normalize angles in state vector
        self.x = self.x + np.dot(K, self.residual_z(z, zp))
        self.P = self.P - np.dot(np.dot(K, Pz), K.T)

        self.x[2] = normalize_angle(self.x[2])
        self.x[3] = normalize_angle(self.x[3])
        self.x[6] = normalize_angle(self.x[6])

def uncented_transform(sigmas_f, Wm, Wc, C, mean_fn, residual_fn, dt, q, fx, c = 's'):
    Sp = mean_fn(sigmas_f, Wm)

    if q == 'p':
        Sp[3] = normalize_angle(Sp[3])
        Sp[2] = normalize_angle(Sp[2])
        Pp = np.zeros_like(C[0])
        if c == 's':
            beta = normalize_angle(np.arctan2((fx[0].lr * np.tan(Sp[3])), fx[0].L))
        else:
            beta = 0
        L_km = np.array([[dt*np.cos(Sp[2] + beta), 0],
                        [dt*np.sin(Sp[2]+ beta), 0],
                        [dt*np.cos(beta) *np.tan(Sp[3])/fx[0].L, 0],
                        [0, dt]])
        X = np.dot(np.dot(L_km, C[1][:2, :2]), L_km.T)
        C[0][:4, :4] = X

        C[0][4:5, 4:5] = 0.00001
        #C[0][5:6, 5:6] = C[1][0,0] * (fx[1].y_p * dt)**2 + C[1][0,0]**2 * (dt **2 * fx[1].c_0 / 2)**4 + C[1][0,0]**3 * (dt**3 * fx[1].c_1 / 6)**6 + C[1][0,0] * C[1][2,2]* (dt **2 / 2)**4
        C[0][5:6, 5:6] = 0.02
        #C[0][6:7, 6:7] = C[1][0,0] * (dt * fx[1].c_0)**2 + C[1][0,0]**2 *((dt**2)* fx[1].c_1 / 2)**4 +  C[1][2,2]*(dt )**2
        C[0][6:7, 6:7] = 0.01
        #C[0][7:8, 7:8] = C[1][0,0] * (dt)*2
        C[0][7:8, 7:8] = 0.02
        C[0][8:9, 8:9] = 0.01
        #C[0][9:10, 9:10] = C[1][0,0] *(dt)**2
        C[0][9:10, 9:10] = 0.03
        #C[0][10:11, 10:11] = C[1][0,0]
        C[0][10:11, 10:11] = 0.02
        C[0][11:12, 11:12] = 0.01

        for i in range(sigmas_f.shape[0]):
            Pp += Wc[i] * np.outer(residual_fn(sigmas_f[i], Sp), residual_fn(sigmas_f[i], Sp))
        Pp += C[0]
    else:

        z_n = sigmas_f.shape[1]
        Pp = np.zeros((z_n, z_n))

        #print('shape', C.shape, Pp.shape)
        
        for i in range(sigmas_f.shape[0]):
            Pp += Wc[i] * np.outer(residual_fn(sigmas_f[i], Sp), residual_fn(sigmas_f[i], Sp))
        Pp += C

    return Sp, Pp



def state_mean(sigmas, Wm):
    x = np.zeros(12)
    sum_sin = np.sum(np.dot(np.sin(sigmas[:, 2]), Wm))
    sum_cos = np.sum(np.dot(np.cos(sigmas[:, 2]), Wm))

    sum_sin_3 = np.sum(np.dot(np.sin(sigmas[:, 3]), Wm))
    sum_cos_3= np.sum(np.dot(np.cos(sigmas[:, 3]), Wm))

    sum_sin_6 = np.sum(np.dot(np.sin(sigmas[:, 6]), Wm))
    sum_cos_6 = np.sum(np.dot(np.cos(sigmas[:, 6]), Wm))

    x[0] = np.sum(np.dot(sigmas[:, 0], Wm))

    x[1] = np.sum(np.dot(sigmas[:, 1], Wm))

    x[2] = np.arctan2(sum_sin, sum_cos)
    x[3] = np.arctan2(sum_sin_3, sum_cos_3)

    x[4] = np.sum(np.dot(sigmas[:, 4], Wm))
    x[5] = np.sum(np.dot(sigmas[:, 5], Wm))
    x[6] = np.arctan2(sum_sin_6, sum_cos_6)
    x[7] = np.sum(np.dot(sigmas[:, 7], Wm))
    x[8] = np.sum(np.dot(sigmas[:, 8], Wm))
    x[9] = np.sum(np.dot(sigmas[:, 9], Wm))
    x[10] = np.sum(np.dot(sigmas[:, 10], Wm))
    x[11] = np.sum(np.dot(sigmas[:, 11], Wm))


    return x

def z_mean(sigmas, Wm):
    z_count = sigmas.shape[1]

    x = np.zeros(z_count)
    for z in range(0, z_count, 9):
        sum_sin = np.sum(np.dot(np.sin(sigmas[:, z+2]), Wm))
        sum_cos = np.sum(np.dot(np.cos(sigmas[:, z+2]), Wm))
        x[z] = np.sum(np.dot(sigmas[:, z], Wm))
        x[z+1] = np.sum(np.dot(sigmas[:, z+1], Wm))
        x[z+2] = np.arctan2(sum_sin, sum_cos)
        x[z+3] = np.sum(np.dot(sigmas[:, z+3], Wm))
        x[z+4] = np.sum(np.dot(sigmas[:, z+4], Wm))

        sum_sin_5 = np.sum(np.dot(np.sin(sigmas[:, z+5]), Wm))
        sum_cos_5 = np.sum(np.dot(np.cos(sigmas[:, z+5]), Wm))
        x[z+5] = np.arctan2(sum_sin_5, sum_cos_5)
        x[z+6] = np.sum(np.dot(sigmas[:, z+6], Wm))
        x[z+7] = np.sum(np.dot(sigmas[:, z+7], Wm))
        x[z+8] = np.sum(np.dot(sigmas[:, z+8], Wm))

    return x

def residual_h(a, b):
    y = a - b
    for i in range(0, len(y), 9):
        y[i + 2] = normalize_angle(y[i + 2])
        y[i + 5] = normalize_angle(y[i + 5])
    return y

def residual_x(a, b):
    y = a - b
    y[2] = normalize_angle(y[2])
    y[3] = normalize_angle(y[3])
    y[7] = normalize_angle(y[7])
    return y
