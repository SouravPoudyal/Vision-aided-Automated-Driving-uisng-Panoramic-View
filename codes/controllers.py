import numpy as np
import queue
import math

# to compute the velocity profile for determining desired setpoint along the path
def velocity_profile(w_0, w_1, w_2, max_centripetal_acceleration, cv_pr_p, straight_line_velocity, curvature_threshold, rate_of_change):

    current_velocity_pr_p = cv_pr_p
    dx1 = w_1[0] - w_0[0]
    dy1 = w_1[1] - w_0[1]
    dx2 = w_2[0] - w_1[0]
    dy2 = w_2[1] - w_1[1]

    theta1 = np.arctan2(dy1, dx1)
    theta2 = np.arctan2(dy2, dx2)
    curvature = ((theta2 - theta1) / np.linalg.norm(np.array([dx2, dy2]))) 

    radius_of_curvature = 1 / (curvature + 1e-6)

    # To Check for negative or zero curvature
    if radius_of_curvature < 0:
        target_velocity = math.sqrt(-max_centripetal_acceleration * radius_of_curvature)
        target_velocity  = min(target_velocity, straight_line_velocity)

    elif radius_of_curvature == 0:
        target_velocity = straight_line_velocity

    else:
        target_velocity = math.sqrt(max_centripetal_acceleration * radius_of_curvature)
        target_velocity  = min(target_velocity , straight_line_velocity)

    # To Check curvature to determine if moving in a straight line
    is_straight_line = abs(curvature) < curvature_threshold

    if is_straight_line: 
        if current_velocity_pr_p < straight_line_velocity:
            # To set a constant velocity for straight lines
            velocity_change = rate_of_change * (straight_line_velocity - current_velocity_pr_p)
            
            # To limit the rate of change to avoid spikes
            velocity_change = min(velocity_change, straight_line_velocity - current_velocity_pr_p)
            current_velocity_pr_p += velocity_change
            current_velocity_pr_p = min(current_velocity_pr_p, straight_line_velocity)
        else:
            # To gradually decrease velocity towards straight_line_velocity
            velocity_change = rate_of_change * (straight_line_velocity - current_velocity_pr_p)
            current_velocity_pr_p += velocity_change
            current_velocity_pr_p = min(current_velocity_pr_p, straight_line_velocity)
    else:
        # Update velocity profile based on curvature with a rate of change
        velocity_change = rate_of_change * (target_velocity - current_velocity_pr_p)
        current_velocity_pr_p += velocity_change

    return current_velocity_pr_p, curvature

def get_cte(wp_1, wp_2, cp):

    # Vector from the vehicle's current position to the waypoint
    vector_wp_to_vehicle = np.array([wp_1[0] - cp[0], wp_1[1] - cp[1]])
    
    y_p_1 = calculate_yaw_path(wp_1, wp_2)
    y_p = np.deg2rad(np.rad2deg(y_p_1) - 90)

    # Vector perpendicular to the desired path
    vector_normal_to_path = np.array([np.cos(y_p), np.sin(y_p)])
    # Calculates the cross-track error (dot product of the two vectors)
    cte = np.dot(vector_wp_to_vehicle, vector_normal_to_path)
    cte *= -1

    return cte

#Calculates actual yaw of path           
def calculate_yaw_path(wp_1, wp_2):

    delta_x = wp_2[0] - wp_1[0]
    delta_y = wp_2[1] - wp_1[1]
        
    # Calculate the yaw of the path (tangent angle)
    yaw_path = np.arctan2(delta_y, delta_x)

    return yaw_path

# Wraps angle to (-pi,pi] range
def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x

class PIDLongitudinalControl():

    def __init__(self, vehicle, control, get_speed, max_t,max_b, K_P, K_I, K_D, dt):

        self.vehicle = vehicle
        self.control = control
        self.get_speed = get_speed
        self.max_t = max_t
        self.max_b = max_b
        self.dt = dt

        self.K_P = K_P
        self.K_I = K_I
        self.K_D = K_D

        self.errorBuffer = queue.deque(maxlen = 10)
        self.error_fil_buff = queue.deque(maxlen = 2)

        self.sat_acc = 0
        self.sign = False #clamping indicator

        self.prev_de = 0
        self.ie = 0
        self.de = 0
        
    def run_step(self, target_speed):
        current_speed = self.get_speed(self.vehicle)
        return self.pid_controller(target_speed, current_speed)
        
    def pid_controller(self, target_speed, current_speed):
        error = target_speed - current_speed

        self.errorBuffer.append(error)
        if len(self.errorBuffer) >=2:
            if self.sign == False:
                self.ie = sum(self.errorBuffer)*self.dt
            else:
                self.sign = False
            f_o = self.low_pass_filter_diff(self.errorBuffer) 
            self.error_fil_buff.append(f_o)
            if len(self.error_fil_buff) >= 2:
                self.de = (self.error_fil_buff[-1] - self.error_fil_buff[-2])/self.dt
            else:
                self.de = 0
        else:
            self.de = 0
            self.ie = 0
        
        acc = np.clip(self.K_P*error + self.K_D*self.de + self.K_I*self.ie, -1.0, 1.0)
        acc = self.anti_windup(acc, error)

        return acc

    def anti_windup(self, acc, error):
        if abs(acc) >= self.max_t:
            self.sat_acc = self.max_t
        if self.sat_acc != 0:
            if (error >= 0 and acc >= 0) or (error <= 0 and acc <= 0):
                self.sign = True

        return self.sat_acc

    #To use low pass filtering for differential path of PID
    def low_pass_filter_diff(self, error):

        #low pass filter with 5Hz cutoff frequency
        fil_de =  0.7284895 * self.prev_de + 0.13575525 * error[-1] + 0.13575525 * error[-2]
        self.prev_de = fil_de

        return fil_de
    #This function gives longitudinal control commands to the car in simulator
    def control_sig_long(self, target_speed):

        accleration = self.run_step(target_speed)

        if accleration >=0.0:
            self.control.throttle = min(abs(accleration), self.max_t)
            self.control.brake = 0.0
        else:
            self.control.throttle = 0.0
            self.control.brake = min(abs(accleration), self.max_b)
        
        self.control.hand_brake = False
        self.control.manual_gear_shift = False
        
        return self.control

class Lateral_purePursuite_stanley_controller():
    def __init__(self, vehicle, control, args_driving, get_speed, max_st, K_s, K_s1, L):
        self.K_s = K_s
        self.K_s1 = K_s1
        self.L = L
        self.vehicle = vehicle
        self.control = control
        self.args_driving = args_driving
        self.get_speed = get_speed
        self.ey_a = [[0,0]]

        self.past_steering = 0
        self.max_steering = max_st

    def run_step_pp_d(self, wp_1, wp_2, cp):
        speed = self.get_speed(self.vehicle)
        steer, cte = self.pure_pursuit_controller(speed, wp_1, wp_2, cp)
        return steer, cte
    
    def run_step_sl_d(self, wp_1, wp_2, cp):
        speed = self.get_speed(self.vehicle)
        steer, cte = self.stanley_controller_direct(speed, wp_1, wp_2, cp)
        return steer, cte
    
    def run_step_sl(self, x_est, wp_1, wp_2, cp):
        speed = self.get_speed(self.vehicle)
        steer, cte = self.stanley_controller(speed, x_est, wp_1, wp_2, cp)
        return steer, cte

    def run_step_sl_bev(self, e_bev):
        speed = self.get_speed(self.vehicle)
        steer, cte = self.stanley_controller_bev(speed, e_bev)
        return steer, cte

    def pure_pursuit_controller(self, speed, wp_1, wp_2, cp):
        l_d = self.K_s * speed
        cte = get_cte(wp_1, wp_2, cp)
        steer = np.arctan2(2* self.L * cte, pow(l_d, 2))
        return steer, cte

    def stanley_controller_direct(self, speed, wp_1, wp_2, cp):
        cte = get_cte(wp_1, wp_2, cp)
        y_p = calculate_yaw_path(wp_1, wp_2)
        relative_yaw = normalize_angle(y_p - cp[2])
        steer_output = np.arctan2((self.K_s*cte), speed) + relative_yaw
        return steer_output, cte
    
    def stanley_controller(self, speed, x_est, wp_1, wp_2, cp):

        #For actual cross-track error and relative yaw calculation
        cte_a = get_cte(wp_1, wp_2, cp)
        y_p = calculate_yaw_path(wp_1, wp_2)
        relative_yaw_a = normalize_angle(y_p - cp[2])
        self.ey_a.append([cte_a, relative_yaw_a])

        #Using EKF estimated CTE and relative Yaw for driving
        cte_e  = x_est[5]
        relative_yaw_e = normalize_angle(x_est[6])

        steer_output = np.arctan2((self.K_s*cte_e), speed) + self.K_s1 * relative_yaw_e

        return steer_output, cte_e

    def stanley_controller_bev(self, speed, e_bev):
        cte = e_bev[1]
        relative_yaw = normalize_angle(e_bev[0])
        steer_output = np.arctan2((self.K_s*cte), speed) + relative_yaw
        return steer_output, cte
    
    def control_sig_lat(self, wp_1, wp_2, cp, e_bev, x_est):

        if self.args_driving == 'f':
            current_steering, cte = self.run_step_sl(x_est, wp_1, wp_2, cp)
        elif self.args_driving == 'b':
            current_steering, cte = self.run_step_sl_bev(e_bev)
        elif self.args_driving == 'sl_d':
            current_steering, cte = self.run_step_sl_d(wp_1, wp_2, cp)
        else:
            current_steering, cte = self.run_step_pp_d(wp_1, wp_2, cp)

        if current_steering> self.past_steering + 0.1:
           current_steering = self.past_steering + 0.1

        elif current_steering < self.past_steering - 0.1:
           current_steering = self.past_steering - 0.1

        if current_steering>=0:
            steering = min(self.max_steering, current_steering)
        else:
            steering = max(-self.max_steering, current_steering)

        
        self.control.steer = steering
        self.past_steering = steering 
        
        return self.control, cte