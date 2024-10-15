import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='BEV-AD',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--max_fn', '-m', type=str2bool, default=True, choices=[True, False],
        help='Activate Max function for stitching otherwise use masking'
    )
    parser.add_argument(
        '--segmentation_BEV', '-s', type=str2bool, default=False, choices=[True, False],
        help='Activate segmentation'
    )
    parser.add_argument(
        '--segmentation_BEV_modes', '-s_m', type=str, default='m', choices=['m', 'c'],
        help='Segmentation modes clear or moderate'
    )
    parser.add_argument(
        '--lane', '-t', type=str, default='l', choices=['l', 'r'],
        help='Choose between left lane and right lane driving'
    )
    parser.add_argument(
        '--kf', '-f', type=str, default='ekf', choices=['ekf', 'ukf'],
        help='Activate EKF or use UKF drive'        
    )
    parser.add_argument(
        '--single_camera_BEV', '-s_b', type=str, default='r_s', choices=['r_s', 'fm_orb', 'fm_sift'],
        help='Activate Single Camera BEV'
    )
    parser.add_argument(
        '--driving', '-d', type=str, default='sl_d', choices=['f', 'b', 'sl_d', 'pp_d'],
        help='Activate EKF or use only BEV to drive'
    )
    parser.add_argument(
        '--kdc_cluster', '-k', type=str2bool, default=False, choices=[True, False],
        help='Activate kdc'
    )
    parser.add_argument(
        '--dbscan_cluster', '-c', type=str2bool, default=True, choices=[True, False],
        help='Activate DBSCAN'
    )
    parser.add_argument(
        '--perception', '-p', type=str2bool, default=False, choices=[True, False],
        help='Activate Perception module'
    )
    parser.add_argument('--speed', '-c_s', type=float, default=25.0, help='cars straight line speed')
    parser.add_argument('--time', '-s_t', type=float, default=0.02, help='simulation time')

    parser.add_argument('--plot','-p_lt',
        type=str2bool,
        default=False,
        choices=[True, False],
        help='Plot trajectory')

    # save cluster result
    parser.add_argument(
        '--cluster_result',
        '-c_r',
        type=str2bool,
        default=False,
        choices=[True, False],
        help='Save 50 cluster result to /images/cluster_results'
    )
    return parser.parse_args()

def str2bool(v):
    return v.lower() in ('true', '1', 'yes')

def init_simulation_params(args):
    lane_params = {
        'left': {'x_i': -110.5, 'y_i': 40.0, 'z_i': 0.05,'p_i': 0, 'yaw_i': 100, 'r_i': 0},
        'right': {'x_i': -114.4, 'y_i': 40.0, 'z_i': 0.05,'p_i': 0, 'yaw_i': 100, 'r_i': 0}
    }

    initial_state = lane_params['left'] if args.lane == 'l' else lane_params['right']

    car_state = {
        'c_s': 15,  # speed (m/s)
        'max_t': 0.8,  # throttle
        'max_b': 0.3,  # brake
        'max_st': 0.8,  # steering
        'L': 2.5,  # wheelbase
        'l_r': 1,  # rear section length
    }

    return initial_state, car_state

def bev_parameters():
    return {
        'h_img':480, 'w_img':640, #BEV frame dimension height and width (640x480)
        'pixel_to_meter': 3.5 / 92,  # conversion for BEV
        'meter_to_pixel': 92 / 3.5,  # conversion for BEV
        'W': 184, #Lane width in pixels
        'lane_width_pixels': 184,  # lane width in pixels
        'desired_color' : np.array([50, 234, 157, 255]), # Defining the BGR color value to display in segmentation map (for example, red: B=0, G=0, R=255)
        'desired_color_1' : np.array([128, 64, 128, 255])
    }

def velocity_profile_parameters():
    return {
        #For Velocity profile
        'c_a' : 0.3*9.81,  #Centripetal accleration at curves (m/s^2)
        #Turn Curverature threshold to consider straight line(1/m)
        'curv_t' : 1.27*10**-2,
        #rate of change of velocity profile
        'rt_vp' : 0.03
    }

def init_controller_params():
    # Standard deviations
    v_std = 0.002       # Translation velocity standard deviation
    sdot_std = 0.002    # Steering rate standard deviation
    tdot_std = 0.002    # Yaw rate standard deviation

    # State measurement standard deviations
    x_std = 0.01       # x position standard deviation
    y_std = 0.01       # y position standard deviation
    th_std = 0.001     # Yaw angle standard deviation

    # Lane measurement standard deviations
    lr_std = 0.0001      # Right lane measurement standard deviation
    ll_std = 0.0001      # Left lane measurement standard deviation
    y_std = 0.000001     # Relative yaw measurement standard deviation

    # Covariance matrices (variance = std^2)
    cov_bi = np.diag([v_std**2, sdot_std**2, tdot_std**2])  # Input noise covariance

    Q_a = np.zeros((12, 12))

    cov_bi_z = np.diag([x_std**2, y_std**2, th_std**2])  # State measurement noise covariance
    cov_cl_z = np.diag([lr_std**2, ll_std**2, y_std**2, 0.3])  # Lane parameters measurement covariance
    cov_o_z = np.diag([0.5, 0.5])  # Object measurement covariance

    # Overall measurement noise covariance
    R_a = np.zeros((9, 9))
    R_a[:3, :3] = cov_bi_z
    R_a[3:7, 3:7] = cov_cl_z
    R_a[7:9, 7:9] = cov_o_z

    # Controller parameters
    return {
        'k_s1': 1, 'k_s': 2,  # Lateral controller parameters
        'k_p': 1.8, 'k_d': 0.27, 'k_i': 2,  # Longitudinal PID controller parameters
        'wp':2, #Waypoint Distance (m), 
        'd_th':4,# Setting a threshold distance to consider the waypoint reached (m)
        'Q_a': Q_a,  # Process noise covariance for the UKF
        'cov_bi': cov_bi,  # Input noise covariance
        'cov_bi_z': cov_bi_z,  # State measurement noise covariance
        'cov_cl_z': cov_cl_z,  # Lane parameters measurement covariance
        'cov_o_z': cov_o_z,  # Object measurement covariance
        'v_std':v_std,
        'sdot_std':sdot_std,
        'tdot_std':tdot_std,
        'R_a': R_a,  # Overall measurement noise covariance
        'c_nd': 's' #slip or no slip for bycycle model
    }

