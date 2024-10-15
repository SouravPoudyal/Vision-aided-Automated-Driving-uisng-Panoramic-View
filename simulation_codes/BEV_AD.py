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

#importing libraries
import sys
import math
import glob
import os
import numpy as np 
import cv2
import queue
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import threading
from numpy.linalg import inv
import carla
import utils_c
import stitch_alg_vid_1 as stitch_1
import stitch_alg_vid as stitch
import Bezier_curve as be
import argparse
import copy


#To connect to carla server
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# For execution with desired settings
parser = argparse.ArgumentParser(
    description='BEV-AD',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def str2bool(v):
    return v.lower() in ('true', '1', 'yes')

# Left lane driving ('l') or Right lane Driving('r) configuration
parser.add_argument(
    '--lane',
    '-t',
    type=str,
    default='l',
    choices=['l', 'r'],
    help='Choose between left lane and right lane driving'
)

# Use only BEV('b') or BEV with Filter('f') to control
parser.add_argument(
    '--driving',
    '-d',
    type=str,
    default='f',
    choices=['f', 'b'],
    help='Activate EKF or use only BEV to drive'
)

# True to Activate DBSCAN clustering
parser.add_argument(
    '--dbscan_cluster',
    '-c',
    type=str2bool,
    default=True,
    choices=[True, False],
    help='Activate DBSCAN'
)

# True to use Kernel density clustering, for number of available lanes finding
parser.add_argument(
    '--kdc_cluster',
    '-k',
    type=str2bool,
    default=False,
    choices=[True, False],
    help='Activate kdc'
)

# True to use Segmentation Camera
parser.add_argument(
    '--segmentation_BEV',
    '-s',
    type=str2bool,
    default=False,
    choices=[True, False],
    help='Activate segmentation'
)

parser.add_argument(
    '--segmentation_BEV_modes',
    '-s_m',
    type=str,
    default='m',
    choices=['m', 'c'],
    help='Segmentation modes clear or moderate'
)

# True to run panoramic FR BEV stitching
parser.add_argument(
    '--single_camera_BEV',
    '-s_b',
    type=str,
    default='r_s',
    choices=['r_s','fm_orb', 'fm_sift'],
    help='Activate Single Camera BEV'
)

parser.add_argument(
    '--lane_curve',
    '-l_c',
    type=str,
    default='b_c',
    choices=['b_c', 'p_f'],
    help='Ues Bezier curve/poly_fit for lane curve'
)
# save cluster result
parser.add_argument(
    '--cluster_result',
    '-c_r',
    type=str2bool,
    default=False,
    choices=[True, False],
    help='Save cluster result to /imag'
)
#Set vehicle speed
def valid_speed(value):
    speeds = [str(i) for i in range(5, 31, 5)]
    if value not in speeds:
        raise argparse.ArgumentTypeError(f"Invalid speed: {value}. Choose from {', '.join(speeds)}")
    return value

parser.add_argument(
    '--speed',
    '-s_p',
    type=valid_speed,
    default="25",
    help='Save cluster result to /imag'
)

args = parser.parse_args()

print(f'Lane Configuration: {args.lane}')
print(f'Driving Mode: {args.driving}')
print(f'DBSCAN Cluster Activated: {args.dbscan_cluster}')
print(f'KDC Cluster Activated: {args.kdc_cluster}')
print(f'Segmentation BEV Activated: {args.segmentation_BEV}')
print(f'Single Camera BEV Activated: {args.single_camera_BEV}')
print(f'Lane curve visualization: {args.single_camera_BEV}')
print(f'Save DBSCAN cluster result to /imag: {args.single_camera_BEV}')


# Locking to ensure safe access to shared data
result_lock = threading.Lock() 
# Initialization with empty lists
# To Define a global variable to store the results
waypoint_positions = []  
car_positions = []
car_positions_1 = {'k': [], 'car_pos': []}
tim = []
speed = []
c_st = []
c_t = []
cte = []
ey_a = []
lst_k = []
v_profile = []
states = None
covariance_matrix = None

#initializing iteration index
k = 0

'''Initialization Simulation'''
# Sampling time
dt = 0.02
# Iteration limit
max_iteration = 1600
iteration_limit = max_iteration - 1
#BEV frame dimension height and width (640x480)
h_img = 480
w_img = 640

#cars initial state in (m)
#position
if args.lane == 'l':
    x_i= -110.5
    y_i= 40.0
    z_i= 0.05
else:
    x_i=-114.4
    y_i= 40.0
    z_i= 0.05
   
#orientation (deg)
p_i= 0
yaw_i= 100
r_i= 0

#cars speed (m/s) , straight line velocity
c_s = int(args.speed)
#Maximum throttle, maximum_brake, maximum sterring
max_t = 0.8
max_b = 0.3
max_st = 0.8
#wheel base length in (m)
L = 2.5
l_r = 1 # wheel base rear section length

#Pixels to meter in BEV
pixel_to_meter = 3.5 / 92
#Lane width in pixels
W = 184

#For Velocity profile
c_a = 0.3*9.81  #Centripetal accleration at curves (m/s^2)
#Turn Curverature threshold to consider straight line(1/m)
curv_t = 1.27*10**-2
#rate of change of velocity profile
rt_vp = 0.03

# Waypoint Specification
wp = 2  #Waypoint Distance (m)
d_th = 2  # Setting a threshold distance to consider the waypoint reached (m)

#Controller parameters
# Stanley gain:K_s , PID gain: K_p, K_d, K_I
k_s1 = 1 # relative yaw correction gain
k_s = 2.5 # cross track error correction gain

#PID gain for longitudinal controller
k_p = 1.8
k_d =  0.27
k_i = 2

v_std = 0.02  # translation velocity standard deviation 
sdot_std = 0.02  # steering rate standard deviation 
tdot_std = 0.02 # yaw rate standard deviation 
# State measurement standard deviation 
x_std = 0.01 # x position standard deviation 
y_std = 0.01 # y position standard deviation 
th_std = 0.001 # yaw standard deviation 

lr_std = 0.001  # Right lane measurement standard deviation 
ll_std = 0.001  # Left lane measurement standard deviation 
yaw_std = 0.002    # relative yaw measurement standard deviation

#Raw data dictionary
lan_dist = {}
exp = False

#Measurement data list
zm = []

if args.single_camera_BEV == 'r_s':
    # Number of frame for single panoramic FR bev stitching
    m_st = 30
    st_h = m_st
else:
    # Number of frame for single panoramic FR bev stitching
    # Initializing variables
    BaseImage = None
    BaseImageHash = None
    st_h = 6 

# For execition time calculation
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result
    return wrapper

'''
bev functions
'''
def place_car_on_bev(bev, car_i, x, y):
    # Ensuring both images have 3 channels
    if bev.shape[2] == 4:
        bev = cv2.cvtColor(bev, cv2.COLOR_BGRA2BGR)
    if car_i.shape[2] == 4:
        car_i_rgb = cv2.cvtColor(car_i, cv2.COLOR_BGRA2BGR)
        alpha_mask = car_i[:, :, 3]  # Extract the alpha channel
    else:
        car_i_rgb = car_i
        alpha_mask = None

    # Getting dimensions of the car image
    car_height, car_width = car_i_rgb.shape[:2]

    #ROI
    a, b, c, d = y - int(car_height/2), y + int(car_height/2), x - int(car_width/2), x + int(car_width/2) + 1

    # Ensuring the car image fits within the BEV image dimensions
    if y + car_height > bev.shape[0] or x + car_width > bev.shape[1]:
        raise ValueError("The car image exceeds the dimensions of the BEV image at the specified location.")

    # Define the region of interest (ROI) on the BEV image where the car will be placed
    roi = bev[a:b, c:d]
    #

    if alpha_mask is not None:
        # Normalize the alpha mask to keep values between 0 and 1
        alpha = alpha_mask / 255.0
        alpha = alpha[:, :, np.newaxis]

        # Blend the images using the alpha mask
        blended = cv2.convertScaleAbs(roi * (1 - alpha) + car_i_rgb * alpha)
        bev[a:b, c:d] = blended
    else:
        bev[a:b, c:d] = car_i_rgb

    return bev

def rotate_image_i(image, rotation):
    rows, cols = image.shape[:2]

    # Calculate the rotation matrix
    M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2), math.degrees(rotation), 1)

    # Rotate the image
    rotated_image = cv2.warpAffine(image, M_rotate, (cols, rows))

    return rotated_image

def transform_image(image, distance, dx, rotation):
    rows, cols = image.shape[:2]

    # Create the translation matrix to move the image downward
    M_translate = np.float32([[1, 0, dx], [0, 1, distance]])
    translated_image = cv2.warpAffine(image, M_translate, (cols, rows + int(distance)))

    # Rotate the image
    M_rotate = cv2.getRotationMatrix2D((cols / 2, rows / 2 + distance / 2), math.degrees(rotation), 1)
    transformed_image = cv2.warpAffine(translated_image, M_rotate, (cols, rows + int(distance)))

    return transformed_image

def get_relative_transformation(base_pos, target_pos, m_to_pix = int(1/pixel_to_meter)):
    # Calculate the relative translation
    dx = target_pos[0] - base_pos[0]
    dy = target_pos[1] - base_pos[1]

    # Calculate the distance between the two positions
    distance = np.sqrt(dx**2 + dy**2) * m_to_pix

    # Calculate the relative rotation
    dtheta = (target_pos[2] - base_pos[2])

    return dx, distance, dtheta

def draw_polygon(image_2, distance, rotation):
    d_1 = h_img/2 - (65+60) - distance - w_img/2 * np.sin(rotation)
    d_2 = h_img/2 - (65+60) - distance + w_img/2 * np.sin(rotation)
    
    # Ensure the indices are within valid ranges
    d_1 = int(np.clip(d_1, 0, h_img))
    d_2 = int(np.clip(d_2, 0, h_img))
    
    # Define the points of the polygon
    pts = np.array([[0, d_1], [0, h_img], [w_img, h_img], [w_img, d_2]], dtype=np.int32)
    
    # Draw the polygon on the image
    image_with_polygon = image_2.copy()
    cv2.polylines(image_with_polygon, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    
    return image_with_polygon

def mask_2(image_2, distance, rotation):
    d_1 = h_img/2 - (65+50) - distance - w_img/2 * np.sin(rotation)
    d_2 = h_img/2 - (65+50) - distance + w_img/2 * np.sin(rotation)

    
    # Ensure the indices are within valid ranges
    d_1 = int(np.clip(d_1, 0, h_img))
    d_2 = int(np.clip(d_2, 0, h_img))
    
    # Define the points of the polygon
    pts = np.array([[0, d_1], [0, h_img], [w_img, h_img], [w_img, d_2]], dtype=np.int32)
    
    # Create a mask with the same dimensions as the image, initialized to 1 (white)
    mask = np.ones(image_2.shape[:2], dtype=np.uint8)
    
    # Fill the polygon defined by pts with 0 (black) on the mask
    cv2.fillPoly(mask, [pts], 0)
    
    # Apply the mask to the image
    image_2[mask == 0] = 0
    
    return image_2

def stitch_images(image1, image2, distance,dx, rotation):
    # Apply rotation and translation to image1
    transformed_image1 = transform_image(image1, distance, dx, rotation)

    image2 = mask_2(image2, distance, rotation)
    #image2 = draw_polygon(image2, distance, rotation)

    # Ensure both images have 3 channels (convert from 4 to 3 if necessary)
    if image1.shape[2] == 4:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGRA2BGR)
    if image2.shape[2] == 4:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGRA2BGR)
    if transformed_image1.shape[2] == 4:
        transformed_image1 = cv2.cvtColor(transformed_image1, cv2.COLOR_BGRA2BGR)
    

    # Create a canvas large enough to fit both images
    canvas_height = max(image2.shape[0], transformed_image1.shape[0])
    canvas_width = max(image2.shape[1], transformed_image1.shape[1])
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place image2 on the canvas at the top-left corner
    canvas[:image2.shape[0], :image2.shape[1]] = image2

    # Blend the images using cv2.max
    for c in range(3):  # For each color channel
        canvas[:transformed_image1.shape[0], :transformed_image1.shape[1], c] = cv2.max(
            canvas[:transformed_image1.shape[0], :transformed_image1.shape[1], c],
            transformed_image1[:, :, c]
        )
    #canvas = stitch_1.stitch_images(image2, transformed_image1)

    return canvas

#To visualize the hough line image
def display_lines_(image, lines, c):
    line_image = np.zeros_like(image)
    #print(lines)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), c, 4)
    return line_image

# for flitering lines with the knowledge of the left lane or right lane driving
def filter_m_theta(m_theta, lines_1, W, l_d='r'):
    filtered_m_theta = []
    filtered_lines = []

    for i in range(len(m_theta)):
        if l_d == 'r':
            if m_theta[i][3] > 0 and (1/4 * W - 30) <= m_theta[i][3] <= (1/4 * W + 30):
                filtered_m_theta.append(m_theta[i])
                filtered_lines.append(lines_1[i])
            elif m_theta[i][3] < 0 and -(3/4 * W + 30) <= m_theta[i][3] <= -(3/4 * W - 30):
                filtered_m_theta.append(m_theta[i])
                filtered_lines.append(lines_1[i])
        else:
            if m_theta[i][3] > 0 and (3/4 * W + 30) >= m_theta[i][3] >= (3/4 * W - 30):
                filtered_m_theta.append(m_theta[i])
                filtered_lines.append(lines_1[i])
            elif m_theta[i][3] < 0 and -(1/4 * W - 30) >= m_theta[i][3] >= -(1/4 * W + 30):
                filtered_m_theta.append(m_theta[i])
                filtered_lines.append(lines_1[i])

    return np.array(filtered_m_theta), np.array(filtered_lines)

# Function to make a number of elements in a list odd if its even.
# median of even number of elemnets in a list gives mean of middle elemnts, which might result in inaccurate values.
def odd_Out(m_theta, t = 'h'):
    if t == 'h':
        if len(m_theta)>0:
            sorted_indices = np.argsort(m_theta[:, 2])
            m_theta = m_theta[sorted_indices]
            if len(m_theta)%2 == 0:
                m_theta = m_theta[:-1, :]
        else:
            m_theta = np.empty((0, 4))
    else:
        if len(m_theta)>0:
            sorted_indices = np.argsort(m_theta[:, 3])
            m_theta = m_theta[sorted_indices]
            if len(m_theta)%2 == 0:
                m_theta = m_theta[:-1, :]
        else:
            m_theta = np.empty((0, 4))

    return m_theta 

#Function to for  pre-processing relative measurement from panoramic BEV
def _slope_intercept_revised(lines, y21_lst, k, lan_dist, exp, u_c, canvas_1, n_lanes):

    exp = False
    s_k = 0
    n_l_d = 0
    # the fixed end of heading line
    # Center point in wxh bev image
    x11 = int(w_img/2) 
    y11 = int(h_img/2)

    #list of lines from hough line
    lines = lines.reshape(lines.shape[0], lines.shape[2])

    # list to store heading angle of lines detected in preferred Frame(PF)
    a_abs = []

    # image coordinate (IF) to Preferred Frame(PF) (units pixels)
    x_st = lines[:,0]
    y_st = h_img - lines[:,1]
    x_en = lines[:,2]
    y_en = h_img - lines[:,3]
 
    # mid poits of line in PF
    x_mid = (x_st + x_en)/2
    y_mid = (y_st + y_en)/2

    #using arctan2 to avoide infinite slope problem 
    angle = np.rad2deg(np.arctan2((y_en - y_st),(x_en - x_st)))

    #To bring detected lines orientation in range [0, pi]
    for i in range(len(angle)):
        if angle[i] <= 0:
            a_abs.append(180 + angle[i])
        if angle[i] > 0:
            a_abs.append(angle[i])

    # Stack position and orientation of detected lines together
    m_theta = np.stack((x_mid, y_mid, a_abs), axis=1)
    # Convert m_theta into an nx4 matrix
    n = m_theta.shape[0]
    m_theta_with_distance = np.zeros((n, 4))
    point = np.array([w_img/2, h_img/2])

    # Preparing reference points as midpoints of lines
    ref_points = m_theta[:, :2]

    # Calculates cross lane edge distances
    distances = utils_c.v_perpendicular_cross_distance(point, ref_points, m_theta[:, 2])

    # Populate m_theta_with_distances
    m_theta_with_distance[:, :3] = m_theta
    m_theta_with_distance[:, 3] = distances

    # Seperating front, center and back, lane lines from BEV
    def line_separation(m_theta_data, lines_data):
        indices_1 = np.where((210 <= m_theta_data[:, 1]) & (m_theta_data[:, 1] <= 480))[0]
        indices_2 = np.where((215 <= m_theta_data[:, 1]) & (m_theta_data[:, 1] <= 310))[0]
        indices_3 = np.where((0 <= m_theta_data[:, 1]) & (m_theta_data[:, 1] <= 210))[0]

        lines_1 = lines_data[indices_1]
        lines_2 = list(lines_data[indices_2])
        lines_3 = lines_data[indices_3]

        m_theta_1 = np.array(m_theta_data[indices_1])
        m_theta_2 = np.array(m_theta_data[indices_2])
        m_theta_3 = np.array(m_theta_data[indices_3])

        return lines_1, lines_2, lines_3, m_theta_1, m_theta_2, m_theta_3

    lines_1, lines_2, lines_3, m_theta_1, m_theta_2, m_theta_3 = line_separation(m_theta_with_distance, lines)

    # Initial filtering the line with the knowledge of the left lane or right lane driving
    f_m_theta, f_lines = filter_m_theta(m_theta_1,lines_1, W, args.lane)

    #For DBSCAN clustring of detected lines
    if u_c:    
        if len(f_m_theta) >= 2:
            label_dict, max_sublist, canvas_1, n_l_d = utils_c.cluster_line(f_lines, f_m_theta, k, canvas_1, args.lane_curve, args.cluster_result)
            if len(max_sublist)>0:
                lan_dist = utils_c.raw_data(label_dict, max_sublist)
            else:
                exp == True

    else:
        lan_dist = {}
    
    
    if args.kdc_cluster:
        if len(f_m_theta) >= 2:
            #Kernel Density clustering for number of lanes
            s_k = utils_c.KDC_lanes(f_m_theta, k)
            print('Kernel_density_score', s_k)
        else:
            s_k = 0
    
    #To store the number of lanes predicted by KDC and DBSCAN
    n_lanes.append([s_k, n_l_d])

    m_theta_2, _ = filter_m_theta(m_theta_2, lines_2, W, args.lane)

    # Making sure that the set of lines are odd, for median calculation
    f_m_theta = odd_Out(f_m_theta, 'h')
    m_theta_3 = odd_Out(m_theta_3, 'h')

    #For curvature prediction using bezier curve
    if args.driving == 'b' and u_c == False:
        m_pve_c, m_nve_c, canvas_1 = be.bezier_curve_max_min_curvature_2(f_m_theta[f_m_theta[:, 3]<0], canvas_1, W)
        if m_pve_c is not None:
            c0_m = m_pve_c
        else:
            c0_m = m_nve_c

    a_abs_lst = []
    if f_m_theta.size > 0:
        a_abs_lst.append(f_m_theta[:, 2])
    if m_theta_3.size > 0:
        a_abs_lst.append(m_theta_3[:, 2])
    
    # Lines orientation within the below defined angles are considered for median finding technique
    filtered_abs_angs = [angles[(angles >= 24) & (angles <= 156)] for angles in a_abs_lst]

    med_ang_lst = [np.median(angles) if len(angles) > 0 else np.nan for angles in filtered_abs_angs]

    # Coordinates to visualize heading direction on the image
    def coordinates(med_ang, y11, y21, x11, w_img):
        if not np.isnan(med_ang):
            if med_ang != 90:
                slope = -np.tan(np.deg2rad(med_ang))
                x21 = int(((y21 - y11) + x11 * slope) / slope)
            else:
                x21 = int(w_img / 2)
        else:
            x21 = 0
        return x21

    x21_lst = [coordinates(med_ang_lst[k], y11, y21_lst[k], x11, w_img) for k in range(len(med_ang_lst))]

    x0 = np.array([x11, y11])
    x01_lst = [np.array([x21_lst[k], y21_lst[k]]) for k in range(len(x21_lst))]

    return med_ang_lst, x0, x01_lst, lan_dist, exp, list(lines_1), list(lines_2), list(lines_3), m_theta_2, canvas_1, n_lanes

# To draw the detected lines on the black image
def draw_detected_line_on_image(image, lines, k):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
def desired_color_fn(seg_camera_image, dc):
    # Creates a mask for pixels matching the desired color
    mask = np.all(seg_camera_image == dc, axis=2)
    #mask_f = np.all((camera_f_seg == dc) | (camera_f_seg == dc1) , axis=2)

    # Applies the mask to the original image to display only the desired color
    seg_dc = np.zeros_like(seg_camera_image)
    seg_dc[mask] = seg_camera_image[mask]

    return  seg_dc

#Segmentation BEV with all colors
def seg_bev_colors(c_f_seg,c_r_seg,c_l_seg,c_rr_seg, H_f, H_r,H_l,H_rr, dc, q = 'w_max', d_color = False):
        
    bev_seg_f = cv2.warpPerspective(c_f_seg, H_f , (w_img, h_img))
    bev_seg_rr = cv2.warpPerspective(c_rr_seg, H_rr ,(w_img, h_img))
    bev_seg_r = cv2.warpPerspective(c_r_seg, H_r , (w_img, h_img))
    bev_seg_l = cv2.warpPerspective(c_l_seg, H_l , (w_img, h_img))

    bev_seg_f = bev_seg_f.astype(np.uint8)
    bev_seg_rr = bev_seg_rr.astype(np.uint8)
    bev_seg_r = bev_seg_r.astype(np.uint8)
    bev_seg_l = bev_seg_l.astype(np.uint8)

    #To generate the stitched segmented image with mask
    fb_s = cv2.addWeighted(bev_seg_f, 1, bev_seg_rr, 1, 0)
    rl_s = cv2.addWeighted(bev_seg_r, 1, bev_seg_l, 1, 0)

    if q == 'w_max':
        #seg BEV without masking
        bev_s = cv2.max(fb_s, rl_s)
    else:
        bev_s = cv2.addWeighted(fb_s, 1, rl_s, 1, 0)

    if d_color is True:    
        bev_s = desired_color_fn(bev_s, dc)

    return bev_s

def with_masking_stitched_edge_eroded_canny(warped, warped_1, e_m, e_m1):
    # Convert to grayscale and ensure the type is uint8
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    warped_1 = cv2.cvtColor(warped_1, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    
    # Canny edge detection
    dst = cv2.Canny(warped, 160, 200, None, 3)
    dst_1 = cv2.Canny(warped_1, 160, 200, None, 3)
    
    # Find the coordinates of the edges
    edge_coords_1 = np.nonzero(dst_1)
    edge_coords = np.nonzero(dst)
    
    # Creates a new black image with one channel
    h, w = dst.shape
    black_img = np.zeros((h, w, 1), dtype=np.uint8)
    black_img_1 = np.zeros((h, w, 1), dtype=np.uint8)
    
    # Setting the pixels at the edge coordinates to white
    black_img[edge_coords] = 255
    black_img_1[edge_coords_1] = 255
    
    # Eroding part of canny image to remove the line separating the common region
    img_dst = cv2.bitwise_and(black_img, e_m)
    dst_canny = cv2.Canny(img_dst, 110, 130, None, 3)
    
    img_dst_1 = cv2.bitwise_and(black_img_1, e_m1)
    dst_canny_1 = cv2.Canny(img_dst_1, 110, 130, None, 3)
    
    # Final canny image
    warped = cv2.add(dst_canny, dst_canny_1)
    
    return warped

def to_generate_stitched_image(b_f, b_r, b_l, b_rr, q='w_max'):

    # Convert all images to the same type
    
    b_f = b_f.astype(np.uint8)
    b_r = b_r.astype(np.uint8)
    b_l = b_l.astype(np.uint8)
    b_rr = b_rr.astype(np.uint8)

    # Generates the stitched image
    fb = cv2.addWeighted(b_f, 1, b_rr, 1, 0)
    rl = cv2.addWeighted(b_r, 1, b_l, 1, 0)

    if q == 'w_max':
        # BEV without masking
        bev = cv2.max(fb, rl)
    else:
        # BEV that requires masking
        bev = cv2.addWeighted(fb, 1, rl, 1, 0)

    return bev

def create_masked_BEV_image(bev_m_1, bev_m_2, bev_, q='w_max'):

    # Convert all images to the same type
        
    bev_m_1 = bev_m_1.astype(np.uint8)
    bev_m_2 = bev_m_2.astype(np.uint8)
    bev_ = bev_.astype(np.uint8)

    # To use the mask to create masked BEV image
    bev_am1 = cv2.bitwise_and(bev_m_1, bev_)
    bev_am2 = cv2.bitwise_and(bev_m_2, bev_)

    if q == 'w_max':
        # To add the two masked BEV to generate final BEV image
        bev = cv2.addWeighted(bev_am1, 1, bev_am2, 1, 1)
    else:
        # To add the two masked BEV to generate final BEV image
        bev = cv2.addWeighted(bev_am1, 0.5, bev_am2, 1, 1)

    return bev_am1, bev_am2, bev
    
def without_masking_eroded_canny(b_warp):
    # Ensure the input image is of type uint8
    b_warp = b_warp.astype(np.uint8)

    # Convert to grayscale
    warped = cv2.cvtColor(b_warp, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    warped = cv2.Canny(warped, 110, 130, None, 3)
    
    return warped

'''
controller functions and class
'''

# Calculates actual CTE
def calculate_cte(waypoint, x, y, yaw):
        # waypoint is a list [x_wp, y_wp, v_wp] representing a waypoint
        x_wp, y_wp = waypoint.transform.location.x, waypoint.transform.location.y
        
        # Vector from the vehicle's current position to the waypoint
        vector_wp_to_vehicle = np.array([x_wp - x, y_wp - y])

        y_p_1 = calculate_yaw_path(waypoint)
        
        y_p = np.deg2rad(np.rad2deg(y_p_1) - 90)

        
        # Vector perpendicular to the desired path (path tangent)
        vector_path_tangent = np.array([np.cos(y_p), np.sin(y_p)])
        
        # Calculates the cross-track error (dot product of the two vectors)
        cte = np.dot(vector_wp_to_vehicle, vector_path_tangent)
        cte *= -1

        return cte

#Calculates actual relative yaw of path w.r.t vehicle            
def calculate_yaw_path(waypoint):
        x_wp, y_wp = waypoint.transform.location.x, waypoint.transform.location.y
        next_waypoint = waypoint.next(wp)[0]
        x_next, y_next = next_waypoint.transform.location.x, next_waypoint.transform.location.y

        delta_x = x_next - x_wp
        delta_y = y_next - y_wp
            
        # Calculate the yaw of the path (tangent angle)
        yaw_path = np.arctan2(delta_y, delta_x)

        return yaw_path

# to compute the velocity profile for determining desired setpoint along the path
def velocity_profile(w_0, w_1, w_2, max_centripetal_acceleration, cv_pr_p, straight_line_velocity, curvature_threshold = curv_t, rate_of_change= rt_vp):

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

    return current_velocity_pr_p

# Function to calculate Rise Time, Overshoot, Settling Time, and Steady-State Error
def calculate_metrics(time, speed, target_speed):

    target_s  = np.mean(target_speed)  
    #steady_state_speed = np.mean(speed)

    rise_90_i = np.argmax(speed >= 0.90 * target_s)
    steady_state_speed = np.mean(speed[rise_90_i:])

    # Finding the index where the speed first exceeds 10% of the steady-state speed
    rise_10_index = np.argmax(speed >= 0.1 * steady_state_speed)

    # Finding the index where the speed first exceeds 90% of the steady-state speed
    rise_90_index = np.argmax(speed >= 0.9 * steady_state_speed)

    # Calculating rise time as the difference between the two indices
    rise_time = time[rise_90_index] - time[rise_10_index]
    # To calculate the normalized response
    ynorm = (speed - speed[0]) / ( speed[-1]- speed[0])

    # To calculate the percentage overshoot using the normalized response
    overshoot = max(np.max(ynorm) - 1, 0) * 100

    # To calculate the percentage undershoot using the normalized response
    undershoot = min(np.min(ynorm), 0) * -100

    # Overshoot alternatively
    #overshoot = (max(speed) - steady_state_speed) / steady_state_speed * 100

    # Settling Time (assuming the signal is settling)
    settling_time_index = np.argmax(np.abs(speed - steady_state_speed) <= 0.05 * steady_state_speed)
    settling_time = time[settling_time_index]

    # Steady-State Error
    # Calculates the mean absolute difference between target_speed and actual speed
    steady_state_error = np.abs(target_s - steady_state_speed)

    return rise_time, overshoot,undershoot, settling_time, steady_state_error

#Calculating the LLE and RLE distance to car
def process_m_theta(m_theta_2):
    def filter(m_t):
        # Create a boolean mask for the condition
        mask = (m_t[:, 2] >= 20) & (m_t[:, 2] <= 170)
        f_mt = m_t[mask]
        
        return f_mt
    m_theta_2 = filter(m_theta_2)
    if m_theta_2.size > 0:
        # Step 1: Filter rows where the 4th column is negative
        m_1 = m_theta_2[m_theta_2[:, 3] < 0]
        if m_1.size > 0:
            m_1 = odd_Out(m_1, 'p')
            dl_m = np.median(m_1[:, 3])
        else:
            dl_m = 0

        # Step 2: Filter rows where the 4th column is positive
        m_2 = m_theta_2[m_theta_2[:, 3] > 0]
        if m_2.size > 0:
            m_2 = odd_Out(m_2, 'p')
            dr_m = np.median(m_2[:, 3])
        else:
            dr_m = 0
    else:
        dl_m, dr_m = 0, 0
    return dl_m, dr_m
#Calculates cross-track error using relative distances of lane
def find_cte_rl(dl_m , dr_m, ct_e, W, pix_t_meter, c = 'l'):
    if c == 'r':
        if dr_m == 0 and dl_m != 0:
            ct_e = (3*W/4 + dl_m)
        elif dr_m == 0 and dl_m == 0:
            ct_e = 0
        else:
            ct_e = (dr_m - W/4)
    else:
        if dl_m == 0 and dr_m != 0:
            ct_e = (dr_m - 3*W/4)
        elif dr_m == 0 and dl_m == 0:
            ct_e = 0
        else:
            ct_e = (W/4 + dl_m)

    ct_e = ct_e * pix_t_meter

    return ct_e

def n_lanes_plot(n_lanes):
    # Creating the x-axis values
    n = list(range(len(n_lanes)))
    n_lanes = np.array(n_lanes)
    
    # Plotting the data
    plt.figure(figsize=(10, 6))
    plt.scatter(n, n_lanes[:, 0], color='blue', marker='o', s=100, label='KDC')
    plt.scatter(n, n_lanes[:, 1], color='red', marker='x', s=100, label='DBSCAN')

    # Adding labels and title
    plt.xlabel('Iteration Index, k')
    plt.ylabel('Number of Lanes')
    plt.title('Predicted Number of Lanes')
    plt.legend()

    # Displaying the plot
    plt.show()

'''Longitudinal PID controller class'''          
class PIDLongitudinalControl():

    def __init__(self, vehicle, x_est, k, control, K_P, K_D, K_I, dt, max_throttle = max_t, max_break = max_b):
        self.vehicle = vehicle
        self.max_break = max_break
        self.max_throttle = max_throttle

        self.K_D = K_D
        self.K_I = K_I
        self.K_P = K_P
        self.dt = dt

        self.x_est = x_est
        self.k = k
        self.control = control

        self.errorBuffer = queue.deque(maxlen = 10)
        self.error_fil_buff = queue.deque(maxlen = 2)

        self.sign = False #clamping indicator

        self.prev_de = 0
        self.ie = 0
        self.de = 0
    
    def run_step(self, target_speed, waypoint):
        current_speed = get_speed(self.vehicle) + np.random.normal(loc=0, scale=v_std)
        return self.pid_controller(target_speed, current_speed, waypoint)
    
    def pid_controller(self, target_speed, current_speed, waypoint):

        self.sign = False
        error = (target_speed - current_speed)

        self.errorBuffer.append(error)

        if len(self.errorBuffer)>=2:
            if self.sign == False:
                self.ie = sum(self.errorBuffer)*self.dt
            else:
                self.sign = False
            f_o = self.low_pass_filter_diff(self.errorBuffer)
            self.error_fil_buff.append(f_o)
            if len(self.error_fil_buff)>=2:
                self.de = (self.error_fil_buff[-1] - self.error_fil_buff[-2])/self.dt
            else:
                self.de = 0 
        else:
            self.de = 0.0
            self.ie = 0.0
        
        acc = np.clip(self.K_P*error + self.K_D*self.de + self.K_I*self.ie, -1.0, 1.0)

        acc = self.anti_windup(acc, error)

        return acc

    def anti_windup(self, acc, error):
        if abs(acc) > max_t:
            acc = max_t
            if (error >= 0 and acc >= 0) or (error < 0 and acc < 0):
                self.sign = True

        return acc

    #To use low pass filtering for differential path of PID
    def low_pass_filter_diff(self, error):

        #low pass filter with 5Hz cutoff frequency
        fil_de =  0.7284895 * self.prev_de + 0.13575525 * error[-1] + 0.13575525 * error[-2]
        self.prev_de = fil_de

        return fil_de
    #This function gives longitudinal control commands to the car in simulator
    def control_sig_long(self, target_speed, waypoint):

        accleration = self.run_step(target_speed, waypoint)

        if accleration >=0.0:
            self.control.throttle = min(abs(accleration), self.max_throttle)
            self.control.brake = 0.0
        else:
            self.control.throttle = 0.0
            self.control.brake = min(abs(accleration), self.max_break)
        
        self.control.hand_brake = False
        self.control.manual_gear_shift = False
        return self.control
    
'''Lateral controller class'''    
class Lateral_purePursuite_stanley_controller():

    def __init__(self, vehicle, x_est_bev, x_est, k, control, K_s, K_s1, L, max_steering = max_st):
        
        self.x_est= x_est
        self.x_est_bev= x_est_bev
        self.k = k
        self.control = control
        self.vehicle = vehicle
        self.K_s = K_s
        self.K_s1 = K_s1
        self.L = L
        self.max_steering = max_steering
        self.past_steering = self.vehicle.get_control().steer
        self.ey_a = [[0,0]]

    def run_step_pp(self, waypoint):
        return self.pure_pursuit_controller(waypoint, self.x_est, self.k)
    
    def run_step_d(self, waypoint):
        return self.stanley_controller_direct(waypoint, self.vehicle.get_transform())
    
    def run_step_sl(self, waypoint):
        return self.stanley_controller(waypoint, self.x_est, self.k)
    
    def run_step_sl_bev(self, waypoint):
        return self.stanley_controller_bev(waypoint, self.x_est_bev)

    def pure_pursuit_controller(self, waypoint, x_est, k):

        cte = calculate_cte(waypoint, x_est[k, 0], x_est[k, 1], x_est[k, 2])

        l_d = self.K_s * get_speed(self.vehicle)
        sin_alpha = cte/(dt*l_d)
        steer_output = np.arctan2((2*self.L*sin_alpha), l_d)

        return steer_output, cte
    
    # Function used to drive the car using actual relative lane state measurements using waypoints
    def stanley_controller_direct(self, waypoint, vehicle_transform):

        cte = calculate_cte(waypoint, vehicle_transform.location.x, vehicle_transform.location.y, math.radians(vehicle_transform.rotation.yaw))
        y_p = calculate_yaw_path(waypoint)
        yaw_diff_heading = normalize_angle(y_p - math.radians(vehicle_transform.rotation.yaw))
        v = get_speed(self.vehicle)

        steer_output = np.arctan2((self.K_s*cte), v) + yaw_diff_heading

        return steer_output, cte

    #Uses EKF estimation for driving
    def stanley_controller(self, waypoint, x_est, k):
        
        #For actual cross-track error and relative yaw calculation
        cte_a = calculate_cte(waypoint, self.vehicle.get_transform().location.x, self.vehicle.get_transform().location.y, math.radians(self.vehicle.get_transform().rotation.yaw))
        y_p_a = calculate_yaw_path(waypoint)
        yaw_diff_heading_a = normalize_angle(y_p_a - math.radians(self.vehicle.get_transform().rotation.yaw))
        self.ey_a.append([cte_a, yaw_diff_heading_a])

        #Using EKF estimated CTE and relative Yaw for driving
        cte = x_est[k, 4]
        v = get_speed(self.vehicle)
        yaw_diff_heading = normalize_angle(x_est[k, 5])

        steer_output = np.arctan2((self.K_s*cte), v) + self.K_s1 * yaw_diff_heading

        return steer_output, cte
    
    #Uses BEV measurements for driving, which is filtered either from DBSCAN or median finding technique 
    def stanley_controller_bev(self ,waypoint , x_est_bev):

        cte = x_est_bev[1]
        yaw_diff_heading = normalize_angle(x_est_bev[0])

        v = get_speed(self.vehicle) + np.random.normal(loc=0, scale=v_std)

        steer_output = np.arctan2((self.K_s*cte), v) + self.K_s1 * yaw_diff_heading

        return steer_output, cte
    
    #This function gives lateral control commands to the car in simulator
    def control_sig_lat(self, waypoint):

        if args.driving == 'f':
            current_steering, cte = self.run_step_sl(waypoint)
        elif args.driving == 'b':
            current_steering, cte = self.run_step_sl_bev(waypoint)
        else:
            current_steering, cte = self.run_step_d(waypoint)


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

'''EKF functions'''     
#To get the cars velocity from the simulator in m/s
def get_speed(vehicle):
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

# computing correction step of EKF
def measurement_update(vehicle, lr_m, ll_m, phi_m, P_check, x_check, cov_y ,W, c_f):
    x_k, y_k, th_k, _ , yo_k, phi_k = x_check[0, 0], x_check[0, 1], x_check[0, 2], x_check[0, 3], x_check[0, 4], x_check[0, 5]
    x_check_copy = np.copy(x_check)
    x_g, y_g, z_g = vehicle.get_location().x + np.random.normal(loc=0, scale=x_std), vehicle.get_location().y + np.random.normal(loc=0, scale=y_std), math.radians(vehicle.get_transform().rotation.yaw) + np.random.normal(loc=0, scale=th_std)

    #Left lane driving or right lane driving setting
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

    #Computing Kalman Gain
    K_k = np.dot(np.dot(P_check, H_k.T), inv(np.dot(np.dot(H_k, P_check), H_k.T) + np.dot(np.dot(M_k, cov_y), M_k.T)))

    #Correcting predicted state (wrapping the angles to [-pi,pi])
    z_k = np.array([x_k, y_k, th_k, normalize_angle(p_m), l_r, l_l])
    z_m =np.array([x_g, y_g, z_g, phi_m, lr_m, ll_m])
    x_check_copy +=np.dot(K_k, (z_m - z_k).reshape(-1, 1)).T
    x_check_copy[0,2] = normalize_angle(x_check_copy[0,2])
    #Correcting covariance
    P_check =  np.dot((np.eye(6) - np.dot(K_k, H_k)), P_check)

    return x_check_copy, P_check

# Wraps angle to (-pi,pi] range
def normalize_angle(x):
    x = x % (2 * np.pi)    # force in range [0, 2 pi)
    if x > np.pi:          # move to [-pi, pi)
        x -= 2 * np.pi
    return x

#Longitudinal controller thread
@measure_time
def long_control_loop(long_controller, world):

    global  c_t, v_profile, lst_k, k, tim, speed, waypoint_positions, car_positions, next_waypoint
    #For EXECUTION time CALCULATION
    iteration = 0

    reached_waypoints = 0
    v_profile.append(c_s)
    tim.append(0)
    speed.append(0)
    waypoint_distance = wp

    t_init_1 = 0
    t_prev_1 = time.time()

    t_init_2 = 0
    t_prev_2 = time.time()

    d_tim = 0
    c_t.append(0)
    current_waypoint = world.get_map().get_waypoint(long_controller.vehicle.get_location())
    next_waypoint = current_waypoint.next(waypoint_distance)
    waypoint_positions.append((next_waypoint[0].transform.location.x, next_waypoint[0].transform.location.y))
    car_positions.append((long_controller.vehicle.get_location().x, long_controller.vehicle.get_location().y, math.radians(long_controller.vehicle.get_transform().rotation.yaw)))
    while True:
        # Start the timer
        start_time = time.perf_counter()

        t_init_1  = time.time()
        d_tim = (t_init_1 - t_prev_1)
        long_controller.dt = d_tim 
        
        current_location = long_controller.vehicle.get_location()
        
        distance_to_waypoint = current_location.distance(next_waypoint[0].transform.location)
        control = long_controller.control_sig_long(v_profile[-1], next_waypoint[0])
    
        long_controller.vehicle.apply_control(control)
        c_t.append(control.throttle)

        if distance_to_waypoint <= d_th:  # Setting a threshold distance to consider the waypoint reached
            
            reached_waypoints += 1
            next_waypoint = next_waypoint[0].next(waypoint_distance)
            
            # for visualization 
            waypoint_positions.append((next_waypoint[0].transform.location.x, next_waypoint[0].transform.location.y))
            car_positions.append((long_controller.vehicle.get_location().x, long_controller.vehicle.get_location().y, math.radians(long_controller.vehicle.get_transform().rotation.yaw)))#
            
            #For computing speed profile as a setpoint speed
            third_nxt_wp = np.array([next_waypoint[0].next(waypoint_distance)[0].transform.location.x,next_waypoint[0].next(waypoint_distance)[0].transform.location.y])
            v_pr_p = velocity_profile(np.array(waypoint_positions[-2]), np.array(waypoint_positions[-1]), third_nxt_wp, c_a, v_profile[-1], c_s)
            v_profile.append(v_pr_p)

            #storing data for visualization
            speed.append(get_speed(long_controller.vehicle))
            #For calculating actual time recorded in a loop or time instance 'k'
            t_init_2 = time.time()
            tim.append(tim[-1]+(t_init_2 - t_prev_2))
            t_prev_2 = t_init_2

            lst_k.append(k)


        t_prev_1 = t_init_1

        # Delay of 0.02 seconds to match simulator sampling rate
        time.sleep(0.02)
        # End the timer
        end_time = time.perf_counter()
        # Calculate the elapsed time
        execution_time = end_time - start_time
        iteration += 1
        print(f"longitudinal Iteration {iteration} execution time: {execution_time} seconds")

        if k >= iteration_limit:
            break

    with result_lock:
        lst_k = lst_k
        v_profile = v_profile
        c_t = c_t
        speed = speed
        tim = tim       
        waypoint_positions = waypoint_positions
        car_positions = car_positions

#Lateral controller and EKF thread
@measure_time
def filter_control_loop(P_est, x_est, dt, world, vehicle, lan_dist, control):

    global states, covariance_matrix, c_st, c_t, lst_k, cte, h_img, w_img, l_r, exp, k, st_h, zm, next_waypoint, ey_a, car_positions_1
    '''
    bev initialization
    '''
    #Homegraphy matrix
    H_Front = np.array([[ 1.07925700e+00,  6.09261809e+00, -2.60161731e+01],
                        [-8.67471471e-04,  4.06327717e+00, -1.51819861e+02],
                        [-7.77621391e-06,  1.90725299e-02,  1.00000000e+00]])

    H_Right = np.array([[-2.35002754e-01,  4.06414289e+01,  3.68431154e+03],
                        [ 1.12359731e+01,  3.00506236e+01, -3.41145949e+03],
                        [-5.77704629e-04,  1.39711320e-01,  1.00000000e+00]])

    H_Left = np.array([[-1.82550719e-01, -2.33310221e+02,  1.61839443e+04],
                    [ 5.35024410e+01, -1.44871891e+02, -1.69284305e+04],
                    [-9.40466687e-04, -6.66967739e-01,  1.00000000e+00]])

    H_Rear = np.array([[ 1.10704364e-01, -5.98863379e-01,  2.82972013e+02],
                    [ 3.47819477e-04, -4.29284318e-01,  2.64289244e+02],
                    [ 1.02370047e-06, -1.88186677e-03,  1.00000000e+00]])
    
    
    # Defining the BGR color value to display in segmentation map (for example, red: B=0, G=0, R=255)
    desired_color = np.array([50, 234, 157, 255])
    desired_color_1 = np.array([128, 64, 128, 255])

    #Creating a white image of Size w_imgxh_img
    img_w = np.zeros([h_img,w_img,4],dtype=np.uint8)
    img_w.fill(255)

    #Using respective homography matrix to perspective transform the white image
    #to create mask for each BEV 
    bev_f_m = cv2.warpPerspective(img_w, H_Front, (w_img, h_img), flags=cv2.INTER_LINEAR)
    bev_r_m = cv2.warpPerspective(img_w, H_Right, (w_img, h_img), flags=cv2.INTER_LINEAR)
    bev_l_m = cv2.warpPerspective(img_w, H_Left, (w_img, h_img), flags=cv2.INTER_LINEAR)
    bev_b_m = cv2.warpPerspective(img_w, H_Rear, (w_img, h_img), flags=cv2.INTER_LINEAR)

    #Generating masks 1 (with common region) and mask 2(without common region) for BEV
    fb_ = cv2.addWeighted(bev_f_m, 1, bev_b_m, 1, 0)
    rl_ = cv2.addWeighted(bev_r_m, 1, bev_l_m, 1, 0)
    bev_m_1 = cv2.bitwise_and(fb_, rl_)
    bev_m_2 = cv2.bitwise_xor(fb_, rl_)


    # Applying erosion to masks 1
    gray = cv2.cvtColor(bev_m_2, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5), np.uint8)
    eroded_img = cv2.erode(gray, kernel, iterations=1)


    # Applying erosion to masks 2
    gray_1 = cv2.cvtColor(bev_m_1, cv2.COLOR_BGR2GRAY)
    kernel_1 = np.ones((5,5), np.uint8)
    eroded_img_1 = cv2.erode(gray_1, kernel_1, iterations=1)

    # Creating segmentation map blueprints for front, left, right, and rear cameras
    if args.segmentation_BEV:
        sem_camera_bp_f = world.get_blueprint_library().find('sensor.camera.semantic_segmentation') 
        sem_camera_bp_f.set_attribute('image_size_x', str(w_img))
        sem_camera_bp_f.set_attribute('image_size_y', str(h_img))

        sem_camera_bp_f.set_attribute('fov', '110')  

        sem_camera_bp_r = world.get_blueprint_library().find('sensor.camera.semantic_segmentation') 
        sem_camera_bp_r.set_attribute('image_size_x', str(w_img))
        sem_camera_bp_r.set_attribute('image_size_y', str(h_img))

        sem_camera_bp_r.set_attribute('fov', '110')  

        sem_camera_bp_l = world.get_blueprint_library().find('sensor.camera.semantic_segmentation') 
        sem_camera_bp_l.set_attribute('image_size_x', str(w_img))
        sem_camera_bp_l.set_attribute('image_size_y', str(h_img))

        sem_camera_bp_l.set_attribute('fov', '110')  

        sem_camera_bp_rr = world.get_blueprint_library().find('sensor.camera.semantic_segmentation') 
        sem_camera_bp_rr.set_attribute('image_size_x', str(w_img))
        sem_camera_bp_rr.set_attribute('image_size_y', str(h_img))

        sem_camera_bp_rr.set_attribute('fov', '110') 

    # Creating camera blueprints for front, left, right, and rear cameras
    camera_bp_front = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp_front.set_attribute('image_size_x', str(w_img))
    camera_bp_front.set_attribute('image_size_y', str(h_img))

    camera_bp_front.set_attribute('fov', '110')

    camera_bp_left = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp_left.set_attribute('image_size_x', str(w_img))
    camera_bp_left.set_attribute('image_size_y', str(h_img))

    camera_bp_left.set_attribute('fov', '110') 

    camera_bp_right = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp_right.set_attribute('image_size_x', str(w_img))
    camera_bp_right.set_attribute('image_size_y', str(h_img))

    camera_bp_right.set_attribute('fov', '110')  

    camera_bp_rear = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp_rear.set_attribute('image_size_x', str(w_img))
    camera_bp_rear.set_attribute('image_size_y', str(h_img))

    camera_bp_rear.set_attribute('fov', '110')

    # To initialize camera transforms for front, left, right, and rear cameras
    camera_trans_front = carla.Transform(carla.Location(x = 2.2, y = 0, z = 1.2), carla.Rotation(pitch=-52.5, yaw=0, roll=0))  # Front camera
    camera_trans_left = carla.Transform(carla.Location(x = 0.83, y = -1, z = 1.59), carla.Rotation(pitch=-46, yaw=-90, roll=0))  #  Left camera
    camera_trans_right = carla.Transform(carla.Location(x = 0.83, y = 1, z = 1.59), carla.Rotation(pitch=-46, yaw=90, roll=0))  # right camera
    camera_trans_rear = carla.Transform(carla.Location(x = -2.2, y = -0.1, z = 1.2), carla.Rotation(pitch=-127.5, yaw=0, roll=0))  # rear camera                         
    

    # Creating a black canvas (image) of desired size
    height, width = h_img, w_img 
    black_image = np.zeros((height, width, 3), dtype=np.uint8)

    ##############################################################

    ######BEV to vehicle###################
    if args.segmentation_BEV:
        camera_seg_f = world.spawn_actor(sem_camera_bp_f, camera_trans_front, attach_to=vehicle)
        camera_seg_r = world.spawn_actor(sem_camera_bp_r, camera_trans_right, attach_to=vehicle)
        camera_seg_l = world.spawn_actor(sem_camera_bp_l, camera_trans_left, attach_to=vehicle)
        camera_seg_rr = world.spawn_actor(sem_camera_bp_rr, camera_trans_rear, attach_to=vehicle)

    # Creates cameras and attach them to the vehicle
    camera_front = world.spawn_actor(camera_bp_front, camera_trans_front, attach_to=vehicle)
    camera_left = world.spawn_actor(camera_bp_left, camera_trans_left, attach_to=vehicle)
    camera_right = world.spawn_actor(camera_bp_left, camera_trans_right, attach_to=vehicle)
    camera_rear = world.spawn_actor(camera_bp_left, camera_trans_rear, attach_to=vehicle)

    if args.segmentation_BEV:
        def sem_callback(image, data_dict, camera_name):
            image.convert(carla.ColorConverter.CityScapesPalette)
            data_dict[camera_name] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

    # Callback stores sensor data in a dictionary for use outside the callback
    def camera_callback(image, data_dict, camera_name):
        data_dict[camera_name] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

    # To get camera dimensions and initialize dictionaries
    image_w = camera_bp_front.get_attribute("image_size_x").as_int()
    image_h = camera_bp_front.get_attribute("image_size_y").as_int()

    if args.segmentation_BEV:
        camera_data = {
            'seg_image_f': np.zeros((image_h, image_w, 4)),
            'seg_image_r': np.zeros((image_h, image_w, 4)),
            'seg_image_l': np.zeros((image_h, image_w, 4)),
            'seg_image_rr': np.zeros((image_h, image_w, 4)),  
            'front_image': np.zeros((image_h, image_w, 4)),
            'left_image': np.zeros((image_h, image_w, 4)),
            'right_image': np.zeros((image_h, image_w, 4)),
            'rear_image': np.zeros((image_h, image_w, 4))
        }
    else:
        camera_data = { 
            'front_image': np.zeros((image_h, image_w, 4)),
            'left_image': np.zeros((image_h, image_w, 4)),
            'right_image': np.zeros((image_h, image_w, 4)),
            'rear_image': np.zeros((image_h, image_w, 4))
        }


    # Start camera recording
    if args.segmentation_BEV:
        camera_seg_f.listen(lambda image: sem_callback(image, camera_data, 'seg_image_f'))
        camera_seg_r.listen(lambda image: sem_callback(image, camera_data, 'seg_image_r'))
        camera_seg_l.listen(lambda image: sem_callback(image, camera_data, 'seg_image_l'))
        camera_seg_rr.listen(lambda image: sem_callback(image, camera_data, 'seg_image_rr'))

    camera_front.listen(lambda image: camera_callback(image, camera_data, 'front_image'))
    camera_left.listen(lambda image: camera_callback(image, camera_data, 'left_image'))
    camera_right.listen(lambda image: camera_callback(image, camera_data, 'right_image'))
    camera_rear.listen(lambda image: camera_callback(image, camera_data, 'rear_image'))

    ####################################################################

    #Initializing variables for EKF and controller
    Q_km = np.diag([v_std**2, sdot_std**2, tdot_std**2]) # input noise covariance 
    cov_y = np.diag([x_std**2, y_std**2, th_std**2, lr_std**2, ll_std**2, y_std**2])  # measurement noise covariance

    #Initializing prediction
    phi_prev = x_est[:, 5]
    st_prev = x_est[:, 3]
    #print('st_prev', st_prev)
    th_prev = x_est[:, 2]
    
    x_check = x_est[0].reshape((1,6))
    P_check = P_est[0]

    ##########
    #relative yaw and crosstrack measurement values initialization
    y_d_h , ct_e = 0 , 0
    x_est_bev = np.array([y_d_h, ct_e])

    ## Previous vehicle yaw and steering initialization
    prev_th = 0
    prev_str = 0

    ## Timing calculation initialization
    t_init = 0
    t_prev = time.time()
    t_diff = 0

    ## FPS initialization
    fps = 0
    #Number of lanes KDC vs DBSCAN
    n_lanes = []


    #Desired List initilization for result
    c_st.append(0)
    cte.append(0)
    lst_k.append(k)
    #For execution time calculation
    iteration_1 = 0

    text_1 = ''

    #Creating vehicle lateral controller object
    args_lateral={'K_s': k_s,'K_s1': k_s1, 'L': L}
    lat_controller = Lateral_purePursuite_stanley_controller(vehicle, x_est_bev, x_est, k, control, **args_lateral)


    cv2.namedWindow('stitched', cv2.WINDOW_AUTOSIZE)
    if args.segmentation_BEV:
        cv2.namedWindow('seg_bev', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('line_bev', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Lane_approx", cv2.WINDOW_NORMAL)
    if args.single_camera_BEV:
        cv2.namedWindow('Stitched Video', cv2.WINDOW_AUTOSIZE)
    cv2.waitKey(1)

    ###################################################### Main Loop##########################
    while True:
        # Starting the timer
        start_time = time.perf_counter()
        # to display the approximate lane curve
        canvas_1 = np.zeros((h_img, w_img, 3), dtype=np.uint8)

        t_init = time.time()
        t_diff = t_init - t_prev
        fps = 1/t_diff
        t_prev = t_init
        #Recorded change in time
        dt = t_diff

        # Updating iteration index and tracked states to the lateral controller
        lat_controller.k = k
        lat_controller.x_est = x_est

        ########################BEV main loop##########################                   
        # Creating a black canvas (image) of desired size to visualize hough lines
        height, width = h_img, w_img 
        black_image = np.zeros((height, width, 3), dtype=np.uint8)

        d_c = desired_color
        q = 'w_max' #'w_max', using max function for stitching otherwise uses masking
        
        #Camera sensor image from simulation
        camera_image_f = camera_data['front_image']
        camera_image_l = camera_data['left_image']
        camera_image_rt = camera_data['right_image']
        camera_image_rr = camera_data['rear_image']
        #segmentation map camera sensor image
        if args.segmentation_BEV:
            camera_f_seg = camera_data['seg_image_f']
            camera_l_seg = camera_data['seg_image_l']
            camera_r_seg = camera_data['seg_image_r']
            camera_rr_seg = camera_data['seg_image_rr']


        bev_f = cv2.warpPerspective(camera_image_f, H_Front , (w_img, h_img))
        bev_l = cv2.warpPerspective(camera_image_l, H_Left, (w_img, h_img))
        bev_rt = cv2.warpPerspective(camera_image_rt, H_Right, (w_img, h_img))
        bev_rr = cv2.warpPerspective(camera_image_rr, H_Rear, (w_img, h_img))   
        
        if args.segmentation_BEV:
            dc_f_seg = desired_color_fn(camera_f_seg, d_c)      
            bev_seg_f = cv2.warpPerspective(dc_f_seg, H_Front , (w_img, h_img))
        
            dc_l_seg = desired_color_fn(camera_l_seg, d_c)
            bev_seg_l = cv2.warpPerspective(dc_l_seg, H_Left , (w_img, h_img))
            
            dc_r_seg = desired_color_fn(camera_r_seg, d_c)
            bev_seg_r = cv2.warpPerspective(dc_r_seg, H_Right , (w_img, h_img))
                
            dc_rr_seg = desired_color_fn(camera_rr_seg, d_c)
            bev_seg_rr = cv2.warpPerspective(dc_rr_seg, H_Rear , (w_img, h_img))
        
        
        bev = to_generate_stitched_image(bev_f, bev_rt,bev_l, bev_rr, q)
        bev_am1, bev_am2, bev = create_masked_BEV_image(bev_m_1,bev_m_2, bev, q)

        if args.segmentation_BEV:
            bev_s = to_generate_stitched_image(bev_seg_f, bev_seg_r,bev_seg_l, bev_seg_rr, q)
            #To use the mask to create masked segmented BEV image for removing noise at stitched edges
            bev_am1_s, bev_am2_s, bev_s = create_masked_BEV_image(bev_m_1,bev_m_2, bev_s, q)

            bev_s_wm_ac = seg_bev_colors(camera_f_seg,camera_r_seg,camera_l_seg,camera_rr_seg, H_Front, H_Right,H_Left,H_Rear,d_c, q, False)
                    
            #Implementing algorithm to refine the canny edge image at the common region or remove noise
            #First two arguments could be bev_am2, bev_am1 to use without segmentation otherwise bev_am2_s, bev_am1_s
            warped = with_masking_stitched_edge_eroded_canny(bev_am2_s, bev_am1_s, eroded_img, eroded_img_1)
        else:
            warped = with_masking_stitched_edge_eroded_canny(bev_am2, bev_am1, eroded_img, eroded_img_1)

        # Example usage
        car_i = cv2.imread('car_i.png', cv2.IMREAD_UNCHANGED)  # Ensure alpha channel is read

        x_w, y_h = int(2 + w_img/2), int(-18 + h_img/2)  # Position to place the car
        bev = place_car_on_bev(bev, car_i, x_w, y_h)
        
        if args.segmentation_BEV:
            #Argument could be:
            #bev_s_wm_ac : for all color segmented BEV
            #bev_s: for desired color(dc), segmented BEV            
            if args.segmentation_BEV_modes == 'c':
                warped = without_masking_eroded_canny(bev_s)
            else:
                warped = without_masking_eroded_canny(bev_s_wm_ac)

            
        else:
            #bev : for BEV camera
            warped = without_masking_eroded_canny(bev)

        #Slicing a region around the car to include only essential edges
        warped[155:285, 280:355].fill(0)
        #Final refined canny image
        canny_image = warped

        #Hough line transformation
        lines = cv2.HoughLinesP(canny_image, 2, np.pi/180, 100, np.array([]), minLineLength=38, maxLineGap=13)

        #initializing center lanes
        m_theta_2 = np.array([])
        if lines is not None:
            draw_detected_line_on_image(black_image, lines, k)
            #To save bev image if uncommented    
            #cv2.imwrite("image_f/imag/{}.png".format(k), black_image)   
            print('k:', k)
            me_a, x_1, x_2, lan_dist, exp, lines_1, lines_2, lines_3, m_theta_2, canvas_1, n_lanes = _slope_intercept_revised(lines, [int(60), int(h_img-60)], k, lan_dist, exp, args.dbscan_cluster, canvas_1, n_lanes)
            if exp == False and args.dbscan_cluster and len(lan_dist) >0:
                try:
                    zm = utils_c.get_z(lan_dist, k, pixel_to_meter)
                except Exception as e:
                    exp = True
                    raise ValueError("Error: {}".format(str(e)))
        else:
            exp = True
            me_a, x_1, x_2 =  None, None, None
            t = 'No lines detected'
        
        # function for vehicle relative yaw calculation
        def heading_display(me_a):
            if me_a > 90:
                y_d_h = -np.deg2rad(me_a - 90)
                s = ''
                r = y_d_h
            elif me_a < 90:
                y_d_h = +np.deg2rad(90 - me_a)
                s = ''
                r = y_d_h
            else:
                #print('TRUE')
                y_d_h = np.deg2rad(0)
                s = ''
                r = 0
            return r, s, y_d_h
        
        if me_a is not None:
            if len(me_a) > 0:
                r, s, y_d_h = heading_display(me_a[0])
            else:
                y_d_h = 0
                r = 0
                s = ''
        else:
            y_d_h = 0
            r = 0
            s = ''
    
        #Panoramic view from single camera using relative states
        if args.single_camera_BEV == 'r_s':
            if k >= 30:
                if st_h == m_st:
                    st_h = 0
                    base_image = bev_f
                    if len(car_positions_1['car_pos']) == 0:
                        car_positions_1['car_pos'].append((vehicle.get_location().x, vehicle.get_location().y, math.radians(vehicle.get_transform().rotation.yaw)))
                        car_positions_1['k'].append(k)    
                if k % 4 == 0:
                    #cv2.imwrite("EKF/1/imag/{}.png".format(k), bev_f)
                    car_positions_1['car_pos'].append((vehicle.get_location().x, vehicle.get_location().y, math.radians(vehicle.get_transform().rotation.yaw)))
                    car_positions_1['k'].append(k)
                    # Get the transformation between the current base image and the next image
                    base_pos = car_positions_1['car_pos'][len(car_positions_1['car_pos']) - 2]
                    next_pos = car_positions_1['car_pos'][len(car_positions_1['car_pos']) - 1]
                    dx, distance, rotation = get_relative_transformation(base_pos, next_pos)

                    # Stitch the images
                    stitched_image = stitch_images(base_image, bev_f, distance, dx, rotation)

                    # Set the stitched image as the new base image
                    base_image = stitched_image
                    # Make a deep copy of the base image for visualization
                    base_image_1 = copy.deepcopy(base_image)
                    car_i_r = rotate_image_i(car_i, -rotation)
                    base_image_1  = place_car_on_bev(base_image_1 , car_i_r, x_w, y_h)
                                    
                    st_h = st_h + 1
                    # Display the stitched image in real-time
                    cv2.imshow('Stitched Video', base_image_1[:480, :640])
      
        if args.single_camera_BEV == 'fm_orb':
            if k >= 30:
                if st_h == 6:
                    st_h = 0
                    BaseImage = bev_f
                    BaseImageHash = stitch_1.dhash(BaseImage)
                    
                if k % 4 == 0:
                    try:
                        new_image = bev_f
                        new_image_hash = stitch_1.dhash(new_image)

                        if BaseImage is None or new_image_hash != BaseImageHash:
                            try:
                                if BaseImage is None or BaseImage.shape[0] <= 700:
                                    StitchedImage = stitch_1.stitch_images(BaseImage, new_image)
                                else:
                                    BaseImage = bev_f
                            except Exception as e:
                                BaseImage = bev_f

                            BaseImage = StitchedImage.copy()
                            BaseImageHash = new_image_hash
                            st_h += 1

                            display_image = cv2.resize(BaseImage, (640, 480))
                            cv2.imshow('Stitched Video', display_image)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                    except Exception as e:
                        raise ValueError("Error: {}".format(str(e)))
        
        #Panoramic view from single camera using feature matching
        if args.single_camera_BEV == 'fm_sift':
            if k >= 30:
                if st_h == 6 :
                    st_h = 0
                    BaseImage, _, _ = stitch.project_onto_cylinder(bev_f)
                if k % 4 == 0:
                    try:
                        if BaseImage.shape[0] <= 700:
                            StitchedImage = stitch.stitch_images(bev_f, BaseImage)
                        else:
                            BaseImage, _, _ = stitch.project_onto_cylinder(bev_f)
                    except Exception as e:
                        BaseImage, _, _ = stitch.project_onto_cylinder(bev_f)
                        #raise ValueError("Error: {}".format(str(e)))
                    
                    BaseImage = StitchedImage.copy()
                    st_h = st_h + 1
                    # Display the stitched image in real-time
                    cv2.imshow('Stitched Video', BaseImage[:480, :640])

        if x_1 is not None and x_2[0] is not None:
            cv2.line(bev, tuple(x_1), tuple(x_2[0]), (0, 0, 255), 4)

        t = ''
        rounded_r = np.round(r, 2)
        text_ = 'Angle FR: '+str(rounded_r) +' rad, ' + s
        cv2.putText(bev, t, (80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(bev, 'Relative Yaw:', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(bev, text_, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        # Filtering measurement input heading position used in visualization and driving using only BEV measurements
        if args.dbscan_cluster:
            if len(m_theta_2)>0:
                dl_m, dr_m = np.median(zm[:, 0])*1/pixel_to_meter, np.median(zm[:, 1])*1/pixel_to_meter
                ct_e = find_cte_rl(dl_m , dr_m, ct_e, W, pixel_to_meter, args.lane)
            else:
                dl_m, dr_m = 0,0
                ct_e = 0
        else:
            if len(m_theta_2)>0:
                dl_m, dr_m = process_m_theta(m_theta_2)
                ct_e = find_cte_rl(dl_m , dr_m, ct_e, W, pixel_to_meter, args.lane)
            else:
                dl_m, dr_m = 0,0
                ct_e = 0

        ############# for visualization#######
        p22 = np.array([int(w_img/2), int(h_img/2)])
        p23 = np.array([int(w_img/2)+ int(dl_m * np.cos(y_d_h)), int(h_img/2 + dl_m * np.sin(y_d_h))])
        p32 = np.array([int(w_img/2), int(h_img/2)])
        p33 = np.array([int(w_img/2)+ int(dr_m* np.cos(y_d_h)), int(h_img/2 + dr_m * np.sin(y_d_h))])

        #Draws left line for positioning
        cv2.line(bev, tuple(p22), tuple(p23), (255,0,0), 4)
        #Draws right line for positioning
        cv2.line(bev, tuple(p32), tuple(p33), (255,0,0), 4)

        #Calculates line in meter
        text_l = str(round(dl_m, 2))
        t_l = 'LLE: ' + text_l + ' px'
        # Add text near the left line
        cv2.putText(bev, 'Position:', (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(bev, t_l, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(bev, text_l, (int(w_img/2)+ int(dl_m * np.cos(y_d_h) - 60), int(h_img/2 + dl_m * np.sin(y_d_h))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Add text near the right line
        text_r = str(round(dr_m, 2))
        t_r = 'RLE: ' + text_r + ' px'
        # Add text near the left line
        cv2.putText(bev, t_r, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(bev, text_r, (int(w_img/2)+ int(dr_m* np.cos(y_d_h)), int(h_img/2 + dr_m * np.sin(y_d_h))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv2.LINE_AA)

        # Visualizing the back heading information
        if  me_a is not None:
            if len(me_a) > 1:
                r, s, _ = heading_display(me_a[1])
            else:
                r, s, x_1, x_2 = 0, "", None, None
                t = 'No lines detected'
        else:
            #print("No lines detected in the image. Adjusting parameters...")
            r, s, x_1, x_2 = 0, "", None, None
            t = 'No lines detected'

        text_ = 'Angle RR: '+str(round(r, 2)) +' rad, ' + s
        

        cv2.putText(bev, text_, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(bev, text_1, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        if x_1 is not None and x_2[1] is not None:
            cv2.line(bev, tuple(x_1), tuple(x_2[1]), (0, 0, 255), 4)

        # measurent data if only bev measurement is used to contol the vehicle
        if args.dbscan_cluster and  exp == False:
            x_est_bev = np.array([np.median(zm[:, 2]), ct_e])
        else:
            x_est_bev = np.array([y_d_h, ct_e])

        lat_controller.x_est_bev = x_est_bev

        ###############################################################
        # This sends reference path i.e., waypoint coordinated to the lateral controller
        control_signal, ct = lat_controller.control_sig_lat(next_waypoint[0])

        if control_signal is not None:
            vehicle.apply_control(control_signal)

        # for visualization    
        c_st.append(control_signal.steer) 
        cte.append(ct)

        #increment in iteration index
        k += 1

        ########################################################### Control input to EKF ############
        v = get_speed(vehicle) + np.random.normal(loc=0, scale=v_std)
        th_dot = (normalize_angle(math.radians(vehicle.get_transform().rotation.yaw) - prev_th)) /dt + np.random.normal(loc=0, scale=tdot_std)
        prev_th = math.radians(vehicle.get_transform().rotation.yaw)
        s_dot = (normalize_angle(normalize_angle(control_signal.steer) - prev_str))/dt + np.random.normal(loc=0, scale=sdot_std)
        prev_str = normalize_angle((control_signal.steer))

        #############################################################################
        if args.driving == 'f':
            #################filter prediction#####################
            # Updating the state with odometry readings ( also wraps the angles to [-pi,pi])
            th =  normalize_angle(th_prev[k-1])
            st = normalize_angle(st_prev[k-1])
            phi = normalize_angle(phi_prev[k-1])


            #Computes Motion model jacobian with respect to noise
            L_m = np.array([[np.cos(th + np.arctan(l_r * np.tan(st) / L)), 0, 0], 
                        [np.sin(th + np.arctan(l_r * np.tan(st) / L)), 0, 0], 
                        [np.tan(st) / (L * np.sqrt(1 + l_r ** 2 * np.tan(st) ** 2 / L ** 2)), 0, 0], 
                        [0, 1, 0], 
                        [np.sin(phi), 0, 0], 
                        [0, 0, -1]])
            U_k = np.mat([v, s_dot, th_dot])
            x_check +=  dt*(np.dot(L_m,U_k.T)).T

            x_check[0,2] = normalize_angle(x_check[0,2])

            #Motion model jacobian with respect to last state
            F_km = np.array([[1, 0, -dt * v * np.sin(th + np.arctan(l_r * np.tan(st) / L)), 
                        -dt * l_r * v * (np.tan(st) ** 2 + 1) * np.sin(th + np.arctan(l_r * np.tan(st) / L)) / (L * (1 + l_r ** 2 * np.tan(st) ** 2 / L ** 2)), 
                        0, 0], 
                        [0, 1, dt * v * np.cos(th + np.arctan(l_r * np.tan(st) / L)), 
                        dt * l_r * v * (np.tan(st) ** 2 + 1) * np.cos(th + np.arctan(l_r * np.tan(st) / L)) / (L * (1 + l_r ** 2 * np.tan(st) ** 2 / L ** 2)), 
                        0, 0], 
                        [0, 0, 1, dt * v * (np.tan(st) ** 2 + 1) / (L * np.sqrt(1 + l_r ** 2 * np.tan(st) ** 2 / L ** 2)) - dt * l_r ** 2 * v * (2 * np.tan(st) ** 2 + 2) * np.tan(st) ** 2 / (2 * L ** 3 * (1 + l_r ** 2 * np.tan(st) ** 2 / L ** 2) ** (3 / 2)), 
                        0, 0], 
                        [0, 0, 0, 1, 0, 0], 
                        [0, 0, 0, 0, 1, dt * v * np.cos(phi)], 
                        [0, 0, 0, 0, 0, 1]])
            
            # Computes Motion model jacobian with respect to noise
            L_km =dt * L_m

            # Propagates uncertainty  
            P_check = np.dot(np.dot(F_km, P_check), F_km.T) + np.dot(np.dot(L_km, Q_km), L_km.T)

            # Updates state estimate using available measurements
            if exp == False and args.dbscan_cluster:
                #Only considerng 5 meansuremnts per frame for faster fps
                if len(zm) >= 4:
                    no_data = 4
                else:
                    no_data = len(zm)
                for i in range(len(zm[:no_data])):
                    x_check, P_check = measurement_update(vehicle, zm[i, 1] + np.random.normal(loc=0, scale=lr_std), zm[i, 0] + np.random.normal(loc=0, scale=ll_std), zm[i, 2]+ np.random.normal(loc=0, scale=yaw_std), P_check, x_check, cov_y , W * pixel_to_meter, args.lane)
            else:
                x_check, P_check = measurement_update(vehicle, dr_m*pixel_to_meter + np.random.normal(loc=0, scale=lr_std), dl_m*pixel_to_meter + np.random.normal(loc=0, scale=ll_std), y_d_h + np.random.normal(loc=0, scale=yaw_std), P_check, x_check, cov_y , W * pixel_to_meter, args.lane)    
            # Setting final state predictions for timestep
            x_est[k, :6] = x_check.flatten()
            P_est[k] = P_check

        cv2.imshow("Canny", canny_image)
        if args.segmentation_BEV:
            if args.segmentation_BEV_modes == 'c':
                cv2.imshow('seg_bev', bev_s)
            else:
                cv2.imshow('seg_bev', bev_s_wm_ac)
            
        cv2.imshow('stitched', bev)
        cv2.imshow('line_bev', black_image)
        cv2.imshow('Lane_approx', canvas_1)

        # End the timer
        end_time = time.perf_counter()
        # Calculate the elapsed time
        execution_time_1 = end_time - start_time
        iteration_1 += 1
        fps = 1/execution_time_1
        text_1 = 'FPS: '+str(round(fps, 2))
        #print(f"Lateral Iteration {iteration_1} execution time: {execution_time_1} seconds")
        if cv2.waitKey(1) == ord('q'):
            break

        if k >= iteration_limit:

            states = x_est
            covariance_matrix = P_est
            #n_lanes_plot(n_lanes)

            # Closing OpenCV window when finished
            if args.segmentation_BEV:
                camera_seg_f.stop()
                camera_seg_r.stop()
                camera_seg_l.stop()
                camera_seg_rr.stop()
            camera_front.stop()
            camera_left.stop()
            camera_right.stop()
            camera_rear.stop()
            if args.segmentation_BEV:
                #destroying the actors associated with the sensors
                camera_seg_f.destroy()
                camera_seg_r.destroy()
                camera_seg_l.destroy()
                camera_seg_rr.destroy()
            camera_front.destroy()
            camera_left.destroy()
            camera_right.destroy()
            camera_rear.destroy()
            cv2.destroyAllWindows()         
            break

    with result_lock:
        cte = cte
        c_st = c_st 
        states = x_est
        covariance_matrix = P_est
        ey_a = np.array(lat_controller.ey_a)
        


def main():
    actor_list = []

    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(5.0)
        world = client.get_world()

        settings = world.get_settings()

        settings.fixed_delta_seconds = dt
        world.apply_settings(settings)

        # Get the map's spawn points
        map = world.get_map()

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('*mini*')
        spawnpoint = carla.Transform(carla.Location(x=x_i, y=y_i,z=z_i), carla.Rotation(pitch=p_i, yaw=yaw_i, roll=r_i))
        vehicle = world.try_spawn_actor(vehicle_bp[0], spawnpoint)

        # Adjusts spectator's view to focus on the spawned vehicle's location
        spectator = world.get_spectator()
        spawnpoint.location.z = spawnpoint.location.z + 1 
        spectator.set_transform(spawnpoint)

        actor_list.append(vehicle)

        x_init = spawnpoint.location.x
        y_init = spawnpoint.location.y
        th_init = math.radians(spawnpoint.rotation.yaw)


        x_est = np.zeros([max_iteration, 6])  # estimated states, x, y, and theta
        P_est = np.zeros([max_iteration, 6, 6])  # state covariance matrices

        x_est[0] = np.array([x_init, y_init, th_init , 0 , 0, 0]) # initial state
        P_est[0] = np.diag([0.01, 0.01, 0.01, 0.5, 0.5 , 0.5]) # initial state covariance

        # vechile control signal object
        control = carla.VehicleControl()

        #Initializing lateral controller object
        args_longitudinal={'K_P': k_p, 'K_D': k_d, 'K_I': k_i, 'dt': dt}
        long_controller = PIDLongitudinalControl(vehicle, x_est, k, control, **args_longitudinal)
        #### 5. Main Filter Loop #######################################################################
        filter_control_thread = threading.Thread(target=filter_control_loop, args=(P_est, x_est, dt, world, vehicle, lan_dist, control))
        filter_control_thread.start()
        # Starting the longitudinal control loop in a separate thread
        long_control_thread = threading.Thread(target=long_control_loop, args = (long_controller, world))
        long_control_thread.start()

        # Waiting for the control loop to finish
        long_control_thread.join()
        filter_control_thread.join()

        # Processing the obtained results
        with result_lock:
            ###############################Plotting results#########################

            def plot_velocity_profile(velocity_profile, tim):
                plt.plot(tim, velocity_profile, label='Velocity Profile', color='b')
                
                # Customize the plot
                plt.title('Velocity Profile Along Waypoint Path')
                plt.xlabel('Waypoint Index')
                plt.ylabel('Velocity (m/s)')
                plt.legend()
                plt.grid(True)


            def plot_car_trajectory(waypoint_positions, car_positions):

                plt.title('Car Waypoints and Trajectory')
                plt.xlabel('X-axis')
                plt.ylabel('Y-axis')

                # To extract x and y coordinates of the waypoints
                waypoint_x = [pos[0] for pos in waypoint_positions]
                waypoint_y = [pos[1] for pos in waypoint_positions]

                plt.plot(waypoint_x, waypoint_y, 'b-x', label='Waypoints')

                # To extract x and y coordinates from the car positions tuples
                car_x = [pos[0] for pos in car_positions]
                car_y = [pos[1] for pos in car_positions]
                plt.plot(car_x, car_y, 'r-*', label='Car Trajectory') 
                plt.legend()
                plt.grid(True)


            def cov_ellipse(state_matrix, covariance_matrix, lst_k):
                plt.figure(figsize=(10, 6))

                # Ensure covariance matrices are positive definite
                covariance_matrix = np.matmul(covariance_matrix, covariance_matrix.transpose(0, 2, 1))

                # Increased scaling factor for visualization
                scaling_factor = 200  # Adjust as needed

                # Plot ellipses for each state
                for k in lst_k:
                    state = state_matrix[k]
                    cov_matrix = covariance_matrix[k]
                    
                    # Compute eigenvalues and eigenvectors of the covariance matrix
                    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                    
                    # Sort eigenvalues and eigenvectors in descending order
                    idx = np.argsort(eigenvalues)[::-1]
                    eigenvalues = eigenvalues[idx] * scaling_factor
                    eigenvectors = eigenvectors[:, idx]

                    # Determine major and minor axes
                    major_axis = 2 * np.sqrt(2. * eigenvalues[0])
                    minor_axis = 2 * np.sqrt(2. * eigenvalues[1])
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

                    # Plot the state point
                    plt.scatter(state[0], state[1], color='blue', label='State' if k == 0 else None)
                    
                    # Plot the covariance ellipse
                    ellipse = Ellipse(xy=state[:2], width=major_axis, height=minor_axis, angle=angle, edgecolor='r', fc='None', lw=2)
                    plt.gca().add_patch(ellipse)

                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title('Covariance Ellipses for Multiple States')
                plt.axis('equal')
                plt.xlim(min(state_matrix[:, 0])-1, max(state_matrix[:, 0])+1)  # Set limits close to the data range
                plt.ylim(min(state_matrix[:, 1])-1, max(state_matrix[:, 1])+1)
                plt.legend()
                plt.show()



            rise_time, overshoot, undershoot, settling_time, steady_state_error = calculate_metrics(np.array(tim), np.array(speed), v_profile)
            plt.figure(figsize=(8, 6))
            plt.plot(tim, speed, label='System Response')
            plot_velocity_profile(v_profile, tim)
            #plt.axhline(20, color='r', linestyle='--', label='Target Speed')
            plt.title('Longitudinal PID Controller Response')
            plt.xlabel('Time (s)')
            plt.ylabel('Speed(m/s)')
            plt.legend()
            plt.grid(True)
            # Annotate the plot with metrics
            plt.annotate(f'Rise Time: {rise_time:.3f} s', xy=(0.6, 0.5), xycoords='axes fraction', fontsize=10)
            #plt.annotate(f'Overshoot: {overshoot:.2f}%', xy=(0.6, 0.3), xycoords='axes fraction', fontsize=10)
            #plt.annotate(f'undershoot: {undershoot:.2f}%', xy=(0.6, 0.4), xycoords='axes fraction', fontsize=10)
            #plt.annotate(f'Settling Time: {settling_time:.3f} seconds', xy=(0.6, 0.2), xycoords='axes fraction', fontsize=10)
            plt.annotate(f'Steady-State Error: {steady_state_error:.3f}', xy=(0.6, 0.6), xycoords='axes fraction', fontsize=10)

            plt.figure(figsize=(10, 6))
            plot_velocity_profile(v_profile, tim)

            diffs = np.diff(np.array(waypoint_positions), axis=0)
            distances = np.sqrt((diffs ** 2).sum(axis=1))
            summed_dist = np.concatenate(([0], np.cumsum(distances)))
            if args.driving == 'f':
                # Calculate distances
                recorded_cte_values = [x_est[k, 4] for k in lst_k]
                recorded_cte_actual = [ey_a[k, 0] for k in lst_k]

                residue = np.array(recorded_cte_values) - np.array(recorded_cte_actual)
                std_residue = np.sqrt([P_est[k, 4, 4] for k in lst_k])

                # Plotting the CTE
                plt.figure(figsize=(10, 6))
                plt.plot(summed_dist, recorded_cte_values, label='CTE Estimated', marker='o', linestyle='-')
                plt.plot(summed_dist, recorded_cte_actual, label='CTE Actual', marker='x', linestyle='-')
                plt.title('Cross Track Error along Track (m)')
                plt.xlabel('Path Distance, (m)')
                plt.ylabel('CTE (m)')
                plt.legend()
                plt.grid(True)
                plt.show()

                # Plotting the Residue
                plt.figure(figsize=(10, 6))
                plt.plot(summed_dist, residue, label='Residue', marker='o', linestyle='-')
                plt.plot(summed_dist, 3 * std_residue, color='r', linestyle='--', label='3 * STD')
                plt.plot(summed_dist, -3 * std_residue, color='r', linestyle='--')
                plt.fill_between(summed_dist, -3 * std_residue, 3 * std_residue, color='r', alpha=0.1)
                plt.title('Residue between Estimated and Actual CTE')
                plt.xlabel('Path Distance, (m)')
                plt.ylabel('Residue (m)')
                plt.legend()
                plt.grid(True)
                plt.show()

                recorded_yd_values = [states[k, 5] for k in lst_k]
                recorded_yd_actual = [ey_a[k, 1] for k in lst_k]

                # Plotting the relative yaw
                plt.figure(figsize=(10, 6))
                plt.plot(summed_dist, recorded_yd_values, label='Relative yaw Estimated', marker='o', linestyle='-')
                plt.plot(summed_dist, recorded_yd_actual, label='Relative yaw Actual', marker='x', linestyle='-')
                plt.title('Relative Yaw along Track')
                plt.xlabel('Path Distance, (m)')
                plt.ylabel('Relative Yaw (rad)')
                plt.legend()
                plt.grid(True)
                plt.show()
                # Plotting the Residue for Relative Yaw

                residue_yd = np.array(recorded_yd_values) - np.array(recorded_yd_actual)
                std_residue = np.sqrt([P_est[k, 5, 5] for k in lst_k])
                plt.figure(figsize=(10, 6))
                plt.plot(summed_dist, residue_yd, label='Residue', marker='o', linestyle='-')
                plt.plot(summed_dist, 3 * std_residue, color='r', linestyle='--', label='3 * STD')
                plt.plot(summed_dist, -3 * std_residue, color='r', linestyle='--')
                plt.fill_between(summed_dist, -3 * std_residue, 3 * std_residue, color='r', alpha=0.1)
                plt.title('Residue between Estimated and Actual Relative Yaw')
                plt.xlabel('Path Distance (m)')
                plt.ylabel('Residue (rad)')
                plt.legend()
                plt.grid(True)  # Adding grid for better readability
                plt.show()

            

            print('x_est', states.shape)
            print('cov_mat', covariance_matrix.shape)

            #covariance ellipse plot
            cov_ellipse(states[:iteration_limit+1,:], covariance_matrix[:iteration_limit+1, :], lst_k)

            plt.figure(figsize=(10, 6))
            # Create an empty plot
            plt.subplot(1, 2, 1)

            recorded_st_values = [c_st[k] for k in lst_k]
            recorded_t_values = [c_t[k] for k in lst_k]

            # Plotting the steering control output
            plt.plot(tim, recorded_st_values, label='Steering Control')
            plt.title('Steering Control Intput')
            plt.xlabel('Time (s)')
            plt.ylabel('Steering Intput')
            plt.legend()

            plt.subplot(1, 2, 2)
            # Plotting the throttle control output
            plt.plot(tim, recorded_t_values, label='Throttle Control', color='orange')
            plt.title('Throttle Control Intput')
            plt.xlabel('Time (s)')
            plt.ylabel('Throttle Intput')
            plt.legend()

            # Assuming states is your state matrix
            # Extract x, y coordinates from states
            x_values = [row[0] for row in states]
            y_values = [row[1] for row in states]
            orientations_rad = [row[2] for row in states]
            # Extracting x, y, and orientations values at the recorded points
            recorded_x_values = [x_values[k] for k in lst_k]
            recorded_y_values = [y_values[k] for k in lst_k]
            recorded_orientations_rad = [orientations_rad[k] for k in lst_k]

            plt.figure(figsize=(10, 6))
            # Convert orientations from radians to degrees
            recorded_orientations_deg = np.degrees(recorded_orientations_rad)
            # Plot for Vehicle Positions and Orientations
            plt.scatter(recorded_x_values, recorded_y_values, c=recorded_orientations_deg, cmap='hsv', label='Vehicle Positions and Orientations')

            plt.figure(figsize=(8, 6))
            if args.driving =='f':
                plt.plot(recorded_x_values, recorded_y_values, 'g-o', label='Estimated Positions')
            plot_car_trajectory(waypoint_positions, car_positions)
            plt.xlabel('X-axis, (m)')
            plt.ylabel('Y-axis, (m)')
            plt.title('Vehicle trajectory')
            plt.legend()
            
            if args.driving == 'f':
                residue = np.array(recorded_x_values) - np.array(car_positions)[:, 0]
                std_residue = np.sqrt([P_est[k, 0, 0] for k in lst_k])


                plt.figure(figsize=(10, 6))
                plt.plot(summed_dist, residue_yd, label='Residue', marker='o', linestyle='-')
                plt.plot(summed_dist, 3 * std_residue, color='r', linestyle='--', label='3 * STD')
                plt.plot(summed_dist, -3 * std_residue, color='r', linestyle='--')
                plt.fill_between(summed_dist, -3 * std_residue, 3 * std_residue, color='r', alpha=0.1)
                plt.title('Residue between Estimated and Actual x-position')
                plt.xlabel('Path Distance (m)')
                plt.ylabel('Residue (m)')
                plt.legend()
                plt.grid(True)  # Adding grid for better readability
                plt.show()

                residue = np.array(recorded_y_values) - np.array(car_positions)[:, 1]
                std_residue = np.sqrt([P_est[k, 1, 1] for k in lst_k])

                
                plt.figure(figsize=(10, 6))
                plt.plot(summed_dist, residue_yd, label='Residue', marker='o', linestyle='-')
                plt.plot(summed_dist, 3 * std_residue, color='r', linestyle='--', label='3 * STD')
                plt.plot(summed_dist, -3 * std_residue, color='r', linestyle='--')
                plt.fill_between(summed_dist, -3 * std_residue, 3 * std_residue, color='r', alpha=0.1)
                plt.title('Residue between Estimated and Actual y-position')
                plt.xlabel('Path Distance (m)')
                plt.ylabel('Residue (m)')
                plt.legend()
                plt.grid(True)  # Adding grid for better readability
                plt.show()

                residue = np.array(recorded_orientations_rad) - np.array(car_positions)[:, 2]
                std_residue = np.sqrt([P_est[k, 2, 2] for k in lst_k])

                
                plt.figure(figsize=(10, 6))
                plt.plot(summed_dist, residue_yd, label='Residue', marker='o', linestyle='-')
                plt.plot(summed_dist, 3 * std_residue, color='r', linestyle='--', label='3 * STD')
                plt.plot(summed_dist, -3 * std_residue, color='r', linestyle='--')
                plt.fill_between(summed_dist, -3 * std_residue, 3 * std_residue, color='r', alpha=0.1)
                plt.title('Residue between Estimated and Actual yaw')
                plt.xlabel('Path Distance (m)')
                plt.ylabel('Residue (rad)')
                plt.legend()
                plt.grid(True)  # Adding grid for better readability
                plt.show()

                

            plt.figure(figsize=(8, 6))
            cars_yaw = [pos[2] for pos in car_positions]
            plt.plot(tim, cars_yaw, label='Actual Yaw(rad)')
            plt.plot(tim, recorded_orientations_rad, label='Estimated Yaw(rad)')

            plt.xlabel('Time (s)')
            plt.ylabel('Orientation in rad/s')
            plt.title('Vehicle Orientations')
            plt.legend()


            plt.tight_layout()
            plt.show()


    finally:
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])

if __name__ == '__main__':
    main()