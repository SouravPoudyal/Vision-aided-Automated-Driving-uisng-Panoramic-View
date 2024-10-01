
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

#importing packages
import serial
from collections import Counter
import numpy as np
import time
import bev
import cv2
import Bezier_curve as be
import EKF as e
import time
import argparse


# specifies serial port to communicate with arudino for sending commands
ser = serial.Serial('/dev/ttyACM0', 9600)

parser = argparse.ArgumentParser(
description='BEV-AD',
formatter_class=argparse.ArgumentDefaultsHelpFormatter)

def str2bool(v):
    return v.lower() in ('true', 'l', 'yes')
    
# Left lane driving ('l') or Right lane Driving('r) configuration
parser.add_argument(
    '--lane',
    '-t',
    type=str,
    default='l',
    choices=['l', 'r'],
    help='Choose between left lane and right lane driving'
)
parser.add_argument(
    '--cluster_result',
    '-c_r',
    type=str2bool,
    default=False,
    choices=[True, False],
    help='save cluster results to /imags'
)

args = parser.parse_args()

#initializing iteration index and time variables
k = 0
t_1 = 0
t_2 = time.time()
diff = 0

prev_is = ""
input_string = ""

prev_frame_time = 0
new_frame_time = 0
fps = 0

#Frame Siye
h_img = 480
w_img = 640

#Pixel to meter on BEV
pixel_to_meter = 0.68/102

#Width of the road
W = 204

if args.lane == 'l':
    #Driving scenerio
    c_f = 'l'
else:
    c_f = 'r'

#activate Filter 
ub_f = 'f'

#true if error in measurement data
exp = False

#Max iteration
max_iteration = 6000
#EKF initialization
dt = 0
L = 0.67
l_r = 0.385
x_init = 0
y_init = 0
th_init = 0

#steering initialization
steer = 0

x_est = np.zeros([max_iteration+1, 6])  # estimated states, x, y, and theta
P_est = np.zeros([max_iteration+1, 6, 6])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init , 0 , 0, 0]) # initial state
P_est[0] = np.diag([0.01, 0.01, 0.01, 0.5, 0.5 , 0.5]) # initial state covariance


### define a video capture object
# Define a video capture object and setup cameras
cameras = {
    "FR": cv2.VideoCapture("v4l2src device=/dev/video4 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"),
    "RT": cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"),
    "LT": cv2.VideoCapture("v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"),
    "RR": cv2.VideoCapture("v4l2src device=/dev/video6 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink")
}

#Calibration matrices
K=np.array([[230.37912994378186, 0.0, 326.381228901319], [0.0, 230.61814452975904, 236.9152648878691], [0.0, 0.0, 1.0]])
D=np.array([[-0.007844798956048664], [-0.01887864867083691], [0.019987919856503687], [-0.006890329594431897]])

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (640,480),cv2.CV_16SC2)
mask_img_white = bev.mask()


img_bev_m1,img_bev_m2 = bev.mask_1()
car = cv2.imread('./car.png')
    
def read_serial():
    global k, cameras, map1, map2, mask_img_white,img_bev_m1,img_bev_m2, car, fps, new_frame_time, prev_frame_time,diff,t_1,t_2,prev_is, c, W, c_f, w_img, h_img, pixel_to_meter, ub_f, exp, x_est, P_est, steer
    #Assumed speed of UGV
    v = 0.37 # in m/s
    #EKF object declaration
    ekf = e.EKF(max_iteration, dt, x_est, P_est, L, l_r, W, pixel_to_meter, v, c_f)	
    while True:
        
        #This function gets relative yaw and CTE from the BEV, similar to that in simulation
        y_d_h, ct_e = bev.bev_readings(cameras['RT'], cameras['RR'], cameras['FR'], cameras['LT'], map1, map2, mask_img_white,img_bev_m1,img_bev_m2, car, fps, k, W, c_f, w_img, h_img, pixel_to_meter, ub_f, ekf, exp, steer, args.cluster_result)
        print("yaw_d, ct_err", y_d_h, ct_e)
        print('ekf',ekf.x_est[k, 5])
        print('bv',y_d_h)
        steer_output = stanley_controller(1, y_d_h, ct_e, v)
        steer = round(steer_map(steer_output), 2)
        steer_output = int(steer_output)
        
        #For loop time calculation 
        new_frame_time = time.time()
        t_diff = (new_frame_time-prev_frame_time)
        ekf.dt = t_diff
        fps = 1/t_diff
        prev_frame_time = new_frame_time
        
        #Sending steering cammand to arudino
        input_string = "on" + str(steer_output) + '\n'
                                                
        if diff > 0.9 and prev_is != input_string:    
            ser.write(input_string.encode())
            prev_is = input_string
            t_1 = time.time()
            diff = 0
        else:
            t_2 = time.time()
            diff = t_2 -t_1
             
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        k = k + 1
        #print("Frame Rate per second is :", fps)  
        if k > max_iteration:
            # After the loop release the cap object
            vid_FR.release()
            vid_RR.release()
            vid_RT.release()
            vid_LT.release()
            # Destroy all the windows
            cv2.destroyAllWindows()
            break
            
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

#Maps for getting steering command in VF(Vehicle Frame)
def steer_map(steer_output):
        
    if steer_output < 90:
        steer_map = -(90 - steer_output)
    else:
        steer_map = (steer_output - 90)
             
    return steer_map

#Function to compute Lateral controller command (uses Stanley controller law) 
@measure_time        
def stanley_controller(K_s, y_dh, c_te, v):
    cte_a = 0
    yaw_diff_heading = 0
    steer_map = 0
    try:
        c_te = float(c_te)
        y_dh = float(y_dh)
    except ValueError:
        raise ValueError("Input values must be convertible to float.")

    #Stanley control law
    steer_output = np.rad2deg(np.arctan2((K_s*c_te), v)) + np.rad2deg(y_dh)
    steer_output = 90 + steer_output

    ###Alternative way of calculation with priority to yaw correction or CTE correction
    cte_a = 90 + np.rad2deg(np.arctan2((K_s*c_te), v))
    yaw_diff_heading = 90 + np.rad2deg(y_dh)
        
    if y_dh < 0 and yaw_diff_heading < 80:
        steer_output_1 =  yaw_diff_heading
        
    elif y_dh > 0 and yaw_diff_heading > 100:
        steer_output_1 = yaw_diff_heading
    else:
        steer_output_1 = cte_a

    print('yaw_diff_heading, CTE_a, steer', yaw_diff_heading, cte_a, steer_output_1)
    
    def saturation_limit(steer_output):

        if steer_output <= 60:
            steer_output = 60
        elif steer_output >= 120:
            steer_output = 120
            
        return steer_output
    
    steer_output = saturation_limit(steer_output) #Stanley law output
    steer_output_1 = saturation_limit(steer_output_1) #Alternative
      
    
    return steer_output_1

def main():
    try:
        read_serial()
    

    finally:
        # Close serial port
        ser.close()
    
if __name__ == '__main__':
    main()

