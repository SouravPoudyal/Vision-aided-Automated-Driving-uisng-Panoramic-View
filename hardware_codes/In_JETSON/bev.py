
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

import cv2
import numpy as np
import time
import math
import utils_c
import Bezier_curve as be

    
def odd_Out(m_theta):
    if len(m_theta)>0:
        sorted_indices = np.argsort(m_theta[:, 2])
        m_theta = m_theta[sorted_indices]
        if len(m_theta)%2 == 0:
            m_theta = m_theta[:-1, :]
    else:
        m_theta = np.empty((0, 4))

    return m_theta 

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
#Calculating the LLE and RLE distance to car
def process_m_theta(m_theta_2, W):
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
            m_1 = odd_Out(m_1)
            dl_m = np.median(m_1[:, 3])
        else:
            dl_m = 0

        # Step 2: Filter rows where the 4th column is positive
        m_2 = m_theta_2[m_theta_2[:, 3] > 0]
        if m_2.size > 0:
            m_2 = odd_Out(m_2)
            dr_m = np.median(m_2[:, 3])
        else:
            dr_m = 0
    else:
        dl_m, dr_m = 0, 0
    return dl_m, dr_m
    
def find_cte_rl(dl_m , dr_m, ct_e, W, pix_t_meter, c):
    print('dl_m,d_rm', dl_m, dr_m)
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
        
def distance_between_lines(line1):
    """
    Calculates the shortest distance between two lines, represented by two points each.
    """
    x1, y1, x2, y2 = line1[0]
    center = 345 

    distance_l = 0
    distance_r = 0

    if x1 < 345 and x2 < 345:
        distance_l = (abs((x1 - center)) + abs((x2 - center)))/2       
        
    else:
        distance_r = (abs((x1 - center)) + abs((x2 - center)))/2


    return distance_l, distance_r
    
def region_of_interest_side(image):

    square = np.array([[
    (0, 180), (0,250), (640,250),
    (640, 180),]], np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, square, 255)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image
    

def correction(img):

	# Define the four coordinates of the white patch
	polygon_coords = np.array([[0, 0], [80, 0], [286, 188], [0, 188]])
	
	# Create a white image
	height, width = 480, 640
	white_image = 255 * np.ones((height, width, 3), dtype=np.uint8)  # 3 channels


	# Create a black mask on the white image
	mask = np.zeros_like(white_image, dtype=np.uint8)  # Use dtype=np.uint8 for 3 channels
	cv2.fillPoly(mask, [polygon_coords], (255, 255, 255))  # Specify white color in 3 channels

	# Invert the mask
	inverted_mask = cv2.bitwise_not(mask)
	
	img = cv2.bitwise_and(img, inverted_mask)
	
	return img

def mask_1():
    '''
    H_RT = np.array([[-3.81647795e-02, -2.73450001e+00, 1.85501649e+02],
    [-5.67335091e-01, -1.82407848e+00, 3.90218129e+02],
    [-1.72331485e-04, -8.23002718e-03, 1.00000000e+00]])'''
    
    H_RT = np.array([[-2.52557944e-01, -2.65273980e+00,  1.77195733e+02],
               [-7.14508775e-01, -1.83875605e+00,  4.04946716e+02],
               [-7.06433900e-04, -8.19688320e-03,  1.00000000e+00]])

    H_LT = np.array([[ 2.46195210e-01, -4.84896316e+00,  6.26248247e+02],
               [ 1.17222387e+00, -3.28596661e+00, -9.34254002e+01],
               [ 7.86184689e-04, -1.41654144e-02,  1.00000000e+00]])

    H_FR = np.array([[-4.75158459e-01, -2.44217376e+00,  4.68541695e+02],
               [-1.71757698e-02, -1.68425324e+00,  3.37000263e+02],
               [-1.07462179e-04, -7.52344437e-03,  1.00000000e+00]])

    H_RR = np.array([[ 7.20959071e-01, -3.91261868e+00,  8.84899153e+01],
               [-3.36934035e-02, -3.01027967e+00,  5.28725330e+01],
               [-6.98076839e-05, -1.17649448e-02,  1.00000000e+00]])
    

    img_m = np.ones((480, 640, 3), dtype=np.uint8)
    img_m = 255 * img_m

    img_m_f = cv2.warpPerspective(img_m, H_FR, (640, 480))
    img_m_f[200:,:].fill(0)
    img_m_f = correction(img_m_f)

    img_m_r = cv2.warpPerspective(img_m, H_RT, (640, 480))
    img_m_r[:,:200].fill(0)

    img_m_b = cv2.warpPerspective(img_m, H_RR, (640, 480))
    img_m_b[:200,:].fill(0)
    img_m_l = cv2.warpPerspective(img_m, H_LT, (640, 480))
    img_m_l[:,400:].fill(0)

    img_fb = cv2.addWeighted(img_m_f, 1, img_m_b, 1, 0)
    img_rl = cv2.addWeighted(img_m_r, 1, img_m_l, 1, 0)

    img_bev_m1 = cv2.bitwise_and(img_fb, img_rl)
    img_bev_m2 = cv2.bitwise_xor(img_fb, img_rl)

    return img_bev_m1, img_bev_m2

		  
def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
def undistortion(frame, map1, map2):
    undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def warp(undistorted_image, homography):
    warp1 = cv2.warpPerspective(undistorted_image, homography, (640, 480))
    return warp1
    

def draw_the_lines(img,lines): 
  imge=np.copy(img)     
  blank_image=np.zeros((imge.shape[0],imge.shape[1],3),\
                                                   dtype=np.uint8)
  for line in lines:  
    for x1,y1,x2,y2 in line:
      cv2.line(blank_image,(x1,y1),(x2,y2),(0,0,255),thickness=1)
      imge = cv2.addWeighted(imge,0.8,blank_image,1,0.0) 
  return imge


def mask():
    image_FR = np.zeros((480, 640, 3), dtype = "uint8")
    image_FR.fill(255)
    

    image_RR = np.zeros((480, 640, 3), dtype = "uint8")
    image_RR.fill(255)
    
    H_FR = np.array([
        [-4.96178351e-01, -2.82467889e+00,  4.99210739e+02],
        [1.33453231e-02, -1.99918029e+00,  3.68485622e+02],
        [2.52709652e-05, -8.51362910e-03,  1.00000000e+00]
    ])

    H_RR = np.array([
        [4.89195258e-01, -3.05400803e+00, 1.35349199e+02],
        [-1.13066123e-01, -2.46231783e+00, 1.23621684e+02],
        [-2.79264811e-04, -9.19904399e-03, 1.00000000e+00]
    ])

 
 
    warp1 = cv2.warpPerspective(image_FR, H_FR, (640, 480))
    warp1[200:,:].fill(0)
    warp1 = cv2.GaussianBlur(warp1, (0, 0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

    warp4 = cv2.warpPerspective(image_RR, H_RR, (640, 480))
    warp4[:200,:].fill(0)
    warp4 = cv2.GaussianBlur(warp4, (0, 0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
    #warp4[:600,:].fill(0)
    
    return cv2.add(warp1,warp4)

def color_balance(image):
    b, g, r = cv2.split(image)
    B = np.mean(b)
    G = np.mean(g)
    R = np.mean(r)
    K = (R + G + B) / 3
    Kb = K / B
    Kg = K / G
    Kr = K / R
    cv2.addWeighted(b, Kb, 0, 0, 0, b)
    cv2.addWeighted(g, Kg, 0, 0, 0, g)
    cv2.addWeighted(r, Kr, 0, 0, 0, r)
    return cv2.merge([b,g,r])
    
def luminance_balance(images):
    [front,back,left,right] = [cv2.cvtColor(image,cv2.COLOR_BGR2HSV) 
                               for image in images]
    hf, sf, vf = cv2.split(front)
    hb, sb, vb = cv2.split(back)
    hl, sl, vl = cv2.split(left)
    hr, sr, vr = cv2.split(right)
    V_f = np.mean(vf)
    V_b = np.mean(vb)
    V_l = np.mean(vl)
    V_r = np.mean(vr)
    V_mean = (V_f + V_b + V_l +V_r) / 4
    vf = cv2.add(vf,(V_mean - V_f))
    vb = cv2.add(vb,(V_mean - V_b))
    vl = cv2.add(vl,(V_mean - V_l))
    vr = cv2.add(vr,(V_mean - V_r))
    front = cv2.merge([hf,sf,vf])
    back = cv2.merge([hb,sb,vb])
    left = cv2.merge([hl,sl,vl])
    right = cv2.merge([hr,sr,vr])
    images = [front,back,left,right]
    images = [cv2.cvtColor(image,cv2.COLOR_HSV2BGR) for image in images]
    return images
   
def stiching(image1,image2,image3,image4, common):
    
    dst= cv2.add(image2, image4)
    dst = cv2.add(dst, image3)
    outImage = cv2.add(dst, image1)
    
    
    return outImage
    
def region_of_interest_front(image):

    square = np.array([[
    (0, 0), (0,180), (640,180),
    (640, 0),]], np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, square, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
    
def region_of_interest_back(image):

    square = np.array([[
    (0, 310), (0,480), (640,480),
    (640, 310),]], np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, square, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def region_of_interest_side(image):

    square = np.array([[
    (0, 130), (0,290), (640,290),
    (640, 130),]], np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, square, 255)
    masked_image = cv2.bitwise_and(image, mask)

    return masked_imag
   
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
    a, b, c, d = y - int(car_height/2), y + int(car_height/2), x - int(car_width/2), x + int(car_width/2)
    
    # Ensuring the car image fits within the BEV image dimensions
    if y + car_height > bev.shape[0] or x + car_width > bev.shape[1]:
       raise ValueError("The car image exceeds the dimensions of the BEV image at the specified location.")
    
    # Define the region of cccinterest (ROI) on the BEV image where the car will be placed
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
    

    
#This function is similar to simulation code _slope_intercept_revised
def _slope_intercept_visualize(lines, y21_lst, a, final_image_st, canvas_1, c, W, c_f, w_img, h_img, pixel_to_meter, ub_f, ekf, exp, args_cluster_result):

    lines = lines.reshape(lines.shape[0], lines.shape[2])
    a_abs = []
    
    x11 = int(w_img/2) + 15
    y11 = int(h_img/2)
    ct_e = 0
    zm = 0
    lan_dist = {}
    m_theta_with_distance = np.array([])
    

    #img coordinate to world coordinate
    x_st = lines[:,0]
    y_st = h_img - lines[:,1]
    x_en = lines[:,2]
    y_en = h_img - lines[:,3]
    
    #mid poits of line in wc
    x_mid = (x_st + x_en)/2
    y_mid = (y_st + y_en)/2
    
    angle = np.rad2deg(np.arctan2((y_en - y_st),(x_en - x_st)))
    a_abs = np.zeros_like(angle)
    mask_a = (angle[:] <= 0)
    mask_b = (angle[:] > 0)
    a_abs[mask_a] = 180 + angle[mask_a]
    a_abs[mask_b] = angle[mask_b]
    
    # Stack them together
    m_theta = np.stack((x_mid, y_mid, a_abs), axis=1)
    # Convert m_theta into an nx4 matrix
    n = m_theta.shape[0]
    m_theta_with_distance = np.zeros((n, 4))
    point = np.array([int(w_img)/2, int(h_img)/2])
    
    ref_points = m_theta[:, :2]
    angles = m_theta[:, 2]
    
    # Calculate distances
    distances = utils_c.perpendicular_cross_distance_v(point, ref_points, angles)
    # Combine results into the output array
    m_theta_with_distance[:, :3] = m_theta
    m_theta_with_distance[:, 3] = distances
    
    #seperating front, center and back, lane lines from BEV
    def line_separation(m_theta_data, lines_data):
        indices_1 = np.where((210 <= m_theta_data[:, 1]) & (m_theta_data[:, 1] <= 480))[0]
        indices_2 = np.where((210 <= m_theta_data[:, 1]) & (m_theta_data[:, 1] <= 310))[0]
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
    f_m_theta, f_lines = m_theta_1, lines_1

    if ub_f == 'f':    
        if len(f_m_theta) >= 2:
            label_dict, max_sublist, canvas_1 = utils_c.cluster_line(f_lines, f_m_theta, c, canvas_1, W, args_cluster_result)
            if len(max_sublist)>0:
                lan_dist = utils_c.raw_data(label_dict, max_sublist, W)
            else:
                exp == True

    else:
        lan_dist = {}
    
    #####################EKF###########
    if exp == False and ub_f == 'f' and len(lan_dist) >0:
        try:
            zm = utils_c.get_z(lan_dist, c, pixel_to_meter)
            ekf.ekf_step(zm, c)
            
        except Exception as e:
            exp = True
            raise ValueError("Error: {}".format(str(e)))
            
    #######################

    # Making sure that the set of lines are odd, for median calculation
    f_m_theta = odd_Out(f_m_theta)
    m_theta_3 = odd_Out(m_theta_3)

    if ub_f == 'b':
        m_pve_c, m_nve_c, canvas_1 = be.bezier_curve_max_min_curvature_2(f_m_theta[f_m_theta[:, 3]<0], canvas_1, W)
        if m_pve_c is not None:
            c0_m = m_pve_c
        else:
            c0_m = 0

    a_abs_lst = []
    if f_m_theta.size > 0:
        a_abs_lst.append(f_m_theta[:, 2])
    if m_theta_3.size > 0:
        a_abs_lst.append(m_theta_3[:, 2])

    filtered_abs_angs = [angles[(angles >= 44) & (angles <= 136)] for angles in a_abs_lst]

    med_ang_lst = [np.median(angles) if len(angles) > 0 else np.nan for angles in filtered_abs_angs]
    #coordinates to visualize heading on the image
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
    '''
    print('Median Angles:', med_ang_lst)
    print('Coordinates:', x01_lst)
    print('Lines Group 1:', lines_1)
    print('Lines Group 2:', lines_2)
    print('Lines Group 3:', lines_3)'''

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
            print('TRUE')
            y_d_h = np.deg2rad(0)
            s = ''
            r = 0
        return r, s, y_d_h
    
    if  med_ang_lst is not None:
        r, s, y_d_h = heading_display( med_ang_lst[0])
    else:
        y_d_h = 0
        r = 0
        s = ''
    if x0 is not None and x01_lst is not None:
        cv2.line(final_image_st, tuple(x0), tuple(x01_lst[0]), (0, 0, 255), 4)

    t = ''
    rounded_r = np.round(r, 2)
    text_ = 'Angle FR: '+str(rounded_r) +' rad, ' + s
    cv2.putText(final_image_st, t, (80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(final_image_st, 'Relative Yaw:', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(final_image_st, text_, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    
    # Filtering measurement input heading position used in visualization and driving using only BEV measurements
    if len(m_theta_2)>0:
        dl_m, dr_m = process_m_theta(m_theta_2, W)
        ct_e = find_cte_rl(dl_m , dr_m, ct_e, W, pixel_to_meter, c_f)
    else:
        dl_m, dr_m = 0,0
        ct_e = 0

    if np.isnan(dl_m):
        dl_m = 0
    if np.isnan(dr_m):
        dr_m = 0
        
    #print("cross track error", ct_e)
    
    
    p22 = np.array([int(w_img/2 + 15), int(h_img/2)])
    p23 = np.array([int(w_img/2 + 15)+int(dl_m*np.cos(y_d_h)), int(h_img/2 + dl_m * np.sin(y_d_h))])
    p32 = np.array([int(w_img/2 + 15), int(h_img/2)])
    p33 = np.array([int(w_img/2 + 15)+int(dr_m*np.cos(y_d_h)), int(h_img/2 + dr_m * np.sin(y_d_h))])

    #Draws left line for positioning
    cv2.line(final_image_st, tuple(p22), tuple(p23), (255,0,0), 3)
    #Draws right line for positioning
    cv2.line(final_image_st, tuple(p32), tuple(p33), (255,0,0), 3)

    #Calculates line in meter
    text_l = str(round(dl_m, 2))
    t_l = 'LLE: ' + text_l + ' px'
    # Add text near the left line
    cv2.putText(final_image_st, 'Position:', (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(final_image_st, t_l, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(final_image_st, text_l, (300+int(dl_m), 238), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    #Draws left line for positioning
    cv2.line(final_image_st, tuple(p22), tuple(p23), (255,0,0), 3)
    #Draws right line for positioning
    cv2.line(final_image_st, tuple(p32), tuple(p33), (255,0,0), 3)

    # Add text near the right line
    text_r = str(round(dr_m, 2))
    t_r = 'RLE: ' + text_r + ' px'
    # Add text near the left line
    cv2.putText(final_image_st, t_r, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(final_image_st, text_r, (int(w_img/2)+int(dr_m), 238), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv2.LINE_AA)

    #Visualizing the back heading information
    if med_ang_lst is not None:
        if len(med_ang_lst)>1:
            r, s, _ = heading_display(med_ang_lst[1])
        else:
            #print("No lines detected in the image. Adjusting parameters...")
            r, s, x0, x01_lst = 0, "", None, None
            t = 'No lines detected'
    else:
        #print("No lines detected in the image. Adjusting parameters...")
        r, s, x0, x01_lst = 0, "", None, None
        t = 'No lines detected'

    text_ = 'Angle RR: '+str(round(r, 2)) +' rad, ' + s


    cv2.putText(final_image_st, text_, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    if x0 is not None and x01_lst[1] is not None:
        cv2.line(final_image_st, tuple(x0), tuple(x01_lst[1]), (0, 0, 255), 4)
        
    return y_d_h, ct_e, canvas_1, c   

@utils_c.measure_time
def bev_readings(vid_RT, vid_RR, vid_FR, vid_LT, map1, map2, mask_img_white,img_bev_m1,img_bev_m2, car, fps, c, W, c_f, w_img, h_img, pixel_to_meter, ub_f, ekf, exp, steer, args_cluster_result):

    # Capture the video frame
    # by frame
    retFR, frameFR = vid_FR.read()
    retRT, frameRT = vid_RT.read()
    retLT, frameLT = vid_LT.read()
    retRR, frameRR = vid_RR.read()
    '''
    frameFR = color_balance(frameFR)
    frameRT = color_balance(frameRT)
    frameRR = color_balance(frameRR)
    frameLT = color_balance(frameLT)
    '''
    #[frameFR, frameRT, frameRR, frameLT] = luminance_balance([frameFR, frameRT, frameRR, frameLT])
    

# Image with removing Distortion for all cameras

    undistorted_img_FR = undistortion(frameFR,map1, map2)
    
    undistorted_img_RR =  undistortion(frameRR,map1, map2)

    undistorted_img_RT = undistortion(frameRT,map1, map2)
    
    undistorted_img_LT = undistortion(frameLT,map1, map2)

# Homography Matrix

    H_RT = np.array([[-2.52557944e-01, -2.65273980e+00,  1.77195733e+02],
               [-7.14508775e-01, -1.83875605e+00,  4.04946716e+02],
               [-7.06433900e-04, -8.19688320e-03,  1.00000000e+00]])

    H_LT = np.array([[ 2.46195210e-01, -4.84896316e+00,  6.26248247e+02],
               [ 1.17222387e+00, -3.28596661e+00, -9.34254002e+01],
               [ 7.86184689e-04, -1.41654144e-02,  1.00000000e+00]])

    H_FR = np.array([[-4.75158459e-01, -2.44217376e+00,  4.68541695e+02],
               [-1.71757698e-02, -1.68425324e+00,  3.37000263e+02],
               [-1.07462179e-04, -7.52344437e-03,  1.00000000e+00]])

    H_RR = np.array([[ 7.20959071e-01, -3.91261868e+00,  8.84899153e+01],
               [-3.36934035e-02, -3.01027967e+00,  5.28725330e+01],
               [-6.98076839e-05, -1.17649448e-02,  1.00000000e+00]])
        
    
    # Wrap Prespective using the above Homography Matrixfinal_image_st
    warp_FR = warp(undistorted_img_FR, H_FR)
    warp_FR[200:,:].fill(0)
    warp_FR = correction(warp_FR)
    warp_RT = warp(undistorted_img_RT, H_RT)
    warp_RT[:,:200].fill(0)
    #warp_RT[:,:400].fill(0)
    warp_RR = warp(undistorted_img_RR, H_RR)
    warp_RR[:200,:].fill(0)
    #warp_RR[:600,:].fill(0)
    warp_LT = warp(undistorted_img_LT, H_LT)
    #warp_LT[:,400:].fill(0)
    warp_LT[:,400:].fill(0)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    
    final_image_st = stiching(warp_FR,warp_RT,warp_RR,warp_LT, mask_img_white.astype(float)/255)
    
    img_fb = cv2.addWeighted(warp_FR, 1, warp_RR, 1, 0)
    img_rl = cv2.addWeighted(warp_RT, 1, warp_LT, 1, 0)
    
    
    #final_image_st = cv2.addWeighted(final_image_st_1,0.5,final_image_st_2,1,0)
    final_image_st = cv2.max(img_fb, img_rl)
    final_image_st = resize_image(final_image_st, 640, 480)
    
    warped = cv2.cvtColor(final_image_st, cv2.COLOR_BGR2GRAY)
    dst = cv2.Canny(warped, 160, 200, None, 3)
        
    # Creating the ROI for Lane Detection
    canny_image = np.copy(dst)
    canny_image[:, 0:150] = 0
    canny_image[:, 500:640] = 0
    canny_image[193:310, 300:350].fill(0)
        
    # Find the coordinates of the edges
    canny_final = np.nonzero(canny_image)

    # Create a new black image with one channels
    h, w = canny_image.shape
    blank_img = np.zeros((h, w, 3), dtype=np.uint8)
    blank_img[193:310, 300:375].fill(255)
    
    # Set the pixels at the edge coordinates to white
    blank_img[canny_final] = 255
      	
    #for lane curve visualization
    canvas_1 = np.zeros((h, w, 3), dtype=np.uint8)
    
    lane_image_1  = np.copy(final_image_st)
    
    lines_1 = cv2.HoughLinesP(canny_image, 2, np.pi/90, 40, np.array([]), minLineLength=50, maxLineGap=10)
    y_d_h, ct_e, canvas_1, c = _slope_intercept_visualize(lines_1, [50, 460] ,0, blank_img, canvas_1, c, W, c_f, w_img, h_img, pixel_to_meter, ub_f, ekf, exp, args_cluster_result)
    cv2.putText(blank_img, 'Steer Output:' + str(steer) + ' deg.' , (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 55), 1, cv2.LINE_AA)
    
    # converting the fps into integer
    fps = int(fps)
    fps = str(fps)
    
    car_i = cv2.imread('car_h.png', cv2.IMREAD_UNCHANGED)  # Ensure alpha channel is read
    x_w, y_h = int(12 + w_img/2), int(5 + h_img/2)  # Position to place the car
    blank_img = place_car_on_bev(blank_img, car_i, x_w, y_h)
    final_image_st = place_car_on_bev(final_image_st, car_i, x_w, y_h)
    
    cv2.namedWindow('Line Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Line Image', blank_img)   
    cv2.namedWindow('Stiched Image Final', cv2.WINDOW_NORMAL)
    cv2.putText(final_image_st, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('Stiched Image Final', final_image_st)
    cv2.namedWindow('lane curve', cv2.WINDOW_NORMAL)
    cv2.imshow('lane curve', canvas_1)
    
    return y_d_h, ct_e


    


