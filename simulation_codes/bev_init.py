import numpy as np
import carla
import cv2
import utils_clustering
import Bezier_curve as be
import math
import utils_parameters as p

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


def get_relative_transformation(base_pos, target_pos, m_to_pix):
    # Calculate the relative translation
    dx = target_pos[0] - base_pos[0]
    dy = target_pos[1] - base_pos[1]

    print(base_pos, target_pos)

    # Calculate the distance between the two positions
    distance = np.sqrt(dx**2 + dy**2) * m_to_pix

    # Calculate the relative rotation
    dtheta = (target_pos[2] - base_pos[2])

    return dx, distance, dtheta

def mask_2(image_2, distance, rotation, bev_parameters):
    d_1 = bev_parameters['h_img']/2 - (65+50) - distance - bev_parameters['w_img']/2 * np.sin(rotation)
    d_2 = bev_parameters['h_img']/2 - (65+50) - distance + bev_parameters['w_img']/2 * np.sin(rotation)

    
    # Ensure the indices are within valid ranges
    d_1 = int(np.clip(d_1, 0, bev_parameters['h_img']))
    d_2 = int(np.clip(d_2, 0, bev_parameters['h_img']))
    
    # Define the points of the polygon
    pts = np.array([[0, d_1], [0, bev_parameters['h_img']], [bev_parameters['w_img'], bev_parameters['h_img']], [bev_parameters['w_img'], d_2]], dtype=np.int32)
    
    # Create a mask with the same dimensions as the image, initialized to 1 (white)
    mask = np.ones(image_2.shape[:2], dtype=np.uint8)
    
    # Fill the polygon defined by pts with 0 (black) on the mask
    cv2.fillPoly(mask, [pts], 0)
    
    # Apply the mask to the image
    image_2[mask == 0] = 0
    
    return image_2

def stitch_images(image1, image2, distance, dx, rotation, bev_parameters):
    # Apply rotation and translation to image1
    transformed_image1 = transform_image(image1, distance, dx, rotation)

    image2 = mask_2(image2, distance, rotation, bev_parameters)
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

def to_generate_stitched_image_nm(b_f, b_r, b_l, b_rr):

    # Convert all images to the same type  
    b_f = b_f.astype(np.uint8)
    b_r = b_r.astype(np.uint8)
    b_l = b_l.astype(np.uint8)
    b_rr = b_rr.astype(np.uint8)

    # Generates the stitched image
    fb = cv2.add(b_f, b_rr)
    rl = cv2.add(b_r, b_l)

    return fb, rl

def create_masked_BEV_image(bev_m_1,bev_m_2, bev_):

    # Convert all images to the same type   
    bev_m_1 = bev_m_1.astype(np.uint8)
    bev_m_2 = bev_m_2.astype(np.uint8)
    bev_ = bev_.astype(np.uint8)

    # To use the mask to create masked BEV image
    bev_am1 = cv2.bitwise_and(bev_m_1, bev_)
    bev_am2 = cv2.bitwise_and(bev_m_2, bev_)

    # To add the two masked BEV to generate final BEV image
    bev = cv2.addWeighted(bev_am1, 0.5, bev_am2, 1, 1)

    return bev_am1, bev_am2, bev

def desired_color_fn(seg_camera_image, dc):
    # Creates a mask for pixels matching the desired color
    mask = np.all(seg_camera_image==dc, axis=2)

    # Applies the mask to the original image to display only the desired color
    seg_dc = np.zeros_like(seg_camera_image) 
    seg_dc[mask] = seg_camera_image[mask]
    
    return seg_dc

#Segmentation BEV with all colors
def seg_bev_colors(c_f_seg, c_r_seg, c_l_seg, c_rr_seg, H_f, H_r, H_l, H_rr, w_img, h_img, dc, m_x, d_color = False):
        
    bev_seg_f = cv2.warpPerspective(c_f_seg, H_f , (w_img, h_img))
    bev_seg_rr = cv2.warpPerspective(c_rr_seg, H_rr ,(w_img, h_img))
    bev_seg_r = cv2.warpPerspective(c_r_seg, H_r , (w_img, h_img))
    bev_seg_l = cv2.warpPerspective(c_l_seg, H_l , (w_img, h_img))

    bev_seg_f = bev_seg_f.astype(np.uint8)
    bev_seg_rr = bev_seg_rr.astype(np.uint8)
    bev_seg_r = bev_seg_r.astype(np.uint8)
    bev_seg_l = bev_seg_l.astype(np.uint8)

    #To generate the stitched segmented image with mask
    fb_s = cv2.add(bev_seg_f, bev_seg_rr)
    rl_s = cv2.add(bev_seg_r, bev_seg_l)

    if m_x:
        bev_s = cv2.max(fb_s, rl_s)
    else:
        bev_s = cv2.add(fb_s, rl_s)

    if d_color is True:    
        bev_s = desired_color_fn(bev_s, dc)

    return bev_s

def without_masking_eroded_canny(b_warp):
    # Ensure the input image is of type uint8
    b_warp = b_warp.astype(np.uint8)

    # Convert to grayscale
    warped = cv2.cvtColor(b_warp, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    warped = cv2.Canny(warped, 110, 130, None, 3)
    
    return warped

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

# To draw the detected lines on the black image
def draw_detected_line_on_image(image, lines):
    # Use efficient tuple unpacking
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


#Function to get heading information
def slope_intercept_revised(lines, y21_lst, k, w_img, h_img, W, args_lane, args_dbscan_cluster, canvas_1, args_kdc_cluster, args_driving, args_cluster_result):

    #Initializing local variables used in this function
    exp = False
    s_k = 0
    # center point of a car to define 
    # the fixed end of heading line
    x11 = int(w_img/2) 
    y11 = int(h_img/2)
    point = np.array([w_img/2, h_img/2])
    lan_dist = {}
    c0_m = 0

    #list of lines from hough line
    lines = lines.reshape(lines.shape[0], lines.shape[2])

    # image coordinate to PF (units pixels)
    x_st = lines[:, 0]
    y_st = h_img - lines[:, 1]
    x_en = lines[:,2]
    y_en = h_img - lines[:,3]

    # mid poits of line in PF
    x_mid = (x_st + x_en)/2
    y_mid = (y_st + y_en)/2

    angle = np.rad2deg(np.arctan2((y_en - y_st), (x_en - x_st)))

    #This operations avoids unwated infinite slope values in case of vertical lines
    mask_1 = (angle <= 0)
    #mask_2 = (angle > 0)
    angle[mask_1] = 180 + angle[mask_1]
    #angle[mask_2] = angle

    ###
    m_theta_with_distances = np.zeros((lines.shape[0], 4))

    m_theta_with_distances[:, 0] = x_mid
    m_theta_with_distances[:, 1] = y_mid
    m_theta_with_distances[:, 2] = angle

    # Calculates  perpendicular cross distances
    distances = utils_clustering.v_perpendicular_cross_distance(point, m_theta_with_distances[:, :2], m_theta_with_distances[:, 2])
    m_theta_with_distances[:, 3] = distances

    # Seperating front, center and back, lane lines from BEV
    def line_separation(m_theta_data, lines_data):
        indices_1 = np.where((210 <= m_theta_data[:, 1]) & (m_theta_data[:, 1] <= 480))[0]
        indices_2 = np.where((215 <= m_theta_data[:, 1]) & (m_theta_data[:, 1] <= 310))[0]
        indices_3 = np.where((0 <= m_theta_data[:, 1]) & (m_theta_data[:, 1] <= 210))[0]

        lines_1 = lines_data[indices_1]
        lines_2 = lines_data[indices_2]
        lines_3 = lines_data[indices_3]

        m_theta_1 = np.array(m_theta_data[indices_1])
        m_theta_2 = np.array(m_theta_data[indices_2])
        m_theta_3 = np.array(m_theta_data[indices_3])

        return lines_1, lines_2, lines_3, m_theta_1, m_theta_2, m_theta_3

    lines_1, lines_2, lines_3, m_theta_1, m_theta_2, m_theta_3 = line_separation(m_theta_with_distances, lines)
    # Initial filtering the line with the knowledge of the left lane or right lane driving
    f_m_theta, f_lines = utils_clustering.filter_m_theta(m_theta_1, lines_1, W, args_lane)

    if args_dbscan_cluster:    
        if len(f_m_theta) >= 2:
            label_dict, max_sublist, canvas_1, n_l_d, c0_m, k = utils_clustering.cluster_line(f_lines, f_m_theta, k, canvas_1, args_cluster_result)
            if len(max_sublist)>0:
                lan_dist = utils_clustering.raw_data(label_dict, max_sublist)
            else:
                exp == True
    
    if args_kdc_cluster:
        if len(f_m_theta) >= 2:
            #Kernel Density clustering for number of lanes
            s_k = utils_clustering.KDC_lanes(f_m_theta, k)
            #print('Kernel_density_score', s_k)
        else:
            s_k = 0
    m_theta_2, _ = utils_clustering.filter_m_theta(m_theta_2, lines_2, W, args_lane)
    # Making sure that the set of lines are odd, for median calculation
    f_m_theta = utils_clustering.odd_Out(f_m_theta, 'h')
    m_theta_3 = utils_clustering.odd_Out(m_theta_3, 'h')

    if args_driving == 'b' and args_dbscan_cluster == False:
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

    #a_abs_lst = [f_m_theta[:, 2], m_theta_3[:, 2]]
    filtered_abs_angs = [angles[(angles >= 24) & (angles <= 156)] for angles in a_abs_lst]

    med_ang_lst = [np.median(angles) if len(angles) > 0 else np.nan for angles in filtered_abs_angs]
        # Coordinates to visualize heading on the image
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

    return med_ang_lst, x0, x01_lst, lan_dist, exp, list(lines_1), list(lines_2), list(lines_3), m_theta_2, canvas_1, c0_m, k

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

def control_reference_visiualization(me_a, x_1, x_2, t,text_1, bev, dl_m, dr_m, bev_parameters, fps):


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
    
    if x_1 is not None and x_2[0] is not None:
        cv2.line(bev, tuple(x_1), tuple(x_2[0]), (0, 0, 255), 4)
    
    rounded_r = np.round(r, 2)
    text_ = 'Angle FR: '+str(rounded_r) +' rad, ' + s
    cv2.putText(bev, t, (80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(bev, 'Relative Yaw:', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(bev, text_, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    ############# for visualization#######
    p22 = np.array([int(bev_parameters['w_img']/2), int(bev_parameters['h_img']/2)])
    p23 = np.array([int(bev_parameters['w_img']/2)+ int(dl_m * np.cos(y_d_h)), int(bev_parameters['h_img']/2 + dl_m * np.sin(y_d_h))])
    p32 = np.array([int(bev_parameters['w_img']/2), int(bev_parameters['h_img']/2)])
    p33 = np.array([int(bev_parameters['w_img']/2)+ int(dr_m* np.cos(y_d_h)), int(bev_parameters['h_img']/2 + dr_m * np.sin(y_d_h))])

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
    cv2.putText(bev, text_l, (int(bev_parameters['w_img']/2)+ int(dl_m * np.cos(y_d_h) - 60), int(bev_parameters['h_img']/2 + dl_m * np.sin(y_d_h))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Add text near the right line
    text_r = str(round(dr_m, 2))
    t_r = 'RLE: ' + text_r + ' px'
    # Add text near the left line
    cv2.putText(bev, t_r, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(bev, text_r, (int(bev_parameters['w_img']/2)+ int(dr_m* np.cos(y_d_h)), int(bev_parameters['h_img']/2 + dr_m * np.sin(y_d_h))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv2.LINE_AA)


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
    
    return bev, y_d_h

# Callback stores sensor data in a dictionary for use outside the callback
def camera_callback(image, data_dict, camera_name):
    data_dict[camera_name] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

def sem_callback(image, data_dict, camera_name):
    image.convert(carla.ColorConverter.CityScapesPalette)
    data_dict[camera_name] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

# Convert a CARLA raw image to a BGRA numpy array
def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    if not isinstance(image, carla.Image):
        raise ValueError("Argument must be a carla.Image")
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    return array
# Convert depth image to a normalized 2D depth array
def depth_to_array(image):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the depth value of each pixel normalized between [0.0, 1.0].
    """
    array = to_bgra_array(image)
    array = array.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    return normalized_depth*1000 # Convert to meters (assuming max depth is 1000 meters)
    # Convert a CARLA raw image to a RGB numpy array
def to_rgb_array(image):
    """Convert a CARLA raw image to a RGB numpy array."""
    array = to_bgra_array(image)
    # Convert BGRA to RGB.
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array
# Define the callback function for the depth camera
def depth_camera_callback(image, data_dict, camera_name):
    # Convert the depth image to a normalized 2D array
    depth_array = depth_to_array(image)

    # Normalize the depth for visualization (convert to grayscale 0-255)
    depth_visualized = np.clip(depth_array/1000, 0, 1) * 255.0
    depth_visualized = depth_array.astype(np.uint8)

    # Apply the 'jet' colormap to the depth image
    depth_colormap = cv2.applyColorMap(depth_visualized, cv2.COLORMAP_JET)

    # The original depth image (RGB channels as raw depth encoding)
    original_depth_image = to_rgb_array(image)

    # Store the depth_map in the data_dict with the camera_name as key
    data_dict[camera_name] =  depth_colormap


def depth_camera_callback_1(image, data_dict, camera_name):
    # Convert raw data to a numpy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4)).astype(np.float32)
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    # Extract the R, G, B channels from the depth image
    R = array[:, :, 0]
    G = array[:, :, 1]
    B = array[:, :, 2]
    # Decode the depth value according to CARLA documentation
    normalized_depth = (R + G * 256.0 + B * 256.0 * 256.0) / (256.0 * 256.0 * 256.0 - 1)
    depth_in_meters = 1000.0 * normalized_depth  # Convert to meters (assuming max depth is 1000 meters)

    # Normalize the depth for visualization (convert to grayscale 0-255)
    depth_visualized = np.clip(depth_in_meters / 1000.0, 0, 1) * 255.0
    depth_visualized = depth_visualized.astype(np.uint8)

    # Convert the depth data to meters (depth is usually in logarithmic scale)
    depth_image = array[:, :, :3].astype(np.uint8)
    
    # Store the depth_map in the data_dict with the camera_name as key
    data_dict[camera_name] = depth_image

def compute_camera_intrinsics(image_size_x, image_size_y, fov):
    """
    Computes the camera intrinsic matrix for a pinhole camera model.
    
    Parameters:
    - image_size_x: int, horizontal image resolution (width)
    - image_size_y: int, vertical image resolution (height)
    - fov: float, horizontal field of view (in degrees)
    
    Returns:
    - K: 3x3 numpy array, intrinsic camera matrix
    """
    # Compute the focal length in pixels using the field of view
    f_x = image_size_x / (2 * np.tan(np.radians(fov / 2)))
    
    # The aspect ratio for the focal length
    f_y = f_x * (image_size_y / image_size_x)
    
    # Principal point (assuming the camera center is in the middle of the image)
    c_x = image_size_x / 2
    c_y = image_size_y / 2
    
    # The camera intrinsic matrix
    K = np.array([
        [f_x,  0,    c_x],
        [0,    f_y,  c_y],
        [0,    0,    1]
    ])
    
    return K

# Step 1: Create road mask by matching RGBA values for road
def extract_road_mask(segmentation_image, road_color_rgba):
    """
    Extract a binary mask where road pixels are present, based on the road color RGBA.
    
    Parameters:
    - segmentation_image: 3D numpy array (H, W, 4) representing the RGBA segmentation image
    - road_color_rgba: 1D numpy array of the road color in RGBA format
    
    Returns:
    - road_mask: 2D binary mask where road pixels are True
    """
    road_mask = np.all(segmentation_image == road_color_rgba, axis=-1)
    return road_mask


def compute_xy_coordinates_from_depth(depth, K, road_mask):
    """
    Computes the x and y camera frame coordinates from the depth map and intrinsic matrix.

    Parameters:
    - depth: 2D numpy array, depth map with shape (height, width)
    - K: 3x3 numpy array, intrinsic camera matrix

    Returns:
    - x: 2D numpy array, x coordinates in the camera frame
    - y: 2D numpy array, y coordinates in the camera frame
    """
    ### START CODE HERE ###

    # Get the shape of the depth tensor
    sh_depth = np.shape(depth)

    # Grab required parameters from the K matrix
    f = K[0, 0]  # Focal length
    c_u = K[0, 2]  # Principal point (u-coordinate)
    c_v = K[1, 2]  # Principal point (v-coordinate)

    # Generate a grid of coordinates corresponding to the shape of the depth map
    x_coords = np.arange(0, sh_depth[1])  # u coordinates (width)
    y_coords = np.arange(0, sh_depth[0])  # v coordinates (height)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # Filter out road pixels using the road mask
    depth_road_only = depth[road_mask]
    x_grid_road_only = x_grid[road_mask]
    y_grid_road_only = y_grid[road_mask]

    # Compute x, y, z coordinates in the camera frame for road pixels
    x_ground = (x_grid_road_only- c_u) * depth_road_only / f
    y_ground = (y_grid_road_only - c_v) * depth_road_only / f
    z_ground = depth_road_only  # z is the depth value itself

    # Stack the x, y, z coordinates to form the xyz_ground array
    xyz_ground = np.stack((x_ground, y_ground, z_ground), axis=0)

    return xyz_ground, x_ground, y_ground, z_ground

# Save depth data to a .dat file
def save_depth_as_dat(depth, filename):
    """
    Save the depth data as a .dat file.
    
    Parameters:
    - depth: 2D numpy array, depth map in meters
    - filename: str, file name for saving the .dat file
    """
    # Save the depth array as a .dat file in binary format
    np.savetxt(filename, depth)

def bev_init(h_img, w_img, world, vehicle, args_max_fn, args_segmentation_BEV, args_perception):
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
    
    # Creating a black canvas (image) of desired size
    height, width = h_img, w_img 
    black_image = np.zeros((height, width, 3), dtype=np.uint8)
    bev_m_1 = np.zeros([h_img, w_img, 3], dtype=np.uint8)
    bev_m_2 = np.zeros([h_img, w_img, 3], dtype=np.uint8)
    eroded_img = np.zeros([h_img, w_img, 3], dtype=np.uint8)
    eroded_img_1 = np.zeros([h_img, w_img, 3], dtype=np.uint8)
    
    if args_max_fn ==  False:
        #Creating a white image of Size w_imgxh_img
        img_w = np.zeros([h_img, w_img, 4], dtype=np.uint8)
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
        eroded_img_1 = cv2.erode(gray_1, kernel, iterations=1)

    if args_segmentation_BEV:
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

    if args_perception:
        # Creating camera blueprints for front camera
        camera_bp_front_p = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp_front_p.set_attribute('image_size_x', str(1280))
        camera_bp_front_p.set_attribute('image_size_y', str(720))

        camera_bp_front_p.set_attribute('fov', '110')


        depth_camera_bp_f = world.get_blueprint_library().find('sensor.camera.depth')
        depth_camera_bp_f.set_attribute('image_size_x', str(1280))
        depth_camera_bp_f.set_attribute('image_size_y', str(720))

        depth_camera_bp_f.set_attribute('fov', '110')

        seg_camera_bp_front_p = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        seg_camera_bp_front_p.set_attribute('image_size_x', str(1280))
        seg_camera_bp_front_p.set_attribute('image_size_y', str(720))

        seg_camera_bp_front_p.set_attribute('fov', '110')

    # To initialize camera transforms for front, left, right, and rear cameras
    camera_trans_front = carla.Transform(carla.Location(x = 2.2 , y = 0 , z = 1.2), carla.Rotation(pitch=-52.5, yaw=0, roll=0)) # Front camera
    camera_trans_left = carla.Transform(carla.Location(x = 0.83, y = -1, z = 1.59), carla.Rotation(pitch=-46, yaw=-90, roll=0))  #  Left camera
    camera_trans_right = carla.Transform(carla.Location(x = 0.83, y = 1, z = 1.59), carla.Rotation(pitch=-46, yaw=90, roll=0))  # right camera
    camera_trans_rear = carla.Transform(carla.Location(x = -2.2, y = -0.1, z = 1.2), carla.Rotation(pitch=-127.5, yaw=0, roll=0))  # rear camera 

    if args_perception:
        camera_trans_front_p = carla.Transform(carla.Location(x = 2.2 , y = 0 , z = 1.2), carla.Rotation(pitch=-5.5, yaw=0, roll=0)) # perception camera 

    if args_segmentation_BEV:
        camera_seg_f = world.spawn_actor(sem_camera_bp_f, camera_trans_front, attach_to=vehicle)
        camera_seg_r = world.spawn_actor(sem_camera_bp_r, camera_trans_right, attach_to=vehicle)
        camera_seg_l = world.spawn_actor(sem_camera_bp_l, camera_trans_left, attach_to=vehicle)
        camera_seg_rr = world.spawn_actor(sem_camera_bp_rr, camera_trans_rear, attach_to=vehicle)
    
    if args_perception:
        camera_depth_f = world.spawn_actor(depth_camera_bp_f, camera_trans_front_p, attach_to=vehicle)
        camera_front_p = world.spawn_actor(camera_bp_front_p, camera_trans_front_p, attach_to=vehicle)
        seg_camera_front_p = world.spawn_actor(seg_camera_bp_front_p, camera_trans_front_p, attach_to=vehicle)

        image_w_p = camera_bp_front_p.get_attribute("image_size_x").as_int()
        image_h_p = camera_bp_front_p.get_attribute("image_size_y").as_int()
        image_fov_p = camera_bp_front_p.get_attribute("fov").as_float() 

        K = compute_camera_intrinsics(image_w_p, image_h_p, image_fov_p)


    # Creates cameras and attach them to the vehicle
    camera_front = world.spawn_actor(camera_bp_front, camera_trans_front, attach_to=vehicle)
    camera_left = world.spawn_actor(camera_bp_left, camera_trans_left, attach_to=vehicle)
    camera_right = world.spawn_actor(camera_bp_left, camera_trans_right, attach_to=vehicle)
    camera_rear = world.spawn_actor(camera_bp_left, camera_trans_rear, attach_to=vehicle)

    # To get camera dimensions and initialize dictionaries
    image_w = camera_bp_front.get_attribute("image_size_x").as_int()
    image_h = camera_bp_front.get_attribute("image_size_y").as_int()


    '''
    # Callback stores sensor data in a dictionary for use outside the callback
    def camera_callback(image, data_dict, camera_name):
        data_dict[camera_name] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))


    def sem_callback(image, data_dict, camera_name):
        image.convert(carla.ColorConverter.CityScapesPalette)
        data_dict[camera_name] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

    if args_segmentation_BEV:
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
    if args_segmentation_BEV:
        camera_seg_f.listen(lambda image: sem_callback(image, camera_data, 'seg_image_f'))
        camera_seg_r.listen(lambda image: sem_callback(image, camera_data, 'seg_image_r'))
        camera_seg_l.listen(lambda image: sem_callback(image, camera_data, 'seg_image_l'))
        camera_seg_rr.listen(lambda image: sem_callback(image, camera_data, 'seg_image_rr'))

    camera_front.listen(lambda image: camera_callback(image, camera_data, 'front_image'))
    camera_left.listen(lambda image: camera_callback(image, camera_data, 'left_image'))
    camera_right.listen(lambda image: camera_callback(image, camera_data, 'right_image'))
    camera_rear.listen(lambda image: camera_callback(image, camera_data, 'rear_image'))
    '''
    if args_segmentation_BEV and not args_perception:
        return image_h, image_w, camera_front, camera_left, camera_right, camera_rear, camera_seg_f, camera_seg_r,  camera_seg_l, camera_seg_rr, H_Front, H_Left, H_Rear, H_Right, bev_m_1, bev_m_2, eroded_img, eroded_img_1
    elif args_segmentation_BEV and args_perception:
        return image_h, image_w, image_w_p, image_h_p, camera_front, camera_left, camera_right, camera_rear, camera_seg_f, camera_seg_r,  camera_seg_l, camera_seg_rr, camera_depth_f, camera_front_p, seg_camera_front_p, H_Front, H_Left, H_Rear, H_Right, bev_m_1, bev_m_2, eroded_img, eroded_img_1, K
    else:
        return image_h, image_w, camera_front, camera_left, camera_right, camera_rear, H_Front, H_Left, H_Rear, H_Right, bev_m_1, bev_m_2, eroded_img, eroded_img_1