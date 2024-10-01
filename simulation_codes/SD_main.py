import numpy as np
import matplotlib.pyplot as plt
import carla
import controllers as c
import threading
import time
import glob
import os
import sys
import bev_init as b
import cv2
import utils_clustering
import utils_parameters
import utils_UKF
import EKF as utils_EKF
import math


#Either use Windows or Linux for implementation
def append_carla_path():
    try:
        sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    except IndexError:
        pass


#Spawn vehicle and spectator to the scene
def setup_vehicle(world, initial_state):
    """ Setup vehicle and spectator in the simulation """
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('*mini')
    spawn_point = carla.Transform(carla.Location(x=initial_state['x_i'], y=initial_state['y_i'], z=initial_state['z_i']), carla.Rotation(pitch=initial_state['p_i'], yaw=initial_state['yaw_i'], roll=initial_state['r_i']))
    vehicle = world.try_spawn_actor(vehicle_bp[0], spawn_point)
    spectator = world.get_spectator()
    spawn_point.location.z += 1
    spectator.set_transform(spawn_point)

    return vehicle, spectator

result_lock = threading.Lock()
stop_threads = False #Shared variable

def long_control_loop(long_controller, world, args, controller_params, vel_profile_params):
    global stop_threads, cp, wp_1, wp_2, e_time_long, c0_a, v_pr_p

    #Initialization
    reached_waypoints = 0
    c0_a = 0 # actual Curvature along the path
    v_pr_p = 0 #velocity profile variable

    current_waypoint = world.get_map().get_waypoint(long_controller.vehicle.get_location())
    next_waypoint = current_waypoint.next(controller_params['wp'])

    while not stop_threads:
        # Start the timer
        start_time = time.perf_counter()
        
        long_controller.dt = e_time_long


        current_location = long_controller.vehicle.get_location()
        distance_to_waypoint = current_location.distance(next_waypoint[0].transform.location)
        control = long_controller.control_sig_long(v_pr_p)
        long_controller.vehicle.apply_control(control)

        # Wapoint position at distance 'wp' from the car
        wp_1 = np.array([next_waypoint[0].transform.location.x, next_waypoint[0].transform.location.y])
        wp_2 = np.array([(next_waypoint[0].next(controller_params['wp']))[0].transform.location.x, (next_waypoint[0].next(controller_params['wp']))[0].transform.location.y])
        # Car's Position
        cp = np.array([long_controller.vehicle.get_location().x, long_controller.vehicle.get_location().y, math.radians(long_controller.vehicle.get_transform().rotation.yaw)])
        if distance_to_waypoint <= controller_params['d_th']:  # Setting a threshold distance to consider the waypoint reached
            reached_waypoints += 1
            next_waypoint = next_waypoint[0].next(controller_params['wp'])

            #For computing speed profile as a setpoint speed
            wp_3 = np.array([(next_waypoint[0].next(controller_params['wp']))[0].transform.location.x,(next_waypoint[0].next(controller_params['wp']))[0].transform.location.y])
            v_pr_p, c0_a = c.velocity_profile(wp_1, wp_2, wp_3, vel_profile_params['c_a'], v_pr_p, args.speed, vel_profile_params['curv_t'], vel_profile_params['rt_vp'])
        
        # Delay of 0.02 seconds to match simulator sampling rate
        time.sleep(args.time)
        # End the timer
        end_time = time.perf_counter()

        # Calculate the elapsed time
        with result_lock:      
            e_time_long = end_time - start_time
            #print(f"execution time Longitudinal controller: {e_time_long} seconds")

def  get_speed(vehicle):
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def filter_LatControl_loop(world, vehicle, control, args, car_state, bev_parameters, initial_state, controller_params):
    global stop_threads, next_waypoint, cp, wp_1, wp_2, e_time_long, c0_a, v_pr_p

    #longitudinal controller execution time variable
    e_time_long = 0
    
    #relative yaw and crosstrack measurement values initialization
    y_d_h , ct_e = 0 , 0

    #initializing iteration index
    k = 0
    #Previous orientation and steering initializations
    prev_th = 0
    prev_str = 0

    ## FPS initialization for visualization
    fps = 0
    t = ''
    text_1 = ''

    #execution Time for lateral controller
    ## Timing calculation initialization
    t_init = 0
    t_prev = time.time()
    t_diff = 0


    #Measurement data list
    zm = []
    #Raw data dictionary
    lan_dist = {}
    exp = False

    dt = e_time_long

    if dt != 0:
        dot_w_max = car_state['max_st']/dt
    else:
        dot_w_max = 0

    #BEV initialization
    if args.segmentation_BEV and not args.perception:
        image_h, image_w, camera_front, camera_left, camera_right, camera_rear, camera_seg_f, camera_seg_r, camera_seg_l, camera_seg_rr, H_Front, H_Left, H_Rear, H_Right, bev_m_1, bev_m_2, eroded_img, eroded_img_1 = b.bev_init(bev_parameters['h_img'], bev_parameters['w_img'], world, vehicle, args.max_fn, args.segmentation_BEV, args.perception, args.cluster_result)
    elif args.segmentation_BEV and args.perception:
        image_h, image_w, image_w_p, image_h_p, camera_front, camera_left, camera_right, camera_rear, camera_seg_f, camera_seg_r,  camera_seg_l, camera_seg_rr, camera_depth_f, camera_front_p, seg_camera_front_p, H_Front, H_Left, H_Rear, H_Right, bev_m_1, bev_m_2, eroded_img, eroded_img_1, K = b.bev_init(bev_parameters['h_img'], bev_parameters['w_img'], world, vehicle, args.max_fn, args.segmentation_BEV, args.perception) 
    else:
        image_h, image_w, camera_front, camera_left, camera_right, camera_rear, H_Front, H_Left, H_Rear, H_Right, bev_m_1, bev_m_2, eroded_img, eroded_img_1 = b.bev_init(bev_parameters['h_img'], bev_parameters['w_img'], world, vehicle, args.max_fn, args.segmentation_BEV, args.perception)


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
    if args.segmentation_BEV and args.perception:
        camera_data = {
            'seg_image_f': np.zeros((image_h, image_w, 4)),
            'seg_image_r': np.zeros((image_h, image_w, 4)),
            'seg_image_l': np.zeros((image_h, image_w, 4)),
            'seg_image_rr': np.zeros((image_h, image_w, 4)),  
            'front_image': np.zeros((image_h, image_w, 4)),
            'left_image': np.zeros((image_h, image_w, 4)),
            'right_image': np.zeros((image_h, image_w, 4)),
            'rear_image': np.zeros((image_h, image_w, 4)),
            'depth_image_f': np.zeros((image_h_p, image_w_p, 4)),
            'image_front_p': np.zeros((image_h_p, image_w_p, 4)),
            'seg_image_front_p': np.zeros((image_h_p, image_w_p, 4))
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
        camera_seg_f.listen(lambda image: b.sem_callback(image, camera_data, 'seg_image_f'))
        camera_seg_r.listen(lambda image: b.sem_callback(image, camera_data, 'seg_image_r'))
        camera_seg_l.listen(lambda image: b.sem_callback(image, camera_data, 'seg_image_l'))
        camera_seg_rr.listen(lambda image: b.sem_callback(image, camera_data, 'seg_image_rr'))

    if args.perception:
        camera_depth_f.listen(lambda image: b.depth_camera_callback(image, camera_data, 'depth_image_f'))
        camera_front_p.listen(lambda image: b.camera_callback(image, camera_data, 'image_front_p'))
        seg_camera_front_p.listen(lambda image: b.sem_callback(image, camera_data, 'seg_image_front_p'))

    camera_front.listen(lambda image: b.camera_callback(image, camera_data, 'front_image'))
    camera_left.listen(lambda image: b.camera_callback(image, camera_data, 'left_image'))
    camera_right.listen(lambda image: b.camera_callback(image, camera_data, 'right_image'))
    camera_rear.listen(lambda image: b.camera_callback(image, camera_data, 'rear_image'))

    #Initializing lateral control object
    lat_controller = c.Lateral_purePursuite_stanley_controller(vehicle, control, args.driving, get_speed, car_state['max_st'], controller_params['k_s'], controller_params['k_s1'], car_state['L'], args.kf)
    
    if args.kf == 'ukf':
        #Initializing UKF modules
        bicycle = utils_UKF.Bicycle(initial_state['x_i'], initial_state['y_i'], initial_state['z_i'], np.deg2rad(initial_state['yaw_i']), 0 , car_state['L'] , car_state['l_r'], dot_w_max, dt)  # Instantiate with initial state and parameters
        clothoid_ln = utils_UKF.clothoid_lane(bev_parameters['W'] * bev_parameters['pixel_to_meter'], 0 ,0 ,0 ,0 , dt, args.lane)
        obj = utils_UKF.object(0, 0 ,0, dt)

        fx_lst = [bicycle, clothoid_ln, obj]
        hx_lst = utils_UKF.Hx
        sigma_points = utils_UKF.MerweScaledSigmaPoints(12)

        ukf = utils_UKF.UKF(dim_x = 12, dim_z = 9, fx=fx_lst, hx=hx_lst,
            dt=dt, points=sigma_points, x_mean_fn=utils_UKF.state_mean, 
            z_mean_fn=utils_UKF.z_mean, residual_x=utils_UKF.residual_x, 
            residual_z=utils_UKF.residual_h, Q=[controller_params['Q_a'], controller_params['cov_bi']], R=controller_params['R_a'], c = controller_params['c_nd'])

        ukf.points.sigma_points(ukf.x, ukf.P)
    else:
        ekf =utils_EKF.EKF(dt, car_state['L'], car_state['l_r'], bev_parameters['W'], bev_parameters['pixel_to_meter'], 0, 0, 0, cp,initial_state, c_f='l', dim_x = 6)

    cv2.namedWindow('stitched', cv2.WINDOW_AUTOSIZE)
    if args.segmentation_BEV:
        cv2.namedWindow('seg_bev', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow('line_bev', cv2.WINDOW_AUTOSIZE)
    #cv2.namedWindow("Canny", cv2.WINDOW_NORMAL)
    #cv2.namedWindow("Lane_approx", cv2.WINDOW_NORMAL)
    if args.perception:
        cv2.namedWindow("Front_Depth_camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Front_camera_p", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Front_segmentation_camera_p", cv2.WINDOW_NORMAL)

    cv2.waitKey(1)

    fig, ax = plt.subplots()
    plt.ion()
    while not stop_threads:
        #Recorded change in time and fps
        t_init = time.time()
        t_diff = t_init - t_prev
        fps = 1/t_diff
        text_1 = 'FPS: '+str(round(fps, 2))
        t_prev = t_init
        dt = t_diff

        # to display the approximate lane curve
        canvas_1 = np.zeros((bev_parameters['h_img'], bev_parameters['w_img'], 3), dtype=np.uint8)

        # Updating iteration index and tracked states to the lateral controller

        ########################BEV main loop##########################                   
        # Creating a black canvas (image) of desired size to visualize hough lines
        black_image = np.zeros((bev_parameters['h_img'], bev_parameters['w_img'], 3), dtype=np.uint8)

        #Camera sensor image from simulation
        camera_image_f = camera_data['front_image']
        camera_image_l = camera_data['left_image']
        camera_image_rt = camera_data['right_image']
        camera_image_rr = camera_data['rear_image']

        bev_f = cv2.warpPerspective(camera_image_f, H_Front , (bev_parameters['w_img'], bev_parameters['h_img']))
        bev_l = cv2.warpPerspective(camera_image_l, H_Left, (bev_parameters['w_img'], bev_parameters['h_img']))
        bev_rt = cv2.warpPerspective(camera_image_rt, H_Right, (bev_parameters['w_img'], bev_parameters['h_img']))
        bev_rr = cv2.warpPerspective(camera_image_rr, H_Rear, (bev_parameters['w_img'], bev_parameters['h_img']))
        #segmentation map camera sensor image
        if args.segmentation_BEV:
            camera_f_seg = camera_data['seg_image_f']
            camera_l_seg = camera_data['seg_image_l']
            camera_r_seg = camera_data['seg_image_r']
            camera_rr_seg = camera_data['seg_image_rr']
        
        #depth map camera sensor image
        if args.perception:
            camera_f_depth = camera_data['depth_image_f']
            camera_f_p = camera_data['image_front_p']
            seg_camera_f_p = camera_data['seg_image_front_p']
            dc_f_seg = b.desired_color_fn(seg_camera_f_p, bev_parameters['desired_color_1'])        

        if args.segmentation_BEV_modes == 'c':
            dc_f_seg = b.desired_color_fn(camera_f_seg, bev_parameters['desired_color'])
            bev_seg_f = cv2.warpPerspective(dc_f_seg, H_Front , (bev_parameters['w_img'], bev_parameters['h_img']))
        
            dc_l_seg = b.desired_color_fn(camera_l_seg, bev_parameters['desired_color'])
            bev_seg_l = cv2.warpPerspective(dc_l_seg, H_Left , (bev_parameters['w_img'], bev_parameters['h_img']))
            
            dc_r_seg = b.desired_color_fn(camera_r_seg, bev_parameters['desired_color'])
            bev_seg_r = cv2.warpPerspective(dc_r_seg, H_Right , (bev_parameters['w_img'], bev_parameters['h_img']))
                
            dc_rr_seg = b.desired_color_fn(camera_rr_seg, bev_parameters['desired_color'])
            bev_seg_rr = cv2.warpPerspective(dc_rr_seg, H_Rear , (bev_parameters['w_img'], bev_parameters['h_img']))        
        
        fb, rl = b.to_generate_stitched_image_nm(bev_f, bev_rt, bev_l, bev_rr)
        if args.segmentation_BEV_modes == 'c':
            fb_s, rl_s = b.to_generate_stitched_image_nm(bev_seg_f, bev_seg_r,bev_seg_l, bev_seg_rr)
        if args.max_fn:
            # BEV without masking
            bev = cv2.max(fb, rl)
            if args.segmentation_BEV:
                if args.segmentation_BEV_modes == 'c':
                    bev_s = cv2.max(fb_s, rl_s)
                else:
                    bev_s = b.seg_bev_colors(camera_f_seg, camera_r_seg, camera_l_seg, camera_rr_seg, H_Front, H_Right, H_Left, H_Rear, bev_parameters['w_img'], bev_parameters['h_img'], bev_parameters['desired_color'], args.max_fn, False)
                
                warped = b.without_masking_eroded_canny(bev_s)
            else:
                #bev : for normal camera
                warped = b.without_masking_eroded_canny(bev)
            
        else:
            # BEV that requires masking
            bev = cv2.add(fb, rl)
            bev_am1, bev_am2, bev = b.create_masked_BEV_image(bev_m_1, bev_m_2, bev)

            if args.segmentation_BEV:
                # Segmentation BEV that requires masking
                if args.segmentation_BEV_modes == 'c':
                    bev_s = cv2.add(fb_s, rl_s)
                    #To use the mask to create masked segmented BEV image for removing noise at stitched edges
                    bev_am1_s, bev_am2_s, bev_s = b.create_masked_BEV_image(bev_m_1,bev_m_2, bev_s)
                else:
                    bev_s = b.seg_bev_colors(camera_f_seg, camera_r_seg, camera_l_seg, camera_rr_seg, H_Front, H_Right, H_Left, H_Rear, bev_parameters['w_img'], bev_parameters['h_img'], bev_parameters['desired_color'], args.max_fn, False)
                    bev_am1_s, bev_am2_s, bev_s = b.create_masked_BEV_image(bev_m_1, bev_m_2, bev_s)
            #Implementing algorithm to refine the canny edge image at the common region or remove noise
            #First two arguments could be bev_am2, bev_am1 to use without segmentation otherwise bev_am2_s, bev_am1_s
                warped = b.with_masking_stitched_edge_eroded_canny(bev_am2_s, bev_am1_s, eroded_img, eroded_img_1)
            else:
                warped = b.with_masking_stitched_edge_eroded_canny(bev_am2, bev_am1, eroded_img, eroded_img_1)

        # place car on BEV
        car_i = cv2.imread('images/car_i.png', cv2.IMREAD_UNCHANGED)  # Ensure alpha channel is read
        x_w, y_h = int(2 +bev_parameters['w_img']/2), int(-18 + bev_parameters['h_img']/2)  # Position to place the car
        bev = b.place_car_on_bev(bev, car_i, x_w, y_h)

        #Slicing a region around the car to include only essential edges
        warped[155:285, 280:355].fill(0)

        #Final refined canny image
        canny_image = warped

        #Hough line transformation
        lines = cv2.HoughLinesP(canny_image, 2, np.pi/180, 100, np.array([]), minLineLength=38, maxLineGap=13)

        #initializing center lanes
        m_theta_2 = np.array([])

        if lines is not None:
            b.draw_detected_line_on_image(black_image, lines)
            #cv2.imwrite("image_f/imag/{}.png".format(k), black_image)   
            me_a, x_1, x_2, lan_dist, exp, lines_1, lines_2, lines_3, m_theta_2, canvas_1, c0_m, k = b.slope_intercept_revised(lines, [int(60), int(bev_parameters['h_img']-60)], k, bev_parameters['w_img'], bev_parameters['h_img'], bev_parameters['W'], args.lane, args.dbscan_cluster, canvas_1, args.kdc_cluster, args.driving, args.cluster_result)
            if exp == False and args.dbscan_cluster and len(lan_dist) > 0:
                try:
                    if args.kf == 'ukf':
                        zm = utils_clustering.get_z(lan_dist, k, bev_parameters['pixel_to_meter'], controller_params['wp'], ukf.x, c0_a, args.kf)
                    else:
                        zm = utils_clustering.get_z(lan_dist, k, bev_parameters['pixel_to_meter'], controller_params['wp'], ekf.x_est, c0_a, args.kf)
                except Exception as e:
                    exp = True
                    raise ValueError("Error: {}".format(str(e)))

        else:
            exp = True
            me_a, x_1, x_2 =  None, None, None
            t = 'No lines detected'
        
        # Filtering measurement input heading position used in visualization and driving using only BEV measurements
        if args.dbscan_cluster:
            if len(m_theta_2)>0:
                dl_m, dr_m = np.median(zm[:, 0])*1/bev_parameters['pixel_to_meter'], np.median(zm[:, 1])*1/bev_parameters['pixel_to_meter']
                ct_e = utils_clustering.find_cte_rl(dl_m , dr_m, ct_e, bev_parameters['W'], bev_parameters['pixel_to_meter'], args.lane)
            else:
                dl_m, dr_m = 0,0
                ct_e = 0
        else:
            if len(m_theta_2)>0:
                dl_m, dr_m = utils_clustering.process_m_theta(m_theta_2)
                ct_e = utils_clustering.find_cte_rl(dl_m , dr_m, ct_e, bev_parameters['W'], bev_parameters['pixel_to_meter'], args.lane)
            else:
                dl_m, dr_m = 0,0
                ct_e = 0
        
        bev, y_d_h = b.control_reference_visiualization(me_a, x_1, x_2, t,text_1, bev, dl_m, dr_m, bev_parameters, fps)

        # measurent data if only bev measurement is used to contol the vehicle
        if args.dbscan_cluster and  exp == False:
            x_est_bev = np.array([np.median(zm[:, 2]), ct_e])
        else:
            x_est_bev = np.array([y_d_h, ct_e])
        
        if args.kf == 'ukf':
            control_signal, ct = lat_controller.control_sig_lat(wp_1, wp_2, cp, x_est_bev, ukf.x)
        else:
            control_signal, ct = lat_controller.control_sig_lat(wp_1, wp_2, cp, x_est_bev, ekf.x_est)
        if control_signal is not None:
            vehicle.apply_control(control_signal)
        
        ########################################################### Control input to UKF ############
        v = get_speed(vehicle) + np.random.normal(loc=0, scale=controller_params['v_std'])
        th_dot = (c.normalize_angle(math.radians(vehicle.get_transform().rotation.yaw) - prev_th)) /dt + np.random.normal(loc=0, scale=controller_params['tdot_std'])
        prev_th = math.radians(vehicle.get_transform().rotation.yaw)
        s_dot = (c.normalize_angle(c.normalize_angle(control_signal.steer) - prev_str))/dt + np.random.normal(loc=0, scale=controller_params['sdot_std'])
        prev_str = c.normalize_angle((control_signal.steer))

        #############################################################################################
        if args.driving == 'f':
            if args.kf == 'ukf':
                u = np.array([v, s_dot, th_dot])
                # Perform predict step
                ukf.predict(u)
                if exp == False:
                    z_c = np.empty((zm.shape[0], cp.shape[0]+zm.shape[1]))
                    z_c[:, :cp.shape[0]] = cp
                    z_c[:, cp.shape[0]:] = zm
                    z_c = np.concatenate(z_c)
                    # Perform update step
                    ukf.update(z_c)
                    ukf.points.sigma_points(ukf.x, ukf.P)
                    ukf.dt = dt
            else:
                ekf.v = v 
                ekf.st_dot = s_dot
                ekf.th_dot = th_dot
                ekf.dt = dt
                ekf.cp = cp
                ekf.ekf_step(zm)

        #cv2.imshow("Canny", canny_image)
        if args.segmentation_BEV:
            cv2.imshow('seg_bev', bev_s)            
        cv2.imshow('stitched', bev)
        #cv2.imshow('line_bev', black_image)
        #cv2.imshow('Lane_approx', canvas_1)

        if args.perception:
            cv2.imshow('Front_Depth_camera', camera_f_depth)
            cv2.imshow('Front_camera_p', camera_f_p)
            cv2.imshow('Front_segmentation_camera_p', dc_f_seg)

        with result_lock:
            ax.plot(wp_1[0], wp_1[1], 'go')
            ax.plot(cp[0], cp[1], 'ro')
            if args.kf == 'ukf':
                ax.plot(ukf.x[0], ukf.x[1], 'bo')
            else:
                ax.plot(ekf.x_est[0], ekf.x_est[1], 'bo')
            #ax.plot(t_init, v_pr_p, 'bo')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.draw()
            plt.pause(0.00000001)

        if cv2.waitKey(1) == ord('q'):
            break 

def main():
    global stop_threads

    append_carla_path()
    args = utils_parameters.parse_args()

    initial_state, car_state = utils_parameters.init_simulation_params(args)
    controller_params = utils_parameters.init_controller_params()
    bev_parameters = utils_parameters.bev_parameters()
    vel_profile_parameters = utils_parameters.velocity_profile_parameters()

    actor_list = []
    try:
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        
        # Setting the simulation time-step
        settings = world.get_settings()
        settings.fixed_delta_seconds = args.time
        world.apply_settings(settings)

        vehicle, spectator = setup_vehicle(world, initial_state)
        actor_list.append(vehicle)

        # Vehicle control signal object
        control = carla.VehicleControl()

        # Initializing longitudinal controller object
        long_controller = c.PIDLongitudinalControl(vehicle, control,  get_speed, car_state['max_t'], car_state['max_b'], controller_params['k_p'], controller_params['k_i'], controller_params['k_d'], args.time)

        # Starting the lateral control and UKF loop in a separate thread
        filter_Lat_control_thread = threading.Thread(target=filter_LatControl_loop, args=(world, vehicle, control, args, car_state, bev_parameters, initial_state, controller_params))
        filter_Lat_control_thread.start()

        # Starting the longitudinal control loop in a separate thread
        long_control_thread = threading.Thread(target=long_control_loop, args=(long_controller, world, args, controller_params, vel_profile_parameters))
        long_control_thread.start()

        input("Press Enter to stop the threads...\n")
        stop_threads = True

        long_control_thread.join()
        filter_Lat_control_thread.join()

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        plt.ioff()

if __name__ == '__main__':
    main()
