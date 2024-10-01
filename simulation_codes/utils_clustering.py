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

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import argrelmin, argrelmax
import numpy as np
import time
import Bezier_curve as be
import utils_parameters
import cv2

#Lane width
W = utils_parameters.bev_parameters()['W']
#Initializing previous mean orientation of lines detected on the closest lane edge cluster to the car
#Utilized just for visualization
prev_head = 90 #degrees

#KDC algorithm to find number of lanes available for navigation
def kernel_density_estimation(data, bandwidth):
    """Perform kernel density estimation using Gaussian kernel."""
    x = np.linspace(np.min(data), np.max(data), 1000)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data[:, None])
    log_density = kde.score_samples(x[:, None])
    pdf = np.exp(log_density)
    return x, pdf

# For execition time calculation
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        #print(f"Execution time of {func.__name__}: {execution_time} seconds")
        return result
    return wrapper

@measure_time #to measure the execution time of the functio
def KDC_lanes(m_t, k):
    m_t = np.array(m_t, dtype=np.float32)  # Convert to NumPy array

    x, pdf = kernel_density_estimation(m_t[:, 3], bandwidth=3)
    # Find local minima and maxima
    mi, ma = argrelmin(pdf), argrelmax(pdf)

    # Getting the selected elements from x using the mask ma
    s_m = x[ma]
    s_1, s_2 = l_count(s_m, [], 'k')
    if len(s_1) == 0 or len(s_2) == 0:
        s_l = 0 
        print("s_1 or s_2 is empty. Cannot compute np.max on an empty array.")
    else:
        s_l = np.max(np.add(s_1, s_2))

    return s_l*2

# Calculates circular mean of angles
def circular_mean(angles):
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    return np.arctan2(sin_sum, cos_sum)

# Function to compute means of detected lines features
def _mean(data, a):
    p_theta = []
    angles_rad = np.deg2rad(data[:, 2])  # Convert angles to radians
    mean_angle_rad = circular_mean(angles_rad)
    mean_angle_deg = np.rad2deg(mean_angle_rad) % 360
    # mean of orientation of the lines
    p_theta.append(mean_angle_deg) 

    # mean of x-coordinate or y-coordinate of midpoint depending on the degree of turn of the mid-point of the lines
    p_mean = np.mean(data[:, a])
    # mean of relative distance of the lane line edge from the car
    mean_c = np.mean(data[:, 3])
    p_theta.append(p_mean)
    p_theta.append(mean_c)

    return p_theta

# conditional test for cluster to determine cluster label that satisfies
# if the cluster belong to LLE or RLE of the lane pair where the car is drving
def l_count(s_m, sig_mask, c ='d'):
    if c == 'd':
        positive_mask = (s_m[:, None] > 0) & (s_m > 0)
        negative_mask = (s_m[:, None] < 0) & (s_m < 0)
        combined_mask = ~positive_mask & sig_mask & ~negative_mask

        d_matrix = np.zeros((len(s_m), len(s_m)))
        d_matrix_1 = np.zeros((len(s_m), len(s_m)))

        abs_s_m = np.abs(s_m)
        d_matrix_full = abs_s_m[:, None] + abs_s_m
        d_matrix[combined_mask] = d_matrix_full[combined_mask]
    else:
        positive_mask = (s_m[:, None] > 0) & (s_m > 0)
        negative_mask = (s_m[:, None] < 0) & (s_m < 0)
        combined_mask = ~positive_mask & ~negative_mask
        d_matrix = np.zeros((len(s_m), len(s_m)))
        d_matrix_full = np.abs(s_m[:, None]) + abs(s_m)
        d_matrix[combined_mask] = d_matrix_full[combined_mask]

    d_matrix_1 = np.abs(np.abs(s_m[:, None]) - abs(s_m))
    r_matrix = abs(W - d_matrix)
    r_matrix_1 = abs(W - d_matrix_1)

    condition_matrix =(r_matrix <= 25)
    c_vector = condition_matrix.sum(axis=1)

    condition_matrix_1 =(r_matrix_1 <= 25)
    c_vector_1 = condition_matrix_1.sum(axis=1)

    s_1 = c_vector.tolist()
    # Replacing values greater than 1 with 1
    s_1 = [1 if x > 1 else x for x in s_1]
    s_2 = c_vector_1.tolist()
    s_2 = [1 if x > 1 else x for x in s_2]

    return s_1, s_2

@measure_time
def cluster_line(lines, data, k, canvas_1, args_cluster_result):
    global prev_head
    c0_m = 0
    a = 0
    no_of_lanes = 0
    closest_label = -1
    max_sublist = []
    closest_pairs = []
    # weights for features of detected lines (position(x or y), orientation and relative distance from the car in BEV) used for DBSCAN clustering
    weights = np.array([0, 0.5, 1])
    x_and_circular_means = {}

    m_theta = data
    c_mean_mt = _mean(m_theta, a)
    
    # If orientation is within the below defined range choose x-position for clustering otherwise y-position
    if 30 <= c_mean_mt[0] <= 150:
        a = 0
    else:
        a = 1
        
    # Extracting only the first and third elements from the data array
    reduced_data = m_theta[:, [a, 2, 3]]

    # Normalize using Min-Max Scaler
    scaler = MinMaxScaler()
    reduced_data[:, 0] = scaler.fit_transform(reduced_data[:, 0].reshape(-1, 1)).ravel()
    reduced_data[:, 1]  = scaler.fit_transform(reduced_data[:, 1].reshape(-1, 1)).ravel()
    reduced_data[:, 2]  = scaler.fit_transform(reduced_data[:, 2].reshape(-1, 1)).ravel()

    # Apply weights to the normalized features
    weighted_data = reduced_data * weights
    
    # Instantiating and fitting DBSCAN
    dbscan = DBSCAN(eps=0.04, min_samples=2)
    
    #fitting
    clusters = dbscan.fit_predict(weighted_data)
    unique_labels = np.unique(clusters)
    label_dict = {label: [] for label in np.unique(clusters)}
    label_dict[-1] = []

    # assing labels to each lines
    for i, label in enumerate(clusters):
        label_dict[label].append(m_theta[i])

    #counts lines in each labels
    count = {key: len(value) for key, value in label_dict.items()}
    # to compute means of each cluster detected lines features and store in a dictionary
    try:
        x_and_circular_means = {key: _mean(np.array(value), a) for key, value in label_dict.items() if len(value) != 0 and key != -1}
    except Exception as e:
         x_and_circular_means = {}

    means = np.array(list(x_and_circular_means.values()))
    
    if means.size > 0:  # Check if means is not empty
        mean_0_diffs = np.abs(means[:, None, 0] - means[:, 0])
        significant_mask = (mean_0_diffs <= 20)

        s_1, s_2 = l_count(means[:, 2], significant_mask, 'd')
    else:
        s_1, s_2 = [], []

    n_l = np.add(s_1, s_2)
    no_of_lanes = np.max(n_l) if n_l.size > 0 else 0

    #########################Furthere processing of the clusters######
    max_sublist = [index for index, value in enumerate(s_1) if value > 0]
    ang_list =  [x_and_circular_means[l][0] for l in max_sublist]

    if ang_list:
        median_angle = np.median(ang_list)
        mad_angle = [abs(angle - median_angle) for angle in ang_list]

        max_sublist = [max_sublist[index] for index, value in enumerate(mad_angle) if value <= 15]

        # Out of the clusters it determines if two cluster belong to same lane edge as closest pairs
        def find_closest_pairs(max_sublist, x_and_circular_means):
            if len(max_sublist) > 2:
                # Extract the distance values from x_and_circular_means for the given indices in max_sublist
                dist_list = np.array([x_and_circular_means[l][2] for l in max_sublist])

                # Sort the distances and get the sorted indices
                sorted_indices = np.argsort(dist_list)
                sorted_dist_list = dist_list[sorted_indices]

                # Calculate the differences between consecutive elements in the sorted list
                diff = np.diff(sorted_dist_list)

                # Find the minimum difference
                min_diff = np.min(diff)

                # Find pairs with the minimum difference
                closest_pairs_indices = np.where(diff == min_diff)[0]
                
                # Retrieve the corresponding indices from max_sublist
                closest_pairs = [(max_sublist[sorted_indices[i]], max_sublist[sorted_indices[i + 1]]) for i in closest_pairs_indices]

                return closest_pairs
            else:
                return []
        closest_pairs = find_closest_pairs(max_sublist, x_and_circular_means)
    else:
        pass

    # The below conditional test is utilized just for visualization
    # this is executed if the LLE cluster and RLE cluster (their labels are stored in max_sublist) is detected by satisfing the condition 
    # (sum of absolute distance of car from LLE and RLE is approximately equal to road width) for the lane pair where the car is driving.
    # Specifically, it finds the cluster of lane edge lines that is left side to the car(if no lines are detected on LLE then it takes RLE), however it need to be also
    # containing highest number of lines compared to other lane edge lines cluster. Priority of these two conditions is (1, 0.5)

    if len(max_sublist) > 0:
        dist = {key: x_and_circular_means[key][2] for key in max_sublist}
        counts = {key: count[key] for key in max_sublist}

        sorted_dist = dict(sorted(dist.items(), key=lambda item: item[1]))
        sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

        dist_values = np.array(list(sorted_dist.values())).reshape(-1, 1)
        count_values = np.array(list(sorted_counts.values())).reshape(-1, 1)

        scaler = MinMaxScaler()
        scaled_dist = scaler.fit_transform(dist_values).ravel()
        scaled_counts = scaler.fit_transform(count_values).ravel()

        sorted_dist_keys = list(sorted_dist.keys())
        sorted_count_keys = list(sorted_counts.keys())

        scaled_dist_dict = dict(zip(sorted_dist_keys, scaled_dist))
        scaled_count_dict = dict(zip(sorted_count_keys, scaled_counts))

        #Relative yaw of the lane w.r.t car is the closest label
        closest_label = min(scaled_dist_dict.keys(), key=lambda k: (scaled_dist_dict[k] + (1 - scaled_count_dict[k])) / 2)

        prev_head = x_and_circular_means[closest_label][0]

        #bezier curve for curvature estimation for clusters if they belong to same lane edge 
        if len(closest_pairs) > 0:
            data_c_1 = np.vstack(label_dict[closest_pairs[0][0]])
            data_c_2 = np.vstack(label_dict[closest_pairs[0][1]])

            m_pve_c, m_nve_c, canvas_1 = be.bezier_curve_max_min_curvature(data_c_1, data_c_2,canvas_1, W)
            if m_pve_c is not None:
                c0_m = m_pve_c
            else:
                c0_m = m_nve_c
        #bezier curve for curvature estimation for cluster in LLE
        else:
            data_c_1 = np.vstack(label_dict[closest_label])
            m_pve_c, m_nve_c, canvas_1 = be.bezier_curve_max_min_curvature_2(data_c_1, canvas_1, W)
            if m_pve_c is not None:
                c0_m = m_pve_c
            else:
                c0_m = m_nve_c
            pass
    # this is executed if the LLE cluster and RLE cluster condition is not satisfied, in this case it finds the closest cluster to the car, 
    # which also has mean orientation close to the previous relative yaw of the lane w.r.t car
    else:
        dist = {key: abs(value[2]) for key, value in x_and_circular_means.items()}
        if len(dist)>0:
            sorted_dist = dict(sorted(dist.items(), key=lambda item: item[1]))
            a_dist =  {key: abs(value[0]-prev_head) for key, value in x_and_circular_means.items()}
            
            sorted_dist_values = np.array(list(sorted_dist.values())).reshape(-1, 1)
            a_dist_values = np.array(list(a_dist.values())).reshape(-1, 1)

            scaler = MinMaxScaler()
            scaled_sorted_dist = scaler.fit_transform(sorted_dist_values).ravel()
            scaled_a_dist = scaler.fit_transform(a_dist_values).ravel()

            sorted_dist_keys = list(sorted_dist.keys())
            a_dist_keys = list(a_dist.keys())

            scaled_sorted_dist_dict = dict(zip(sorted_dist_keys, scaled_sorted_dist))
            scaled_a_dist_dict = dict(zip(a_dist_keys, scaled_a_dist))

            closest_label = min(scaled_sorted_dist_dict.keys(), key=lambda k: (scaled_sorted_dist_dict[k] + scaled_a_dist_dict[k]) / 2)
            max_sublist = [closest_label]
            di_c = x_and_circular_means[closest_label][2]

            if di_c > 0:
                s_W = -W
            else:
                s_W = W
                
            data_c_1 = np.vstack(label_dict[closest_label])
            m_pve_c, m_nve_c, canvas_1 = be.bezier_curve_max_min_curvature_2(data_c_1, canvas_1, s_W)
            if m_pve_c is not None:
                c0_m = m_pve_c
            else:
                c0_m = m_nve_c
        else:
            pass

    #For visualization of clusters
    if closest_label != -1:    
        di_c = x_and_circular_means[closest_label][2]
    else:
        di_c = 0
    
    if args_cluster_result:
        # Define fixed colors for clusters
        fixed_colors = [
            (255, 0, 0),    # blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # red
            (255, 255, 0),  # Yellow
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (0, 255, 128),  # Spring green
            (128, 0, 255),  # Purple
            (128, 255, 0),   # Lime
            (255, 0, 255),  # Magenta
        ]
        # Create white canvas image
        canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White image

        # Draw lines with their respective label colors
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line
            label = clusters[i]  # Get label for this line
            if label == -1:  # Outlier
                color = (255, 0, 255)  # Magenta for outliers
            else:
                color = fixed_colors[label % len(fixed_colors)]  # Assign fixed color based on label

            # Draw line on canvas
            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness=2)
        # Add text annotations for stats and legend for colors in the bottom right
        if di_c > 0:
            text_stats = [
                f'No. of lanes: {no_of_lanes*2}',
                f'RLE-car: {closest_label}',
                f'Detected LE: {max_sublist}'
            ]
        elif di_c < 0:
            text_stats = [
                f'No. of lanes: {no_of_lanes*2}',
                f'LLE-car: {closest_label}',
                f'Detected LE: {max_sublist}'
            ]
        else:
                text_stats = [
                f'No. of lanes: {no_of_lanes*2}',
                f'Outlier: {closest_label}',
                f'Detected LE: {max_sublist}'
            ]


        # Calculate starting position for the text annotations
        stats_y0, dy = canvas.shape[0] - 20 * (len(text_stats) + len(unique_labels) + 1), 20  # Adjust based on the number of lines
        text_x_position = canvas.shape[1] - 250  # Adjusted x-position to fit within the right edge

        for i, text in enumerate(text_stats):
            y = stats_y0 + i * dy
            cv2.putText(canvas, text, (text_x_position, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Add legend for colors below the stats
        legend_y0 = stats_y0 + len(text_stats) * dy
        for i, label in enumerate(unique_labels):
            if label == -1:
                color = (255, 0, 255)
                label_text = 'Outliers'
            else:
                color = fixed_colors[label % len(fixed_colors)]
                label_text = f'Label {label}'
            y = legend_y0 + i * dy
            cv2.rectangle(canvas, (text_x_position, y - 15), (text_x_position + 20, y + 5), color, -1)
            cv2.putText(canvas, label_text, (text_x_position + 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        if k >= 50 and k<= 70:
            cv2.imwrite("images/cluster_results/{}.png".format(k), canvas)
        k = k + 1

    return label_dict, max_sublist, canvas_1, no_of_lanes*2, c0_m, k

# cross lane edge distance from the car
def perpendicular_cross_distance(point, ref_point, angle, d = 'c'):
    if d == 'c':
        theta = np.radians(angle-90)
    else:
        theta = np.radians(angle)
    direction_vector_normal = np.array([np.cos(theta), np.sin(theta)])
    ref_to_point_vector = ref_point - point
    distance = np.dot(ref_to_point_vector, direction_vector_normal)

    if d != 'c':
        distance = abs(distance)

    return distance

# Vectorized perpendicular_cross_distance calculation
def v_perpendicular_cross_distance(point, ref_points, angles, d='c'):
    if d == 'c':
        thetas = np.radians(angles - 90)
    else:
        thetas = np.radians(angles)
    
    direction_vectors_normal = np.stack((np.cos(thetas), np.sin(thetas)), axis=1)
    ref_to_point_vectors = ref_points - point
    distances = np.sum(ref_to_point_vectors * direction_vectors_normal, axis=1)
    
    if d != 'c':
        distances = np.abs(distances)
    
    return distances

def equalize(len_d_1, len_d_2, k1, k2, lan_dist):
    if len(lan_dist[k2]) == 0:
        lan_dist[k2] = np.empty((0, 2))
    else:
        lan_dist[k2] = np.vstack(lan_dist[k2])

    while len_d_1 != len_d_2:
        diff = abs(len_d_1 - len_d_2)
        last_element_d_1 = lan_dist[k1][-(diff-1) % len_d_1]
        if k1 == 'd_l':
            extra_element_d_2 = np.array([last_element_d_1[0] + W, last_element_d_1[1]])
        elif k1 == 'd_r':
            extra_element_d_2 = np.array([-W + last_element_d_1[0], last_element_d_1[1]])

        lan_dist[k2] = np.vstack((lan_dist[k2], extra_element_d_2))
        len_d_2 = len(lan_dist[k2])

    return lan_dist

#This block seperates left lane edge lines cluster from right lane edge lines clusters, to get their relative position w.r.t car
def raw_data(label_dict, max_sublist):
    lan_dist = {'d_l': [], 'ref_point_l_list': [], 'ref_point_r_list': [], 'd_r': []}

    # Reference points and angles
    for lb in max_sublist:
        data = np.vstack(label_dict[lb]).T 
        ref_points = data[:2]
        angles = data[2]
        distances = data[3]

        mask_1 = (np.abs(distances) > W)
        mask_2 = (np.abs(distances) < W)
        
        m_1 = distances[mask_1]
        a_1 = angles[mask_1]
        p_1 = ref_points[:, mask_1]
      
        m_2 = distances[mask_2]
        a_2 = angles[mask_2]
        p_2 = ref_points[:, mask_2]

        combined_1 = np.empty((0, 2))
        combined_2 = np.empty((0, 2))

        if np.mean(distances) < 0:
            if m_1.size > 0:
                l_d_1 = W + m_1
                combined_1 = np.vstack((l_d_1, a_1)).T
            if m_2.size > 0:
                combined_2 = np.vstack((m_2, a_2)).T

            lan_dist['d_l'] = np.vstack((combined_1, combined_2)) if combined_1.size > 0 or combined_2.size > 0 else np.empty((0, 2))
            lan_dist['ref_point_l_list'] = np.hstack((p_1, p_2)).T if p_1.size > 0 or p_2.size > 0 else np.empty((0, 2))
        else:
            if m_1.size > 0:
                r_d_1 = m_1 - W
                combined_1 = np.vstack((r_d_1, a_1)).T
            if m_2.size > 0:
                combined_2 = np.vstack((m_2, a_2)).T

            lan_dist['d_r'] = np.vstack((combined_1, combined_2)) if combined_1.size > 0 or combined_2.size > 0 else np.empty((0, 2))
            lan_dist['ref_point_r_list'] = np.hstack((p_1, p_2)).T if p_1.size > 0 or p_2.size > 0 else np.empty((0, 2))

    # Checks lengths of lists
    len_d_l = len(lan_dist['d_l'])
    len_d_r = len(lan_dist['d_r'])

    # Iterates until the lengths of 'd_l' and 'd_r' are equal
    if len_d_l > len_d_r:
        lan_dist = equalize(len_d_l, len_d_r, 'd_l', 'd_r', lan_dist)
    else:
        lan_dist = equalize(len_d_r, len_d_l, 'd_r', 'd_l', lan_dist)

    return lan_dist

# This block combines measurements of relative position and relative yaw of lane lines w.r.t car
def get_z(lan_dist, k, scal_factor, wp, x_est, c0_m, args_filter):
    d_y_l = np.vstack(lan_dist['d_l']).T
    d_y_r = np.vstack(lan_dist['d_r']).T

    if len(lan_dist['ref_point_l_list']) > len(lan_dist['ref_point_r_list']):
        y_lr = d_y_l[1, :].copy()
        if np.mean(y_lr) >= 90:
            mask = y_lr >= 90
            y_lr[mask] = 90 - y_lr[mask]
            y_lr = np.deg2rad(y_lr)
        elif np.mean(y_lr) <= 90:
            mask = y_lr <= 90
            y_lr[mask] = 90 - y_lr[mask]
            y_lr = np.deg2rad(y_lr)
        else:
            y_lr = np.zeros(len(y_lr))
    else:
        y_lr = d_y_r[1, :].copy()
        if np.mean(y_lr) >= 90:
            mask = y_lr >= 90
            y_lr[mask] = 90 - y_lr[mask]
            y_lr = np.deg2rad(y_lr)
        elif np.mean(y_lr) <= 90:
            mask = y_lr <= 90
            y_lr[mask] = 90 - y_lr[mask]
            y_lr = np.deg2rad(y_lr)
        else:
            y_lr = np.zeros(len(y_lr))
    dl_m = d_y_l[0, :] * scal_factor
    dr_m = d_y_r[0, :] * scal_factor

    y_lr = y_lr.reshape(len(y_lr), 1)
    dl_m = dl_m.reshape(len(dl_m), 1)
    dr_m = dr_m.reshape(len(dr_m), 1)
    if args_filter == 'ukf':   
        if c0_m is not None:
            c_m = np.ones(len(dl_m)).reshape(len(dl_m), 1) * c0_m
        else:
            c_m = np.ones(len(dl_m)).reshape(len(dl_m), 1) * 0
        x_i = np.ones(len(dl_m)).reshape(len(dl_m), 1) * wp
        y_i = np.ones(len(dl_m)).reshape(len(dl_m), 1) * x_est[11]

        z_m = np.concatenate((dl_m, dr_m, y_lr, c_m, x_i, y_i), axis=1)
    else:
        z_m = np.concatenate((dl_m, dr_m, y_lr), axis=1)

    return z_m

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

def filter_m_theta(m_theta, lines_1, W, l_d='r'):
    # Create masks based on the conditions
    if l_d == 'r':
        mask_1 = (m_theta[:, 3] > 0) & ((1/4 * W - 30) <= m_theta[:, 3]) & (m_theta[:, 3] <= (1/4 * W + 30))
        mask_2 = (m_theta[:, 3] < 0) & (-(3/4 * W + 30) <= m_theta[:, 3]) & (m_theta[:, 3] <= -(3/4 * W - 30))
    else:
        mask_1 = (m_theta[:, 3] > 0) & ((3/4 * W - 30) <= m_theta[:, 3]) & (m_theta[:, 3] <= (3/4 * W + 30))
        mask_2 = (m_theta[:, 3] < 0) & (-(1/4 * W + 30) <= m_theta[:, 3]) & (m_theta[:, 3] <= -(1/4 * W - 30))

    # Combine masks
    combined_mask = mask_1 | mask_2

    # Apply masks to filter arrays
    filtered_m_theta = m_theta[combined_mask]
    filtered_lines = lines_1[combined_mask]

    return filtered_m_theta, filtered_lines  

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
