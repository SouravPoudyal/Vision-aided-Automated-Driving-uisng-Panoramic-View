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

#This utility functions are similar to that defined in the simulation code


import numpy as np
import cv2
import Bezier_curve as be
import learn as le
import time
from collections import Counter

#Initializing previous mean orientation of lines detected on the closest lane edge cluster to the car
#Utilized just for visualization
prev_head = 90

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

def kernel_density_estimation(data, bandwidth):
    """Perform kernel density estimation using Gaussian kernel."""
    x = np.linspace(np.min(data), np.max(data), 1000)
    pdf = np.exp(-0.5 * ((x - data[:, None]) / bandwidth) ** 2).sum(axis=0) / (np.sqrt(2 * np.pi) * bandwidth)
    return x, pdf

def find_local_extrema(data):
    """Find local minima and maxima in a 1D array."""
    minima = (np.diff(np.sign(np.diff(data))) > 0).nonzero()[0] + 1
    maxima = (np.diff(np.sign(np.diff(data))) < 0).nonzero()[0] + 1
    return minima, maxima
@measure_time
def KDC_lanes(m_t, k):
    m_t = np.array(m_t, dtype=np.float32)  # Convert to NumPy array

    x, pdf = kernel_density_estimation(m_t[:, 3], bandwidth=3)
    # Find local minima and maxima
    mi, ma = find_local_extrema(pdf)


    # Getting the selected elements from x using the mask ma
    s_m = x[ma]
    s_1, s_2 = l_count(s_m,[],'k')

    s_l = np.max(np.add(s_1, s_2))
    #print('final lanes', s_l)

    return s_l

def circular_mean(angles):
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    return np.arctan2(sin_sum, cos_sum)

# Function to compute circular mean using PyTorch
def _mean(data, a):
    p_theta = []
    angles_rad = np.deg2rad(data[:, 2])  # Convert angles to radians
    mean_angle_rad = circular_mean(angles_rad)
    mean_angle_deg = np.rad2deg(mean_angle_rad) % 360
    p_theta.append(mean_angle_deg)

    p_mean = np.mean(data[:, a])
    mean_c = np.mean(data[:, 3])
    p_theta.append(p_mean)
    p_theta.append(mean_c)

    return p_theta  # Convert to scalar and return

def l_count(s_m, sig_mask, W, c ='d'):
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
        d_matrix = np.abs(s_m[:, None]) + abs(s_m)

    d_matrix_1 = np.abs(np.abs(s_m[:, None]) - abs(s_m))
    r_matrix = abs(W - d_matrix)
    r_matrix_1 = abs(W - d_matrix_1)

    condition_matrix =(r_matrix <= 22)
    c_vector = condition_matrix.sum(axis=1)

    condition_matrix_1 =(r_matrix_1 <= 22)
    c_vector_1 = condition_matrix_1.sum(axis=1)

    s_1 = c_vector.tolist()
    # Replacing values greater than 1 with 1
    s_1 = [1 if x > 1 else x for x in s_1]
    s_2 = c_vector_1.tolist()
    s_2 = [1 if x > 1 else x for x in s_2]

    return s_1, s_2

@measure_time
def cluster_line(lines, data, k, canvas_1, W, args_cluster_result):
    global prev_head

    a = 0
    no_of_lanes = 0
    closest_label = -1
    max_sublist = []
    closest_pairs = []
    weights = np.array([0.9, 1])
    x_and_circular_means = {}

    # Extracting only the first and third elements from the data array
    m_theta = data
    c_mean_mt = _mean(m_theta, a)
    
    if 30 <= c_mean_mt[0] <= 150:
        a = 0
    else:
        a = 1

    reduced_data = m_theta[:, [2, 3]]

    # Normalize using Min-Max Scaler
    scaler = le.customMinMaxScaler()
    reduced_data[:, 0] = scaler.fit_transform(reduced_data[:, 0].reshape(-1, 1)).ravel()
    reduced_data[:, 1]  = scaler.fit_transform(reduced_data[:, 1].reshape(-1, 1)).ravel()
    #reduced_data[:, 2]  = scaler.fit_transform(reduced_data[:, 2].reshape(-1, 1)).ravel()

    # Apply weights to the normalized features
    weighted_data = reduced_data * weights

    #print('weighted_data', weighted_data)
    
    # Instantiating and fitting DBSCAN
    dbscan = le.customDBSCAN(eps=0.09, min_samples=3)
    
    clusters = dbscan.fit_predict(weighted_data)
    unique_labels = np.unique(clusters)
    label_dict = {label: [] for label in np.unique(clusters)}
    label_dict[-1] = []

    for i, label in enumerate(clusters):
        label_dict[label].append(m_theta[i])

    count = {key: len(value) for key, value in label_dict.items()}
    try:
        x_and_circular_means = {key: _mean(np.array(value), a) for key, value in label_dict.items() if len(value) != 0 and key != -1}
    except Exception as e:
         x_and_circular_means = {}

    means = np.array(list(x_and_circular_means.values()))
    
    if means.size > 0:  # Check if means is not empty
        mean_0_diffs = np.abs(means[:, None, 0] - means[:, 0])
        significant_mask = mean_0_diffs <= 20

        s_1, s_2 = l_count(means[:, 2], significant_mask, W, 'd')
    else:
        s_1, s_2 = [], []

    n_l = np.add(s_1, s_2)
    no_of_lanes = np.max(n_l) if n_l.size > 0 else 0
    #print('Final lanes:', no_of_lanes)

    #########################Furthere processing of the clusters######
    max_sublist = [index for index, value in enumerate(s_1) if value > 0]
    ang_list =  [x_and_circular_means[l][0] for l in max_sublist]

    if ang_list:
        median_angle = np.median(ang_list)
        mad_angle = [abs(angle - median_angle) for angle in ang_list]
        #print('mad_angle', max_sublist)
        max_sublist = [max_sublist[index] for index, value in enumerate(mad_angle) if value <= 15]
        #print('max_sublist', max_sublist)

        closest_pairs = find_closest_pairs(max_sublist, x_and_circular_means)
        #print('closest_pairs:', closest_pairs)
    else:
        pass

    if len(max_sublist) > 0:
        dist = {key: x_and_circular_means[key][2] for key in max_sublist}
        counts = {key: count[key] for key in max_sublist}

        sorted_dist = dict(sorted(dist.items(), key=lambda item: item[1]))
        sorted_counts = dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))

        dist_values = np.array(list(sorted_dist.values())).reshape(-1, 1)
        count_values = np.array(list(sorted_counts.values())).reshape(-1, 1)

        scaler = le.customMinMaxScaler()
        scaled_dist = scaler.fit_transform(dist_values).ravel()
        scaled_counts = scaler.fit_transform(count_values).ravel()

        sorted_dist_keys = list(sorted_dist.keys())
        sorted_count_keys = list(sorted_counts.keys())

        scaled_dist_dict = dict(zip(sorted_dist_keys, scaled_dist))
        scaled_count_dict = dict(zip(sorted_count_keys, scaled_counts))

        closest_label = min(scaled_dist_dict.keys(), key=lambda k: (scaled_dist_dict[k] + (1 - scaled_count_dict[k])) / 2)

        prev_head = x_and_circular_means[closest_label][0]

        if len(closest_pairs) > 0:
            data_c_1 = np.vstack(label_dict[closest_pairs[0][0]])
            data_c_2 = np.vstack(label_dict[closest_pairs[0][1]])
            data_c_m = np.vstack((data_c_1, data_c_2))
            m_pve_c, m_nve_c, canvas_1 = be.bezier_curve_max_min_curvature_2(data_c_m, canvas_1, W)
            if m_pve_c is not None:
                c0_m = m_pve_c
            else:
                c0_m = m_nve_c
        else:
            data_c_1 = np.vstack(label_dict[closest_label])
            m_pve_c, m_nve_c, canvas_1 = be.bezier_curve_max_min_curvature_2(data_c_1, canvas_1, W)
            if m_pve_c is not None:
                c0_m = m_pve_c
            else:
                c0_m = m_nve_c
            pass

    else:
        dist = {key: abs(value[2]) for key, value in x_and_circular_means.items()}
        if len(dist)>0:
            sorted_dist = dict(sorted(dist.items(), key=lambda item: item[1]))
            a_dist =  {key: abs(value[0]-prev_head) for key, value in x_and_circular_means.items()}
            
            
            sorted_dist_values = np.array(list(sorted_dist.values())).reshape(-1, 1)
            a_dist_values = np.array(list(a_dist.values())).reshape(-1, 1)

            scaler = le.customMinMaxScaler()
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
        #print('c0_m', c0_m)

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
        if k < 300:
            cv2.imwrite("imag/{}.png".format(k), canvas)

    return label_dict, max_sublist, canvas_1

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

# Vectorize the calculation of distances
def perpendicular_cross_distance_v(point, ref_points, angles, d='c'):
    if d == 'c':
        thetas = np.radians(angles - 90)
    else:
        thetas = np.radians(angles)
    direction_vectors_normal = np.column_stack((np.cos(thetas), np.sin(thetas)))
    ref_to_point_vectors = ref_points - point
    distances = np.einsum('ij,ij->i', ref_to_point_vectors, direction_vectors_normal)
    if d != 'c':
        distances = np.abs(distances)
    return distances

def equalize(len_d_1, len_d_2, W, k1, k2, lan_dist):
    # Initialize lan_dist[k2] with correct shape if it's empty
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

def raw_data(label_dict, max_sublist, W):
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
        lan_dist = equalize(len_d_l, len_d_r, W, 'd_l', 'd_r', lan_dist)
    else:
        lan_dist = equalize(len_d_r, len_d_l, W,  'd_r', 'd_l', lan_dist)

    return lan_dist


def get_z(lan_dist, k, scal_factor):
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

    z_m = np.concatenate((dl_m, dr_m, y_lr), axis=1)

    return z_m
    
def find_closest_pairs(max_sublist, x_and_circular_means):
    if len(max_sublist) > 2:
        # Extract the distance values from x_and_circular_means for the given indices in max_sublist
        dist_list = np.array([x_and_circular_means[l][2] for l in max_sublist])
        
        # Get the sorted indices based on distance values
        sorted_indices = np.argsort(dist_list)
        
        # Get the sorted distances
        sorted_dist_list = dist_list[sorted_indices]
        
        # Compute the differences between consecutive sorted distances
        diffs = np.diff(sorted_dist_list)
        
        # Find the minimum difference
        min_diff = np.min(diffs)
        
        # Find all pairs with the minimum difference
        closest_pairs_indices = np.where(diffs == min_diff)[0]
        closest_pairs = [(max_sublist[sorted_indices[i]], max_sublist[sorted_indices[i + 1]]) for i in closest_pairs_indices]
        
        return closest_pairs
    else:
        return []   
