
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

import numpy as np
import cv2

def bezier_curve_max_min_curvature(data_c_1, data_c_2, canvas_1, W):

    if data_c_1.size < 3 and data_c_2.size < 3  :
        return None, None, canvas_1 
    # Calculate the mean points and angles
    mean_point_1 = np.mean(data_c_1[:, :2], axis=0)
    mean_point_2 = np.mean(data_c_2[:, :2], axis=0)

    mean_angle_1 = np.mean(data_c_1[:, 2])
    mean_angle_2 = np.mean(data_c_2[:, 2])

    # Convert angles to radians
    theta_1 = np.deg2rad(mean_angle_1)
    theta_2 = np.deg2rad(mean_angle_2)

    # Calculate the direction vectors for the tangents
    dir_1 = np.array([np.cos(theta_1), np.sin(theta_1)])
    dir_2 = np.array([np.cos(theta_2), np.sin(theta_2)])

    # Estimate distance parameter for control points
    d = np.linalg.norm(mean_point_2 - mean_point_1) / 3

    # Calculate control points
    P1_0 = mean_point_1 + d * dir_1
    P1_2 = mean_point_2 - d * dir_2

    # Averaging control points to get a smoother curve
    P1 = (P1_0 + P1_2) / 2

    # Generate the Bézier curve points
    t = np.linspace(0, 1, 100)
    curve = (1 - t)[:, None] ** 2 * mean_point_1 + 2 * (1 - t)[:, None] * t[:, None] * P1 + t[:, None] ** 2 * mean_point_2

    # First derivative of the Bézier curve
    curve_prime = 2 * (1 - t)[:, None] * (P1 - mean_point_1) + 2 * t[:, None] * (mean_point_2 - P1)

    # Second derivative of the Bézier curve
    curve_double_prime = 2 * (mean_point_2 - 2 * P1 + mean_point_1)

    # Calculate the curvature
    curvature = (curve_prime[:, 0] * curve_double_prime[1] - curve_prime[:, 1] * curve_double_prime[0]) / (curve_prime[:, 0] ** 2 + curve_prime[:, 1] ** 2) ** 1.5
    
    # Find the largest positive curvature and the largest negative curvature
    largest_positive_curvature = np.max(curvature) if np.max(curvature) > 0 else None
    largest_negative_curvature = np.min(curvature) if np.min(curvature) < 0 else None

    point_1 = mean_point_1
    point_2 =mean_point_2
    # Scale and translate the points for better visualization
    curve_scaled = (curve * [1, -1] + [0, 500]).astype(np.int32)
    point_1_scaled = (point_1 * [1, -1] + [0, 500]).astype(np.int32)
    point_2_scaled = (point_2 * [1, -1] + [0, 500]).astype(np.int32)
    P1_scaled = (P1 * [1, -1] + [0, 500]).astype(np.int32)

    # Draw the Bézier curve
    for i in range(len(curve_scaled) - 1):
        cv2.line(canvas_1, tuple(curve_scaled[i]), tuple(curve_scaled[i + 1]), (255, 255, 255), 2)

    # Draw control points and lines
    cv2.circle(canvas_1, tuple(point_1_scaled), 5, (0, 0, 255), -1)
    cv2.circle(canvas_1, tuple(point_2_scaled), 5, (0, 0, 255), -1)
    cv2.circle(canvas_1, tuple(P1_scaled), 5, (0, 255, 0), -1)
    cv2.line(canvas_1, tuple(point_1_scaled), tuple(P1_scaled), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(canvas_1, tuple(P1_scaled), tuple(point_2_scaled), (0, 255, 0), 1, cv2.LINE_AA)

    return largest_positive_curvature, largest_negative_curvature, canvas_1


def bezier_curve_max_min_curvature_1(data_c_1, canvas_1, W):

    if data_c_1.size < 3:
        return None, None, canvas_1 
    # Extract x and y coordinates
    x = data_c_1[:, 0]
    y = data_c_1[:, 1]

    # Obtain indices to sort y
    sorted_indices = np.argsort(y)

    # Sort y and corresponding angles theta
    y_sorted = y[sorted_indices]
    theta_sorted = data_c_1[:, 2][sorted_indices]

    # Determine the points and angles
    point_1 = np.array([x[sorted_indices[0]], y_sorted[0]])
    point_2 = np.array([x[sorted_indices[-1]], y_sorted[-1]])

    # Convert angles to radians
    theta_1 = np.deg2rad(theta_sorted[0])
    theta_2 = np.deg2rad(theta_sorted[-1])

    # Calculate the direction vectors for the tangents
    dir_1 = np.array([np.cos(theta_1), np.sin(theta_1)])
    dir_2 = np.array([np.cos(theta_2), np.sin(theta_2)])

    # Estimate distance parameter for control points
    d = np.linalg.norm(point_2 - point_1) / 3

    # Calculate control points
    P1_0 = point_1 + d * dir_1
    P1_2 = point_2 - d * dir_2

    # Averaging control points to get a smoother curve
    P1 = (P1_0 + P1_2) / 2

    # Generate the Bézier curve points
    t = np.linspace(0, 1, 100)
    curve = (1 - t)[:, None] ** 2 * point_1 + 2 * (1 - t)[:, None] * t[:, None] * P1 + t[:, None] ** 2 * point_2

    # First derivative of the Bézier curve
    curve_prime = 2 * (1 - t)[:, None] * (P1 - point_1) + 2 * t[:, None] * (point_2 - P1)

    # Second derivative of the Bézier curve
    curve_double_prime = 2 * (point_2 - 2 * P1 + point_1)

    # Calculate the curvature
    curvature = (curve_prime[:, 0] * curve_double_prime[1] - curve_prime[:, 1] * curve_double_prime[0]) / (curve_prime[:, 0] ** 2 + curve_prime[:, 1] ** 2) ** 1.5

    # Find the largest positive curvature and the largest negative curvature
    largest_positive_curvature = np.max(curvature) if np.max(curvature) > 0 else None
    largest_negative_curvature = np.min(curvature) if np.min(curvature) < 0 else None


    # Scale and translate the points for better visualization
    curve_scaled = (curve * [1, -1] + [0, 500]).astype(np.int32)
    point_1_scaled = (point_1 * [1, -1] + [0, 500]).astype(np.int32)
    point_2_scaled = (point_2 * [1, -1] + [0, 500]).astype(np.int32)
    P1_scaled = (P1 * [1, -1] + [0, 500]).astype(np.int32)

    # Draw the Bézier curve
    for i in range(len(curve_scaled) - 1):
        cv2.line(canvas_1, tuple(curve_scaled[i]), tuple(curve_scaled[i + 1]), (255, 255, 255), 2)

    # Draw control points and lines
    cv2.circle(canvas_1, tuple(point_1_scaled), 5, (0, 0, 255), -1)
    cv2.circle(canvas_1, tuple(point_2_scaled), 5, (0, 0, 255), -1)
    cv2.circle(canvas_1, tuple(P1_scaled), 5, (0, 255, 0), -1)
    cv2.line(canvas_1, tuple(point_1_scaled), tuple(P1_scaled), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(canvas_1, tuple(P1_scaled), tuple(point_2_scaled), (0, 255, 0), 1, cv2.LINE_AA)


    return largest_positive_curvature, largest_negative_curvature, canvas_1

def bezier_curve_max_min_curvature_2(data_c, canvas_1, W):
    if data_c.size < 3:
        return None, None, canvas_1  # Return None for curvatures and the canvas as is

    distance = W
    # Extract x and y coordinates
    x = data_c[:, 0]
    y = data_c[:, 1]

    # Obtain indices to sort y
    sorted_indices = np.argsort(y)

    # Check if sorted_indices is empty
    if sorted_indices.size == 0:
        return None, None, canvas_1

    # Sort y and corresponding angles theta
    y_sorted = y[sorted_indices]
    theta_sorted = data_c[:, 2][sorted_indices]

    # Check if y_sorted is empty
    if y_sorted.size == 0:
        return None, None, canvas_1

    # Determine the points and angles
    point_1 = np.array([x[sorted_indices[0]], y_sorted[0]])
    point_2 = np.array([x[sorted_indices[-1]], y_sorted[-1]])

    # Convert angles to radians
    theta_1 = np.deg2rad(theta_sorted[0])
    theta_2 = np.deg2rad(theta_sorted[-1])

    # Calculate the direction vectors for the tangents
    dir_1 = np.array([np.cos(theta_1), np.sin(theta_1)])
    dir_2 = np.array([np.cos(theta_2), np.sin(theta_2)])

    # Estimate distance parameter for control points
    d = np.linalg.norm(point_2 - point_1) / 3

    # Calculate control points
    P1_0 = point_1 + d * dir_1
    P1_2 = point_2 - d * dir_2

    # Averaging control points to get a smoother curve
    P1 = (P1_0 + P1_2) / 2

    # Generate the Bézier curve points
    t = np.linspace(0, 1, 100)
    curve = (1 - t)[:, None] ** 2 * point_1 + 2 * (1 - t)[:, None] * t[:, None] * P1 + t[:, None] ** 2 * point_2

    # First derivative of the Bézier curve
    curve_prime = 2 * (1 - t)[:, None] * (P1 - point_1) + 2 * t[:, None] * (point_2 - P1)

    # Second derivative of the Bézier curve
    curve_double_prime = 2 * (point_2 - 2 * P1 + point_1)

    # Calculate the curvature
    curvature = (curve_prime[:, 0] * curve_double_prime[1] - curve_prime[:, 1] * curve_double_prime[0]) / (curve_prime[:, 0] ** 2 + curve_prime[:, 1] ** 2) ** 1.5

    # Find the largest positive curvature and the largest negative curvature
    largest_positive_curvature = np.max(curvature) if np.max(curvature) > 0 else None
    largest_negative_curvature = np.min(curvature) if np.min(curvature) < 0 else None

    # Scale and translate the points for better visualization
    curve_scaled = (curve * [1, -1] + [0, 500]).astype(np.int32)
    point_1_scaled = (point_1 * [1, -1] + [0, 500]).astype(np.int32)
    point_2_scaled = (point_2 * [1, -1] + [0, 500]).astype(np.int32)
    P1_scaled = (P1 * [1, -1] + [0, 500]).astype(np.int32)

    # Draw the original Bézier curve
    for i in range(len(curve_scaled) - 1):
        cv2.line(canvas_1, tuple(curve_scaled[i]), tuple(curve_scaled[i + 1]), (255, 255, 255), 2)

    # Draw control points and lines for the original curve
    cv2.circle(canvas_1, tuple(point_1_scaled), 5, (0, 0, 255), -1)
    cv2.circle(canvas_1, tuple(point_2_scaled), 5, (0, 0, 255), -1)
    cv2.circle(canvas_1, tuple(P1_scaled), 5, (0, 255, 0), -1)
    cv2.line(canvas_1, tuple(point_1_scaled), tuple(P1_scaled), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(canvas_1, tuple(P1_scaled), tuple(point_2_scaled), (0, 255, 0), 1, cv2.LINE_AA)

    # Offset the curve points by the specified distance
    normal = np.array([np.sin(theta_1), -np.cos(theta_1)])  # Normal vector to the direction
    offset = distance * normal

    point_1_offset = point_1 + offset
    point_2_offset = point_2 + offset
    P1_offset = P1 + offset

    # Generate the offset Bézier curve points
    curve_offset = (1 - t)[:, None] ** 2 * point_1_offset + 2 * (1 - t)[:, None] * t[:, None] * P1_offset + t[:, None] ** 2 * point_2_offset

    # Scale and translate the offset curve points for better visualization
    curve_offset_scaled = (curve_offset * [1, -1] + [0, 500]).astype(np.int32)
    point_1_offset_scaled = (point_1_offset * [1, -1] + [0, 500]).astype(np.int32)
    point_2_offset_scaled = (point_2_offset * [1, -1] + [0, 500]).astype(np.int32)
    P1_offset_scaled = (P1_offset * [1, -1] + [0, 500]).astype(np.int32)

    # Draw the offset Bézier curve
    for i in range(len(curve_offset_scaled) - 1):
        cv2.line(canvas_1, tuple(curve_offset_scaled[i]), tuple(curve_offset_scaled[i + 1]), (0, 255, 255), 2)

    # Draw control points and lines for the offset curve
    cv2.circle(canvas_1, tuple(point_1_offset_scaled), 5, (255, 0, 0), -1)
    cv2.circle(canvas_1, tuple(point_2_offset_scaled), 5, (255, 0, 0), -1)
    cv2.circle(canvas_1, tuple(P1_offset_scaled), 5, (0, 255, 255), -1)
    cv2.line(canvas_1, tuple(point_1_offset_scaled), tuple(P1_offset_scaled), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(canvas_1, tuple(P1_offset_scaled), tuple(point_2_offset_scaled), (0, 255, 255), 1, cv2.LINE_AA)

    return largest_positive_curvature, largest_negative_curvature, canvas_1