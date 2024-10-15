import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def find_matches(base_image, sec_image):
    sift = cv2.SIFT_create()
    base_kp, base_des = sift.detectAndCompute(cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY), None)
    sec_kp, sec_des = sift.detectAndCompute(cv2.cvtColor(sec_image, cv2.COLOR_BGR2GRAY), None)

    index_params = dict(algorithm=1, trees=5)  # Algorithm 1 is for KDTree for SIFT
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(base_des, sec_des, k=2)

    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

    return good_matches, base_kp, sec_kp

def find_homography(matches, base_kp, sec_kp):
    if len(matches) < 4:
        raise ValueError("Not enough matches found between the images.")
    base_pts = np.float32([base_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    sec_pts = np.float32([sec_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography_matrix, status = cv2.findHomography(sec_pts, base_pts, cv2.RANSAC, 4.0)
    return homography_matrix, status

def get_new_frame_size_and_matrix(homography_matrix, sec_image_shape, base_image_shape):
    height, width = sec_image_shape
    initial_matrix = np.array([[0, width - 1, width - 1, 0], [0, 0, height - 1, height - 1], [1, 1, 1, 1]])
    final_matrix = np.dot(homography_matrix, initial_matrix)
    x, y, c = final_matrix
    x, y = np.divide(x, c), np.divide(y, c)
    min_x, max_x, min_y, max_y = int(round(min(x))), int(round(max(x))), int(round(min(y))), int(round(max(y)))
    new_width, new_height = max_x, max_y
    correction = [0, 0]
    if min_x < 0: new_width -= min_x; correction[0] = abs(min_x)
    if min_y < 0: new_height -= min_y; correction[1] = abs(min_y)
    new_width = max(new_width, base_image_shape[1] + correction[0])
    new_height = max(new_height, base_image_shape[0] + correction[1])
    x, y = np.add(x, correction[0]), np.add(y, correction[1])
    old_initial_points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    new_final_points = np.float32(np.array([x, y]).transpose())
    homography_matrix = cv2.getPerspectiveTransform(old_initial_points, new_final_points)
    return [new_height, new_width], correction, homography_matrix

def convert_xy(x, y, center, f):
    xt = (f * np.tan((x - center[0]) / f)) + center[0]
    yt = ((y - center[1]) / np.cos((x - center[0]) / f)) + center[1]
    return xt, yt

def project_onto_cylinder(initial_image):
    h, w = initial_image.shape[:2]
    center, f = [w // 2, h // 2], 1100
    transformed_image = np.zeros(initial_image.shape, dtype=np.uint8)
    all_coords = np.array([np.array([i, j]) for i in range(w) for j in range(h)])
    ti_x, ti_y = all_coords[:, 0], all_coords[:, 1]
    ii_x, ii_y = convert_xy(ti_x, ti_y, center, f)
    ii_tl_x, ii_tl_y = ii_x.astype(int), ii_y.astype(int)
    good_indices = (ii_tl_x >= 0) & (ii_tl_x <= (w-2)) & (ii_tl_y >= 0) & (ii_tl_y <= (h-2))
    ti_x, ti_y, ii_x, ii_y = ti_x[good_indices], ti_y[good_indices], ii_x[good_indices], ii_y[good_indices]
    ii_tl_x, ii_tl_y = ii_tl_x[good_indices], ii_tl_y[good_indices]
    dx, dy = ii_x - ii_tl_x, ii_y - ii_tl_y
    weight_tl, weight_tr = (1.0 - dx) * (1.0 - dy), dx * (1.0 - dy)
    weight_bl, weight_br = (1.0 - dx) * dy, dx * dy
    transformed_image[ti_y, ti_x, :] = (weight_tl[:, None] * initial_image[ii_tl_y, ii_tl_x, :]) + \
                                       (weight_tr[:, None] * initial_image[ii_tl_y, ii_tl_x + 1, :]) + \
                                       (weight_bl[:, None] * initial_image[ii_tl_y + 1, ii_tl_x, :]) + \
                                       (weight_br[:, None] * initial_image[ii_tl_y + 1, ii_tl_x + 1, :])
    min_x = min(ti_x)
    transformed_image = transformed_image[:, min_x:-min_x, :]
    return transformed_image, ti_x - min_x, ti_y

def stitch_images(base_image, sec_image):
    if base_image.shape[2] == 4: base_image = base_image[:, :, :3]
    if sec_image.shape[2] == 4: sec_image = sec_image[:, :, :3]
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_proj = executor.submit(project_onto_cylinder, sec_image)
        future_matches = executor.submit(find_matches, base_image, sec_image)
        sec_image_cyl, mask_x, mask_y = future_proj.result()
        matches, base_kp, sec_kp = future_matches.result()
    sec_image_mask = np.zeros(sec_image_cyl.shape, dtype=np.uint8)
    sec_image_mask[mask_y, mask_x, :] = 255
    homography_matrix, status = find_homography(matches, base_kp, sec_kp)
    new_frame_size, correction, homography_matrix = get_new_frame_size_and_matrix(homography_matrix, sec_image_cyl.shape[:2], base_image.shape[:2])
    sec_image_transformed = cv2.warpPerspective(sec_image_cyl, homography_matrix, (new_frame_size[1], new_frame_size[0]))
    base_image_transformed = np.zeros((new_frame_size[0], new_frame_size[1], 3), dtype=np.uint8)
    base_image_transformed[correction[1]:correction[1]+base_image.shape[0], correction[0]:correction[0]+base_image.shape[1]] = base_image
    if sec_image_transformed.shape != base_image_transformed.shape:
        sec_image_transformed = cv2.resize(sec_image_transformed, (base_image_transformed.shape[1], base_image_transformed.shape[0]))
    stitched_image = cv2.max(sec_image_transformed, base_image_transformed)
    return stitched_image

def dhash(image, hash_size=8):
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])