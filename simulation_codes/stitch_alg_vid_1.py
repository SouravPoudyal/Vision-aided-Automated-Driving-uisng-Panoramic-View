
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def find_matches(base_image, sec_image):
    orb = cv2.ORB_create()
    base_kp, base_des = orb.detectAndCompute(cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY), None)
    sec_kp, sec_des = orb.detectAndCompute(cv2.cvtColor(sec_image, cv2.COLOR_BGR2GRAY), None)

    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
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

    min_x, max_x = int(round(min(x))), int(round(max(x)))
    min_y, max_y = int(round(min(y))), int(round(max(y)))

    new_width, new_height = max_x, max_y
    correction = [0, 0]
    if min_x < 0:
        new_width -= min_x
        correction[0] = abs(min_x)
    if min_y < 0:
        new_height -= min_y
        correction[1] = abs(min_y)

    new_width = max(new_width, base_image_shape[1] + correction[0])
    new_height = max(new_height, base_image_shape[0] + correction[1])

    x, y = np.add(x, correction[0]), np.add(y, correction[1])
    old_initial_points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    new_final_points = np.float32(np.array([x, y]).transpose())

    homography_matrix = cv2.getPerspectiveTransform(old_initial_points, new_final_points)
    return [new_height, new_width], correction, homography_matrix

def stitch_images(base_image, sec_image):
    if base_image.shape[2] == 4:
        base_image = base_image[:, :, :3]
    if sec_image.shape[2] == 4:
        sec_image = sec_image[:, :, :3]

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_matches = executor.submit(find_matches, base_image, sec_image)
        matches, base_kp, sec_kp = future_matches.result()

    homography_matrix, status = find_homography(matches, base_kp, sec_kp)
    new_frame_size, correction, homography_matrix = get_new_frame_size_and_matrix(homography_matrix, sec_image.shape[:2], base_image.shape[:2])

    sec_image_transformed = cv2.warpPerspective(sec_image, homography_matrix, (new_frame_size[1], new_frame_size[0]))
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