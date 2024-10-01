import cv2
import numpy as np
import time
import bev as b

def undistortion(frame, map1, map2):
    undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def apply_homography(frame, H):
    return cv2.warpPerspective(frame, H, (frame.shape[1], frame.shape[0]))

def create_mask(shape, mask_type):
    mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)  # Ensure the mask is single-channel
    h, w = shape[:2]
    if mask_type == "FR":
        mask[:h//2, :] = 255  # Upper half
    elif mask_type == "LT":
        mask[:, :w//2] = 255  # Left half
    elif mask_type == "RT":
        mask[:, w//2:] = 255  # Right half
    elif mask_type == "RR":
        mask[h//2:, :] = 255  # Lower half
    return mask

# Camera calibration parameters
K = np.array([[235.60683035299465, 0.0, 324.1537938041766], [0.0, 234.23649265395363, 238.9532690544971], [0.0, 0.0, 1.0]])
D = np.array([[-0.006972786170879497], [-0.029847393698910662], [0.016019837503268488], [-0.00015714315310710544]])

# Initialize the map for undistortion
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (640, 480), cv2.CV_16SC2)

# Homography matrices
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

homographies = {"RT": H_RT, "LT": H_LT, "FR": H_FR, "RR": H_RR}

# Setup cameras
cameras = {
    "FR": cv2.VideoCapture("v4l2src device=/dev/video4 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"),
    "RT": cv2.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"),
    "LT": cv2.VideoCapture("v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"),
    "RR": cv2.VideoCapture("v4l2src device=/dev/video6 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink")
}

last_time = {cam_id: time.time() for cam_id in cameras.keys()}
fps = {cam_id: 0 for cam_id in cameras.keys()}
overall_last_time = time.time()

while True:
    bev_images = {}
    for cam_id, cam in cameras.items():
        ret, frame = cam.read()
        if ret:
            current_time = time.time()
            time_diff = current_time - last_time[cam_id]
            fps[cam_id] = 1 / time_diff if time_diff > 0 else 0
            last_time[cam_id] = current_time

            undistorted_img = undistortion(frame, map1, map2)
            bev_img = apply_homography(undistorted_img, homographies[cam_id])
            bev_images[cam_id] = bev_img

            #fps_text = f"FPS: {fps[cam_id]:.2f}"
            #cv2.putText(bev_img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if len(bev_images) == 4:
        masks = {cam_id: create_mask(bev_images[cam_id].shape, cam_id) for cam_id in bev_images.keys()}
        masked_images = {}
        for cam_id in bev_images.keys():
            print(f"Camera {cam_id}: BEV shape: {bev_images[cam_id].shape}, Mask shape: {masks[cam_id].shape}, Mask type: {masks[cam_id].dtype}")
            masked_images[cam_id] = cv2.bitwise_and(bev_images[cam_id], bev_images[cam_id], mask=masks[cam_id])

        # Stitch the images together using cv2.max
        stitched_image_1 = cv2.max(masked_images["FR"], masked_images["RR"])
        stitched_image_2 = cv2.max(masked_images["RT"], masked_images["LT"])
        stitched_image = cv2.max(stitched_image_1, stitched_image_2)
       
        # Calculate overall FPS
        overall_current_time = time.time()
        overall_time_diff = overall_current_time - overall_last_time
        overall_fps = 1 / overall_time_diff if overall_time_diff > 0 else 0
        overall_last_time = overall_current_time

        resized_stitched_image = resize_image(stitched_image, 640, 480)
        overall_fps_text = f"Overall FPS: {overall_fps:.2f}"
        cv2.putText(resized_stitched_image, overall_fps_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("Stitched BEV Image", resized_stitched_image)
        #cv2.imwrite("images/stitched_bev.png", resized_stitched_image)  # Save the stitched BEV image

    if cv2.waitKey(1) == ord('q'):
        break

# Clean up all resources
for cam in cameras.values():
    cam.release()
cv2.destroyAllWindows()

