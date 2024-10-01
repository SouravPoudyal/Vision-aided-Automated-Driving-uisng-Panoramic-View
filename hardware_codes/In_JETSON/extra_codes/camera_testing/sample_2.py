import cv2 as cv
import numpy as np
import time

def undistortion(frame, map1, map2):
    undistorted_img = cv.remap(frame, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    return undistorted_img

def resize_image(image, width, height):
    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

# Camera calibration parameters
K = np.array([[235.60683035299465, 0.0, 324.1537938041766], [0.0, 234.23649265395363, 238.9532690544971], [0.0, 0.0, 1.0]])
D = np.array([[-0.006972786170879497], [-0.029847393698910662], [0.016019837503268488], [-0.00015714315310710544]])

# Initialize the map for undistortion
map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (640,480), cv.CV_16SC2)

# Setup cameras
cameras = {
    "FR": cv.VideoCapture("v4l2src device=/dev/video4 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"),
    "RT": cv.VideoCapture("v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"),
    "LT": cv.VideoCapture("v4l2src device=/dev/video2 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"),
    "RR": cv.VideoCapture("v4l2src device=/dev/video6 ! video/x-raw, width=640, height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink")
}

last_time = {cam_id: time.time() for cam_id in cameras.keys()}
fps = {cam_id: 0 for cam_id in cameras.keys()}

while True:
    for cam_id, cam in cameras.items():
        ret, frame = cam.read()
        if ret:
            current_time = time.time()
            time_diff = current_time - last_time[cam_id]
            fps[cam_id] = 1 / time_diff if time_diff > 0 else 0
            last_time[cam_id] = current_time

            undistorted_img = undistortion(frame, map1, map2)
            resized_img = resize_image(undistorted_img, 800, 600)
            
            fps_text = f"FPS: {fps[cam_id]:.2f}"
            cv.putText(resized_img, fps_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv.imshow(f"Camera {cam_id}", resized_img)
            cv.imwrite(f"images/{cam_id}_800x600.png", resized_img)  # Save the resized image

    if cv.waitKey(1) == ord('q'):
        break

# Clean up all resources
for cam in cameras.values():
    cam.release()
cv.destroyAllWindows()

