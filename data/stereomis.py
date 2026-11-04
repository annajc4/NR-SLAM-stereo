import os
import sys
import glob
import cv2
import numpy as np

current_folder = os.path.dirname(os.path.abspath(__file__))
stereo_mis_folder = os.path.join(current_folder, "../../robust-pose-estimator")
sys.path.insert(0, stereo_mis_folder)
from dataset.rectification import StereoRectifier
from dataset.preprocess.stereo_rectify import rectify_pair

#args should be stereomis seqeuence folder, first frame, last frame
if len(sys.argv) != 4:
    sys.exit(
        "args should be stereomis seqeuence folder, first frame, last frame"
    )

seq_folder = sys.argv[1]
first_frame = int( sys.argv[2])
last_frame = int( sys.argv[3])
#what stereomis original preprocessing resizes to
img_size = (640,512)

camera_calib = os.path.join(seq_folder, "StereoCalibration.ini")
rectifier = StereoRectifier(camera_calib, img_size_new=img_size)
calib = rectifier.get_rectified_calib()
K = calib["intrinsics"]["left"]
print("Camera.fx:", K[0, 0], "Camera.fy:", K[1, 1], "Camera.cx:", K[0, 2], "Camera.cy:", K[1, 2], "Stereo.bf:", calib["bf"])


output_folder = os.path.join(seq_folder, "video_frames")
os.makedirs(output_folder, exist_ok=True)
videos = sorted(glob.glob(os.path.join(seq_folder, "*.mp4")))
video_path = videos[0]
video = cv2.VideoCapture(video_path)
frame_num=0
times = open(os.path.join(output_folder, "times.txt"), "w")
while frame_num<last_frame:
    success, img_bgr = video.read()
    if not success:
        break
    frame_num+=1
    
    times.write(f"{video.get(cv2.CAP_PROP_POS_MSEC) / 1000}\n")

    if frame_num<first_frame:
        continue
    print("Processing frame: ", frame_num)

    halved_height = img_bgr.shape[0]//2
    left_img = cv2.resize(img_bgr[:halved_height], img_size)
    right_img = cv2.resize(img_bgr[halved_height:], img_size)
    left_rectified, right_rectified = rectify_pair(left_img, right_img, rectifier.maps)
    #name like stereomis i.e. 000123l.png 000123r.png
    cv2.imwrite(os.path.join(output_folder, f"{frame_num:06d}l.png"), left_rectified)
    cv2.imwrite(os.path.join(output_folder, f"{frame_num:06d}r.png"), right_rectified)

video.release()
times.close()
print("Images saved")
