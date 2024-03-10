import os
from time import sleep

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from common import visualize
from picture_compare import SimpleFacerec

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the video mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='../models/detector.tflite'),
    running_mode=VisionRunningMode.VIDEO)

VIDEO_FILE = os.path.join("..", "data", "pictures", "Friends.mp4")

sfc = SimpleFacerec()
image_path = os.path.join("..", "data", "pictures")
nr_pictures = sfc.load_encoded_images(image_path)

with FaceDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(VIDEO_FILE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = int(1000/25)
    print(fps)
    print(frame_duration)
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_counter = 0
    frame_duration_counter = 0
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break


            mp_image_temp = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_detector_result = detector.detect_for_video(mp_image_temp, frame_duration_counter)


            for result in face_detector_result.detections:
                if result.categories[0].score > 0.5:
                    result.categories[0].category_name = sfc.face_lowest_distances(result.keypoints)
                    for i in range(len(result.keypoints)):
                        result.keypoints[i].label = str(i)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            image_copy = np.copy(mp_image.numpy_view())
            annotated_image = visualize(image_copy, face_detector_result)
            cv2.imshow("Annotated", annotated_image)


        # Break the loop
        else:
            break

        frame_counter = frame_counter + 1
        frame_duration_counter = frame_duration_counter + frame_duration
