import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from common import visualize


BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the video mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='../models/detector.tflite'),
    running_mode=VisionRunningMode.VIDEO)

with FaceDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)
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
            # Display the resulting frame
            # cv2.imshow('Frame', frame)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            face_detector_result = detector.detect_for_video(mp_image, frame_duration_counter)

            image_copy = np.copy(mp_image.numpy_view())
            annotated_image = visualize(image_copy, face_detector_result)
            cv2.imshow("Annotated", annotated_image)

            # sleep(1)
            # cv2.waitKey(0)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

        frame_counter = frame_counter + 1
        frame_duration_counter = frame_duration_counter + frame_duration
