import math
import os
from collections import Counter

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from common import visualize
from picture_compare import SimpleFacerec
import tkinter as tk
import userinteraction.userinterface as gui


BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the video mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='../models/detector.tflite'),
    running_mode=VisionRunningMode.VIDEO)

VIDEO_FILE = os.path.join("..", "data", "movies", "The Ones With Chandler's Sarcasm _ Friends.mp4")

sfc = SimpleFacerec()
image_path = os.path.join("..", "data", "pictures")
#sfc.save_encodings_images(image_path)
sfc.read_encoded_images()

faces_in_box = []


with FaceDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(VIDEO_FILE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1000/fps
    print(fps)
    print(frame_duration)
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_counter = 0
    frame_duration_counter = 0
    previous_x, previous_y = 0, 0
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()


        if ret:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            mp_image_temp = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_detector_result = detector.detect_for_video(mp_image_temp, round(frame_duration_counter))
            for result in face_detector_result.detections:
                distance = math.sqrt((previous_x - result.bounding_box.origin_x)**2  + (previous_y - result.bounding_box.origin_y)**2)
                if result.categories[0].score > 0.75:
                    result.categories[0].category_name = sfc.face_lowest_distances(result.keypoints)
                else:
                    result.categories[0].category_name = 'xxx'

                if distance < 100:
                    print('same box')
                    if result.categories[0].category_name != 'xxx':
                        faces_in_box.append(result.categories[0].category_name)
                else:
                    print('other box')
                    print(faces_in_box)
                    if len(faces_in_box) > 0:
                        counts = Counter(faces_in_box)
                        most_frequent, the_count = counts.most_common(1)[0]
                        print(most_frequent, the_count)
                        gui.start_gui(most_frequent)
                    faces_in_box = []

                previous_x, previous_y = result.bounding_box.origin_x, result.bounding_box.origin_y

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            image_copy = np.copy(mp_image.numpy_view())
            annotated_image = visualize(image_copy, face_detector_result)
            cv2.imshow("Annotated", annotated_image)
            if frame_counter % round(fps) == 0 and 1==2:
                for result in face_detector_result.detections:
                    if result.categories[0].score > 0.75:
                        gui.start_gui(result.categories[0].category_name)
                        if not gui.is_true:
                            print(gui.name_entry)
                            sfc.write_encoded_images(gui.name_entry, result.keypoints)
                        else:
                            print("Keuze was goed")




        # Break the loop
        else:
            break

        frame_counter = frame_counter + 1
        frame_duration_counter = frame_duration_counter + frame_duration
