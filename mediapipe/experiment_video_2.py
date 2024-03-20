import os

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from common import visualize
from picture_compare import SimpleFacerec
import userinteraction.userinterface as gui



def ask_user(face_detector_result_inp):
    for result in face_detector_result_inp.detections:
        if result.categories[0].score > 0.6:
            gui.start_gui(result.categories[0].category_name)
            if not gui.is_true:
                print(gui.name_entry)
                sfc.write_encoded_images(gui.name_entry, result.keypoints)
            else:
                print("Keuze was goed")












BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face detector instance with the video mode:
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='../models/detector.tflite'),
    min_detection_confidence=0.75,
    running_mode=VisionRunningMode.VIDEO)

VIDEO_FILE = os.path.join("..", "data", "movies", "Friends.mp4")

sfc = SimpleFacerec()
image_path = os.path.join("..", "data", "pictures")
sfc.save_encodings_images(image_path)
sfc.read_encoded_images()
annotation_counter = 0
with (FaceDetector.create_from_options(options) as detector):
    cap = cv2.VideoCapture(VIDEO_FILE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1000 / fps
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
            face_detector_result = detector.detect_for_video(mp_image_temp, round(frame_duration_counter))
            show_annotated = False
            c = 1
            for result in face_detector_result.detections:
                if result.categories[0].score > 0.75:
                    result.categories[0].category_name = str(c) + ' ' + sfc.face_lowest_distances(result.keypoints)
                    show_annotated = True
                    annotation_counter = annotation_counter + 1
                else:
                    result.categories[0].category_name = ''
                c = c + 1

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            image_copy = np.copy(mp_image.numpy_view())
            if show_annotated:
                annotated_image = visualize(image_copy, face_detector_result)
                cv2.imshow("Annotated", annotated_image)
            else:
                cv2.imshow("Annotated", image_copy)

            if show_annotated and annotation_counter % 25 == 0:
                for detection in face_detector_result.detections:
                    bbox = detection.bounding_box
                    name = detection.categories[0].category_name
                    x, y = bbox.origin_x-25, bbox.origin_y-25
                    w, h = bbox.width+25, bbox.height+25
                    box_image = image_copy[y: y + h, x: x + w]
                    cv2.imwrite(f"image_{round(frame_duration_counter)}_{name}.jpg", box_image)

            # ask_user(face_detector_result)





        # Break the loop
        else:
            break

        frame_counter = frame_counter + 1
        frame_duration_counter = frame_duration_counter + frame_duration
