import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from common import visualize
from picture_compare import SimpleFacerec
import userinteraction.userinterface as gui
import pandas as pd
from matplotlib import pyplot as plt
import graphs as grp


dict={'Joey Tribbiani': 0,
      'Chandler Bing': 1,
      'Ross Geller': 2,
      'Monica Geller': 3,
      'Rachel Green': 4,
      'Phoebe Buffay': 5,
      'Other': 6
    }

facecounter =[['Joey Tribbiani',],
              ['Chandler Bing',],
              ['Ross Geller',],
              ['Monica Geller',],
              ['Rachel Green',],
              ['Phoebe Buffay',],
              ['Other']
              ]

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
with FaceDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(VIDEO_FILE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1000/fps
    print(fps)
    print(frame_duration)
    experiment=input("Enter the experiment number: ")
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_counter = 0
    frame_duration_counter = 0
    # Read until video is completed
    while cap.isOpened() and frame_counter<500:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            mp_image_temp = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face_detector_result = detector.detect_for_video(mp_image_temp, round(frame_duration_counter))

            for result in face_detector_result.detections:
                if result.categories[0].score > 0.75:
                    result.categories[0].category_name = sfc.face_lowest_distances(result.keypoints)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            image_copy = np.copy(mp_image.numpy_view())
            annotated_image = visualize(image_copy, face_detector_result)
            cv2.imshow("Annotated", annotated_image)
            #if frame_counter % round(fps) == 0 and 1 == 2:
            if frame_counter % round(fps) == 0:
                for result in face_detector_result.detections:
                    if result.categories[0].score > 0.75:
                        facecounter[dict[result.categories[0].category_name]].append(frame_counter)

        # Break the loop
        else:
            break

        frame_counter = frame_counter + 1
        frame_duration_counter = frame_duration_counter + frame_duration

    fc_DataFrame = pd.DataFrame(facecounter)
    fc_DataFrame = fc_DataFrame.transpose()
    fc_DataFrame = fc_DataFrame.drop(0)
    fc_DataFrame.columns = ['Joey Tribbiani', 'Chandler Bing', 'Ross Geller', 'Monica Geller', 'Rachel Green', 'Phoebe Buffay', 'Other']

    total = [0, 0, 0, 0, 0, 0, 0]
    for i in fc_DataFrame.columns:
        total[dict[i]] = fc_DataFrame.count()[i]

    grp.print_hist_recognized_faces(fc_DataFrame, total, experiment)

    grp.print_frames_of_recognized_faces(fc_DataFrame, experiment)