import os
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
from mediapipe.tasks.python.vision import FaceLandmarkerResult
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from landmark_face_rec import LandmarkFaceRec
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

IMAGE_PATH = os.path.join("..", "..", "data", "pictures")
VIDEO_PATH = os.path.join("..", "..", "data", "movies")
TASK_MODEL = '../../models/face_landmarker.task'
IMAGE_PATH_EMBEDDINGS = os.path.join("..", "..", "data", "embeddings")

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())
    return annotated_image


def get_detection_results(lmf, detection_results: FaceLandmarkerResult):
    for face_landmarks in detection_results.face_landmarks:
        n = lmf.face_k_lowest_distances(face_landmarks, 3)
        print(n)
    print("---")


def initialize_detector():
    base_options = python.BaseOptions(model_asset_path=TASK_MODEL)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           running_mode=VisionTaskRunningMode.VIDEO,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    return vision.FaceLandmarker.create_from_options(options)


def save_encodings_to_file():
    lmf = LandmarkFaceRec()
    lmf.save_encodings_images(IMAGE_PATH_EMBEDDINGS)


def run_face_detector(file_name):
    lmf = LandmarkFaceRec()
    lmf.read_encoded_images()
    detector = initialize_detector()
    capture = cv2.VideoCapture(os.path.join(VIDEO_PATH, file_name))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_duration = 1000 / fps
    frame_duration_counter = 0
    while capture.isOpened():
        ret, frame = capture.read()
        if ret:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            mp_image_temp = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_result = detector.detect_for_video(mp_image_temp, frame_duration_counter)
            get_detection_results(lmf, detection_result)

            annotated_image = draw_landmarks_on_image(mp_image_temp.numpy_view(), detection_result)
            cv2.imshow('Video', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            frame_duration_counter = frame_duration_counter + round(frame_duration)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#
if __name__ == '__main__':
    #save_encodings_to_file()  # enable if you want to save encodings to a file
    run_face_detector('Friends.mp4')
