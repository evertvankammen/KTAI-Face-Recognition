import os

import cv2
import face_recognition

from using_face_recognition import UsingFaceRecognition

IMAGE_PATH = os.path.join("..", "data", "pictures")
IMAGE_PATH_EMBEDDINGS = os.path.join("..", "data", "embeddings")
IMAGE_PATH_EMBEDDINGS_2 = os.path.join("..", "encoding_with_pickle", "dataset")


def save_encodings_to_file():
    sfc = UsingFaceRecognition()
    sfc.save_encodings_images(IMAGE_PATH_EMBEDDINGS_2)


def get_locations_and_names(frame, sfc):
    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=1)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        best_match_face = sfc.face_k_lowest_distances(face_encoding, 3)
        face_names.append(best_match_face)
    return face_locations, face_names


def draw_face_location(face_locations, names, frame):
    for (top, right, bottom, left), name in zip(face_locations, names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


def run_face_detector(file_name):
    capture = cv2.VideoCapture(os.path.join(IMAGE_PATH, file_name))
    sfc = UsingFaceRecognition()
    sfc.read_encoded_images()
    frame_counter = 0
    face_locations = []
    names = []
    while capture.isOpened():
        ret, frame = capture.read()
        if ret:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if frame_counter % 25 == 0 or 1 ==1:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                face_locations, names = get_locations_and_names(small_frame, sfc)

            if len(face_locations) > 0:
                draw_face_location(face_locations, names, frame)

        cv2.imshow('Video', frame)
        frame_counter = frame_counter + 1


if __name__ == '__main__':
    #save_encodings_to_file()  # enable if you want to save the encodings to a file
    run_face_detector('Friends.mp4')
