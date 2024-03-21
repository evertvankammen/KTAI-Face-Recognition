import os
import cv2
import numpy as np

from picture_compare import SimpleFacerec
import face_recognition

image_path = os.path.join("..", "data", "pictures")

sfc = SimpleFacerec()
nr_pictures = sfc.load_encoded_images(image_path)

print(nr_pictures)

VIDEO_FILE = os.path.join("..", "data", "pictures", "Friends.mp4")
capture = cv2.VideoCapture(VIDEO_FILE)
fps = capture.get(cv2.CAP_PROP_FPS)
frame_duration = int(1000 / 25)

memory_db = []


def method_name(frame, frame_counter):
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=2)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        # matches = sfc.detect_faces(face_encoding)
        # if len(matches) > 0:
        #    print(matches)
        # name = "Unknown"

        best_match_face = sfc.face_lowest_distances(face_encoding)
        face_names.append(best_match_face)
    # memory_db.append((frame_counter, face_locations, face_names))
    return face_locations, face_names


def draw_face_location(face_locations, names, frame):
    for (top, right, bottom, left), name in zip(face_locations, names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


frame_counter = 0
frame_duration_counter = 0
face_locations = None
names = None
show_frames = 0
while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_counter % 25 == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            face_locations, names = method_name(small_frame, frame_counter)
            show_frames = 25

        if show_frames > 0:
            draw_face_location(face_locations, names, frame)
            show_frames = show_frames - 1

    cv2.imshow('Video', frame)
    frame_counter = frame_counter + 1

# cap = cv2.VideoCapture(0)
#
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     # Perform face detection on the frame
#     # <INSERT_CODE_HERE>
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
