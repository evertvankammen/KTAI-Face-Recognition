import face_recognition
import numpy as np
import cv2
import os

known_face_encodings = []
known_face_names = []

for actor_dir in os.listdir("friends_actors"):
    for filename in os.listdir(f"friends_actors/{actor_dir}"):
        image = face_recognition.load_image_file(f"friends_actors/{actor_dir}/{filename}")
        face_encoding = face_recognition.face_encodings(image)[0]

        known_face_encodings.append(face_encoding)
        known_face_names.append(f"{actor_dir}/{filename.split('.')[0]}")


video_capture = cv2.VideoCapture("Friends.mp4")

while True:
    ret, frame = video_capture.read()

    # Converteer het frame naar RGB
    rgb_frame = frame[:, :, ::-1]

    # Zoek gezichten in het frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Vergelijk de gevonden gezichten met de bekende gezichten
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Onbekend"

        # Vind de beste match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Teken een kader om het gezicht en de naam
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Toon het frame
    cv2.imshow("Friends gezichtsherkenning", frame)

    # Wacht op een toetsaanslag om te stoppen
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()

