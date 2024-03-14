import cv2
import face_recognition
import pandas as pd
import numpy as np

# Laden van de getrainde modellen uit het CSV-bestand
df = pd.read_csv("actor_encodings.csv")

# Functie om de encoding vanuit de CSV-rij te lezen en om te zetten naar een numpy-array
def lees_encoding(row):
    encoding_str = row["encoding"]
    encoding_list = encoding_str.strip("[]").split()
    encoding_array = np.array(encoding_list, dtype=np.float64)
    return encoding_array

# Omzetten van de encodings in de DataFrame naar numpy-arrays
encodings = df.apply(lees_encoding, axis=1).tolist()
actor_names = df["actor_name"].tolist()

# Open de videoclip van Friends
videoclip_path = "Friends.mp4"
videoclip = cv2.VideoCapture(videoclip_path)
fps = videoclip.get(cv2.CAP_PROP_FPS)
frame_duration = int(1000 / 25)

# Loop door elke frame van de videoclip
while True:
    ret, frame = videoclip.read()

    if not ret:
        break

    # Gezichten herkennen en labelen
    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=2)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    gezicht_labels = []
    for gezichtskenmerk in face_encodings:
        label = "Onbekend"
        for encoding, actor_name in zip(encodings, actor_names):
            afstand = np.linalg.norm(np.subtract(gezichtskenmerk, encoding))
            if afstand < 0.6:  # Aanpassen van de drempelwaarde naar wens
                label = actor_name
                break
        gezicht_labels.append(label)


    # Teken de labels op het frame
    for (top, right, bottom, left), label in zip(face_locations, gezicht_labels):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Toon het frame
    cv2.imshow('Friends Gezichtsherkenning', frame)

    # Stop de loop als 'q' wordt ingedrukt
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sluit de videoclip en de OpenCV vensters
videoclip.release()
cv2.destroyAllWindows()
