import os
import cv2
import pickle
import face_recognition

encodings_path = os.path.join("encodings_hog_10.pickle")
# Laad de opgeslagen encodings van acteurs
with open(encodings_path, 'rb') as f:
    known_encodings = pickle.load(f)

# Extract de encodings en namen uit de geladen dictionary
known_encodings_list = known_encodings['encodings']
actor_names = known_encodings['names']


# Pad naar de map met testafbeeldingen
test_images_path = os.path.join("test")

# Maak een map aan voor de gevalideerde afbeeldingen
validate_images_path = os.path.join("validate")
if not os.path.exists(validate_images_path):
    os.makedirs(validate_images_path)

# Laad de testafbeeldingen en loop er doorheen
for filename in os.listdir(test_images_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(test_images_path, filename)
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Gebruik face_recognition om gezichten in de afbeelding te detecteren
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

        # Loop door elk gedetecteerd gezicht en vergelijk met bekende encodings
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings_list, face_encoding)
            name = "Unknown"

            # Controleer of er een overeenkomst is gevonden
            if True in matches:
                first_match_index = matches.index(True)
                name = actor_names[first_match_index]

            # Teken een rechthoek rond het gedetecteerde gezicht
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            # Voeg de naam van de acteur toe aan de afbeelding
            cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Sla de afbeelding met de getekende rechthoeken op in de map "validate"
        validate_image_path = os.path.join(validate_images_path, filename)
        cv2.imwrite(validate_image_path, image)

print("Validatie voltooid. Gevalideerde afbeeldingen zijn opgeslagen in de map 'validate'.")