import os
import face_recognition
import pandas as pd

# Map met afbeeldingen van de acteurs
actors_dir = os.path.join("friends_actors")
actor_images = []

# Loop door de map en laad afbeeldingen van elke acteur
actor_data = []
for actor_name in os.listdir(actors_dir):
    actor_path = os.path.join(actors_dir, actor_name)
    if os.path.isdir(actor_path):
        images = [os.path.join(actor_path, img) for img in os.listdir(actor_path)]
        actor_images.append(images)
        actor_data.extend([(actor_name, img) for img in images])

# Train een gezichtsherkenningsmodel voor elke acteur
actor_encodings = []
for actor_name, image_path in actor_data:
    img = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(img)[0]
    actor_encodings.append((actor_name, encoding))

# Maak een pandas DataFrame
df = pd.DataFrame(actor_encodings, columns=["actor_name", "encoding"])

# Bewaar de getrainde modellen als een CSV-bestand
df.to_csv("actor_encodings.csv", index=False)
