import face_recognition
import pickle
import cv2
import os

class ImageEncoder:
    def __init__(self, dataset_directory):
        self.dataset_directory = dataset_directory
        self.known_encodings = []
        self.known_names = []
        self.processed_images_per_person = {}

    def get_image_files(self):
        image_files = []
        for root, dirs, files in os.walk(self.dataset_directory):
            for filename in files:
                if filename.lower().endswith(('.jpg', 'jpeg', 'png')):
                    image_files.append(os.path.join(root, filename))
        return image_files

    def encode_images(self, max_images=None, encode_model='hog'):
        print("Extracting all images from dataset...")
        image_paths =self.get_image_files()

        for (i, image_path) in enumerate(image_paths):
            print("Processing image {}/{}".format(i + 1, len(image_paths)))
            name = image_path.split(os.path.sep)[-2]

            if name not in self.processed_images_per_person:
                self.processed_images_per_person[name] = 0

            if max_images is None or self.processed_images_per_person[name] < max_images:
                image = cv2.imread(image_path)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes = face_recognition.face_locations(rgb, model=encode_model)
                encodings = face_recognition.face_encodings(rgb, boxes)

                for encoding in encodings:
                    self.known_encodings.append(encoding)
                    self.known_names.append(name)

                self.processed_images_per_person[name] += 1

                if max_images is not None and all(count>= max_images for count in self.processed_images_per_person.values()):
                    print("Maximum number of images of {} are processed".format(name))
                    continue

    def save_encodings(self, output_file):
        print("Serializing encodings...")
        data = {"encodings": self.known_encodings, "names": self.known_names}
        with open(output_file, "wb") as f:
            pickle.dump(data, f )

