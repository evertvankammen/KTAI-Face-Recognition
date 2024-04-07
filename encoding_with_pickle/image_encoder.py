import face_recognition
import pickle
import cv2
import os

class ImageEncoder:
    """
        Class to encode images using face recognition and save the encodings.

        Args:
            dataset_directory (str): Path to the directory containing the dataset of images.

        Attributes:
            dataset_directory (str): Path to the dataset directory.
            known_encodings (list): List to store face encodings.
            known_names (list): List to store corresponding names of the people.
            processed_images_per_person (dict): Dictionary to keep track of processed images for each person.

        Methods:
            get_image_files: Retrieve image files from the dataset directory.
            encode_images: Encode images using face recognition.
            save_encodings: Serialize and save the face encodings and corresponding names to a file.
    """
    def __init__(self, dataset_directory):
        self.dataset_directory = dataset_directory
        self.known_encodings = []
        self.known_names = []
        self.processed_images_per_person = {}

    def get_image_files(self):
        """
                Retrieve image files from the dataset directory.

                Returns:
                    list: List of file paths for images.
        """
        image_files = []
        for root, dirs, files in os.walk(self.dataset_directory):
            for filename in files:
                if filename.lower().endswith(('.jpg', 'jpeg', 'png')):
                    image_files.append(os.path.join(root, filename))
        return image_files

    def encode_images(self, max_images=None, encode_model='hog'):
        """
            Encode images using face recognition.

            Args:
                max_images (int, optional): Maximum number of images to process per person.
                encode_model (str, optional): Face detection model to use (default is 'hog').

            Returns:
                None
        """
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
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        """
            Serialize and save the face encodings and corresponding names to a file.

            Args:
                output_file (str): Output file path for saving the encodings.

            Returns:
                None
        """
        print("Serializing encodings...")
        data = {"encodings": self.known_encodings, "names": self.known_names}
        with open(output_file, "wb") as f:
            pickle.dump(data, f )

