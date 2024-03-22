import face_recognition
import pickle
import cv2
import os

def get_image_files(directory):
    image_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith(('.jpg', 'jpeg', '.png')):
                image_files.append(os.path.join(root, filename))
    return image_files

# grab the paths to the input images in our dataset
print("[INFO] extracting all images from dataset directory...")
imagePaths = get_image_files("dataset")

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []
# Track the number of images processed for each person
processed_images_per_person = {}
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] Processing image {}/{}".format(i + 1,
                                                 len(imagePaths)))
    # Getting the name of the actor from the directory: dataset/<Actor>?/..
    name = imagePath.split(os.path.sep)[-2]
    # If the person's images processed are less than 3, proceed with processing
    if name not in processed_images_per_person:
        processed_images_per_person[name] = 0
    if processed_images_per_person[name] < 3:
        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model='cnn')
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)
        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)
        processed_images_per_person[name] += 1

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings_cnn.pickle", "wb")
f.write(pickle.dumps(data))
f.close()