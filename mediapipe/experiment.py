import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from common import visualize


IMAGE_FILE = os.path.join( "image_1 Joey Tribbiani_834.jpg")

img = cv2.imread(IMAGE_FILE)
cv2.imshow("image", img)

# STEP 1: Import the necessary modules.


# left eye, right eye, nose tip, mouth, left eye tragion, and right eye tragion.


# STEP 2: Create an FaceDetector object.
base_options = python.BaseOptions(model_asset_path='../models/detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

# STEP 4: Detect faces in the input image.
detection_result = detector.detect(image)

for x in detection_result.detections:
    print(x.bounding_box)
    for y in x.keypoints:
        print(y)
    print(x.categories)

# STEP 5: Process the detection result. In this case, visualize it.
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
cv2.imshow("rotated", annotated_image)

cv2.waitKey(0)
