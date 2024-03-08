import os

import cv2

from picture_compare import get_image_encodings, SimpleFacerec

image_path = os.path.join("..", "data", "pictures")

sfc = SimpleFacerec()
sfc.load_encoded_images(image_path)

name, img, enc = get_image_encodings(image_path,"Joey Tribbiani.jpg")

found = sfc.detect_faces(enc)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    found = sfc.detect_faces(frame, frame)











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