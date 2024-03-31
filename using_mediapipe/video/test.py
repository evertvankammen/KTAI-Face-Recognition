import os

import cv2

from using_mediapipe.video.face_detector import euclidean_distance
from using_mediapipe.video.picture_analyser import PictureAnalyser, get_relative_to_box
import mediapipe as mp


def test_function():
    pa = PictureAnalyser()
    img1 = cv2.imread(os.path.join("", "1.jpg"))
    img2 = cv2.imread(os.path.join("", "14.jpg"))
    image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=img1)
    image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=img2)
    embeddings1 = pa.get_embeddings(image1.numpy_view())
    embeddings2 = pa.get_embeddings(image2.numpy_view())
    get_relative_to_box(embeddings1)
    get_relative_to_box(embeddings2)
    print(euclidean_distance(embeddings1[0].xy_relative_to_bbox, embeddings2[0].xy_relative_to_bbox))
    pass


test_function()
