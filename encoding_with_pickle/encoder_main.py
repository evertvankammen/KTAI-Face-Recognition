from image_encoder import ImageEncoder
import os

image_path = os.path.join("dataset")
encode_model = 'cnn'
number_of_images = 2

pickle_output_path = os.path.join("encodings_{}_{}.pickle".format(encode_model, number_of_images))
image_encoder = ImageEncoder(image_path)

image_encoder.encode_images(encode_model=encode_model, max_images=number_of_images)
image_encoder.save_encodings(pickle_output_path)