from face_recognizer import FaceRecognizer
import os

video_file = os.path.join("..", "data", "pictures", "Friends.mp4")
encodings_file = "encodings_hog.pickle"
output_path = os.path.join("output_result_videos", "test3.avi")

face_recognizer = FaceRecognizer(video_file, encodings_file, output_path=output_path, process_every_nth_frame=5)
face_recognizer.process_video(desired_tolerance=0.53, desired_width=450, desired_model='hog', upsample_times=2)