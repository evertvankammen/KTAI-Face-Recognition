from using_mediapipe.video.video_processor import run_face_detector, save_encodings_to_file

#save_encodings_to_file("embeddings.csv")  # enable if you want to save the encodings to a file
run_face_detector('The Ones With Chandler\'s Sarcasm _ Friends.mp4',"embeddings.csv")
