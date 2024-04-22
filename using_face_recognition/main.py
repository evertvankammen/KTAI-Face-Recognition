from face_recognizer import FaceRecognizer
import os

video_file = os.path.join("..", "data", "pictures", "Friends.mp4")
# video_file = os.path.join("..", "data", "movies", "The Ones With Chandler's Sarcasm _ Friends.mp4")
encodings_file = os.path.join("..", "data", "encodings", "encodings_hog.pickle")
output_path = os.path.join("output_result_videos", "test3.avi")
experiment_results= f'exp_set_from_movie_results_1500.txt'

face_recognizer = FaceRecognizer(video_file, encodings_file, output_path=output_path, process_every_nth_frame=1, show_display=False)
# face_recognizer.process_video(desired_tolerance=0.53, desired_width=800, desired_model='hog', upsample_times=1)
frames, sampled, found_names_list_with_frame_number, counted = face_recognizer.process_video(desired_tolerance=0.53,
                                                                                                 desired_width=750,
                                                                                                 desired_model='hog',
                                                                                                 upsample_times=1,
                                                                                                 sample_probability=1,
                                                                                                 save_images=True)

print(frames, sampled, counted)
with open(experiment_results, 'w') as fp:
    fp.write('sampled: ' + str(sampled) + '\n')
    fp.write(str(counted) + '\n')
    fp.write('\n'.join('%s %s' % x for x in found_names_list_with_frame_number))