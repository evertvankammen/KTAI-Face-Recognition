from collections import Counter

import face_recognition
import pickle
import cv2
import random
import time
import os
import numpy

from take_box_picture import save_partial_image
from video_processor import VideoLoader

found_names_list = []
found_names_list_with_frame_number = []


class FaceRecognizer:
    """
        Class to recognize faces in a video using pre-trained face encodings.

        Args:
            video_file (str): Path to the input video file.
            encodings_file (str): Path to the file containing pre-trained face encodings.
            output_path (str, optional): Path to save the output video with recognized faces.
            show_display (bool, optional): Whether to display the processed video.
            process_every_nth_frame (int, optional): Process every nth frame of the video.

        Attributes:
            video_file (str): Path to the input video file.
            encodings_file (str): Path to the file containing pre-trained face encodings.
            output_path (str): Path to save the output video with recognized faces.
            show_display (bool): Whether to display the processed video.
            process_every_nth_frame (int): Process every nth frame of the video.
            writer: Video writer object for writing processed video.

        Methods:
            _load_encodings: Load pre-trained face encodings from the specified file.
            _resize_frame: Resize the frame to the desired width while maintaining aspect ratio.
            _name_box: Draw a rectangle and put the name label on the detected face.
            process_video: Process the input video, detect faces, and recognize them.
            process_video_threaded: Process the video in a separate thread.
    """

    def __init__(self, video_file, encodings_file,
                 output_path=None, show_display=True, process_every_nth_frame=5):
        self.video_file = video_file
        self.encodings_file = encodings_file
        self.output_path = output_path
        self.show_display = show_display
        self.process_every_nth_frame = process_every_nth_frame
        self.writer = None

    def _load_encodings(self):
        """
            Load pre-trained face encodings from the specified file.

            Returns:
                dict: Dictionary containing pre-trained face encodings and corresponding names.
        """
        print("Loading encodings...")
        data = pickle.loads(open(self.encodings_file, "rb").read())
        return data

    def _resize_frame(self, shape, desired_width):
        """
            Resize the frame to the desired width while maintaining aspect ratio.

            Args:
                shape (tuple): Shape of the input frame.
                desired_width (int): Desired width of the resized frame.

            Returns:
                tuple: Resized shape of the frame.
        """
        height, width, _ = shape
        new_height = int(height * (desired_width / width))
        return desired_width, new_height

    def _name_box(self, frame, left, top, right, bottom, name):
        """
            Draw a rectangle and put the name label on the detected face.

            Args:
                frame: Input frame.
                left (int): Left coordinate of the detected face.
                top (int): Top coordinate of the detected face.
                right (int): Right coordinate of the detected face.
                bottom (int): Bottom coordinate of the detected face.
                name (str): Name to be displayed.

            Returns:
                None
        """
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    def _save_recognition_info(self, actor_recognition_info, output_file):
        """
        Save actor recognition information to a pickle file.

        Args:
            actor_recognition_info (list): List containing actor recognition information per frame.
            output_file (str): Path to the output pickle file.

        Returns:
            None
        """
        with open(output_file, 'wb') as pickle_file:
            pickle.dump(actor_recognition_info, pickle_file)

    def process_video(self, desired_model='hog', upsample_times=2, desired_tolerance=0.6,
                      desired_width=750, desired_frame_rate=30, sample_probability=0.1, save_images=True):
        """
            Process the input video, detect faces, and recognize them.

            Args:
                desired_model (str, optional): Face detection model to use (default is 'hog').
                upsample_times (int, optional): Number of times to upsample the image (default is 2).
                desired_tolerance (float, optional): Tolerance level for face recognition (default is 0.6).
                desired_width (int, optional): Desired width of the resized frame (default is 450).
                desired_frame_rate (int, optional): Desired frame rate of the output video (default is 30).
                sample_probability (float, optional): Probability of checking a frame (default is 0.1).
                save_images (bool, optional): Whether to save the images of faces (default is False)
            Returns:
                None
        """
        experiment_directory = f"experiment_tolerance_{desired_tolerance}_internet_pictures_sample_probability_{sample_probability}"
        if not os.path.exists(experiment_directory):
            os.makedirs(experiment_directory)
        data = self._load_encodings()
        video_loader = VideoLoader(self.video_file, desired_frame_rate)
        video_loader.open_video()
        time.sleep(2.0)
        frame_count = 0
        sample_count = 0
        actor_recognition_info = []

        while video_loader.capture.isOpened():
            _, frame = video_loader.capture.read()

            if frame is None:
                break

            frame_count += 1
            print(frame_count)
            if random.random() >= sample_probability:
                continue

            sample_count += 1
            new_size = self._resize_frame(frame.shape, desired_width)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(frame, dsize=new_size, interpolation=cv2.INTER_CUBIC)

            r = frame.shape[1] / float(rgb.shape[1])

            boxes = face_recognition.face_locations(rgb, model=desired_model,
                                                    number_of_times_to_upsample=upsample_times)
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=desired_tolerance)
                name = "Unknown"

                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)

                names.append(name)
                found_names_list.append(name)
                found_names_list_with_frame_number.append((name, frame_count))

            frame_info = {'frame_number': frame_count, 'actors': names}
            actor_recognition_info.append(frame_info)

            for ((top, right, bottom, left), name) in zip(boxes, names):
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)
                self._name_box(frame, left, top, right, bottom, name)

                if save_images:
                    filename = os.path.join(experiment_directory, f"frame_{frame_count}.jpg")
                    print(f"saving image to {filename}")
                    cv2.imwrite(filename, frame)
                    # save_partial_image(frame, (top, right, bottom, left), name, experiment_directory, frame_count)


                if self.writer is None and self.output_path is not None:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    self.writer = cv2.VideoWriter(self.output_path, fourcc, 20, (frame.shape[1], frame.shape[0]), True)

                if self.writer is not None:
                    self.writer.write(frame)

            if self.show_display:
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        save_recognition = os.path.join(experiment_directory, "frames_information")
        self._save_recognition_info(actor_recognition_info, save_recognition)
        return frame_count, sample_count, found_names_list_with_frame_number, Counter(found_names_list)
