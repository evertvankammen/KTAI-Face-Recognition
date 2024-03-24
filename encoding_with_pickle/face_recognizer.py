import face_recognition
import pickle
import cv2
import threading
import time
from video_processor import VideoLoader

class FaceRecognizer:
    def __init__(self, video_file, encodings_file,
                 output_path=None, show_display=True, process_every_nth_frame=5):
        self.video_file = video_file
        self.encodings_file = encodings_file
        self.output_path = output_path
        self.show_display = show_display
        self.process_every_nth_frame = process_every_nth_frame
        self.writer = None

    def _load_encodings(self):
        print("Loading encodings...")
        data = pickle.loads(open(self.encodings_file, "rb").read())
        return data

    def _resize_frame(self, shape, desired_width):
        height, width, _ = shape
        new_height = int(height * (desired_width/ width))
        return desired_width, new_height

    def _name_box(selfself, frame, left, top, right, bottom, name):
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    def process_video(self, desired_model='hog', upsample_times=2, desired_tolerance=0.6, desired_width=450, desired_frame_rate=30):
        data = self._load_encodings()
        video_loader = VideoLoader(self.video_file, desired_frame_rate)
        video_loader.open_video()
        time.sleep(2.0)
        frame_count = 0

        while video_loader.capture.isOpened():
            _, frame = video_loader.capture.read()

            if frame is None:
                break

            frame_count += 1
            if frame_count % self.process_every_nth_frame != 0:
                continue

            new_size = self._resize_frame(frame.shape, desired_width)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(frame, dsize=new_size, interpolation=cv2.INTER_CUBIC)

            r = frame.shape[1] / float(rgb.shape[1])

            boxes = face_recognition.face_locations(rgb, model=desired_model, number_of_times_to_upsample=upsample_times)
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

            for ((top, right, bottom, left), name) in zip(boxes, names):
                top = int(top * r)
                right = int(right * r)
                bottom = int(bottom * r)
                left = int(left * r)
                self._name_box(frame, left, top, right, bottom, name)

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

    def process_video_threaded(self, *args, **kwargs):
        thread = threading.Thread(target=self.process_video, args=args, kwargs=kwargs)
        thread.start()
        return thread