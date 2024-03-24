import cv2

class VideoLoader:
    def __init__(self, video_file, frame_rate=None):
        self.video_file = video_file
        self.frame_rate = frame_rate
        self.capture = None

    def open_video(self):
        # Create a VideoCapture object and read from input file
        self.capture = cv2.VideoCapture(self.video_file)

        # Check if video file opened successfully
        if not self.capture.isOpened():
            print("Error: Could not open video file.")
            return False

        # If frame_rate is specified, set it
        if self.frame_rate is not None:
            self.capture.set(cv2.CAP_PROP_FPS, self.frame_rate)

        return True

    def close_video(self):
        # Release the video capture object
        if self.capture is not None:
            self.capture.release()

# Example usage:
# video_loader = VideoLoader("video_file.mp4")
# if video_loader.open_video():
#     # Video is opened, do something
#     # ...
#     video_loader.close_video()
