import cv2

class VideoLoader:
    """
        Class to load and manage video files for processing.

        Args:
            video_file (str): Path to the input video file.
            frame_rate (int, optional): Desired frame rate for the video (default is None).

        Attributes:
            video_file (str): Path to the input video file.
            frame_rate (int): Desired frame rate for the video.
            capture: VideoCapture object to read frames from the video file.

        Methods:
            open_video: Open the input video file and initialize the VideoCapture object.
            close_video: Release the VideoCapture object and close the video file.
    """
    def __init__(self, video_file, frame_rate=None):
        self.video_file = video_file
        self.frame_rate = frame_rate
        self.capture = None

    def open_video(self):
        """
            Open the input video file and initialize the VideoCapture object.

            Returns:
             bool: True if video file opened successfully, False otherwise.
        """
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
        """
            Release the VideoCapture object and close the video file.

            Returns:
                None
        """
        # Release the video capture object
        if self.capture is not None:
            self.capture.release()