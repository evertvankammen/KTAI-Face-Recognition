import math
from collections import namedtuple
from typing import Union, Tuple
from recordclass import recordclass
import cv2
from mediapipe.python.solutions import face_detection as mp_faces

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

Embedding = recordclass('Embedding', ['nr', 'shape', 'relative_bounding_box', 'relative_key_points', 'label',
                                      'confidence', 'xy_relative_to_bbox', 'frame_number', 'file_name'])
XY = namedtuple('XY', ['x', 'y'])


def normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int,
                                    image_height: int) -> Union[None, Tuple[int, int]]:
    """
    Converts normalized coordinates to pixel coordinates within an image.

    Parameters:
    normalized_x (float): The normalized x-coordinate, ranging from 0 to 1.
    normalized_y (float): The normalized y-coordinate, ranging from 0 to 1.
    image_width (int): The width of the image in pixels.
    image_height (int): The height of the image in pixels.

    Returns:
    Union[None, Tuple[int, int]]:
        - If the normalized coordinates are valid (within the range of 0 to 1),
          returns a tuple containing the corresponding pixel coordinates (x_px, y_px).
        - If the normalized coordinates are not valid, returns None.

    """
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def draw_label(emb, frame):
    """
    Draws a label on a frame based on the provided embedding and bounding box.

    Parameters:
    emb (numpy.ndarray): The embedding containing the label information.
    frame (numpy.ndarray): The frame on which the label will be drawn.

    Returns:
    None
    """
    relative_bounding_box = emb.relative_bounding_box
    image_rows, image_cols, _ = emb.shape
    rect_start_point = normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols, image_rows)
    if rect_start_point is not None:
        cv2.putText(frame, emb.label, (MARGIN + rect_start_point[0], MARGIN + rect_start_point[1]),
                    cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)


def get_box(emb):
    """
    Returns the bounding box coordinates of the given image embedding.

    Parameters:
        emb (numpy.ndarray): The image embedding.

    Returns:
        tuple: A tuple containing the starting and ending points of the bounding box.

    Example:
        emb = numpy.zeros((224, 224, 3))
        rect_start_point, rect_end_point = get_box(emb)
    """
    image_rows, image_cols, _ = emb.shape
    relative_bounding_box = emb.relative_bounding_box
    rect_start_point = normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
        image_rows)
    rect_end_point = normalized_to_pixel_coordinates(
        relative_bounding_box.xmin + relative_bounding_box.width,
        relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
        image_rows)
    return rect_start_point, rect_end_point


def draw_box(emb, frame):
    """
    Draws a box around a specified region on the given frame.

    Parameters:
        emb (numpy.ndarray): The input image with the region of interest.
        frame (numpy.ndarray): The output image to draw the box on.

    Returns:
        None

    Example usage:
        emb = cv2.imread('input_image.jpg')
        frame = cv2.imread('output_image.jpg')
        draw_box(emb, frame)
    """
    rect_start_point, rect_end_point = get_box(emb)
    image_rows, image_cols, _ = emb.shape
    cv2.rectangle(frame, rect_start_point, rect_end_point, TEXT_COLOR, 3)


def draw_key_points(image, embedding):
    """
    Draws key points on an image.

    Args:
        image (numpy.ndarray): The input image.
        embedding (Embedding): The embedding containing the key points.

    Returns:
        None

    """
    rows, cols, _ = embedding.shape

    for keypoint in embedding.relative_key_points:
        keypoint_px = normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                      cols, rows)
        color, thickness, radius = (0, 255, 0), 2, 2
        cv2.circle(image, keypoint_px, thickness, color, radius)


def annotate(frame, embeddings):
    """Annotate the given frame with the provided embeddings.

    Parameters:
    - frame (ndarray): The frame to annotate.
    - embeddings (list): The embeddings to draw on the frame.

    Returns:
    - ndarray: The annotated frame.

    Example usage:
    frame = annotate(frame, embeddings)
    """
    if len(embeddings) == 0:
        return frame
    for emb in embeddings:
        draw_box(emb, frame)
        draw_label(emb, frame)
        draw_key_points(frame, emb)
    return frame


def find_names(embeddings, sfc, frame_number):
    """
    Find names for the given embeddings using the provided face classifier.

    Parameters:
    embeddings (list): List of embeddings.
    sfc (FaceClassifier): Face classifier object used to classify the faces.
    frame_number (int): The frame number.

    Returns:
    None

    """
    for emb in embeddings:
        name = sfc.face_k_lowest_distances(emb.xy_relative_to_bbox, 3, frame_number)
        emb.label = name


def get_relative_to_box(embeddings):
    """
    Get the relative x and y coordinates of keypoints within a bounding box.

    Parameters:
    - embeddings (list): List of embeddings containing relative bounding box and keypoints.

    Returns:
    - relative_x_ys (list): List of lists of XY objects representing the relative x and y coordinates of keypoints for each embedding.

    """
    relative_x_ys = []
    for emb in embeddings:
        x_min_prc = emb.relative_bounding_box.xmin  # percentage of frame width
        y_min_prc = emb.relative_bounding_box.ymin  # percentage of frame height

        w_bbox_prc = emb.relative_bounding_box.width  # percentage of frame width
        h_bbox_prc = emb.relative_bounding_box.height  # percentage of frame height
        emb_x_ys = []
        for p in emb.relative_key_points:
            x_px_relative = 100 * (p.x - x_min_prc) / w_bbox_prc
            y_px_relative = 100 * (p.y - y_min_prc) / h_bbox_prc
            if x_px_relative < 0:
                print("x < 0")
            if y_px_relative < 0:
                print("y < 0")
            emb_x_ys.append(XY(x=round(x_px_relative), y=round(y_px_relative)))
        relative_x_ys.append(emb_x_ys)
        emb.xy_relative_to_bbox = emb_x_ys
    return relative_x_ys


class PictureAnalyser:
    """

    The PictureAnalyser class represents a picture analyzer that can detect faces in an image, extract their embeddings, and perform various analyses on the detected faces.

    Attributes:
    - model: A string representing the model to be used for face detection.
    - min_detection_confidence: A floating-point value representing the minimum confidence threshold for face detection.

    Methods:
    - __init__(self, min_detection_confidence, model): Initializes a new instance of the PictureAnalyser class.
        - Parameters:
            - min_detection_confidence: A floating-point value representing the minimum confidence threshold for face detection.
            - model: A string representing the model to be used for face detection.
        - Returns: None

    - get_embeddings(self, frame): Extracts embeddings of the faces detected in the provided frame.
        - Parameters:
            - frame: A NumPy array representing an image frame.
        - Returns: A list of Embedding objects representing the embeddings of the detected faces.

    - analyse_frame(self, frame, sfc, frame_number): Analyzes the provided frame, performs various analyses on the detected faces, and annotates the frame with the results.
        - Parameters:
            - frame: A NumPy array representing an image frame.
            - sfc: TBD - please provide details on the meaning of this parameter.
            - frame_number: An integer representing the number of the frame.
        - Returns: A copy of the annotated frame as a NumPy array.

    """
    model = None
    min_detection_confidence = None

    def __init__(self, min_detection_confidence, model):
        self.model = model
        self.min_detection_confidence = float(min_detection_confidence)

    def get_embeddings(self, frame):
        """
            Extracts embeddings from a given frame using face detection.

            Args:
                frame (np.ndarray): The input frame.

            Returns:
                List[Embedding]: List of embeddings extracted from the frame.

            Raises:
                None

            Example:
                frame = cv2.imread('image.jpg')
                embeddings = get_embeddings(frame)
        """
        rows, cols, _ = frame.shape
        with mp_faces.FaceDetection(min_detection_confidence=self.min_detection_confidence,
                                    model_selection=self.model) as faces:
            results = faces.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            embeddings = []

            if results is None or results.detections is None or len(results.detections) == 0:
                return embeddings

            for i in range(len(results.detections)):
                detection = results.detections[i]
                relative_bounding_box = detection.location_data.relative_bounding_box  # relative to whole frame
                relative_key_points = detection.location_data.relative_keypoints  # relative to whole frame
                embeddings.append(
                    Embedding(nr=i, shape=frame.shape, relative_bounding_box=relative_bounding_box,
                              relative_key_points=relative_key_points, confidence=detection.score,
                              label=""))

            return embeddings

    def analyse_frame(self, frame, sfc, frame_number):
        """

        :param frame: The input frame that needs to be analysed.
        :type frame: numpy array

        :param sfc: The Semantic Feature Collection (SFC) that contains the names and embeddings for comparison.
        :type sfc: SemanticFeatureCollection

        :param frame_number: The number of the current frame.
        :type frame_number: int

        Analyse the given frame by performing the following steps:
        1. Get the embeddings of the frame using the get_embeddings method.
        2. Calculate the relative position of each embedding with respect to the bounding box using the get_relative_to_box method.
        3. Find the names for each embedding using the find_names method, using the Semantic Feature Collection (SFC) for comparison.
        4. Annotate the frame with the embeddings using the annotate method.

        :return: The analysed frame with annotations.
        :rtype: numpy array

        """
        rows, cols, _ = frame.shape
        embeddings = self.get_embeddings(frame)
        get_relative_to_box(embeddings)
        find_names(embeddings, sfc, frame_number)
        return annotate(frame.copy(), embeddings)
