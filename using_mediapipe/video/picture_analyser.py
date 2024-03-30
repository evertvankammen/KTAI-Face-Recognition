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
                                      'confidence', 'xy_relative_to_bbox'])
XY = namedtuple('XY', ['x', 'y'])


def normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int,
                                    image_height: int) -> Union[None, Tuple[int, int]]:
    # Checks if the float value is between 0 and 1.
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
    relative_bounding_box = emb.relative_bounding_box
    image_rows, image_cols, _ = emb.shape
    rect_start_point = normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols, image_rows)
    if rect_start_point is not None:
        cv2.putText(frame, emb.label, (MARGIN + rect_start_point[0], MARGIN + rect_start_point[1]),
                    cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)


def draw_box(emb, frame):
    image_rows, image_cols, _ = emb.shape
    relative_bounding_box = emb.relative_bounding_box
    rect_start_point = normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
        image_rows)
    rect_end_point = normalized_to_pixel_coordinates(
        relative_bounding_box.xmin + relative_bounding_box.width,
        relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
        image_rows)
    cv2.rectangle(frame, rect_start_point, rect_end_point, TEXT_COLOR, 3)
    return rect_start_point


def draw_key_points(image, embedding):
    rows, cols, _ = embedding.shape

    for keypoint in embedding.relative_key_points:
        keypoint_px = normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                      cols, rows)
        color, thickness, radius = (0, 255, 0), 2, 2
        cv2.circle(image, keypoint_px, thickness, color, radius)


def annotate(frame, embeddings):
    if len(embeddings) == 0:
        return frame
    for emb in embeddings:
        draw_box(emb, frame)
        draw_label(emb, frame)
        draw_key_points(frame, emb)
    return frame


def find_names(embeddings, sfc):
    for emb in embeddings:
        name = sfc.face_k_lowest_distances(emb.xy_relative_to_bbox, 3)
        emb.label = name


def get_relative_to_box(embeddings):
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
            emb_x_ys.append(XY(x=round(x_px_relative), y=round(y_px_relative)))
        relative_x_ys.append(emb_x_ys)
        emb.xy_relative_to_bbox = emb_x_ys
    return relative_x_ys


class PictureAnalyser:
    model = None
    min_detection_confidence = None

    def __init__(self, min_detection_confidence=0.75, model=('full_range_model', 1)):
        self.model = model
        self.min_detection_confidence = float(min_detection_confidence)

    def get_embeddings(self, frame):
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

    def analyse_frame(self, frame, sfc):
        rows, cols, _ = frame.shape
        embeddings = self.get_embeddings(frame)
        get_relative_to_box(embeddings)
        find_names(embeddings, sfc)
        return annotate(frame.copy(), embeddings)
