import io
import math
import sys
from functools import wraps

from PIL import Image
import numpy as np
from networks.commons import functions
import cv2

def capture_output(func):
    """Выключает логи MTCNN"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout

    return wrapper

class FaceDetector:
    def __init__(self):
        from mtcnn import MTCNN
        self.detector = MTCNN()
        self.detector.detect_faces = capture_output(self.detector.detect_faces)

    def detect_face(self, img, align=True):
        resp = []

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(img_rgb)

        if len(detections) > 0:
            for detection in detections:
                x, y, w, h = detection["box"]
                detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]
                img_region = [x, y, w, h]
                confidence = detection["confidence"]

                if align:
                    keypoints = detection["keypoints"]
                    left_eye = keypoints["left_eye"]
                    right_eye = keypoints["right_eye"]
                    detected_face = FaceDetector.alignment_procedure(detected_face, left_eye, right_eye)

                resp.append((detected_face, img_region, confidence))

        return resp

    def alignment_procedure(img, left_eye, right_eye):
        # Выравнивание фото по координатам глаз

        left_eye_x, left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye

        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # поворот по часовой
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # против


        a = functions.get_distance(np.array(left_eye), np.array(point_3rd))
        b = functions.get_distance(np.array(right_eye), np.array(point_3rd))
        c = functions.get_distance(np.array(right_eye), np.array(left_eye))


        if b != 0 and c != 0:
            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = np.arccos(cos_a)
            angle = (angle * 180) / math.pi

            if direction == -1:
                angle = 90 - angle

            img = Image.fromarray(img)
            img = np.array(img.rotate(direction * angle))


        return img
