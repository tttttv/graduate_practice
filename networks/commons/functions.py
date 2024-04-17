import os
import base64
import numpy as np
import cv2
from networks.commons.FaceDetector import FaceDetector
from keras.preprocessing import image

def get_distance(source_representation, test_representation):
    """Расстояние между эмбеддингами"""
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    dist = source_representation - test_representation
    dist = np.sum(np.multiply(dist, dist))
    dist = np.sqrt(dist)
    return dist


def get_threshhold(model_name):

    base_threshold = 0.55

    thresholds = {
        "VGG-Face": 0.60,
        "Facenet": 10,
        "OpenFace": 0.55,
        "DeepID": 45
    }

    threshold = thresholds.get(model_name, base_threshold)

    return threshold


def loadBase64Img(uri):
    """Изображение из base64"""
    encoded_data = uri.split(",")[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr


def load_image(img):
    """
    Загружает изображение по пути, или бейс64
    """

    # если уже нампай
    if type(img).__module__ == np.__name__:
        return img, None

    # base64
    if img.startswith("data:image/"):
        return loadBase64Img(img), None

    # path
    if os.path.isfile(img) is not True:
        raise ValueError(f"Confirm that {img} exists")

    img_obj_bgr = cv2.imread(img)

    return img_obj_bgr, img


def extract_faces(img, target_size=(224, 224)):
    """Выделяет лица с фотографии"""
    detector_backend = "mtcnn"

    extracted_faces = []


    img, img_name = load_image(img)
    img_region = [0, 0, img.shape[1], img.shape[0]]

    if detector_backend == "skip":
        face_objs = [(img, img_region, 0)]
    else:
        face_detector = FaceDetector()
        face_objs = face_detector.detect_face(img)


    if len(face_objs) == 0: #Пропускаем если нет лиц на фото
        face_objs = [(img, img_region, 0)]

    for current_img, current_region, confidence in face_objs:
        if current_img.shape[0] > 0 and current_img.shape[1] > 0:

            # ресайз
            factor_0 = target_size[0] / current_img.shape[0]
            factor_1 = target_size[1] / current_img.shape[1]
            factor = min(factor_0, factor_1)

            dsize = (
                int(current_img.shape[1] * factor),
                int(current_img.shape[0] * factor),
            )
            current_img = cv2.resize(current_img, dsize)

            diff_0 = target_size[0] - current_img.shape[0]
            diff_1 = target_size[1] - current_img.shape[1]

            current_img = np.pad(
                current_img,
                (
                    (diff_0 // 2, diff_0 - diff_0 // 2),
                    (diff_1 // 2, diff_1 - diff_1 // 2),
                    (0, 0),
                ),
                "constant",
            )

            # Проверка совпадения размеров
            if current_img.shape[0:2] != target_size:
                current_img = cv2.resize(current_img, target_size)

            # нормализация
            img_pixels = image.img_to_array(current_img)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255

            region_obj = {
                "x": int(current_region[0]),
                "y": int(current_region[1]),
                "w": int(current_region[2]),
                "h": int(current_region[3]),
            }

            extracted_face = [img_pixels, region_obj, confidence]
            extracted_faces.append(extracted_face)

    return extracted_faces


def get_target_size(model_name):
    target_sizes = {
        "VGG-Face": (224, 224),
        "Facenet": (160, 160),
        "OpenFace": (100, 100),
        "DeepID": (47, 55),
    }

    target_size = target_sizes.get(model_name)

    return target_size