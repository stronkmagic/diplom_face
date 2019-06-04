import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.helper import get_face_detector
from utils.align import AlignDlib


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def read_img(image_path):
    img = cv2.imread(image_path, 1)
    img = to_rgb(img)
    return img


def img_path_to_encoding(image_path, model):
    img = read_img(image_path)
    img = align_image(img)
    if img is None:
        return
    return img_to_encoding(img, model)


def img_to_encoding(image, model):
    try:
        if image.shape != (96, 96, 3):
            image = cv2.resize(image, (96, 96))
        img = image[..., ::-1]
        img = np.around(np.transpose(img, (2, 0, 1)) / 255.0, decimals=12)
        x_train = np.array([img])
        embedding = model.predict(x_train)
        return embedding
    except:
        return None


def align_image(img):
     alignment = AlignDlib('models/landmarks_5.dat')
     return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img))


def distance(img1, db):

    min_dist = 100
    identity = None
    for (name, encoded_image_name) in db.items():
        dist = np.linalg.norm(img1 - encoded_image_name)
        print(dist)
        print(name)
        if dist < 0.15 and min_dist > dist:
            min_dist = dist
            identity = name
    return min_dist, identity
