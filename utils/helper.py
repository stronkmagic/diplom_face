import os
import tensorflow as tf
from utils import utils
from keras.utils import CustomObjectScope
from keras.models import load_model
from keras import backend as K
import json
import numpy as np
from cv2 import dnn
from utils import data_augmentation


def prepare_face_database(model):
    counter = 0
    face_database = {}
    for name in os.listdir('images'):
        for image in os.listdir(os.path.join('images', name)):

            image_path = os.path.join('images', name, image)
            if os.path.isfile(image_path):
                identity = os.path.splitext(os.path.basename(image))[0]
                face = utils.img_path_to_encoding(image_path, model)
                #face = get_embedding_image(image_path, model, identity)

                counter = counter + 1
                print(counter)
                print(identity)

                if face is not None:
                    face_database[identity] = face

    return face_database


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = K.sum(K.square(anchor - positive), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = K.sum(K.square(anchor - negative), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = pos_dist - neg_dist + alpha
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = K.sum(K.maximum(basic_loss, 0))

    return loss


def load_keras_model(model):
    with CustomObjectScope({'tf': tf}):
        model = load_model(model)
        # compile mannually
        model.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
        return model


def load_caffee_model(protxt, model):
    model = dnn.readNetFromCaffe(protxt, model)
    return model


def get_face_detector():
    return load_caffee_model('model/face_detection_model/deploy.prototxt', 'model/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel')