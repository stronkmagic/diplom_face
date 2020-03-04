from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
import numpy as np
from keras_vggface import utils
from utils import pickle_stuff
from matplotlib import pyplot
import face_recognition
import cv2
import os
from imutils import paths


def pre_compute_features(model='vgg16', db="mydb"):

    def crop_face(imgarray, section, margin=20, size=224):
        """
        :param imgarray: full image
        :param section: face detected area (x, y, w, h)
        :param margin: add some margin to the face detected area to include a full head
        :param size: the result image resolution with be (size x size)
        :return: resized image in numpy array with shape (size x size x 3)
        """
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (top, right, bottom, left) = section

        margin = int(min(right - left, top - bottom) * margin / 100)
        x_a = left - margin
        y_a = top - margin
        x_b = right + margin
        y_b = bottom + margin
        if x_a < 0:
            x_b = min(x_b - x_a, img_w - 1)
            x_a = 0
        if y_a < 0:
            y_b = min(y_b - y_a, img_h - 1)
            y_a = 0
        if x_b > img_w:
            x_a = max(x_a - (x_b - img_w), 0)
            x_b = img_w
        if y_b > img_h:
            y_a = max(y_a - (y_b - img_h), 0)
            y_b = img_h
        cropped = imgarray[y_a: y_b, x_a: x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        resized_img = np.array(resized_img)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    # pooling: None, avg or max
    # model: vgg16, resnet50, sesnet50
    precompute_features = []
    model_obj = VGGFace(model=model, include_top=True, input_shape=(224, 224, 3), pooling='avg')
    DATASET_FOLDER = 'dataset/' + db + '/'
    i = 0
    for name in os.listdir(DATASET_FOLDER):
        userDir = os.path.join(DATASET_FOLDER, name)
        for user in os.listdir(userDir):
            # Load a sample picture and learn how to recognize it.
            imageFaceRec = face_recognition.load_image_file(os.path.join(userDir, user))

            i = i + 1
            print(str(i) + "/" + str(len(list(paths.list_files(DATASET_FOLDER)))))
            print(name)
            faces = np.asarray(face_recognition.face_locations(imageFaceRec))
            if faces.size > 0:
                img, cropped = crop_face(imageFaceRec, np.asarray(faces[0]), margin=5, size=224)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = utils.preprocess_input(x, version=2)  # or version=2
                mean_features = model_obj.predict(x)
                if mean_features is not None and len(mean_features) >= 1:
                    precompute_features.append({"name": name, "features": mean_features[0]})
    pickle_stuff("./data/"+model+"_"+db+"_features.pickle", precompute_features)
