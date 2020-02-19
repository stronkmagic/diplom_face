import face_recognition
from keras_vggface.vggface import VGGFace
import numpy as np
from scipy import spatial
import cv2
from utils import load_stuff
from time import sleep, time
import csv
from .precompute_features import pre_compute_features
import os.path

# csv resutlts file constants
NAME_COL = 'name'
PRECISION_COL = 'precision'
FPS_COL = 'fps'


class FaceIdentify(object):
    """
    Singleton class for real time face identification
    """
    CASE_PATH = "data/haarcascade_frontalface_alt.xml"

    def __new__(cls, precompute_features_file=None, writer_csv=None, model_name='vgg16'):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceIdentify, cls).__new__(cls)
        return cls.instance

    def __init__(self, precompute_features_file=None, writer_csv=None, model_name='vgg16'):
        self.face_size = 224
        self.writer_csv = writer_csv
        self.precompute_features_map = load_stuff(precompute_features_file)
        print("Loading VGG Face model...")
        self.model = VGGFace(model=model_name,
                             include_top=True,
                             input_shape=(224, 224, 3),
                             pooling='avg')  # pooling: None, avg or max
        print("Loading VGG Face model done")

    @classmethod
    def draw_label(cls, image, coords, label, distance, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        (top, right, bottom, left) = coords
        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, label, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # Tolerance
        cv2.putText(image, str(round(distance, 3)), (left + 6, top + 6), font, 1.0, (0, 0, 255), 1)

    def crop_face(self, imgarray, section, margin=20, size=224):
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

    def identify_face(self, features, threshold=10000):
        distances = []
        for person in self.precompute_features_map:
            person_features = person.get("features")
            distance = spatial.distance.euclidean(person_features, features)
            distances.append(distance)
        min_distance_value = min(distances)
        min_distance_index = distances.index(min_distance_value)

        if min_distance_value < threshold:
            name = self.precompute_features_map[min_distance_index].get("name")
            return name, min_distance_value
        else:
            return "", 0

    def detect_face(self):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture(0)
        # infinite loop, break by key ESC
        while True:
            start_time = time()
            if not video_capture.isOpened():
                sleep(5)
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = np.asarray(face_recognition.face_locations(gray))

            # placeholder for cropped faces
            face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
            for i, face in enumerate(faces):
                face_img, cropped = self.crop_face(frame, np.asarray(face), margin=-15, size=self.face_size)
                face_imgs[i, :, :, :] = face_img
            if len(face_imgs) > 0:
                # generate features for each face
                features_faces = self.model.predict(face_imgs)
                predicted_names = [self.identify_face(features_face) for features_face in features_faces]
            end_time = time()
            fps = 1.0/(end_time-start_time)
            # draw results
            for i, face in enumerate(faces):
                face = np.asarray(face)
                name = predicted_names[i][0]
                precision = predicted_names[i][1]
                label = "{}".format(name)
                self.draw_label(frame, face, label, precision)
                row = {NAME_COL: name, PRECISION_COL: precision, FPS_COL: round(fps, 2)}
                print(row)
                self.writer_csv.writerow(row)

            cv2.imshow('Keras Faces', frame)
            if cv2.waitKey(5) == 27:  # ESC key press
                break
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


def face_recognition_start(model="vgg16", db="mydb", stat_file="results.csv", force_pre_compute=False,
                           pre_compute_feature_file="./data/pre_compute_features.pickle"):
    # force pre compute features
    if force_pre_compute or not os.path.exists(pre_compute_feature_file):
        print("Starting pre computing features")
        pre_compute_features(model, db)
        print("Finished pre computing features")
    # write everything to csv file
    with open(stat_file, mode='w') as res:
        fieldnames = [NAME_COL, PRECISION_COL, FPS_COL]
        writer = csv.DictWriter(res, fieldnames=fieldnames)

        # start face rec
        face = FaceIdentify(precompute_features_file=pre_compute_feature_file, writer_csv=writer,
                            model_name=model)
        face.detect_face()
