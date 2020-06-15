import csv
import os
import face_recognition
from keras_vggface.vggface import VGGFace
import cv2
from scipy import spatial
from utils import load_stuff, pickle_stuff, crop_face
from core import face_alignment, face_augmentation
import numpy as np
from imutils import paths
from time import time, sleep
from matplotlib import pyplot as plt
# csv results file constants
NAME_COL = 'name'
PRECISION_COL = 'precision'
FPS_COL = 'fps'


class WebCamFaceRecognition(object):
    def __new__(cls, features_files='features', model_name='resnet50', augm_on=False, align_on=False):
        if not hasattr(cls, 'instance'):
            cls.instance = super(WebCamFaceRecognition, cls).__new__(cls)
        return cls.instance

    def __init__(self, features_files='features', model_name='resnet50', augm_on=False, align_on=False):
        self.face_size = 224
        self.writer_csv = None
        self.features_files = features_files
        self.features_map = None
        self.augm_on = augm_on
        self.align_on = align_on
        if model_name == "resnet50":
            print("Loading VGG Face model...")
            self.model = VGGFace(model=model_name,
                                 include_top=True,
                                 input_shape=(224, 224, 3),
                                 pooling='avg')  # pooling: None, avg or max
            print("Loading VGG Face model done")
        else:
            self.model = None

    def load_features_map(self):
        self.features_map = load_stuff(self.features_files)

    def set_csv_writer(self, writer):
        self.writer_csv = writer

    def pre_compute_features(self, db):
        precompute_features = []
        DATASET_FOLDER = 'dataset/' + db + '/'
        i = 0
        for name in os.listdir(DATASET_FOLDER):
            userDir = os.path.join(DATASET_FOLDER, name)
            for user in os.listdir(userDir):
                # Load a sample picture and learn how to recognize it.
                data_image = face_recognition.load_image_file(os.path.join(userDir, user))

                i = i + 1
                print(str(i) + "/" + str(len(list(paths.list_files(DATASET_FOLDER)))))
                print(user)
                face_locations = np.asarray(face_recognition.face_locations(data_image))
                if len(face_locations) > 0:
                    if self.align_on:
                        faces = np.asarray(face_alignment(data_image))
                        if len(faces) > 0:
                            data_image = faces[0]
                        else:
                            data_image, cropped = crop_face(data_image, np.asarray(face_locations[0]), margin=-10, size=self.face_size)
                        face_locations = np.asarray(face_recognition.face_locations(data_image))

                    if self.augm_on:
                        faces_augm = face_augmentation(data_image, 5)
                        for img in faces_augm:
                            face_encodings = self.predict_encodings(img, face_locations)
                            if face_encodings is not None and len(face_encodings) >= 1:
                                precompute_features.append({"name": user, "features": face_encodings[0]})

                    face_encodings = self.predict_encodings(data_image, face_locations)
                    if face_encodings is not None and len(face_encodings) >= 1:
                        precompute_features.append({"name": user, "features": face_encodings[0]})
        pickle_stuff(self.features_files, precompute_features)

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

    def predict_encodings(self, img, face_locs):
        if self.model is None:
            face_encodings = face_recognition.face_encodings(img, face_locs, num_jitters=1)
        else:
            face_locations = np.asarray(face_locs)
            face_imgs = np.empty((len(face_locations), self.face_size, self.face_size, 3))
            for i, face in enumerate(face_locations):
                face_img, cropped = crop_face(img, np.asarray(face), margin=5, size=self.face_size)
                face_imgs[i, :, :, :] = face_img
            face_encodings = self.model.predict(face_imgs)
        return face_encodings

    def identify_face(self, features, threshold=0.9):
        distances = []
        for person in self.features_map:
            person_features = person.get("features")
            distance = spatial.distance.euclidean(person_features, features)
            distances.append(distance)
        min_distance_value = min(distances)
        min_distance_index = distances.index(min_distance_value)

        if min_distance_value < threshold:
            name = self.features_map[min_distance_index].get("name")
            return name, min_distance_value
        else:
            return "", 0

    def detect_face(self):

        # Load feature map
        self.load_features_map()

        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture(0)

        # init params
        face_locations = []
        face_encodings = []
        predicted_names = []
        fps = 30
        results = 0
        process_this_frame = True
        stop_program = False

        # infinite loop, break by key ESC
        while not stop_program:

            # Grab a single frame of video
            ret, frame = video_capture.read()

            if not video_capture.isOpened():
                sleep(2)

            if process_this_frame:

                start_time = time()

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_frame = frame[:, :, ::-1]

                # Face location
                face_locations = face_recognition.face_locations(rgb_frame)

                # Encodings
                face_encodings = self.predict_encodings(rgb_frame, face_locations)

                # Prediction
                predicted_names = [self.identify_face(features_face, threshold=0.45) for features_face in face_encodings]

                end_time = time()
                fps = 1.0 / (end_time - start_time)

            process_this_frame = not process_this_frame

            # draw results
            for i, face in enumerate(face_locations):
                if len(predicted_names) > 0:
                    face = np.asarray(face)
                    name = predicted_names[i][0]
                    distance = predicted_names[i][1]
                    label = "{}".format(name)
                    self.draw_label(frame, face, label, distance)
                    row = {NAME_COL: name, PRECISION_COL: distance, FPS_COL: round(fps, 2)}
                    results += 1
                    self.writer_csv.writerow(row)
                    print(row)
                    print(results)
                    if results == 1000:
                        stop_program = True

            cv2.imshow('Keras Faces', frame)
            if cv2.waitKey(5) == 27:  # ESC key press
                stop_program = True
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()


def start_face_rec_test(db="mydb", model="dlib", stat_file="results.csv", force_pre_compute=False,
                        features_files='./data/pre_compute_features.pickle', align_on=False, augm_on=False):
    face = WebCamFaceRecognition(features_files=features_files, model_name=model, align_on=align_on, augm_on=augm_on)

    # force pre compute features
    if force_pre_compute or not os.path.exists(features_files):
        print("Starting pre computing features")
        face.pre_compute_features(db)
        print("Finished pre computing features")

    # write everything to csv file
    with open(stat_file, mode='w') as res:
        fieldnames = [NAME_COL, PRECISION_COL, FPS_COL]
        writer = csv.DictWriter(res, fieldnames=fieldnames)
        face.set_csv_writer(writer)
        # start face rec
        face.detect_face()
