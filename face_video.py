import face_recognition
import cv2
import numpy as np

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

protoPath = 'face_detection_model/deploy.prototxt'
modelPath = 'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel'
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

known_face_encodings = []
known_face_names = []
import os
from imutils import paths
i = 0
for name in os.listdir('dataset'):
    userDir = os.path.join('dataset', name)
    for user in os.listdir(userDir):
        # Load a sample picture and learn how to recognize it.
        image = face_recognition.load_image_file(os.path.join(userDir, user))
        i = i + 1
        print(str(i) + "/" + str(len(list(paths.list_files('dataset')))))
        print(name)
        face_encoding = face_recognition.face_encodings(image)
        if face_encoding is not None and len(face_encoding) >= 1:
            # Create arrays of known face encodings and their names
            known_face_encodings.append(face_encoding[0])
            known_face_names.append(user)
known_face_encodings = known_face_encodings
known_face_names = known_face_names

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        detector.setInput(imageBlob)
        detections = detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.9:
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)



                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        # face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        #
        face_names = []
        # for face_encoding in face_encodings:
        #     # See if the face is a match for the known face(s)
        #     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        #     name = "Unknown"
        #
        #     # # If a match was found in known_face_encodings, just use the first one.
        #     # if True in matches:
        #     #     first_match_index = matches.index(True)
        #     #     name = known_face_names[first_match_index]
        #
        #     # Or instead, use the known face with the smallest distance to the new face
        #     face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        #     best_match_index = np.argmin(face_distances)
        #     if matches[best_match_index]:
        #         name = known_face_names[best_match_index]

        #    face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left) in face_locations:
        face2 = frame[top:bottom, right:left]
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (122, 122, 122), 2)

        # # Draw a label with a name below the face
        # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        # font = cv2.FONT_HERSHEY_DUPLEX
        # cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()