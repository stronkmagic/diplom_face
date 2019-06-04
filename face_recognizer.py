import cv2
from utils import utils
from utils import helper
from utils import stat_saver
import imutils
import numpy as np

face_detector = helper.get_face_detector()
model = helper.load_keras_model('model/nn4.small2.v1.h5')
face_database = helper.prepare_face_database(model)


video_capture = cv2.VideoCapture(0)
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	ret, frame = video_capture.read()
	frame = imutils.resize(frame, width=1000)

	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
								 (300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	face_detector.setInput(blob)
	detections = face_detector.forward()
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < 0.9:
			continue

		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# predict
		roi = frame[startX:endX, startY:endY]
		embedding = utils.img_to_encoding(roi, model)
		if embedding is not None:
			(min_dist, identity) = utils.distance(embedding, face_database)
		else:
			(min_dist, identity) = 100, 'None'

		try:
			text = "Face : " + identity[:-1] + " Dist : " + str(min_dist)
		except:
			text = "Unexpected error"

		# draw the bounding box of the face along with the associated
		# probability
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
					  (0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
video_capture.stop()
