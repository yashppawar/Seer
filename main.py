import numpy as np
import time
import cv2 
import pyttsx3
import keyboard

# load the COCO class labels our YOLO model was trained on
LABELS = open("coco.names").read().strip().split("\n")

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
print("Done!")

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize
cap = cv2.VideoCapture(0)
frame_count = 0
start = time.time()
first = True
frames = []

def say(out_say):
    engine = pyttsx3.init()
    engine.say(out_say)
    engine.runAndWait()

while True:
	frame_count += 1

	if keyboard.is_pressed("q"):
		break
	
	# Capture frame-by-frame
	ret, frame = cap.read()
	frame = cv2.flip(frame, 1)
	frames.append(frame)

	if not ret: continue # if no frame, go to next iteration

	cv2.imshow('Video Feed', cv2.flip(frame, 1))
	key = cv2.waitKey(1)

	if frame_count % 100: # runs predictor once every 100 frames
		continue  # guard, if its not 100th frame, skip the loop

	end = time.time()

	(H, W) = frame.shape[:2]
	
	blob = cv2.dnn.blobFromImage(
		frame, 
		1 / 255.0, 
		(416, 416),
		swapRB = True, 
		crop = False
	)

	net.setInput(blob)
	layerOutputs = net.forward(ln)
	
	boxes = []
	confidences = []
	classIDs = []
	centers = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > 0.5:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				centers.append((centerX, centerY))

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

	texts = []

	if len(idxs) <= 0:  # no idxs so dont run the next code
		continue

	for i in idxs.flatten():
		centerX, centerY = centers[i]

		if centerX <= W/3:
			W_pos = "left "
		elif centerX <= (W/3 * 2):
			W_pos = "center "
		else:
			W_pos = "right "

		if centerY <= H/3:
			H_pos = "top "
		elif centerY <= (H/3 * 2):
			H_pos = "mid "
		else:
			H_pos = "bottom "

		texts.append(H_pos + W_pos + LABELS[classIDs[i]])


	if texts:
		description = ', '.join(texts)
		say(description)

cap.release()
cv2.destroyAllWindows()
