# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import subprocess
import pyglet
from gtts import gTTS
#from pygame import mixer
from pydub import AudioSegment
import playsound
import random


AudioSegment.converter = "C:\ffmpeg-20191126-59d264b-win64-static\bin\ffmpeg.exe"
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, fps and
# frame count
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
#time.sleep(2.0)
fps = FPS().start()
frame_count = 0
#start = time.time()
first = True
frames = []

while True:
	frame_count += 1
    # Capture frame-by-frameq
	ret, frame = vs.read()
	frame = cv2.flip(frame,1)
	frames.append(frame)

	if frame_count == 30000000:
		break
	if ret:
		key = cv2.waitKey(1)
		if frame_count % 60 == 0:
			end = time.time()
			# grab the frame dimensions and convert it to a blob
			(H, W) = frame.shape[:2]
			# construct a blob from the input image and then perform a forward
			# pass of the YOLO object detector, giving us our bounding boxes and
			# associated probabilities
			blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
				swapRB=True, crop=False)
			net.setInput(blob)
			layerOutputs = net.forward(ln)

			# initialize our lists of detected bounding boxes, confidences, and
			# class IDs, respectively
			boxes = []
			confidences = []
			classIDs = []
			centers = []

			# loop over each of the layer outputs
			for output in layerOutputs:
				# loop over each of the detections
				for detection in output:
					# extract the class ID and confidence (i.e., probability) of
					# the current object detection
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]

					# filter out weak predictions by ensuring the detected
					# probability is greater than the minimum probability
					if confidence > 0.5:
						# scale the bounding box coordinates back relative to the
						# size of the image, keeping in mind that YOLO actually
						# returns the center (x, y)-coordinates of the bounding
						# box followed by the boxes' width and height
						box = detection[0:4] * np.array([W, H, W, H])
						(centerX, centerY, width, height) = box.astype("int")

						# use the center (x, y)-coordinates to derive the top and
						# and left corner of the bounding box
						x = int(centerX - (width / 2))
						y = int(centerY - (height / 2))

						# update our list of bounding box coordinates, confidences,
						# and class IDs
						boxes.append([x, y, int(width), int(height)])
						confidences.append(float(confidence))
						#print(classID)
						classIDs.append(classID)
						centers.append((centerX, centerY))
						cv2.rectangle(frame, (x, y), (x + int(width), y + int(height)), (255, 0, 0), thickness=5)

						#cv2.putText(frame, , (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



			# apply non-maxima suppression to suppress weak, overlapping bounding
			# boxes
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
			#cv2.putText(frame, idxs, (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 4)
			'''cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break'''

			texts = []
			sp=[]

			# ensure at least one detection exists
			if len(idxs) > 0:
				# loop over the indexes we are keeping
				for i in idxs.flatten():
					# find positions
					centerX, centerY = centers[i][0], centers[i][1]

					#divides into 3 segments for each vertical and horizontal dimensions
					# use the center (x, y)-coordinates to derive the top
					# and and left corner of the bounding box
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
					sp.append(LABELS[classIDs[i]])

					#cv2.putText(frame, LABELS[classIDs[i]], (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 4)

			for i in range(len(sp)):
				print(sp[i])
				cv2.putText(frame, sp[i], (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 4)

			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
                        #printing the output objects 
			print(texts)

                        #converting texts to speech using gTTS
			if texts:
				description = ', '.join(texts)
				my_variable = 'I see ' + description
				audio = gTTS(my_variable, lang='en')
				r1 = random.randint(1, 10000000)
				filename = "audio" + str(r1) + ".mp3"

				#filename= "audio.mp3"
				audio.save(filename)
				playsound.playsound(filename)
                            
                            
                            
#subprocess.call(["ffplay", "-nodisp", "-autoexit", "tts.mp3"])

'''                            mixer.init()
                            with open("tts.mp3", 'r'):
                                mixer.music.load("tts.mp3")
                                mixer.music.play()
                                
'''

	#fps.update()

vs.release()
os.remove("audio.mp3")
cv2.destroyAllWindows()
